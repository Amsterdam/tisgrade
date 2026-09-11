# Standard library
import logging
import math

# Third-party geospatial and numerical libraries
import geopandas as gpd
import numpy as np
import pyproj
import shapely

# Distance and geometry tools
from geopy.distance import geodesic
from haversine import haversine
from numpy.typing import NDArray
from scipy.stats import circmean
from shapely import GeometryType, STRtree
from shapely.geometry import LineString, Point
from shapely.ops import transform

# Clustering tools
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

# Local configuration values from .env / config module
import tisgrade_config as tsgcf


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_classes
logger = logging.getLogger(f"tisgrade.{__name__}")


# ===========================================================================
# Classes
# ===========================================================================

class ObjectAnnotation:
    """
    Represents one detected object in one panoramic image.

    An ObjectAnnotation stores:
    - the object type/code;
    - the bounding box in image pixels;
    - the camera location;
    - image metadata, such as image size, field of view and azimuth;
    - optional source IDs.

    The class can calculate:
    - the viewing direction from the camera to the object;
    - the angular object size in degrees;
    - a possible geographic line where the object may be located.

    The class uses lazy calculation:
    calculated values are cached and only recalculated when needed.
    """

    # Class-level counter used to assign a unique internal ID to each annotation.
    __next_id = 0

    def __init__(self, 
                 object_code: str, 
                 bbox: list, 
                 origin: Point, 
                 pic_size_hor_px: int, 
                 pic_size_ver_px: int, 
                 pic_field_of_view: float | int, 
                 pic_azimuth: float | int, 
                 picture_id: str = None,
                 annotation_id: str = None):
        """
        Initialize an ObjectAnnotation.

        Parameters
        ----------
        object_code:
            Code or type of the detected object.

        bbox:
            Bounding box of the object in image pixels.
            Expected format: a list of [x, y] coordinate pairs.

        origin:
            Camera location as a Shapely Point.
            Point.x is longitude. Point.y is latitude.

        pic_size_hor_px:
            Horizontal image size in pixels.

        pic_size_ver_px:
            Vertical image size in pixels.

        pic_field_of_view:
            Horizontal field of view of the image in degrees.

        pic_azimuth:
            Camera viewing direction in degrees.

        picture_id:
            Optional ID of the source picture.

        annotation_id:
            Optional ID of the source annotation.
        """
                 
        # Assign a unique internal ID to this object annotation.
        self.__id = ObjectAnnotation.__next_id
        ObjectAnnotation.__next_id += 1

        # Validate object_code.
        if not isinstance(object_code, str):
            raise TypeError("object_code must be a string.")
        self.__object_code = object_code

        # Validate bbox.
        # It must be a list with at least 3 coordinate pairs.
        # Each coordinate pair must be a list with two integers: [x, y].
        if (
            not isinstance(bbox, list) or
            len(bbox) < 3 or
            not all(isinstance(coord, list) and len(coord) == 2 for coord in bbox) or
            not all(isinstance(x, int) and isinstance(y, int) for coord in bbox for x, y in [coord])
        ):
            raise TypeError("bbox must be a list of [x, y] coordinates of atleast 3 with integers.")
        self.__bbox = bbox
        
        # Validate origin.
        # Shapely Point uses x/y coordinates.
        # In this project: x = longitude, y = latitude.
        if not isinstance(origin, Point):
            raise TypeError("origin must be a shapely.geometry.Point.")
        self.__origin = origin  # x = longitude, y = latitude  

        # Validate horizontal image size in pixels.
        if not isinstance(pic_size_hor_px, int):
            raise TypeError("pic_size_hor_px must be an integer.")
        self.__pic_size_hor_px = pic_size_hor_px

        # Validate vertical image size in pixels.
        if not isinstance(pic_size_ver_px, int):
            raise TypeError("pic_size_ver_px must be an integer.")
        self.__pic_size_ver_px = pic_size_ver_px

        # Validate image field of view in degrees.
        if not isinstance(pic_field_of_view, (int, float)):
            raise TypeError("pic_field_of_view must be a number (int or a float).")
        self.__pic_field_of_view = pic_field_of_view

        # Validate camera azimuth in degrees.
        if not isinstance(pic_azimuth, (int, float)):
            raise TypeError("pic_azimuth must be a number (int or a float).")
        self.__pic_azimuth = pic_azimuth

        # Validate optional picture ID.
        if not isinstance(picture_id, str | None):
            raise TypeError('picture_id must be a string or None')
        self.__picture_id = picture_id

        # Validate optional annotation ID.
        if not isinstance(annotation_id, str | None):
            raise TypeError('annotation_id must be a string or None')
        self.__object_annotation_id = annotation_id

        # Minimum and maximum real-world object size in metres.
        # These values are needed to calculate the possible object location line.
        self.__min_object_size_m = None
        self.__max_object_size_m = None        

        # Observer list.
        # Other objects, such as Intersection objects, can subscribe to changes.
        self.__observers = []

        # Cache state.
        # True means calculated values must be recalculated before use.
        self.__is_stale = True

        # Cached calculated direction in degrees.
        self.__direction_deg = None

        # Cached calculated angular object size in degrees.
        self.__size_deg = None

        # Cached possible object location line.
        self.__line = None


    def add_observer(self, observer):
        """
        Add an observer that must be notified when this annotation changes.

        The observer must implement an on_child_changed(...) method.
        """
        self.__observers.append(observer)


    def __notify_observers(self):
        """
        Notify all registered observers that this annotation has changed.

        This is used to mark dependent objects as stale.
        """
        for observer in self.__observers:
            observer.on_child_changed(self)  # Call observer callback.

    def __calculate_properties(self)->None:
        """
        Recalculate all cached geometric properties.

        This method updates:
        - direction in degrees;
        - angular object size in degrees;
        - possible object location line.

        It is called only when cached values are stale.
        """
        self.__is_stale = False
        self.__direction_deg = self.__calculate_direction_deg()
        self.__size_deg = self.__calculate_size_deg()
        self.__line = self.__calculate_line()



    def set_min_object_size(self, min_object_size_m: float)-> None:
        """
        Set the minimum real-world object size in metres.

        Changing this value affects the possible object location line.
        Therefore the annotation is marked as stale and observers are notified.
        """
        if not isinstance(min_object_size_m, float):
            raise TypeError("min object size must be a float.")
        self.__is_stale = True
        self.__min_object_size_m = min_object_size_m
        self.__notify_observers()
        

    def set_max_object_size(self, max_object_size_m: float)-> None:
        """
        Set the maximum real-world object size in metres.

        Changing this value affects the possible object location line.
        Therefore the annotation is marked as stale and observers are notified.
        """
        if not isinstance(max_object_size_m, float):
            raise TypeError("max object size must be a float.")
        self.__is_stale = True
        self.__max_object_size_m = max_object_size_m
        self.__notify_observers()

    def get_id(self)-> int:
        """Return the internal unique ID of this object annotation."""
        return self.__id

    def get_object_code(self)->str:
        """Return the object code."""
        return self.__object_code

    def get_picture_id(self)-> str | None:
        """Return the source picture ID, if available."""
        return self.__picture_id

    def get_object_annotation_id(self)-> str | None:
        """Return the source annotation ID, if available."""
        return self.__object_annotation_id

    def print(self)-> None:
        """
        Print a short debug representation of this object annotation.

        This includes the internal ID, object code, bbox, origin and calculated line.
        """
        print(f'object_id: {self.__id}, object_code: {self.__object_code} bbox: {self.__bbox}, origin: {self.__origin}, line:{self.get_line()}')

    def object_x_min_px(self)-> int:
        """Return the minimum x-coordinate of the bbox in pixels."""
        return min(p[0] for p in self.__bbox)
    
    def object_x_max_px(self)-> int:
        """Return the maximum x-coordinate of the bbox in pixels."""
        return max(p[0] for p in self.__bbox)

    def object_y_min_px(self)-> int:
        """Return the minimum y-coordinate of the bbox in pixels."""
        return min(p[1] for p in self.__bbox)
    
    def object_y_max_px(self)-> int:
        """Return the maximum y-coordinate of the bbox in pixels."""
        return max(p[1] for p in self.__bbox)

    def x_center_px(self) -> float:
        """Return the horizontal centre pixel of a sign, accounting for wrap-around.
    
        Panoramic images wrap horizontally. When the two edge pixels of a sign
        straddle the 0°/360° seam, a naive average would place the centre on the
        wrong side of the image. This function detects that case and corrects for
        it.

        The method uses the minimum and maximum x pixel values of the bbox.
    
        Returns
        -------
        float
            Horizontal centre pixel in the image range.
        """
        if abs(self.object_x_max_px() - self.object_x_min_px()) <= self.__pic_size_hor_px / 2:
            # Normal case: the bbox does not cross the panorama seam.
            x_loc_px = (self.object_x_min_px() + self.object_x_max_px()) / 2
        else:
            # Seam case: the bbox crosses the horizontal image boundary.
            # Shift by half the image width to find the wrapped centre.
            x_loc_px = (self.object_x_min_px() + self.object_x_max_px()) / 2 - self.__pic_size_hor_px/ 2
    
        # Normalize the centre pixel to the image range.
        return x_loc_px % (self.__pic_size_hor_px - 1)


    def __calculate_direction_deg(self)-> float:

        """Convert the horizontal object position to a compass bearing.
    
        The horizontal centre pixel of the object is compared with the centre of
        the image. The pixel offset is converted to degrees using the image field
        of view. The camera azimuth is then added to get the absolute compass
        bearing.
    
        Returns
        -------
        float
            Absolute compass bearing in degrees, in the range [0, 360).
        """
        # Pixel distance from the image centre.
        delta_px = self.x_center_px() - (0.5 * self.__pic_size_hor_px) + 1

        # Convert pixel offset to angular offset.
        delta_deg = (delta_px / self.__pic_size_hor_px) * self.__pic_field_of_view

        # Add camera azimuth and normalize to [0, 360).
        self.__direction_deg = (delta_deg + self.__pic_azimuth) % 360
        return self.__direction_deg

    def direction_deg(self)-> float:
        """
        Return the calculated compass direction in degrees.

        Recalculates cached values first if they are stale.
        """
        if self.__is_stale:
            self.__calculate_properties()
        return self.__direction_deg


    def __calculate_size_deg(self) -> float:
        """Return the angular object size in degrees.
    
        The current implementation uses the vertical bbox size in pixels and
        converts it to degrees using the horizontal image size and field of view.

        Returns
        -------
        float
            Object size in degrees.
        """
        
        # Check whether this is still needed now that the data comes directly
        # from Panoramax. It may be possible to use the original pixel bounds
        # directly.
        size = self.object_y_max_px() - self.object_y_min_px()

        # If the size would cross the image seam, use the shorter span.
        if size > self.__pic_size_hor_px / 2:
            size = self.__pic_size_hor_px - size

        # Convert pixel size to angular size.
        size_deg = size/self.__pic_size_hor_px * self.__pic_field_of_view

        return size_deg

    def size_deg(self) -> float:
        """
        Return the calculated angular object size in degrees.

        Recalculates cached values first if they are stale.
        """
        if self.__is_stale:
            self.__calculate_properties()        
        return self.__size_deg

    def __calculate_line(self) -> LineString | None:
        """
        Calculate the line representing the object's possible locations.

        The line connects:
        - the estimated location based on the minimum object size;
        - the estimated location based on the maximum object size.

        If minimum or maximum object size is missing, no line can be calculated.

        Returns
        -------
        LineString | None
            A LineString between the minimum and maximum distance points,
            or None if min/max object sizes are not set.
        """
        if self.__min_object_size_m is None or self.__max_object_size_m is None:
            return None
        
        # Estimate distance for the minimum and maximum physical object size.
        distance_min = self.calculate_object_distance(self.__min_object_size_m)
        distance_max = self.calculate_object_distance(self.__max_object_size_m)

        # Convert both distances to geographic endpoints and connect them.
        return LineString([self.calculate_endpoint(distance_min), self.calculate_endpoint(distance_max)])

    def get_line(self) -> LineString | None:
        """
        Get the line representing the object's possible locations.

        Recalculates cached values first if they are stale.

        Returns
        -------
        LineString | None
            A LineString connecting the min and max distance points from the origin,
            or None if min/max object sizes are not set.
        """
        if self.__is_stale:
            self.__calculate_properties()        
        return self.__line

    
    # Now that the data comes directly from Panoramax, the field of view can be
    # used directly in the distance calculation.
    def calculate_object_distance(self,
        object_size_m: float
    ) -> float:
        """Estimate the camera-to-object distance from known object size.
    
        The calculation uses the angular size of the object in the image.

        A larger angular size means the object is closer to the camera.
        A smaller angular size means the object is farther from the camera.
    
        Parameters
        ----------
        object_size_m:
            Physical size of the object in metres.
    
        Returns
        -------
        float
            Estimated distance in metres.
        """
        return object_size_m / (2 * math.pi) * (360 / self.size_deg())    

    def calculate_endpoint(
        self, distance_meters: float
    ) -> Point:
        """Compute the endpoint of a bearing line.
    
        The endpoint is calculated from:
        - the camera origin;
        - the calculated object direction;
        - the given distance in metres.
    
        Parameters
        ----------
        distance_meters:
            Length of the bearing line in metres.
    
        Returns
        -------
        Point
            Endpoint as a Shapely Point.
            Point.x = longitude. Point.y = latitude.
        """
        # Shapely uses Point(longitude, latitude).
        # geopy expects coordinates as (latitude, longitude).
        start_latlon = (self.__origin.y, self.__origin.x) #origin = (long (x), lat (y)) geodesic expets (Lat, long)

        # Move from the origin along the calculated compass bearing.
        dest = geodesic(meters=distance_meters).destination(start_latlon, self.direction_deg())

        # Convert back to a Shapely Point.
        return Point(dest.longitude, dest.latitude)


    def object_size_m(self, object_location: Point) -> float:
        """Estimate the real-world object size at a given location.
    
        The method calculates the distance between the camera and the given
        object location. It then uses the angular object size to estimate the
        physical object size in metres.
    
        Parameters
        ----------
        object_location:
            Target object location as a Shapely Point.
            Point.x = longitude. Point.y = latitude.
    
        Returns
        -------
        float
            Estimated object size in metres.
        """ 
        if not isinstance(object_location, Point):
            raise TypeError("objct_location must be an Point.")

        # Distance from the camera to the given object location.
        distance = self.distance_m(object_location)

        # Angular object size in degrees.
        size_deg = self.size_deg()

        # Convert angular size at this distance to physical size.
        return 2 * math.pi * distance * size_deg/360


    def distance_m(self, object_location: Point) -> float:
        """Return the geodesic distance between the camera and an object location.
    
        Parameters
        ----------
        object_location:
            Target location as a Shapely Point.
            Point.x = longitude. Point.y = latitude.
    
        Returns
        -------
        float
            Distance in metres.
        """
        if not isinstance(object_location, Point):
            raise TypeError("object_location must be an Point.")


        # Read coordinates from Shapely Points.
        # Shapely uses x = longitude, y = latitude.
        origin_lon, origin_lat = self.__origin.x, self.__origin.y
        object_lon, object_lat = object_location.x, object_location.y

        # geopy expects coordinates as (latitude, longitude).
        return geodesic(
            (origin_lat, origin_lon), (object_lat, object_lon)
        ).meters

    def get_origin(self) -> Point:
        """Return the camera origin as a Shapely Point."""
        return self.__origin


    def has_line(self) -> bool:
        """
        Check whether this annotation can generate a location line.

        A line can only be generated when both minimum and maximum object sizes
        are available.
        """
        return self.__min_object_size_m is not None and self.__max_object_size_m is not None
    

    def min_distance_to_line_m(self, point: Point)->float:
        """
        Calculate the minimum geodesic distance from a point to this object's line.

        The method:
        1. Gets the calculated possible object location line.
        2. Loops through all segments of that line.
        3. Finds the nearest point on each segment using Shapely.
        4. Measures the geodesic distance to that nearest point.
        5. Returns the smallest distance.

        Parameters
        ----------
        point:
            Point to compare with this object's line.

        Returns
        -------
        float
            Minimum distance in metres.
        """

        if not isinstance(point, Point):
            raise TypeError("point must be an Point.")    

        # Get the possible object location line.
        line = self.get_line()

        # Distance cannot be calculated when no line is available.
        if line is None:
            raise ValueError("LineString is None, cannot calculate distance.")            
        
        # Start with infinity so every real distance will be smaller.
        min_dist = float('inf')

        # Coordinates are expected as [(lon0, lat0), (lon1, lat1), ...].
        coords = list(line.coords)  # e.g. [(lon0,lat0), (lon1,lat1), ...]

        # Check each segment of the line.
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])

            # Find the nearest point on this segment.
            nearest = seg.interpolate(seg.project(point))   # closest point on segment

            # Calculate geodesic distance in metres.
            # geopy expects coordinates as (latitude, longitude).
            dist = geodesic((point.y, point.x), (nearest.y, nearest.x)).meters

            # Keep the smallest distance found.
            min_dist = min(min_dist, dist)

        return min_dist


class Intersection:
    """
    Represents the possible intersection between two ObjectAnnotation lines.

    Each ObjectAnnotation can produce a LineString that represents the possible
    real-world location of the observed object. When two of these lines cross,
    their crossing point is a possible object location.

    This class stores:
    - the two ObjectAnnotation objects;
    - the calculated intersection point;
    - the estimated object size at that point;
    - a score for how well both size estimates match;
    - the angle between the two observation lines.

    The class uses lazy calculation:
    values are only calculated when requested. If one of the child annotations
    changes, this object is marked as stale and recalculated on the next request.
    """

    # Class-level counter used to assign a unique internal ID to each intersection.
    __next_id = 0
    
    def __init__(self, annotation1: ObjectAnnotation, annotation2: ObjectAnnotation):
        """
        Initialize an Intersection object with two ObjectAnnotation instances.

        Parameters
        ----------
        annotation1:
            First ObjectAnnotation instance.

        annotation2:
            Second ObjectAnnotation instance.

        Raises
        ------
        TypeError
            If either input is not an ObjectAnnotation.
        """

        # Validate first annotation.
        if not isinstance(annotation1, ObjectAnnotation):
            raise TypeError("annotation1 must be an ObjectAnnotation instance")

        # Validate second annotation.
        if not isinstance(annotation2, ObjectAnnotation):
            raise TypeError("annotation2 must be an ObjectAnnotation instance")

        # Store the first annotation and register this Intersection as observer.
        # If the annotation changes, this Intersection must recalculate its cache.
        self.__annotation1 = annotation1
        self.__annotation1.add_observer(self)

        # Store the second annotation and register this Intersection as observer.
        self.__annotation2 = annotation2
        self.__annotation2.add_observer(self)

        # Assign a unique internal ID.
        self.__id = Intersection.__next_id

        # Cache state.
        # True means the calculated values must be recalculated before use.
        self.__is_stale = True

        # Cached calculated values.
        self.__intersection = None
        self.__object_size = None
        self.__object_size_score = None
        self.__angle_between_lines = None

        # Increase the class-level counter for the next Intersection.
        Intersection.__next_id += 1

    def on_child_changed(self, changed_child):
        """
        Callback that is called when one of the child annotations changes.

        The changed_child argument is currently not used, but it can be useful
        later if different handling is needed for different child objects.
        """

        # Mark cached values as stale.
        # They will be recalculated the next time they are requested.
        self.__is_stale = True

    def get_id(self) -> int | None:
        """Return the internal unique ID of this intersection."""
        return self.__id

    def print(self):
        """
        Print a short debug representation of this intersection.

        This includes:
        - the intersection point;
        - the estimated object size;
        - the angle between the two lines;
        - the IDs of both source annotations.
        """
        print(f'intersection point: {self.intersection()}, size: {self.object_size()}, hoek: {self.angle_between_lines()}, id anno 1: {self.__annotation1.get_id()}, id anno 2: {self.__annotation2.get_id()}')

    def __calculate_properties(self) -> None:
        """
        Calculate and cache all intersection properties.

        The method:
        1. Gets the possible location lines from both annotations.
        2. Resets all cached calculated values.
        3. Stops if one or both lines are missing.
        4. Calculates the geometric intersection of both lines.
        5. If the result is a Point, stores it as the intersection.
        6. Estimates the object size from both annotations at that point.
        7. Calculates a size agreement score.
        8. Calculates the smallest angle between both observation directions.

        Returns
        -------
        None
            The calculated values are stored on the instance.
        """

        # Only recalculate if the cached values are stale.
        if self.__is_stale:
            self.__is_stale = False

            # Get the possible object location lines from both annotations.
            line1 = self.__annotation1.get_line()
            line2 = self.__annotation2.get_line()

            # Reset cached values before recalculation.
            # This prevents old values from remaining after invalid input.
            self.__intersection = None
            self.__object_size = None
            self.__object_size_score = None
            self.__angle_between_lines = None

            # If one or both lines are missing, no intersection can be calculated.
            if line1 is None or line2 is None:
                return None

            # Calculate the geometric intersection of both lines.
            intersection = line1.intersection(line2)

            # Only accept real point intersections.
            # Other results, such as empty geometry or overlapping line parts,
            # are ignored here.
            if isinstance(intersection, Point):
                self.__intersection = intersection

                # Estimate object size from both observations at the same point.
                size_1 = self.__annotation1.object_size_m(self.__intersection)
                size_2 = self.__annotation2.object_size_m(self.__intersection)

                # Store the average estimated object size.
                self.__object_size = (size_1 + size_2)/2

                # Store a score that describes how similar both size estimates are.
                self.__object_size_score =  object_size_score(size_1, size_2)

                # Calculate the smallest angle between the two viewing directions.
                a1 = self.__annotation1.direction_deg()
                a2 = self.__annotation2.direction_deg()
                diff = abs(a1 - a2)
                self.__angle_between_lines = min(diff, 360 - diff)

            return None
            

    def intersection(self) -> Point | None:
        """
        Return the intersection point of the two annotation lines.

        Recalculates cached values first if they are stale.

        Returns
        -------
        Point | None
            The intersection Point if both lines intersect as a point.
            None if no valid point intersection exists.
        """
        if self.__is_stale:
            self.__calculate_properties()

        return self.__intersection

    def object_size(self) -> float | None:
        """
        Return the estimated object size at the intersection point.

        The value is the average of the object size estimates from both
        ObjectAnnotation instances.

        Recalculates cached values first if they are stale.

        Returns
        -------
        float | None
            Estimated object size in metres, or None if no valid intersection exists.
        """

        if self.__is_stale:
            self.__calculate_properties()

        return self.__object_size            

    def object_size_score(self)  -> float:
        """
        Return the score for how well both object size estimates match.

        A higher score means the two annotations estimate a more similar object size.

        Recalculates cached values first if they are stale.

        Returns
        -------
        float
            Object size similarity score.
        """
        if self.__is_stale:
            self.__calculate_properties()

        return self.__object_size_score

    def angle_between_lines(self) -> float:
        """
        Return the smallest angle between the two annotation directions.

        The angle is calculated from the compass bearings of both annotations.
        The result is always the smaller angle, in the range 0 to 180 degrees.

        Recalculates cached values first if they are stale.

        Returns
        -------
        float
            Unsigned angular difference in degrees, in the range [0, 180].
        """
        if self.__is_stale:
            self.__calculate_properties()

        return self.__angle_between_lines

    def get_annotation1(self) -> ObjectAnnotation:
        """Return the first ObjectAnnotation."""
        return self.__annotation1

    def get_annotation2(self) -> ObjectAnnotation:
        """Return the second ObjectAnnotation."""
        return self.__annotation2

    @staticmethod
    def __local_aeqd_crs(lon_lat_coords: np.ndarray) -> pyproj.CRS:
        """
        Build a local azimuthal-equidistant CRS for a set of lon/lat coordinates.

        The CRS is centered on the bounding box of the input coordinates.
        This allows distance and tiling calculations to be done in metres.

        The method is antimeridian-safe:
        if the data crosses the 180/-180 longitude boundary, longitudes are
        temporarily unwrapped before calculating the center.

        Parameters
        ----------
        lon_lat_coords:
            NumPy array with coordinates in longitude/latitude order.

        Returns
        -------
        pyproj.CRS
            Local azimuthal-equidistant coordinate reference system.
        """

        # Split coordinates into longitude and latitude arrays.
        lons, lats = lon_lat_coords[:, 0], lon_lat_coords[:, 1]

        # If the longitude range is larger than 180 degrees, the area likely
        # crosses the antimeridian. Shift negative longitudes into the 0-360 range.
        if lons.max() - lons.min() > 180:
            lons = np.where(lons < 0, lons + 360, lons)

        # Calculate center longitude and wrap it back to the [-180, 180] range.
        lon0 = (lons.min() + lons.max()) / 2.0
        lon0 = ((lon0 + 180) % 360) - 180  # wrap back to [-180, 180]

        # Calculate center latitude.
        lat0 = (lats.min() + lats.max()) / 2.0

        # Build a local metre-based CRS centered on the data.
        return pyproj.CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )

    @staticmethod
    def __project_lines(lines: np.ndarray, transformer: pyproj.Transformer) -> np.ndarray:
        """
        Reproject an array of Shapely LineStrings.

        The input lines are transformed with a vectorized pyproj call.
        This avoids looping over every geometry manually.

        Parameters
        ----------
        lines:
            NumPy array of Shapely LineString geometries.

        transformer:
            pyproj Transformer used to convert coordinates.

        Returns
        -------
        np.ndarray
            Reprojected LineString geometries.
        """

        def _xy(coords: np.ndarray) -> np.ndarray:
            """
            Transform coordinate arrays from source CRS to target CRS.

            Parameters
            ----------
            coords:
                Coordinate array where column 0 is x/lon and column 1 is y/lat.

            Returns
            -------
            np.ndarray
                Transformed coordinates as two columns: x and y.
            """

            # Transform all coordinates in one vectorized call.
            x, y = transformer.transform(coords[:, 0], coords[:, 1])

            # Return transformed coordinates in the format expected by Shapely.
            return np.column_stack([x, y])

        return shapely.transform(lines, _xy)

    @staticmethod
    def __crossing_pairs(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find pairs of lines that cross each other.

        The method uses:
        - STRtree to quickly find candidate line pairs;
        - Shapely vectorized geometry operations to keep only true crossings;
        - a mask to remove duplicate symmetric pairs.

        Parameters
        ----------
        lines:
            NumPy array of Shapely LineString geometries.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two arrays with matching indices.
            Each pair i[k], j[k] represents two lines that cross.
        """

        # Build a spatial index for fast candidate lookup.
        tree = STRtree(lines)

        # Find candidate line pairs that cross.
        i, j = tree.query(lines, predicate="crosses")

        # Remove duplicate symmetric pairs.
        # For example, keep (2, 5), but remove (5, 2).
        mask = j > i
        i, j = i[mask], j[mask]

        # If no candidate pairs remain, return empty arrays.
        if len(i) == 0:
            return i, j

        # Calculate the actual intersection geometries for candidate pairs.
        pts = shapely.intersection(lines[i], lines[j])

        # Keep only intersections that are points.
        is_point = shapely.get_type_id(pts) == GeometryType.POINT

        return i[is_point], j[is_point]

    @staticmethod
    def __process_tile(
        tile_lines: np.ndarray,
        tile_annotations: list,
        core_minx: float,
        core_miny: float,
        core_maxx: float,
        core_maxy: float,
    ) -> list[tuple]:
        """
        Process one tile and return annotation pairs whose lines cross inside it.

        The tile contains a buffered selection of lines. The buffer prevents
        missing intersections near tile borders. To avoid double counting, only
        intersections inside the unbuffered tile core are kept.

        This method is written as a pure function:
        it does not depend on shared state outside its parameters. That makes it
        easier to run in parallel later, for example with ProcessPoolExecutor or Dask.

        Parameters
        ----------
        tile_lines:
            Lines selected for this tile.

        tile_annotations:
            ObjectAnnotation objects matching tile_lines by index.

        core_minx, core_miny, core_maxx, core_maxy:
            Bounds of the unbuffered tile core in projected metres.

        Returns
        -------
        list[tuple]
            List of annotation pairs whose projected lines cross inside the tile core.
        """

        # Find crossing line index pairs within this tile.
        i, j = Intersection.__crossing_pairs(tile_lines)

        # No crossings in this tile.
        if len(i) == 0:
            return []

        # Calculate the actual intersection points.
        pts = shapely.intersection(tile_lines[i], tile_lines[j])
        xs, ys = shapely.get_x(pts), shapely.get_y(pts)

        # Only keep intersections whose point falls in this tile's unbuffered core.
        # This prevents double-counting the same intersection from overlapping
        # buffered tiles.
        in_core = (
            (xs >= core_minx) & (xs < core_maxx) &
            (ys >= core_miny) & (ys < core_maxy)
        )
        i, j = i[in_core], j[in_core]

        # Convert line index pairs back to ObjectAnnotation pairs.
        return [(tile_annotations[a], tile_annotations[b]) for a, b in zip(i, j)]

    @staticmethod
    def find_intersections(annotations: list["ObjectAnnotation"]) -> list["Intersection"]:
        """
        Find all pairwise crossing points between ObjectAnnotation bearing lines.

        The input annotation lines are stored in longitude/latitude coordinates
        using EPSG:4326. For tiling and buffer calculations, the lines are first
        projected into a local azimuthal-equidistant CRS. This local CRS uses
        metres as units.

        The method:
        1. Filters annotations to keep only valid LineStrings.
        2. Projects the valid lines to a local metre-based CRS.
        3. Splits the projected area into tiles.
        4. Selects lines per tile, with a buffer around the tile.
        5. Finds crossing line pairs inside each tile.
        6. Keeps only intersections inside the unbuffered tile core.
        7. Creates Intersection objects for the resulting annotation pairs.

        The tile approach helps limit memory use and avoids checking every line
        against every other line for large datasets.

        Notes
        -----
        The local projection approach is suitable for areas of limited size.
        The original comment assumes roughly up to 100 km x 100 km and latitude
        between about -60 and 80 degrees.

        Returns
        -------
        list[Intersection]
            List of Intersection objects for all detected crossing annotation pairs.
        """

        # Tile size in metres.
        # Larger tiles mean fewer tiles, but more memory use per tile.
        TILE_SIZE = 2000.0

        # Buffer size in metres.
        # The buffer should be at least the maximum line length to avoid missing
        # intersections near tile boundaries.
        BUFFER = 200.0

        # No annotations means no intersections.
        if not annotations:
            return []

        # Keep only annotations that have a valid LineString.
        valid_lines = []
        valid_annotations = []
        for annotation in annotations:
            line = annotation.get_line()
            if line is not None and line.is_valid and line.geom_type == "LineString":
                valid_lines.append(line)
                valid_annotations.append(annotation)

        # No valid lines means no intersections.
        if not valid_lines:
            return []

        # Convert list of valid lines to a NumPy object array for vectorized Shapely operations.
        lines_arr = np.array(valid_lines, dtype=object)

        # --- Reproject lon/lat to local metre-based coordinates -----------------

        # Get all coordinates from all lines.
        # Coordinates are in lon/lat order.
        all_coords = shapely.get_coordinates(lines_arr)  # (N, 2) lon, lat

        # Build a local CRS centered on the data extent.
        local_crs = Intersection.__local_aeqd_crs(all_coords)

        # Create transformer from WGS84 lon/lat to the local CRS.
        transformer = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)

        # Project all lines to local metre-based coordinates.
        projected_lines = Intersection.__project_lines(lines_arr, transformer)

        # --- Tile the projected extent -----------------------------------------

        # Get bounds for every projected line.
        # Shape: (N, 4), with columns minx, miny, maxx, maxy.
        bounds = shapely.bounds(projected_lines)  # (N, 4): minx, miny, maxx, maxy

        # Overall extent of all projected lines.
        minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
        maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

        # Collected Intersection objects.
        intersections = []

        # Loop over tiles from left to right.
        x = minx
        while x < maxx:

            # Loop over tiles from bottom to top.
            y = miny
            while y < maxy:

                # Bounds of the unbuffered tile core.
                core_minx, core_miny = x, y
                core_maxx, core_maxy = x + TILE_SIZE, y + TILE_SIZE

                # Select all lines whose bounds intersect the buffered tile.
                # This includes lines just outside the tile core, so crossings
                # near tile boundaries are not missed.
                sel = (
                    (bounds[:, 2] >= core_minx - BUFFER) & (bounds[:, 0] <= core_maxx + BUFFER) &
                    (bounds[:, 3] >= core_miny - BUFFER) & (bounds[:, 1] <= core_maxy + BUFFER)
                )

                # Indices of selected lines.
                idx = np.where(sel)[0]

                # At least two lines are needed to form an intersection.
                if len(idx) > 1:
                    # Find crossing annotation pairs in this tile.
                    pairs = Intersection.__process_tile(
                        projected_lines[idx],
                        [valid_annotations[k] for k in idx],
                        core_minx, core_miny, core_maxx, core_maxy,
                    )

                    # Convert each annotation pair to an Intersection object.
                    for a, b in pairs:
                        intersections.append(Intersection(a, b))

                # Move to the next tile in y direction.
                y += TILE_SIZE

            # Move to the next tile in x direction.
            x += TILE_SIZE

        return intersections


class Cluster:
    """
    Represents a group of valid Intersection objects.

    A cluster is used as a candidate real-world object location. It contains
    multiple line intersections that are close to each other and may point to
    the same physical object.

    The class can:
    - store and remove intersections;
    - return the coordinates of all valid intersections;
    - calculate the cluster centre;
    - calculate the maximum distance from intersections to the centre;
    - calculate a cluster quality score;
    - recursively split clusters with DBSCAN.
    """

    # Class-level counter used to assign a unique internal ID to each cluster.
    __next_id = 0

    # Earth radius is now read from the configuration file.
    # __earth_radius_m = 6_371_000

    def __init__(self)->None:
        """
        Initialize an empty Cluster.

        Each Cluster gets a unique internal ID.
        Intersections are stored by their intersection ID.
        """

        # Assign a unique internal cluster ID.
        self.__id = Cluster.__next_id
        Cluster.__next_id += 1

        # Store intersections by intersection ID.
        # Note: this is used as a dictionary, with:
        # key   = intersection ID
        # value = Intersection object
        self.__intersections: list[Intersection] = {}


    def get_id(self)->None:
        """Return the internal unique ID of this cluster."""
        return self.__id

    def add_intersection(self, intersection: Intersection) -> None:
        """
        Add a valid Intersection to this cluster.

        Parameters
        ----------
        intersection:
            Intersection object to add.

        Raises
        ------
        TypeError
            If the input is not an Intersection.
            If the Intersection does not contain a valid Point.
        """

        # Validate input type.
        if not isinstance(intersection, Intersection):
            raise TypeError("Expected an Intersection object")

        # Validate that the Intersection has a valid point geometry.
        if not isinstance(intersection.intersection(), Point):
            raise TypeError("Not a valid intersection")

        # Use the intersection ID as dictionary key.
        intersection_id = intersection.get_id()

        # Store the Intersection in the cluster.
        intersection_id = intersection.get_id()
        self.__intersections[intersection_id] = intersection

    def get_intersections(self) -> list[Intersection]:
        """
        Return all intersections with valid coordinate data.

        Returns
        -------
        list[Intersection]
            List of Intersection objects where intersection.intersection()
            is not None. Returns an empty list if no valid intersections exist.
        """

        # Only return intersections that have a valid intersection point.
        return [intersection
                for intersection in self.__intersections.values()
                if intersection.intersection() is not None]

    def remove_intersection(self, intersection: Intersection) -> bool:
        """
        Remove an Intersection from this cluster.

        Parameters
        ----------
        intersection:
            Intersection object to remove.

        Returns
        -------
        bool
            True if the intersection was found and removed.
            False if the intersection was not present in the cluster.
        """

        # Get the ID used as key in the internal dictionary.
        intersection_id = intersection.get_id()

        # Remove the intersection if it exists.
        if intersection_id in self.__intersections:
            del self.__intersections[intersection_id]
            return True

        return False

    def amount_intersections(self) -> int:
        """
        Count the number of valid intersections in this cluster.

        Returns
        -------
        int
            Number of intersections with a valid intersection point.
        """

        # Count only intersections with a valid point.
        return sum(1 for intersection in self.__intersections.values()
                if intersection.intersection() is not None)

    def get_coordinates(self) -> NDArray[np.float64]:
        """
        Return coordinates of valid intersections.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n, 2), containing coordinates in decimal degrees.

            The code returns:
            - column 0: x-coordinate, longitude;
            - column 1: y-coordinate, latitude.

            Returns an empty array if no valid intersections exist.
        """

        # Extract coordinates from valid intersection points.
        # Shapely uses Point.x = longitude and Point.y = latitude.
        return np.array([(intersection.intersection().x, intersection.intersection().y) 
                                for intersection_id, intersection in self.__intersections.items()
                                if intersection.intersection() is not None]) 

    def get_center(self) -> Point|None:
        """
        Calculate the geometric centre of all valid intersection points.

        The centre is calculated as the average longitude and average latitude.

        Returns
        -------
        Point | None
            Cluster centre as a Shapely Point.
            Point.x = longitude. Point.y = latitude.
            Returns None if the cluster has no valid intersections.
        """

        # Get all valid intersection coordinates.
        coordinates = self.get_coordinates()

        # No coordinates means no centre can be calculated.
        if len(coordinates) == 0:
            return None

        # Average all x-values and y-values.
        avg_x = np.mean(coordinates[:, 0])  # average x-value, longitude
        avg_y = np.mean(coordinates[:, 1])  # average y-value, latitude

        # Return centre as Point(longitude, latitude).
        return Point(avg_x, avg_y)   # Point(longitude, latitude)


    def get_coordinates_rad(self):
        """
        Return valid intersection coordinates in radians.

        This is needed for haversine distance calculations.

        Returns
        -------
        np.ndarray
            Coordinates converted from degrees to radians.
        """

        return np.radians(self.get_coordinates())
         

    def get_max_distance_intersection_center_rad(self) -> float:
        """
        Return the maximum haversine distance from an intersection to the cluster centre.

        The value returned by this method is in radians.
        To get the distance in metres, use get_max_distance_intersection_center_m().

        Returns
        -------
        float
            Maximum distance in radians.
            Returns 0.0 if no valid intersections exist.
        """

        # Get coordinates in radians for haversine calculation.
        coordinates_rad = self.get_coordinates_rad()

        # Return 0 if no coordinates are available.
        if coordinates_rad.size == 0:
            return 0.0

        # Calculate the centroid in radians.
        centroid_rad = coordinates_rad.mean(axis=0, keepdims=True)

        # Calculate haversine distance from each point to the centroid.
        dists_rad = haversine_distances(coordinates_rad, centroid_rad).flatten()

        # Return the maximum distance in radians.
        # Conversion to metres happens in get_max_distance_intersection_center_m().
        return dists_rad.max()

    def get_max_distance_intersection_center_m(self) -> float:
        """
        Return the maximum distance from an intersection to the cluster centre in metres.

        Returns
        -------
        float
            Maximum distance in metres.
        """

        # Convert radians to metres by multiplying with Earth radius.
        return self.get_max_distance_intersection_center_rad()*tsgcf.EARTH_RADIUS_M


    def score_cluster(self) -> float:
        """
        Compute a quality score for the cluster.
    
        The score combines three signals:
    
        - density_score:
          Average pairwise KDE density of the intersection points inside the
          cluster. This is higher when points are closer together.

        - points_score:
          Ratio of observed intersection points to the theoretical maximum
          number of possible line pairs.

          The theoretical maximum is:
              0.5 * L * (L - 1)

          where L is the number of unique object annotations.

        - size factor:
          log10(N), where N is the number of valid intersection points.
          This rewards larger clusters without letting size dominate completely.

        Final score:
            points_score * density_score * log10(N)
    
        Returns
        -------
        float
            Cluster quality score.
        """

        # Calculate spatial density score of the intersection points.
        density_score = self.kde_density()
 
        # Count the unique object annotations that contributed to this cluster.
        obj_anno_count = len(self.get_object_annotation())

        # Calculate the theoretical maximum number of line-pair intersections.
        # Avoid division by zero when the cluster has fewer than two annotations.
        max_pairs = 0.5 * obj_anno_count * (obj_anno_count - 1)

        # Ratio between observed intersections and theoretical maximum intersections.
        points_score = self.amount_intersections() / max_pairs if max_pairs > 0 else 0.0
 
        # Combine the score components.
        score = points_score * density_score * math.log10(self.amount_intersections())
 
        return score



    def get_object_annotation(self) -> list[ObjectAnnotation]:
        """
        Return all unique ObjectAnnotation objects used in this cluster.

        Each Intersection contains two ObjectAnnotation objects. This method
        collects both annotations from all valid intersections and removes
        duplicates by using a set.

        Returns
        -------
        list[ObjectAnnotation]
            Unique object annotations that contributed to this cluster.
        """

        # Use a set to avoid duplicate ObjectAnnotation objects.
        obj_anno: set[ObjectAnnotation]  = set()

        # Add both annotations from each intersection.
        for inters in self.get_intersections():
            obj_anno.add(inters.get_annotation1())
            obj_anno.add(inters.get_annotation2())

        # Convert back to a list before returning.
        return list(obj_anno) # Converteer terug naar een lijst

    def __calculate_view_direction(self) -> float | None:
        """
        Calculate the average viewing direction of all object annotations.

        Because directions are circular values, this method uses circular mean.
        For example, directions 359° and 1° should average to 0°, not 180°.

        Returns
        -------
        float | None
            Average viewing direction in degrees, in the range [0, 360).
            Returns None if there are no object annotations.
        """

        # Get all unique object annotations in this cluster.
        lst_obj_anno = self.get_object_annotation()

        # Collect the viewing direction of each object annotation.
        line_direction: float = []
        for obj_anno in lst_obj_anno:
            line_direction.append(obj_anno.direction_deg())

        # No directions means no average can be calculated.
        if not line_direction:
            return None

        # Calculate circular mean and normalize to [0, 360).
        return (circmean(line_direction, high=360, low=0).item())% 360

    def get_perpendicular(self) -> float | None:
        """
        Return the direction perpendicular to the average viewing direction.

        Returns
        -------
        float | None
            Direction in degrees.
        """

        # Add 180 degrees to get the opposite/perpendicular direction and
        # normalize to [0, 360).
        return (self.__calculate_view_direction() + 180)% 360


    def get_object_size(self) -> tuple[float | None, float | None]:
        """
        Estimate the average object size for this cluster.

        The method estimates the object size for each object annotation at the
        cluster centre. It then returns the mean and standard deviation.

        Returns
        -------
        tuple[float | None, float | None]
            First value:
                Mean object size in metres.

            Second value:
                Standard deviation of object sizes in metres.

            Returns (None, None) if there are fewer than two object annotations.
        """

        # Get all unique object annotations in this cluster.
        lst_obj_anno = self.get_object_annotation()

        # Store object size estimates.
        object_sizes: list[float] = []

        # Estimate object size for each annotation at the cluster centre.
        for obj_anno in lst_obj_anno:
            size = obj_anno.object_size_m(self.get_center())
            if size is not None:
                object_sizes.append(obj_anno.object_size_m(self.get_center()))

        # Calculate mean and sample standard deviation if enough annotations exist.
        if len(lst_obj_anno) >= 2:
            size = float(np.mean(object_sizes))
            size_sd = float(np.std(object_sizes, ddof=1))
            return size, size_sd

        return None, None


    def kde_density(self) -> float:
        """
        Estimate the average pairwise KDE density of the cluster points.
    
        For every unique pair of intersection points, the haversine distance is
        calculated. That distance is passed through a Gaussian kernel.

        The result is normalized by the number of point pairs, so clusters with
        different numbers of points can be compared.
    
        Returns
        -------
        float
            Average kernel density.
            Returns 0.0 for clusters with fewer than two points.
        """

        # Get valid intersection coordinates.
        points = self.get_coordinates()

        # Number of valid points.
        n = len(points)

        # At least two points are needed for pairwise density.
        if n < 2:
            return 0.0
    
        # Sum kernel values for all unique point pairs.
        density = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # haversine() expects coordinates and returns kilometres.
                # Multiplication by 1000 converts kilometres to metres.
                distance_m = haversine(points[i], points[j]) * 1000

                # Add Gaussian kernel value for this pair distance.
                density += self.gaussian_kernel(distance_m, h=0.5)
    
        # Number of unique point pairs.
        num_pairs = 0.5 * n * (n - 1)

        # Return average density per pair.
        return density / num_pairs

    @staticmethod
    def gaussian_kernel(d: float, h: float = 0.5) -> float:
        """
        Evaluate a Gaussian kernel for a given distance.

        Parameters
        ----------
        d:
            Distance from the kernel centre in metres.

        h:
            Bandwidth in metres.
            Smaller values produce sharper peaks.
    
        Returns
        -------
        float
            Kernel weight.
        """

        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (d / h) ** 2)

    @staticmethod
    def funct_cluster(
        cluster: 'Cluster',
        max_radius_m: float = 3,
        epsilon: float = 1.5,
        epsilon_decrease: float = .75,
        max_dept: int = 1,
        current_dept: int = 1,
        min_samples: int = 1,
    ) -> list['Cluster']:
        """
        Recursively cluster line-intersection points into candidate object locations.

        The method uses DBSCAN with a haversine distance metric.

        Main steps
        ----------
        1. Stop if the maximum recursion depth is reached.

        2. Stop if all intersections are already close enough to the cluster
           centre. In that case, the input cluster is returned as-is.

        3. Run DBSCAN on the intersection coordinates.

        4. Create new Cluster objects for each DBSCAN label.
           Noise points with label -1 are ignored.

        5. Score clusters and sort them from best to worst.

        6. Resolve conflicts:
           each ObjectAnnotation line may belong to at most one cluster.
           Higher-scoring clusters claim their lines first.

        7. Remove clusters that do not meet the minimum size requirement.

        8. Decrease epsilon and recursively process the remaining clusters.

        Parameters
        ----------
        cluster:
            Cluster to split or accept.

        max_radius_m:
            Maximum allowed distance in metres from any intersection to the
            cluster centre. If the cluster is already within this radius,
            recursion stops for this cluster.

        epsilon:
            DBSCAN neighbourhood radius in metres before conversion to radians.

        epsilon_decrease:
            Factor used to reduce epsilon at each recursion level.

        max_dept:
            Maximum recursion depth.

        current_dept:
            Current recursion depth. Callers normally leave this at 1.

        min_samples:
            Minimum cluster size passed to DBSCAN and used to filter small clusters.
    
        Returns
        -------
        list[Cluster]
            Final list of accepted clusters.
        """

        # logger.info(f"start cluster, epsilo: {epsilon}, current_dept: {current_dept}, intersection count: {cluster.amount_intersections()}")

        # List to collect clusters found at this recursion level.
        cluster_lst : list[Cluster] = []

        # --- Base case: maximum depth reached ---------------------------------- #
        if current_dept > max_dept:
            cluster_lst.append(cluster)
            logger.warning(f"max depth reached; returning cluster {cluster.get_id()} ({cluster.amount_intersections()} pts)")
            return cluster_lst
    
        # Increase recursion depth for child calls.
        current_dept += 1

        # If all points are already close enough to the centre, keep this cluster.
        if cluster.get_max_distance_intersection_center_m() < max_radius_m:
            # All points are already within the acceptable radius → single cluster.
            cluster_lst.append(cluster)
            return cluster_lst
    
        # --- DBSCAN clustering ------------------------------------------------- #

        # Get coordinates and intersections.
        # These lists are expected to have the same order and length.
        coordinates_rad = cluster.get_coordinates_rad()
        intersections = cluster.get_intersections()

        # Create DBSCAN clusterer.
        # eps is converted from metres to radians for the haversine metric.
        clusterer = DBSCAN(
            eps=epsilon*2*math.pi/tsgcf.EARTH_RADIUS_M,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )

        # Assign a DBSCAN label to each coordinate.
        labels = clusterer.fit_predict(coordinates_rad)


        # Create new clusters for each DBSCAN label.
        # Label -1 is DBSCAN noise and is excluded.
        unique_labels = set(labels) - {-1}    
        
        for label in unique_labels:
            new_cluster = Cluster()

            # Add all intersections with the current DBSCAN label.
            for idx, current_label in enumerate(labels):
                if current_label == label:
                    new_cluster.add_intersection(intersections[idx])

            cluster_lst.append(new_cluster)

    
        # Sort clusters by quality score.
        # Highest scoring clusters are processed first during conflict resolution.
        clusters_sorted = sorted(
            cluster_lst,
            key=lambda cluster: cluster.score_cluster(),
            reverse=True
        )

        # --- Conflict resolution ----------------------------------------------- #
        # Each bearing line may be assigned to at most one cluster.
        # Clusters are processed in descending score order, so high-quality
        # clusters claim their lines first.
        line_to_cluster: dict[int, int] = {}
    
        for cluster in clusters_sorted:
            cluster_id = cluster.get_id()

            # Loop through all intersections in this cluster.
            for inters in cluster.get_intersections():

                # Get the IDs of the two ObjectAnnotation lines that form this intersection.
                id_a, id_b = inters.get_annotation1().get_id(), inters.get_annotation2().get_id()
    
                # A conflict exists if either line was already claimed by another cluster.
                conflict_a = id_a in line_to_cluster and line_to_cluster[id_a] != cluster_id
                conflict_b = id_b in line_to_cluster and line_to_cluster[id_b] != cluster_id

                # Remove the intersection from this cluster if it conflicts.
                if conflict_a or conflict_b:
                    cluster.remove_intersection(inters)
    
                # Mark both lines as claimed by this cluster.
                line_to_cluster[id_a] = cluster_id
                line_to_cluster[id_b] = cluster_id
    
        # Remove clusters that fall below the minimum size requirement.
        clusters_filtered = [
            cluster for cluster in clusters_sorted
            if cluster.amount_intersections() >= min_samples
        ]

        # Stop if no valid clusters remain.
        if not clusters_filtered:
            logger.warning("cluster_filtered is empty; aborting recursion.")
            return clusters_filtered

        # --- Recurse with tighter epsilon -------------------------------------- #

        # Make DBSCAN stricter at the next recursion level.
        epsilon *= epsilon_decrease
        
        # Collect final clusters from recursive calls.
        cluster_result_lst : list[Cluster] = []

        for cluster in clusters_filtered:
            cluster_result_lst.extend(Cluster.funct_cluster(
                cluster, max_radius_m, epsilon, epsilon_decrease, max_dept, current_dept, min_samples))

        return cluster_result_lst    

  
class Centroid:
    """
    Represents a final candidate object location based on an original Cluster.

    A Centroid links:
    - one original Cluster;
    - the calculated centre point of that cluster;
    - the object annotations that are assigned to this centroid.

    The centroid can provide:
    - its location;
    - the estimated object size;
    - the object size standard deviation;
    - the viewing/perpendicular direction;
    - source picture and annotation IDs.
    """

    # Class-level counter used to assign a unique internal ID to each centroid.
    __next_id = 0

    def __init__(self, original_cluster: Cluster)->None:
        """
        Initialize a Centroid from an original Cluster.

        Parameters
        ----------
        original_cluster:
            Cluster from which this centroid is derived.

        Raises
        ------
        TypeError
            If original_cluster is not a Cluster instance.
        """

        # Assign a unique internal centroid ID.
        self.__id = Centroid.__next_id        
        Centroid.__next_id += 1

        # Validate the source cluster.
        if not isinstance(original_cluster, Cluster):
            raise TypeError("original_cluster must be an Cluster instance")

        # Store the cluster from which this centroid was created.
        self.__original_cluster = original_cluster

        # Store ObjectAnnotation objects assigned to this centroid.
        self.__lst_object_annotation: list[ObjectAnnotation] = []

    def add_object_annotation(self, object_annotation: ObjectAnnotation)->None:
        """
        Add an ObjectAnnotation to this centroid.

        Parameters
        ----------
        object_annotation:
            ObjectAnnotation to assign to this centroid.

        Raises
        ------
        TypeError
            If object_annotation is not an ObjectAnnotation instance.
        """

        # Validate input type.
        if not isinstance(object_annotation, ObjectAnnotation):
            raise TypeError("object_annotation must be an ObjectAnnotation instance")

        # Add the annotation to the centroid.
        self.__lst_object_annotation.append(object_annotation)

    def get_object_annotations(self)->list[ObjectAnnotation]:
        """
        Return all ObjectAnnotation objects assigned to this centroid.

        Returns
        -------
        list[ObjectAnnotation]
            List of assigned object annotations.
        """
        return self.__lst_object_annotation

    def get_id(self)->int:
        """
        Return the internal unique ID of this centroid.

        Returns
        -------
        int
            Centroid ID.
        """
        return self.__id

    def get_original_cluster(self)->Cluster:
        """
        Return the original Cluster from which this centroid was created.

        Returns
        -------
        Cluster
            Source cluster.
        """
        return self.__original_cluster

    def get_center(self)->Point:
        """
        Return the centre point of the original cluster.

        Returns
        -------
        Point
            Centre of the original cluster.
            Point.x = longitude. Point.y = latitude.
        """
        return self.__original_cluster.get_center()

    def get_object_size(self)->tuple[float | None, float | None]:
        """
        Return the estimated object size from the original cluster.

        The original cluster returns both size and standard deviation.
        This method returns only the size.

        Returns
        -------
        float | None
            Estimated object size in metres.
        """
        size, _ =  self.__original_cluster.get_object_size()
        return size

    def get_object_size_sd(self)->tuple[float | None, float | None]:
        """
        Return the standard deviation of the estimated object size.

        The original cluster returns both size and standard deviation.
        This method returns only the standard deviation.

        Returns
        -------
        float | None
            Standard deviation of the estimated object size in metres.
        """
        _, size_sd =  self.__original_cluster.get_object_size()
        return size_sd

    def get_perpendicular(self)->float:
        """
        Return the perpendicular direction calculated from the original cluster.

        Returns
        -------
        float
            Direction in degrees.
        """
        return self.__original_cluster.get_perpendicular()

    def get_object_annotation_id(self)->str:
        """
        Return the annotation ID of the first assigned ObjectAnnotation.

        This assumes that at least one ObjectAnnotation is assigned.

        Returns
        -------
        str
            Source annotation ID.
        """
        return self.__lst_object_annotation[0].get_object_annotation_id()

    def get_picture_id(self)->str:
        """
        Return the picture ID of the first assigned ObjectAnnotation.

        This assumes that at least one ObjectAnnotation is assigned.

        Returns
        -------
        str
            Source picture ID.
        """
        return self.__lst_object_annotation[0].get_picture_id()

    # def add_valid_object_annotations(self, list_object_annotation: list[ObjectAnnotation], MAX_CLUSTER_RADIUS_M):
    #     # Note for future improvement.
    #     # Possible recalculate new centre.
    #     # Possible add line score to line so best line can be selected.

    #     tree = STRtree([obj_anna.get_line() for obj_anna in list_object_annotation])

    #     DEGREE_BUFFER = MAX_CLUSTER_RADIUS_M * 360 / (EARTH_RADIUS_M * 2 * math.pi)

    #     p = self.get_center()

    #     # Coarse filter: bounding box in degrees around the centroid.
    #     search_box = box(
    #         p.x - DEGREE_BUFFER, p.y - DEGREE_BUFFER,
    #         p.x + DEGREE_BUFFER, p.y + DEGREE_BUFFER
    #     )
    #     candidate_indices = tree.query(search_box)          # returns indices into line_geoms
    #     candidates = [list_object_annotation[i] for i in candidate_indices]

    #     # Fine filter: exact geodesic distance for each candidate.
    #     for obj_anna in candidates:
    #         dist = obj_anna.min_distance_to_line_m(p)
    #         if dist <= MAX_CLUSTER_RADIUS_M:
    #             self.add_object_annotation(obj_anna)

    #     return None

    def get_lst_object_annotation(self) -> list[ObjectAnnotation]:
        """
        Return all ObjectAnnotation objects assigned to this centroid.

        This method returns the same internal list as get_object_annotations().

        Returns
        -------
        list[ObjectAnnotation]
            List of assigned object annotations.
        """
        return self.__lst_object_annotation


    # Possible handy for the future, not in use at the moment.
    def update_links_and_sizes(self) -> None:
        """
        Update the link and size fields for all object annotations in this cluster.

        This method is not currently used.

        The link points to the Panoramax endpoint with the annotation ID and
        picture ID as query parameters.
        """

        # Loop through assigned object annotations.
        for obj_anna in self._object_annotations:

            # Build a Panoramax link for the object annotation.
            obj_anna.link = (
                f"{tsgcf.PANORAMAX_END_POINT}"
                f"?annot={obj_anna.get_object_annotation_id()}&pic={obj_anna.get_picture_id()}"
            )

            # Store the calculated object size on the object annotation.
            obj_anna.size = obj_anna.get_object_size()

    @staticmethod
    def add_valid_object_annotations_all_centroids(
    lst_centroids: list['Centroid'],
    list_object_annotation: list['ObjectAnnotation'],
    MAX_CLUSTER_RADIUS_M: float
    ) -> None:
        """
        Add all valid ObjectAnnotations to the matching centroids.

        This method processes all centroids at once. It uses geodesic buffers
        around centroid locations and a spatial join to find object annotation
        lines that intersect those buffers.

        Main steps
        ----------
        1. Create a geodesic buffer around each centroid.
        2. Convert centroid buffers to a GeoDataFrame.
        3. Convert ObjectAnnotation lines to a GeoDataFrame.
        4. Use a spatial join to find lines that intersect centroid buffers.
        5. Add the matching ObjectAnnotations to the matching Centroids.

        Parameters
        ----------
        lst_centroids:
            List of centroids to process.

        list_object_annotation:
            List of all object annotations. Each annotation should have a line.

        MAX_CLUSTER_RADIUS_M:
            Maximum distance in metres for an annotation line to belong to a centroid.

        Returns
        -------
        None
        """

        def geodesic_point_buffer(lon: float, lat: float, radius_m: float):
            """
            Create a geodesic buffer around a longitude/latitude point.

            The buffer is created in a local azimuthal-equidistant projection
            centered on the input point. It is then transformed back to EPSG:4326.

            Parameters
            ----------
            lon:
                Longitude of the centre point.

            lat:
                Latitude of the centre point.

            radius_m:
                Buffer radius in metres.

            Returns
            -------
            shapely geometry
                Buffer polygon in longitude/latitude coordinates.
            """

            # Create a local projection centered on the centroid.
            proj = pyproj.Proj(
                proj="aeqd",
                ellps="WGS84",
                datum="WGS84",
                lat_0=lat,
                lon_0=lon
            )

            # Create a circular buffer around local point (0, 0).
            # In this local projection, units are metres.
            buf = Point(0, 0).buffer(radius_m)  # Buffer in meters

            # Transform the buffer back to longitude/latitude coordinates.
            return transform(lambda x, y: proj(x, y, inverse=True), buf)

        # --- Step 1: Convert centroids to a GeoDataFrame with geodesic buffers ---

        # Lists used to build the centroid GeoDataFrame.
        centroids = []
        geometries = []

        # Create one geodesic buffer for each centroid.
        for centroid in lst_centroids:
            center = centroid.get_center()
            buffer = geodesic_point_buffer(center.x, center.y, MAX_CLUSTER_RADIUS_M)
            centroids.append(centroid)
            geometries.append(buffer)

        # GeoDataFrame with centroid objects and their buffer geometries.
        centroids_gdf = gpd.GeoDataFrame(
            {"centroid": centroids},
            geometry=geometries,
            crs="EPSG:4326"
        )

        # --- Step 2: Convert object annotation lines to a GeoDataFrame ---

        # GeoDataFrame with ObjectAnnotation objects and their calculated lines.
        lines_gdf = gpd.GeoDataFrame(
            {
                "object_annotation": list_object_annotation,
                "geometry": [obj_anna.get_line() for obj_anna in list_object_annotation]
            },
            crs="EPSG:4326"
        )

        # --- Step 3: Spatial join ---

        # Reset indexes to avoid duplicate index problems during the spatial join.
        centroids_gdf = centroids_gdf.reset_index(drop=True)
        lines_gdf = lines_gdf.reset_index(drop=True)

        # Find all annotation lines that intersect centroid buffers.
        joined = gpd.sjoin(
            lines_gdf,
            centroids_gdf,
            how="inner",
            predicate="intersects"
        )

        # --- Step 4: Add matching lines to the matching centroids ---

        # Each row contains one object annotation and one matching centroid.
        for _, row in joined.iterrows():
            centroid = row["centroid"]
            obj_anna = row["object_annotation"]
            centroid.add_object_annotation(obj_anna)


    @staticmethod
    def resolve_centroid_assignments(lst_centroid: list['Centroid'], MAX_CLUSTER_RADIUS_M: float) -> list['Centroid']:
        """
        Resolve ObjectAnnotation assignments to centroids based on scoring.

        One ObjectAnnotation can initially match multiple centroids. This method
        assigns each ObjectAnnotation to at most one centroid: the centroid with
        the highest score.

        The score combines:
        - distance score;
        - object size score;
        - cluster score.

        Parameters
        ----------
        lst_centroid:
            List of centroids with preliminary ObjectAnnotation assignments.

        MAX_CLUSTER_RADIUS_M:
            Maximum distance in metres used to calculate the distance score.

        Returns
        -------
        list[Centroid]
            New list of Centroids with only the winning ObjectAnnotation assignments.
        """

        # Step 1: Calculate scores for all possible centroid/object annotation combinations.
        score_map = {}  # {(centroid_id, obj_anna_id): score}

        for centroid in lst_centroid:
            # Estimated object size for this centroid.
            centroid_size = centroid.get_object_size()

            # Quality score of the original cluster behind this centroid.
            cluster_score = centroid.get_original_cluster().score_cluster()
            
            for obj_anna in centroid.get_lst_object_annotation():
                # Calculate distance score.
                # A smaller distance gives a higher score.
                distance = obj_anna.min_distance_to_line_m(centroid.get_center())
                distance_score = (1 - distance / MAX_CLUSTER_RADIUS_M) ** 2

                # Calculate size score.
                # This compares the centroid object size with the object size
                # estimated from this annotation at the centroid location.
                object_size = obj_anna.object_size_m(centroid.get_center())
                size_score = object_size_score(centroid_size, object_size)

                # Combine all score parts into one total score.
                total_score = distance_score * size_score * cluster_score
                score_map[(centroid.get_id(), obj_anna.get_id())] = total_score

        # Step 2: Select the best centroid for each ObjectAnnotation.
        best_centroid_per_annotation = {}  # {obj_anna_id: (best_centroid_id, best_score)}

        for (centroid_id, obj_anna_id), score in score_map.items():
            # Keep the centroid with the highest score for this annotation.
            if obj_anna_id not in best_centroid_per_annotation or score > best_centroid_per_annotation[obj_anna_id][1]:
                best_centroid_per_annotation[obj_anna_id] = (centroid_id, score)

        # Step 3: Create new Centroids with only the winning ObjectAnnotations.
        lst_new_centroids = []

        for centroid in lst_centroid:
            # Create a new centroid based on the same original cluster.
            new_centroid = Centroid(centroid.get_original_cluster())

            # Add only annotations for which this centroid is the winning centroid.
            for obj_anna in centroid.get_lst_object_annotation():
                best_centroid_id, _ = best_centroid_per_annotation.get(obj_anna.get_id(), (None, -1))
                if best_centroid_id == centroid.get_id():
                    new_centroid.add_object_annotation(obj_anna)

            # Keep only centroids with at least two assigned ObjectAnnotations.
            if len(new_centroid.get_object_annotations())>=2:
                lst_new_centroids.append(new_centroid)

        return lst_new_centroids


# ===========================================================================
# Support functions
# ===========================================================================

def object_size_score(size_1: float, size_2: float)  -> float:
    """
    Calculate a similarity score between two object size estimates.

    The score is close to 1 when both sizes are similar.
    The score becomes lower when the difference between the sizes becomes larger.

    Parameters
    ----------
    size_1:
        First object size estimate.

    size_2:
        Second object size estimate.

    Returns
    -------
    float
        Similarity score.
    """
    return 1 - abs(size_1 - size_2) / (size_1 + size_2)
