
import math
from geopy.distance import geodesic
from shapely.geometry import LineString, Point, box
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import haversine_distances
from haversine import haversine
from scipy.stats import circmean
from shapely import STRtree, GeometryType
import os
# from tisgrade_config import EARTH_RADIUS_M, PANORAMAX_END_POINT

import geopandas as gpd
from shapely.ops import transform
import shapely
import pyproj
from sklearn.cluster import DBSCAN

import logging
from dotenv import load_dotenv

load_dotenv()

EARTH_RADIUS_M      = int(os.environ["EARTH_RADIUS_M"])
PANORAMAX_END_POINT = os.environ["PANORAMAX_END_POINT"]


logger = logging.getLogger(f"tisgrade.{__name__}")



# ===========================================================================
# Classes
# ===========================================================================


class ObjectAnnotation:
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
                 
        self.__id = ObjectAnnotation.__next_id
        ObjectAnnotation.__next_id += 1

        # Validate object_code (str)
        if not isinstance(object_code, str):
            raise TypeError("object_code must be a string.")
        self.__object_code = object_code

        # Validate bbox (list of lists with ints)
        if (
            not isinstance(bbox, list) or
            len(bbox) < 3 or
            not all(isinstance(coord, list) and len(coord) == 2 for coord in bbox) or
            not all(isinstance(x, int) and isinstance(y, int) for coord in bbox for x, y in [coord])
        ):
            raise TypeError("bbox must be a list of [x, y] coordinates of atleast 3 with integers.")
        self.__bbox = bbox
        
        # Validate origin (Point)
        if not isinstance(origin, Point):
            raise TypeError("origin must be a shapely.geometry.Point.")
        self.__origin = origin  #x = longitude, y = latitude  

        # Validate pic_size_hor_px (int)
        if not isinstance(pic_size_hor_px, int):
            raise TypeError("pic_size_hor_px must be an integer.")
        self.__pic_size_hor_px = pic_size_hor_px

        # Validate pic_size_ver_px (int)
        if not isinstance(pic_size_ver_px, int):
            raise TypeError("pic_size_ver_px must be an integer.")
        self.__pic_size_ver_px = pic_size_ver_px

        # Validate pic_field_of_view (float)
        if not isinstance(pic_field_of_view, (int, float)):
            raise TypeError("pic_field_of_view must be a number (int or a float).")
        self.__pic_field_of_view = pic_field_of_view

        # Validate pic_azimuth (float)
        if not isinstance(pic_azimuth, (int, float)):
            raise TypeError("pic_azimuth must be a number (int or a float).")
        self.__pic_azimuth = pic_azimuth

        # Validate picture_id (str or None)
        if not isinstance(picture_id, str | None):
            raise TypeError('picture_id must be a string or None')
        self.__picture_id = picture_id

        # Validate annotation_id (str or None)
        if not isinstance(annotation_id, str | None):
            raise TypeError('annotation_id must be a string or None')
        self.__object_annotation_id = annotation_id

        self.__min_object_size_m = None
        self.__max_object_size_m = None        
        
        self.__observers = []
        self.__is_stale = True
        
        self.__direction_deg = None

        self.__size_deg = None

        self.__line = None


    def add_observer(self, observer):
        self.__observers.append(observer)


    def __notify_observers(self):
        for observer in self.__observers:
            observer.on_child_changed(self)  # Roep callback aan

    def __calculate_properties(self)->None:
        self.__is_stale = False
        self.__direction_deg = self.__calculate_direction_deg()
        self.__size_deg = self.__calculate_size_deg()
        self.__line = self.__calculate_line()



    def set_min_object_size(self, min_object_size_m: float)-> None:
        if not isinstance(min_object_size_m, float):
            raise TypeError("min object size must be a float.")
        self.__is_stale = True
        self.__min_object_size_m = min_object_size_m
        self.__notify_observers()
        

    def set_max_object_size(self, max_object_size_m: float)-> None:        
        if not isinstance(max_object_size_m, float):
            raise TypeError("max object size must be a float.")
        self.__is_stale = True
        self.__max_object_size_m = max_object_size_m
        self.__notify_observers()

    def get_id(self)-> int:
        return self.__id

    def get_object_code(self)->str:
        return self.__object_code

    def get_picture_id(self)-> str | None:
        return self.__picture_id

    def get_object_annotation_id(self)-> str | None:
        return self.__object_annotation_id

    def print(self)-> None:
        print(f'object_id: {self.__id}, object_code: {self.__object_code} bbox: {self.__bbox}, origin: {self.__origin}, line:{self.get_line()}')

    def object_x_min_px(self)-> int:
        return min(p[0] for p in self.__bbox)
    
    def object_x_max_px(self)-> int:
        return max(p[0] for p in self.__bbox)

    def object_y_min_px(self)-> int:
        return min(p[1] for p in self.__bbox)
    
    def object_y_max_px(self)-> int:
        return max(p[1] for p in self.__bbox)

    def x_center_px(self) -> float:
        """Return the horizontal centre pixel of a sign, accounting for wrap-around.
    
        Panoramic images wrap horizontally.  When the two edge pixels of a sign
        straddle the 0°/360° seam, a naive average would place the centre on the
        wrong side of the image.  This function detects that case and corrects for
        it.
    
        Parameters
        ----------
        x_pos_1, x_pos_2:
            Horizontal pixel positions of the left and right edges of the sign.
        picture_size_px:
            Total horizontal resolution of the panoramic image.
    
        Returns
        -------
        float
            Horizontal centre pixel in [0, picture_size_px).
        """
        if abs(self.object_x_max_px() - self.object_x_min_px()) <= self.__pic_size_hor_px / 2:
            x_loc_px = (self.object_x_min_px() + self.object_x_max_px()) / 2
        else:
            # Edges straddle the seam → shift by half the image width
            x_loc_px = (self.object_x_min_px() + self.object_x_max_px()) / 2 - self.__pic_size_hor_px/ 2
    
        return x_loc_px % (self.__pic_size_hor_px - 1)


    def __calculate_direction_deg(self)-> float:

        """Convert a horizontal pixel position to a compass bearing.
    
        Computes the angular offset of *x_pos* from the image centre, converts it
        to degrees using the camera's field of vision, then adds the camera azimuth
        to obtain an absolute bearing.
    
        Parameters
        ----------
        x_pos:
            Horizontal pixel position within the image (0 = left edge).
        picture_size_px:
            Total horizontal resolution of the panoramic image in pixels.
        field_of_vision:
            Horizontal field of vision of the image in degrees.
        azimuth:
            Camera heading (azimuth) in degrees at the time of capture.
        picture_rotation:
            ``"clockwise"`` (default) if pixel x increases with bearing;
            ``"counter_clockwise"`` otherwise.
    
        Returns
        -------
        float
            Absolute compass bearing in [0, 360).
        """
        delta_px = self.x_center_px() - (0.5 * self.__pic_size_hor_px) + 1
        delta_deg = (delta_px / self.__pic_size_hor_px) * self.__pic_field_of_view
        self.__direction_deg = (delta_deg + self.__pic_azimuth) % 360
        return self.__direction_deg

    def direction_deg(self)-> float:
        if self.__is_stale:
            self.__calculate_properties()
        return self.__direction_deg


    def __calculate_size_deg(self) -> float:
        """Return the width of a sign in pixels, handling wrap-around.
    
        Parameters
        ----------
        start_px, end_px:
            Horizontal pixel positions of the sign's left and right edges.
        picture_px:
            Total horizontal resolution of the image.
    
        Returns
        -------
        float
            Sign width in pixels (always the *smaller* of the two possible spans).
        """
        
        #  Check has to be made now we get the data directly from Panoramax this part of the function is still necessary
        #  Most likely we now can us directly start_px and end_px
        size = self.object_y_max_px() - self.object_y_min_px()
        # When the sign straddles the image seam, the true span is the complement
        if size > self.__pic_size_hor_px / 2:
            size = self.__pic_size_hor_px - size

        size_deg = size/self.__pic_size_hor_px * self.__pic_field_of_view

        return size_deg

    def size_deg(self) -> float:
        if self.__is_stale:
            self.__calculate_properties()        
        return self.__size_deg

    def __calculate_line(self) -> LineString | None:
        """
        Get the line representing the object's possible locations.

        Returns:
            A LineString connecting the min and max distance points from the origin,
            or None if min/max object sizes are not set.
        """
        if self.__min_object_size_m is None or self.__max_object_size_m is None:
            return None
        
        distance_min = self.calculate_object_distance(self.__min_object_size_m)
        distance_max = self.calculate_object_distance(self.__max_object_size_m)

        return LineString([self.calculate_endpoint(distance_min), self.calculate_endpoint(distance_max)])

    def get_line(self) -> LineString | None:
        """
        Get the line representing the object's possible locations.

        Returns:
            A LineString connecting the min and max distance points from the origin,
            or None if min/max object sizes are not set.
        """
        if self.__is_stale:
            self.__calculate_properties()        
        return self.__line

    
    # Now we get the data directly field of view can be added to the function call.
    #  x field_of_view / 360
    def calculate_object_distance(self,
        object_size_m: float
    ) -> float:
        """Estimate the camera-to-sign distance from known sign dimensions.
    
        Uses the pinhole camera model: the image wraps a full 360 °, so the
        effective circumference at distance *d* is ``2π·d``.
    
        Parameters
        ----------
        sign_size_m:
            Physical width of the sign in metres (known from standards).
        sign_size_px:
            Observed width of the sign in the panoramic image in pixels.
        picture_size_px:
            Total horizontal resolution of the panoramic image in pixels.
    
        Returns
        -------
        float
            Estimated distance in metres.
        """
        return object_size_m / (2 * math.pi) * (360 / self.size_deg())    

    def calculate_endpoint(
        self, distance_meters: float
    ) -> Point:
        """Compute the endpoint of a bearing line given a start, direction, and length.
    
        Parameters
        ----------
        start_point:
            Camera location as a Shapely Point (longitude, latitude).
        direction_degrees:
            Bearing from north in degrees (0–360).
        distance_meters:
            Length of the bearing line in metres.
    
        Returns
        -------
        Point
            End-point as a Shapely Point (longitude, latitude).
        """
        start_latlon = (self.__origin.y, self.__origin.x) #origin = (long (x), lat (y)) geodesic expets (Lat, long)
        dest = geodesic(meters=distance_meters).destination(start_latlon, self.direction_deg())
        return Point(dest.longitude, dest.latitude)


    def object_size_m(self, object_location: Point) -> float:
        """Return the vertical size of the object in meters if it would be locatated at the given point geodesic distance in metres between two Shapely Points.
    
        Parameters
        ----------
        object location (Point):
            Target location as a Shapely Point (longitude, latitude).
    
        Returns
        -------
        float
            vertical object size in meters.
        """ 
        if not isinstance(object_location, Point):
            raise TypeError("objct_location must be an Point.")
        distance = self.distance_m(object_location)
        size_deg = self.size_deg()
        return 2 * math.pi * distance * size_deg/360


    def distance_m(self, object_location: Point) -> float:
        """Return the geodesic distance in metres between two Shapely Points.
    
        Parameters
        ----------
        object_location:
            Camera location as a Shapely Point (longitude, latitude).
        point:
            Target location as a Shapely Point (longitude, latitude).
    
        Returns
        -------
        float
            Distance in metres.
        """
        if not isinstance(object_location, Point):
            raise TypeError("object_location must be an Point.")


        origin_lon, origin_lat = self.__origin.x, self.__origin.y
        object_lon, object_lat = object_location.x, object_location.y

        return geodesic(
            (origin_lat, origin_lon), (object_lat, object_lon)
        ).meters

    def get_origin(self) -> Point:
        return self.__origin


    # Add a method to check if line is available
    def has_line(self) -> bool:
        """Check if the object has min/max sizes set to generate a line."""
        return self.__min_object_size_m is not None and self.__max_object_size_m is not None
    

    def min_distance_to_line_m(self, point: Point)->float:
        """
        Minimum geodesic distance in metres between a Point and a LineString.
        Walks each segment, finds the nearest point on that segment via Shapely
        (in degree-space, fine at this scale), then measures with geodesic.
        """

        if not isinstance(point, Point):
            raise TypeError("point must be an Point.")    

        line = self.get_line()
        if line is None:
            raise ValueError("LineString is None, cannot calculate distance.")            
        
        min_dist = float('inf')
        coords = list(line.coords)  # e.g. [(lon0,lat0), (lon1,lat1), ...]

        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            nearest = seg.interpolate(seg.project(point))   # closest point on segment
            dist = geodesic((point.y, point.x), (nearest.y, nearest.x)).meters
            min_dist = min(min_dist, dist)

        return min_dist


class Intersection:
    __next_id = 0
    
    def __init__(self, annotation1: ObjectAnnotation, annotation2: ObjectAnnotation):
        """
        Initialize an Intersections object with two ObjectAnnotation instances.

        Args:
            annotation1: First ObjectAnnotation instance
            annotation2: Second ObjectAnnotation instance

        Raises:
            TypeError: If either input is not an ObjectAnnotation
        """
        if not isinstance(annotation1, ObjectAnnotation):
            raise TypeError("annotation1 must be an ObjectAnnotation instance")
        if not isinstance(annotation2, ObjectAnnotation):
            raise TypeError("annotation2 must be an ObjectAnnotation instance")

        self.__annotation1 = annotation1
        self.__annotation1.add_observer(self)
        self.__annotation2 = annotation2
        self.__annotation2.add_observer(self)
        self.__id = Intersection.__next_id
        self.__is_stale = True
        self.__intersection = None
        self.__object_size = None
        self.__object_size_score = None
        self.__angle_between_lines = None
        Intersection.__next_id += 1

    def on_child_changed(self, changed_child):
        """Callback die wordt aangeroepen als een Child wijzigt."""
        self.__is_stale = True  # Markeer cache als stale

    def get_id(self) -> int | None:
        return self.__id

    def print(self):
        print(f'intersection point: {self.intersection()}, size: {self.object_size()}, hoek: {self.angle_between_lines()}, id anno 1: {self.__annotation1.get_id()}, id anno 2: {self.__annotation2.get_id()}')

    def __calculate_properties(self) -> None:
        if self.__is_stale:
            self.__is_stale = False
            line1 = self.__annotation1.get_line()
            line2 = self.__annotation2.get_line()

            self.__intersection = None
            self.__object_size = None
            self.__object_size_score = None
            self.__angle_between_lines = None


            if line1 is None or line2 is None:
                return None

            intersection = line1.intersection(line2)

            if isinstance(intersection, Point):
                self.__intersection = intersection

                size_1 = self.__annotation1.object_size_m(self.__intersection)
                size_2 = self.__annotation2.object_size_m(self.__intersection)
                self.__object_size = (size_1 + size_2)/2
                self.__object_size_score =  object_size_score(size_1, size_2)

                a1 = self.__annotation1.direction_deg()
                a2 = self.__annotation2.direction_deg()
                diff = abs(a1 - a2)
                self.__angle_between_lines = min(diff, 360 - diff)
            return None
            

    def intersection(self) -> Point | None:
        """
        Calculate the intersection point of the two annotation lines.

        Returns:
            The intersection Point if lines intersect and both annotations have lines,
            None otherwise.
        """
        if self.__is_stale:
            self.__calculate_properties()

        return self.__intersection

    def object_size(self) -> float | None:

        if self.__is_stale:
            self.__calculate_properties()

        return self.__object_size            

    def object_size_score(self)  -> float:
        if self.__is_stale:
            self.__calculate_properties()

        return self.__object_size_score

    def angle_between_lines(self) -> float:
        """Return the smallest angle between two compass bearings (0–180°).
    
        Parameters
        ----------
        angle1, angle2:
            Compass bearings in degrees.
    
        Returns
        -------
        float
            Unsigned angular difference in [0, 180].
        """
        if self.__is_stale:
            self.__calculate_properties()

        return self.__angle_between_lines

    def get_annotation1(self) -> ObjectAnnotation:
        """Get the first annotation"""
        return self.__annotation1

    def get_annotation2(self) -> ObjectAnnotation:
        """Get the second annotation"""
        return self.__annotation2

    @staticmethod
    def __local_aeqd_crs(lon_lat_coords: np.ndarray) -> pyproj.CRS:
        """Build an azimuthal-equidistant CRS centered on the data's bounding box.

        Antimeridian-safe: unwraps longitudes before finding the center so an
        AOI straddling 180/-180 (e.g. Fiji, the Bering Strait) doesn't get a
        bogus center from naive min/max.
        """
        lons, lats = lon_lat_coords[:, 0], lon_lat_coords[:, 1]

        if lons.max() - lons.min() > 180:
            lons = np.where(lons < 0, lons + 360, lons)

        lon0 = (lons.min() + lons.max()) / 2.0
        lon0 = ((lon0 + 180) % 360) - 180  # wrap back to [-180, 180]
        lat0 = (lats.min() + lats.max()) / 2.0

        return pyproj.CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )

    @staticmethod
    def __project_lines(lines: np.ndarray, transformer: pyproj.Transformer) -> np.ndarray:
        """Bulk-reproject an array of shapely LineStrings via shapely.transform
        (one vectorized pyproj call over all vertices, no per-geometry looping)."""

        def _xy(coords: np.ndarray) -> np.ndarray:
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])

        return shapely.transform(lines, _xy)

    @staticmethod
    def __crossing_pairs(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized crossing-pair detection: STRtree for candidates, then
        shapely ufuncs (no Python-level loop) to keep only true point crossings."""
        tree = STRtree(lines)
        i, j = tree.query(lines, predicate="crosses")

        mask = j > i  # dedupe symmetric pairs
        i, j = i[mask], j[mask]
        if len(i) == 0:
            return i, j

        pts = shapely.intersection(lines[i], lines[j])
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
        """Pure function, no closures over shared state — deliberately shaped so
        it can be handed to ProcessPoolExecutor.map / Dask later without changes."""
        i, j = Intersection.__crossing_pairs(tile_lines)
        if len(i) == 0:
            return []

        pts = shapely.intersection(tile_lines[i], tile_lines[j])
        xs, ys = shapely.get_x(pts), shapely.get_y(pts)

        # Only keep intersections whose point falls in this tile's unbuffered
        # core — this is what prevents double-counting the same intersection
        # from two overlapping buffered tiles.
        in_core = (
            (xs >= core_minx) & (xs < core_maxx) &
            (ys >= core_miny) & (ys < core_maxy)
        )
        i, j = i[in_core], j[in_core]

        return [(tile_annotations[a], tile_annotations[b]) for a, b in zip(i, j)]

    @staticmethod
    def find_intersections(annotations: list["ObjectAnnotation"]) -> list["Intersection"]:
        """Find all pairwise crossing points between bearing lines from ObjectAnnotations.

        Lines are given in lon/lat (EPSG:4326). They're reprojected into a local,
        dynamically-centered azimuthal-equidistant CRS so tile/buffer math can be
        done in meters, then processed tile-by-tile to bound peak memory.

        Valid for AOIs roughly up to 100km x 100km, latitude within [-60, 80]
        (outside that range AEQD distortion grows and this approach should be
        revisited). Currently runs single-process; _process_tile is written as a
        pure function so it can be swapped into a process pool or Dask later.
        """
        TILE_SIZE = 2000.0   # meters — tune based on profiling; bigger tiles = fewer tiles but more memory per tile
        BUFFER = 200.0        # meters, >= max line length: guarantees no missed cross-tile intersection

        if not annotations:
            return []

        valid_lines = []
        valid_annotations = []
        for annotation in annotations:
            line = annotation.get_line()
            if line is not None and line.is_valid and line.geom_type == "LineString":
                valid_lines.append(line)
                valid_annotations.append(annotation)

        if not valid_lines:
            return []

        lines_arr = np.array(valid_lines, dtype=object)

        # --- reproject lon/lat -> local meters ---
        all_coords = shapely.get_coordinates(lines_arr)  # (N, 2) lon, lat
        local_crs = Intersection.__local_aeqd_crs(all_coords)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        projected_lines = Intersection.__project_lines(lines_arr, transformer)

        # --- tile the projected extent ---
        bounds = shapely.bounds(projected_lines)  # (N, 4): minx, miny, maxx, maxy
        minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
        maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()

        intersections = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                core_minx, core_miny = x, y
                core_maxx, core_maxy = x + TILE_SIZE, y + TILE_SIZE

                sel = (
                    (bounds[:, 2] >= core_minx - BUFFER) & (bounds[:, 0] <= core_maxx + BUFFER) &
                    (bounds[:, 3] >= core_miny - BUFFER) & (bounds[:, 1] <= core_maxy + BUFFER)
                )
                idx = np.where(sel)[0]

                if len(idx) > 1:
                    pairs = Intersection.__process_tile(
                        projected_lines[idx],
                        [valid_annotations[k] for k in idx],
                        core_minx, core_miny, core_maxx, core_maxy,
                    )
                    for a, b in pairs:
                        intersections.append(Intersection(a, b))

                y += TILE_SIZE
            x += TILE_SIZE

        return intersections


class Cluster:
    __next_id = 0
    # __earth_radius_m = 6_371_000

    def __init__(self)->None:
        self.__id = Cluster.__next_id
        Cluster.__next_id += 1
        self.__intersections: list[Intersection] = {}


    def get_id(self)->None:
        return self.__id

    def add_intersection(self, intersection: Intersection) -> None:
        if not isinstance(intersection, Intersection):
            raise TypeError("Expected an Intersection object")

        if not isinstance(intersection.intersection(), Point):
            raise TypeError("Not a valid intersection")

        intersection_id = intersection.get_id()

        intersection_id = intersection.get_id()
        self.__intersections[intersection_id] = intersection

    def get_intersections(self) -> list[Intersection]:
        """Returns a list of intersections that have valid coordinate data.

        Returns:
            list[Intersection]: List of Intersection objects where intersection.intersection()
            is not None. Returns empty list if no valid intersections exist.
        """
        return [intersection
                for intersection in self.__intersections.values()
                if intersection.intersection() is not None]

    def remove_intersection(self, intersection: Intersection) -> bool:
        intersection_id = intersection.get_id()
        if intersection_id in self.__intersections:
            del self.__intersections[intersection_id]
            return True
        return False

    def amount_intersections(self) -> int:
        return sum(1 for intersection in self.__intersections.values()
                if intersection.intersection() is not None)

    def get_coordinates(self) -> NDArray[np.float64]:
        """Returns coordinates of valid intersections.

        Returns:
            NDArray[np.float64]: Array with shape (n, 2) containing [latitude, longitude]
            in decimal degrees. Returns empty array with shape (0, 2) if no valid
            intersections exist.
        """
        return np.array([(intersection.intersection().x, intersection.intersection().y) 
                                for intersection_id, intersection in self.__intersections.items()
                                if intersection.intersection() is not None]) 

    def get_center(self) -> Point|None:

        coordinates = self.get_coordinates()
        if len(coordinates) == 0:
            return None

        avg_x = np.mean(coordinates[:, 0])  # gemiddelde van alle x-waarden (longitude)
        avg_y = np.mean(coordinates[:, 1])  # gemiddelde van alle y-waarden (latitude)
        return Point(avg_x, avg_y)   # Point(longitude, latitude)


    def get_coordinates_rad(self):
        return np.radians(self.get_coordinates())
         

    def get_max_distance_intersection_center_rad(self) -> float:
        """Returns the maximum distance (in meters) from any intersection to the cluster's centroid.

        Returns:
            float: Maximum distance in meters. Returns 0.0 if no valid intersections exist.
        """
        coordinates_rad = self.get_coordinates_rad()

        # Return 0 if no coordinates available
        if coordinates_rad.size == 0:
            return 0.0

        centroid_rad = coordinates_rad.mean(axis=0, keepdims=True)
        dists_rad = haversine_distances(coordinates_rad, centroid_rad).flatten()

        # Convert from radians to meters (multiply by Earth radius in meters)
        # earth_radius_m = 6_371_000  # Earth radius in meters
        return dists_rad.max()

    def get_max_distance_intersection_center_m(self) -> float:
        return self.get_max_distance_intersection_center_rad()*EARTH_RADIUS_M


    def score_cluster(self) -> float:
        """Compute a quality score for the cluster.
    
        The score combines three signals:
    
        * **density_score** – average pairwise KDE density of the intersection
        points inside the cluster (high when points are tightly packed).
        * **points_score** – ratio of observed intersection points to the
        theoretical maximum ``0.5 * L * (L-1)``, where *L* is the number of
        unique bearing lines (high when many line pairs actually intersect).
        * **size factor** – ``log10(N)`` where *N* is the number of intersection
        points (rewards larger clusters without dominating the score).
    
        Final score = ``points_score × density_score × log10(N)``
    
        Parameters
        ----------
        clusters:
            Mapping ``{cluster_label: [intersection_item, ...]}``.
            Each *intersection_item* is a dict with keys ``"point"`` (Shapely
            Point) and ``"lines"`` (list of bearing-line dicts, each with an
            ``"id"``).
    
        Returns
        -------
        dict
            Mapping ``{cluster_label: score}``.
        """

        density_score = self.kde_density()
 
        # Gather the unique bearing lines that contributed to this cluster
        obj_anno_count = len(self.get_object_annotation())

        # Avoid division by zero for single-line clusters <-- I like how carefully the AI is ;-p
        max_pairs = 0.5 * obj_anno_count * (obj_anno_count - 1)
        points_score = self.amount_intersections() / max_pairs if max_pairs > 0 else 0.0
 
        score = points_score * density_score * math.log10(self.amount_intersections())
 
        return score



    def get_object_annotation(self) -> list[ObjectAnnotation]:

        obj_anno: set[ObjectAnnotation]  = set()
        for inters in self.get_intersections():
            obj_anno.add(inters.get_annotation1())
            obj_anno.add(inters.get_annotation2())

        return list(obj_anno) # Converteer terug naar een lijst

    def __calculate_view_direction(self) -> float | None:
        lst_obj_anno = self.get_object_annotation()
        line_direction: float = []
        for obj_anno in lst_obj_anno:
            line_direction.append(obj_anno.direction_deg())

        if not line_direction:
            return None
        return (circmean(line_direction, high=360, low=0).item())% 360

    def get_perpendicular(self) -> float | None:
        return (self.__calculate_view_direction() + 180)% 360


    def get_object_size(self) -> tuple[float | None, float | None]:
        lst_obj_anno = self.get_object_annotation()
        object_sizes: list[float] = []
        for obj_anno in lst_obj_anno:
            size = obj_anno.object_size_m(self.get_center())
            if size is not None:
                object_sizes.append(obj_anno.object_size_m(self.get_center()))

        if len(lst_obj_anno) >= 2:
            size = float(np.mean(object_sizes))
            size_sd = float(np.std(object_sizes, ddof=1))
            return size, size_sd
        return None, None


    def kde_density(self) -> float:
        """Estimate the average pairwise KDE density of a set of (lat, lon) points.
    
        For every unique pair (i, j) the haversine distance is computed and fed
        through :func:`gaussian_kernel`. The result is normalised by the number
        of pairs so that clusters of different sizes are comparable.
    
        Parameters
        ----------
        points:
            Array of shape (N, 2) containing (latitude, longitude) in degrees.
    
        Returns
        -------
        float
            Average kernel density. Returns 0 for clusters with fewer than 2 points.
        """
        points = self.get_coordinates()
        n = len(points)
        if n < 2:
            return 0.0
    
        density = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # haversine() expects (lat, lon) and returns kilometres
                distance_m = haversine(points[i], points[j]) * 1000
                density += self.gaussian_kernel(distance_m, h=0.5)
    
        num_pairs = 0.5 * n * (n - 1)
        return density / num_pairs

    @staticmethod
    def gaussian_kernel(d: float, h: float = 0.5) -> float:
        """Evaluate a Gaussian kernel at distance *d* with bandwidth *h*.
    
        Parameters
        ----------
        d:
            Distance from the kernel centre (metres).
        h:
            Bandwidth in metres. Smaller values produce sharper peaks.
    
        Returns
        -------
        float
            Kernel weight (always positive).
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
        """Recursively cluster a set of line-intersection points into sign locations.
    
        Cluster function:

        1
        Check in the list of intersections already match the criteria to form a cluster.

        2 a
        If this is the case create a defaultdict, set the dict value to next_group_label, add the items and return the list. end of fuction.

        2 b
        else call the cluster algorithm and create a defaultdict for each found cluster and add the intersections. First label is next_group_label, next cluser: next_group_label + 1 and so on.
        The function applies DBSCAN with a haversine metric.

        3
        Remove all cluster that have are not valid (label -1)

        4
        Score the quality of all valid clusters

        5
        Loop to valid clusters
            Loop to all intersections
                Keep only those intersections that are not already in a other cluster

        6
        Remove cluster that don’t meet the minimum requirements any more

        7
        Decrease epsilon
        After each level the
        clusters are scored, conflicts between bearing lines are resolved (each
        line may belong to at most one sign), and then *epsilon* is tightened by
        25 % before recursing.


        8
        Create a dictionary list to add all the cluster to.
        For each cluster
            call this function recursive
            Add the returned dictionary (label and intersections) to the dictionary list
            Set next label to the highest label + 1

        9
        Return the dictionary clusters with their intersections
        

        Recursion terminates when either:
    
        * *current_dept* exceeds *max_dept*, **or**
        * all points already lie within *max_radius_rad* of their centroid.
    
        Parameters
        ----------
        cluster:
            cluster. The cluster contains a List of object of the class Intersection
        max_radius_rad:
            Maximum allowed distance (in radians) from a cluster centroid.
            Points closer than this threshold are already considered well-clustered.
        epsilon:
            DBSCAN neighbourhood radius in radians (≈ metres / EARTH_RADIUS_M).
        next_group_label:
            Integer label assigned to the first cluster at this recursion level.
            Subsequent clusters get incrementing labels to avoid collisions across
            recursive calls.
        max_dept:
            Maximum recursion depth.
        current_dept:
            Current recursion depth (internal counter; callers should leave at 1).
        min_samples:
            Minimum cluster size passed to DBSCAN and used to filter tiny clusters.
    
        Returns
        -------
        List of clusters
        """
        # logger.info(f"start cluster, epsilo: {epsilon}, current_dept: {current_dept}, intersection count: {cluster.amount_intersections()}")

        cluster_lst : list[Cluster] = []
        # --- Base case: maximum depth reached ---------------------------------- #
        if current_dept > max_dept:
            cluster_lst.append(cluster)
            logger.warning(f"max depth reached; returning cluster {cluster.get_id()} ({cluster.amount_intersections()} pts)")
            return cluster_lst
    
        current_dept += 1

        if cluster.get_max_distance_intersection_center_m() < max_radius_m:
            # All points are already within the acceptable radius → single cluster
            cluster_lst.append(cluster)
            return cluster_lst
    
        # --- DBSCAN clustering ------------------------------------------------- #
        # Get coordinates and intersections (guaranteed same order and length)
        coordinates_rad = cluster.get_coordinates_rad()
        intersections = cluster.get_intersections()

        clusterer = DBSCAN(
            eps=epsilon*2*math.pi/EARTH_RADIUS_M,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = clusterer.fit_predict(coordinates_rad)


        # Create new clusters for each label (excluding noise label -1)
        unique_labels = set(labels) - {-1}    
        
        for label in unique_labels:
            new_cluster = Cluster()
            # Add all intersections with current label
            for idx, current_label in enumerate(labels):
                if current_label == label:
                    new_cluster.add_intersection(intersections[idx])
            cluster_lst.append(new_cluster)

    
        clusters_sorted = sorted(
            cluster_lst,
            key=lambda cluster: cluster.score_cluster(),
            reverse=True
        )

        # --- Conflict resolution ----------------------------------------------- #
        # Each bearing line may be assigned to at most one cluster.  We process
        # clusters in descending score order so that high-quality clusters claim
        # their lines first.
        line_to_cluster: dict[int, int] = {}
    
        for cluster in clusters_sorted:
            cluster_id = cluster.get_id()
            for inters in cluster.get_intersections():
                id_a, id_b = inters.get_annotation1().get_id(), inters.get_annotation2().get_id()
    
                # Skip if either line is already owned by a *different* cluster
                conflict_a = id_a in line_to_cluster and line_to_cluster[id_a] != cluster_id
                conflict_b = id_b in line_to_cluster and line_to_cluster[id_b] != cluster_id
                if conflict_a or conflict_b:
                    cluster.remove_intersection(inters)
    
                line_to_cluster[id_a] = cluster_id
                line_to_cluster[id_b] = cluster_id
    
        # Remove clusters that fall below the minimum size
        # Filter clusters to only keep those with enough intersections
        clusters_filtered = [
            cluster for cluster in clusters_sorted
            if cluster.amount_intersections() >= min_samples
        ]

        if not clusters_filtered:
            logger.warning("cluster_filtered is empty; aborting recursion.")
            return clusters_filtered

        # --- Recurse with tighter epsilon -------------------------------------- #
        epsilon *= epsilon_decrease
        
        cluster_result_lst : list[Cluster] = []
        for cluster in clusters_filtered:
            cluster_result_lst.extend(Cluster.funct_cluster(
                cluster, max_radius_m, epsilon, epsilon_decrease, max_dept, current_dept, min_samples))

        return cluster_result_lst    


class Centroid:
    __next_id = 0

    def __init__(self, original_cluster: Cluster)->None:
        self.__id = Centroid.__next_id        
        Centroid.__next_id += 1

        if not isinstance(original_cluster, Cluster):
            raise TypeError("original_cluster must be an Cluster instance")
        self.__original_cluster = original_cluster

        self.__lst_object_annotation: list[ObjectAnnotation] = []

    def add_object_annotation(self, object_annotation: ObjectAnnotation)->None:

        if not isinstance(object_annotation, ObjectAnnotation):
            raise TypeError("object_annotation must be an ObjectAnnotation instance")

        self.__lst_object_annotation.append(object_annotation)

    def get_object_annotations(self)->list[ObjectAnnotation]:
        return self.__lst_object_annotation

    def get_id(self)->int:
        return self.__id

    def get_original_cluster(self)->Cluster:
        return self.__original_cluster

    def get_center(self)->Point:
        return self.__original_cluster.get_center()

    def get_object_size(self)->tuple[float | None, float | None]:
        size, _ =  self.__original_cluster.get_object_size()
        return size

    def get_object_size_sd(self)->tuple[float | None, float | None]:
        _, size_sd =  self.__original_cluster.get_object_size()
        return size_sd

    def get_perpendicular(self)->float:
        return self.__original_cluster.get_perpendicular()

    def get_object_annotation_id(self)->str:
        return self.__lst_object_annotation[0].get_object_annotation_id()

    def get_picture_id(self)->str:
        return self.__lst_object_annotation[0].get_picture_id()

    # def add_valid_object_annotations(self, list_object_annotation: list[ObjectAnnotation], MAX_CLUSTER_RADIUS_M):
    #     # Note for future improvement.
    #     # Possible recalculate new centre.
    #     # Possible add line score to line so best line can be selected.

    #     tree = STRtree([obj_anna.get_line() for obj_anna in list_object_annotation])

    #     DEGREE_BUFFER = MAX_CLUSTER_RADIUS_M * 360 / (EARTH_RADIUS_M * 2 * math.pi)

    #     p = self.get_center()

    #     # Coarse filter: bounding box in degrees around the centroid
    #     search_box = box(
    #         p.x - DEGREE_BUFFER, p.y - DEGREE_BUFFER,
    #         p.x + DEGREE_BUFFER, p.y + DEGREE_BUFFER
    #     )
    #     candidate_indices = tree.query(search_box)          # returns indices into line_geoms
    #     candidates = [list_object_annotation[i] for i in candidate_indices]

    #     # Fine filter: exact geodesic distance for each candidate
    #     for obj_anna in candidates:
    #         dist = obj_anna.min_distance_to_line_m(p)
    #         if dist <= MAX_CLUSTER_RADIUS_M:
    #             self.add_object_annotation(obj_anna)

    #     return None

    def get_lst_object_annotation(self) -> list[ObjectAnnotation]:
        return self.__lst_object_annotation


    # Possible handy for the future, not in use at the moment.
    def update_links_and_sizes(self) -> None:
        """Werk de 'link' en 'size' bij voor alle objectannotaties in dit cluster."""
        for obj_anna in self._object_annotations:

            obj_anna.link = (
                f"{PANORAMAX_END_POINT}"
                f"?annot={obj_anna.get_object_annotation_id()}&pic={obj_anna.get_picture_id()}"
            )
            obj_anna.size = obj_anna.get_object_size()

    @staticmethod
    def add_valid_object_annotations_all_centroids(
    lst_centroids: list['Centroid'],
    list_object_annotation: list['ObjectAnnotation'],
    MAX_CLUSTER_RADIUS_M: float
    ) -> None:
        """
        Voeg alle objectannotaties toe aan de juiste centroids in één keer,
        gebruikmakend van geodesic buffering en spatial join.

        Parameters
        ----------
        lst_centroids : list[Centroid]
            Lijst van alle centroids die moeten worden verwerkt.
        list_object_annotation : list[ObjectAnnotation]
            Lijst van alle objectannotaties (lijnen).
        MAX_CLUSTER_RADIUS_M : float
            Maximale afstand (in meters) voor een lijn om bij een centroid te horen.
        EARTH_RADIUS_M : float, optional
            Straal van de aarde in meters (standaard: 6_371_000).
        """
        def geodesic_point_buffer(lon: float, lat: float, radius_m: float):
            """Create a geodesic buffer around a point."""
            proj = pyproj.Proj(
                proj="aeqd",
                ellps="WGS84",
                datum="WGS84",
                lat_0=lat,
                lon_0=lon
            )
            buf = Point(0, 0).buffer(radius_m)  # Buffer in meters
            return transform(lambda x, y: proj(x, y, inverse=True), buf)

        # --- Stap 1: Converteer centroids naar GeoDataFrame met geodesic buffers ---
        centroids = []
        geometries = []

        for centroid in lst_centroids:
            center = centroid.get_center()
            buffer = geodesic_point_buffer(center.x, center.y, MAX_CLUSTER_RADIUS_M)
            centroids.append(centroid)
            geometries.append(buffer)

        centroids_gdf = gpd.GeoDataFrame(
            {"centroid": centroids},
            geometry=geometries,
            crs="EPSG:4326"
        )

        # --- Stap 2: Converteer lijnen naar GeoDataFrame ---
        lines_gdf = gpd.GeoDataFrame(
            {
                "object_annotation": list_object_annotation,
                "geometry": [obj_anna.get_line() for obj_anna in list_object_annotation]
            },
            crs="EPSG:4326"
        )

        # --- Stap 3: Spatial join ---
        # Reset index om duplicate indexen te voorkomen
        centroids_gdf = centroids_gdf.reset_index(drop=True)
        lines_gdf = lines_gdf.reset_index(drop=True)

        joined = gpd.sjoin(
            lines_gdf,
            centroids_gdf,
            how="inner",
            predicate="intersects"
        )

        # --- Stap 4: Voeg de lijnen toe aan de juiste centroids ---
        for _, row in joined.iterrows():
            centroid = row["centroid"]
            obj_anna = row["object_annotation"]
            centroid.add_object_annotation(obj_anna)


    @staticmethod
    def resolve_centroid_assignments(lst_centroid: list['Centroid'], MAX_CLUSTER_RADIUS_M: float) -> list['Centroid']:
        """
        Resolves ObjectAnnotation assignments to Centroids based on scoring.
        Each ObjectAnnotation is assigned to at most one Centroid (the one with highest score).
        Returns a new list of Centroids with only the winning assignments.
        """

        # Stap 1: Bereken scores voor alle mogelijke combinaties
        score_map = {}  # {(centroid_id, obj_anna_id): score}

        for centroid in lst_centroid:
            centroid_size = centroid.get_object_size()
            cluster_score = centroid.get_original_cluster().score_cluster()
            
            for obj_anna in centroid.get_lst_object_annotation():
                # Bereken afstandsscore
                distance = obj_anna.min_distance_to_line_m(centroid.get_center())
                distance_score = (1 - distance / MAX_CLUSTER_RADIUS_M) ** 2

                # Bereken groottescore. Nog naar classe functie omzetten!!
                object_size = obj_anna.object_size_m(centroid.get_center())
                size_score = object_size_score(centroid_size, object_size)

                # Totale score
                total_score = distance_score * size_score * cluster_score
                score_map[(centroid.get_id(), obj_anna.get_id())] = total_score

        # Stap 2: Bepaal voor elke ObjectAnnotation de beste Centroid
        best_centroid_per_annotation = {}  # {obj_anna_id: (best_centroid_id, best_score)}

        for (centroid_id, obj_anna_id), score in score_map.items():
            if obj_anna_id not in best_centroid_per_annotation or score > best_centroid_per_annotation[obj_anna_id][1]:
                best_centroid_per_annotation[obj_anna_id] = (centroid_id, score)

        # Stap 3: Maak nieuwe Centroids met alleen de winnende ObjectAnnotations
        lst_new_centroids = []

        for centroid in lst_centroid:
            new_centroid = Centroid(centroid.get_original_cluster())

            for obj_anna in centroid.get_lst_object_annotation():
                best_centroid_id, _ = best_centroid_per_annotation.get(obj_anna.get_id(), (None, -1))
                if best_centroid_id == centroid.get_id():
                    new_centroid.add_object_annotation(obj_anna)

            if len(new_centroid.get_object_annotations())>=2:
                lst_new_centroids.append(new_centroid)
        return lst_new_centroids

# support functions
def object_size_score(size_1: float, size_2: float)  -> float:
    return 1 - abs(size_1 - size_2) / (size_1 + size_2)           

