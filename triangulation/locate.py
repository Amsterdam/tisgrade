# Standard library
import logging
from datetime import datetime
from typing import TypedDict, List, Optional
import math

# Third-party geospatial and numerical libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# local
# import triangulation_classes as tricl
from . import classes as tricl


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_classes
# logger = logging.getLogger(f"tisgrade.{__name__}")
logger = logging.getLogger(__name__)

# ===========================================================================
# Classes
# ===========================================================================


class ObjectAnnotationInput(TypedDict):
    """
    Input record for object localization with polygonal bounding shapes

    Attributes:
        origin: Source/origin of the annotation
        picture_field_of_view: Camera field of view in degrees
        picture_azimuth: Camera azimuth angle in degrees
        picture_size_horizontal_px: Image width in pixels
        picture_size_vertical_px: Image height in pixels
        bbox: Polygon as list of points [[x1,y1], [x2,y2], ..., [xn,yn]]
              where first and last point must be identical to form a closed shape
        object_minimum_size_meters: Minimum object size in meters
        object_maximum_size_meters: Maximum object size in meters
    """
    origin: Point
    picture_field_of_view: float | int
    picture_azimuth: float | int
    picture_size_horizontal_px: int
    picture_size_vertical_px: int
    bbox: Polygon 
    object_minimum_size_meters: float | int
    object_maximum_size_meters: float | int



def _to_gdf(records: list[dict]) -> gpd.GeoDataFrame:
    """GeoDataFrame from a list of dicts, or an empty one with a CRS if there are none."""
    if not records:
        return gpd.GeoDataFrame({"id": pd.Series(dtype="int64")}, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def locate_object(  records_input: List[ObjectAnnotationInput],
                    intersection_angle_deg_min : int|float = 15,
                    intersection_angle_deg_max : int|float|None = None,
                    object_size_score : float = 0.8,
                    cluster_radius_m_max : int|float = 1,
                    cluster_min_size : int = 1,
                    cluster_max_depth : int|None = None,
                    cluster_epsilon_start : int|float|None = None,
                    cluster_epsilon_decrease : float = 0.8
                ):

    if cluster_epsilon_start is None:
        cluster_epsilon_start = cluster_radius_m_max

    if intersection_angle_deg_max is None:
        intersection_angle_deg_max = 360
    elif intersection_angle_deg_max < 0:
        intersection_angle_deg_max = 0
    elif intersection_angle_deg_max > 360:
        intersection_angle_deg_max = 360

    if cluster_max_depth is None:
        cluster_max_depth = math.ceil(math.log(0.001)/math.log(0.8))


    def _empty_result(n):
        def empty():
            return gpd.GeoDataFrame({"id": pd.Series(dtype="int64")}, geometry=[], crs="EPSG:4326")
        return [-1] * n, empty(), empty(), empty(), empty(), empty()    
    # ---------------------------------------------------------------------- #
    # Step 2: Find pairwise intersections and enrich them
    # ---------------------------------------------------------------------- #
    start_dt = datetime.now()
    logger.info(f'1 find intersections')


    # Validate input records
    if not records_input:
        logger.warning("Empty records_input provided")
        return _empty_result(len(records_input))

    record_labels = [-1] * len(records_input)
    lst_obj_anno = []
    for idx, rec_anna in enumerate(records_input):
        # Validate polygon
        if len(rec_anna['bbox']) < 3:
            logger.warning(f"Skipping invalid polygon with {len(rec_anna['bbox'])} points")
            continue

        if rec_anna['bbox'][0] != rec_anna['bbox'][-1]:
            logger.warning("Polygon not closed - adding closing point")
            rec_anna['bbox'].append(rec_anna['bbox'][0])

        annObj = tricl.ObjectAnnotation(
                origin=rec_anna['origin'],
                pic_field_of_view=rec_anna['picture_field_of_view'],
                pic_azimuth=rec_anna['picture_azimuth'],
                pic_size_hor_px=rec_anna['picture_size_horizontal_px'],
                pic_size_ver_px=rec_anna['picture_size_vertical_px'],
                bbox=rec_anna['bbox'],
                object_minimum_size_meters=rec_anna['object_minimum_size_meters'],
                object_maximum_size_meters=rec_anna['object_maximum_size_meters'],
                external_id=idx
        )        
        lst_obj_anno.append(annObj)

    lst_obj_anno_valid = []
    for obj_anno in lst_obj_anno:
        try:
            line = obj_anno.get_line()
        except (ZeroDivisionError, ValueError):
            logger.warning(f"Skipping record {obj_anno.get_external_id()}: line could not be calculated")
            continue
        if line is not None and line.is_valid:
            lst_obj_anno_valid.append(obj_anno)



    if not lst_obj_anno_valid:
        logger.warning("No valid annotations created from input records")
        return _empty_result(len(records_input))


    lst_intersection = tricl.Intersection.find_intersections(lst_obj_anno_valid)


    # ---------------------------------------------------------------------- #
    # Step 3: Filter to geometrically reliable intersections
    #
    # Criteria:
    #   - Angle between lines ≥ MIN_INTERSECTION_ANGLE_DEG  (location precision)
    #   - Angle between lines ≤ MAX_INTERSECTION_ANGLE_DEG  (both cameras can
    #     plausibly see the same object within a ±75° viewing cone)
    #   - Relative size discrepancy ≤ MIN_SIZE_SCORE         (same object, not two)
    # ---------------------------------------------------------------------- #

    logger.info(f'2 find reliable intersections')
    lst_inter_reliable = [
        item for item in lst_intersection
        if (
            (size_score := item.object_size_score()) is not None
            and size_score >= object_size_score
            and (angle := item.angle_between_lines()) is not None
            and intersection_angle_deg_min <= angle <= intersection_angle_deg_max
        )
    ]

    if not lst_inter_reliable:
        return _empty_result(len(records_input))

    logger.info(f'3 create base cluster')
    cluster = tricl.Cluster()

    for inter_sec in lst_inter_reliable:
        cluster.add_intersection(inter_sec)


    logger.info(f'{datetime.now()-start_dt} time pre processing')   
    # ---------------------------------------------------------------------- #
    # Step 4: Cluster reliable intersections → object locations
    # ---------------------------------------------------------------------- #
    start_dt = datetime.now()
    logger.info(f'4 start clustering')
    lst_cluster = tricl.Cluster.split_cluster(
        parent_cluster=cluster,
        max_radius_m=cluster_radius_m_max,
        epsilon=cluster_epsilon_start,
        epsilon_decrease=cluster_epsilon_decrease,
        max_depth=cluster_max_depth,
        min_samples=cluster_min_size
    )

    logger.info(f'{datetime.now()-start_dt} time clustering')
    # ghost busters and save the orpins part
    # ghost cluster will be eliminated
    # orpin lines close to a cluster will be connected to a cluster

    
    start_dt = datetime.now()
    logger.info(f'5 start save the orphans and ghostbuster')
    lst_centroid: list[tricl.Centroid] = []

    for cluster in lst_cluster:
        centroid = tricl.Centroid(cluster)                    
        lst_centroid.append(centroid)
    tricl.Centroid.add_valid_object_annotations_all_centroids(lst_centroids=lst_centroid, list_object_annotation=lst_obj_anno_valid, MAX_CLUSTER_RADIUS_M= cluster_radius_m_max)

    lst_clean_centroid = tricl.Centroid.resolve_centroid_assignments(lst_centroid, cluster_radius_m_max)

    centriods_gpkg = _to_gdf([
        {
            "geometry": cent.get_center(),
            "id" : cent.get_id (),             
            "object_size_meter": cent.get_object_size(),
            "object_size_meter_sd": cent.get_object_size_sd(),
            "object_azimuth" : cent.get_azimuth() 
        }
        for cent in lst_clean_centroid
    ])

    line_records = []
    assigned_ids = set()
    for cl_cen in lst_clean_centroid:
        cl_cen_id = cl_cen.get_id()
        center = cl_cen.get_center()        

        for anno in cl_cen.get_object_annotations():
            record_number = anno.get_external_id()
            assigned_ids.add(record_number) 

            # Add to each record the label of the centroid it belongs to.
            if 0 <= record_number < len(record_labels):
                record_labels[record_number] = cl_cen_id
            else:
                logger.warning(f"Invalid record number {record_number}")

            line_records.append({
                "geometry": anno.get_line(),
                "origin_longitude": anno.get_origin().x,
                "origin_latitude": anno.get_origin().y,
                "centroid_id": cl_cen_id,
                "id": record_number,
                "line_azimuth": anno.direction_deg(),
                "object_distance_meter": anno.distance_m(center),
                "object_size_meter": anno.object_size_m(center),
            })

    # Annotations that did not end up in any centroid.
    for anno in lst_obj_anno_valid:
        record_number = anno.get_external_id()
        if record_number in assigned_ids:
            continue
        line_records.append({
            "geometry": anno.get_line(),
            "origin_longitude": anno.get_origin().x,
            "origin_latitude": anno.get_origin().y,
            "centroid_id": None,
            "id": record_number,
            "line_azimuth": anno.direction_deg(),
            "object_distance_meter": None,
            "object_size_meter": None,
        })

    all_annotations_gpkg = _to_gdf(line_records)
    all_annotations_gpkg["centroid_id"] = all_annotations_gpkg["centroid_id"].astype("Int64")

    clusters_gpkg = _to_gdf([
            {
                "geometry": cluster.get_center(),
                "object_size_meter": cluster.get_object_size()[0],
                "object_size_meter_sd": cluster.get_object_size()[1],           
                "object_azimuth" : cluster.get_azimuth()    
            }
            for cluster in lst_cluster
        ])

    intersections_reliable_gpkg = _to_gdf([
        # [
            {
                "geometry": intersection.intersection(),
                "angle": intersection.angle_between_lines(),
                "object_size_average_meter": intersection.object_size(),
                "object_size_score": intersection.object_size_score(),                
                "line1": intersection.get_annotation1().get_external_id(),
                "line2": intersection.get_annotation2().get_external_id()               
            }
            for intersection in lst_inter_reliable
        ])

    intersections_gpkg = _to_gdf([
            {
                "geometry": intersection.intersection(),
                "angle": intersection.angle_between_lines(),
                "object_size_average_meter": intersection.object_size(),
                "object_size_score": intersection.object_size_score(),                  
                "line1": intersection.get_annotation1().get_external_id(),
                "line2": intersection.get_annotation2().get_external_id()                
            }
            for intersection in lst_intersection
        ])


    logger.info(f'{datetime.now()-start_dt} time post processing')
    logger.info(f'done')

    
    logger.info(f'len cluster: {len(lst_cluster)}, len centroid: {len(lst_centroid)} len clean cent: {len(lst_clean_centroid)}')

    return record_labels, centriods_gpkg, clusters_gpkg, intersections_reliable_gpkg, intersections_gpkg, all_annotations_gpkg
