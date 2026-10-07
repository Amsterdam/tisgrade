# Standard library
from datetime import datetime
from pathlib import Path

# Third-party
import psycopg2
import pandas as pd
import geopandas as gpd

# Local
import tisgrade_config as tsgcf
import tisgrade_classes as tsgc
import tisgrade_data_classes as tsgdc
import tisgrade_data_load as tsg_dl
import tisgrade_data_store as tsg_ds

from shapely.geometry import Point, Polygon


from typing import TypedDict, List

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



def lib_locate_object(records_input: List[ObjectAnnotationInput], quality: "tsgdc.QualitySetting"):

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

    # i = 0
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

        annObj = tsgc.ObjectAnnotation(
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


    lst_intersection = tsgc.Intersection.find_intersections(lst_obj_anno_valid)


    # ---------------------------------------------------------------------- #
    # Step 3: Filter to geometrically reliable intersections
    #
    # Criteria:
    #   - Angle between lines ≥ MIN_INTERSECTION_ANGLE_DEG  (location precision)
    #   - Angle between lines ≤ MAX_INTERSECTION_ANGLE_DEG  (both cameras can
    #     plausibly see the same sign within a ±75° viewing cone)
    #   - Relative size discrepancy ≤ MIN_SIZE_SCORE         (same sign, not two)
    # ---------------------------------------------------------------------- #

    logger.info(f'2 find reliable intersections')
    lst_inter_reliable = [
        item for item in lst_intersection
        if (
            (size_score := item.object_size_score()) is not None
            and size_score >= quality.sign_size_score
            and (angle := item.angle_between_lines()) is not None
            and quality.intersection_angle_deg_min <= angle <= quality.intersection_angle_deg_max
        )
    ]

    if not lst_inter_reliable:
        # continue
        return _empty_result(len(records_input))

    logger.info(f'3 create base cluster')
    cluster = tsgc.Cluster()

    for inter_sec in lst_inter_reliable:
        cluster.add_intersection(inter_sec)


    logger.info(f'{datetime.now()-start_dt} time pre processing')   
    # ---------------------------------------------------------------------- #
    # Step 4: Cluster reliable intersections → sign locations
    # ---------------------------------------------------------------------- #
    start_dt = datetime.now()
    logger.info(f'4 start clustering')
    lst_cluster  = tsgc.Cluster.split_cluster(
        parent_cluster=cluster,
        max_radius_m=quality.cluster_radius_m_max,
        epsilon=quality.cluster_epsilon_start,
        epsilon_decrease=quality.cluster_epsilon_decrease,
        max_depth=quality.cluster_max_depth,
        min_samples=quality.cluster_min_size
    )

    logger.info(f'{datetime.now()-start_dt} time clustering')
    # ghost busters and save the orpins part
    # ghost cluster will be eliminated
    # orpin lines close to a cluster will be connected to a cluster

    
    start_dt = datetime.now()
    logger.info(f'5 start save the orphans and ghostbuster')
    lst_centroid: list[tsgc.Centroid] = []

    for cluster in lst_cluster:
        centroid = tsgc.Centroid(cluster)                    
        lst_centroid.append(centroid)
    tsgc.Centroid.add_valid_object_annotations_all_centroids(lst_centroids=lst_centroid, list_object_annotation=lst_obj_anno_valid, MAX_CLUSTER_RADIUS_M= quality.cluster_radius_m_max)

    lst_clean_centroid = tsgc.Centroid.resolve_centroid_assignments(lst_centroid, quality.cluster_radius_m_max)

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
                "sign_size_average_meter": intersection.object_size(),
                "sign_size_score": intersection.object_size_score(),                
                "line1": intersection.get_annotation1().get_external_id(),
                "line2": intersection.get_annotation2().get_external_id()               
            }
            for intersection in lst_inter_reliable
        ])

    intersections_gpkg = _to_gdf([
            {
                "geometry": intersection.intersection(),
                "angle": intersection.angle_between_lines(),
                "sign_size_average_meter": intersection.object_size(),
                "sign_size_score": intersection.object_size_score(),                  
                "line1": intersection.get_annotation1().get_external_id(),
                "line2": intersection.get_annotation2().get_external_id()                
            }
            for intersection in lst_intersection
        ])


    logger.info(f'{datetime.now()-start_dt} time post processing')
    logger.info(f'done')

    
    logger.info(f'len cluster: {len(lst_cluster)}, len centroid: {len(lst_centroid)} len clean cent: {len(lst_clean_centroid)}')

    # return lst_clean_centroid, lst_cluster, lst_inter_reliable, lst_intersection
    return record_labels, centriods_gpkg, clusters_gpkg, intersections_reliable_gpkg, intersections_gpkg, all_annotations_gpkg
    


def locate_panoramax_objects(sign_parameters: "tsgdc.SignParameters", geo_bounds: "tsgdc.GeoBounds", run_parameters: "tsgdc.RunParameters",  quality: "tsgdc.QualitySetting", pg_conn: "psycopg2.extensions.connection"):

    """
    Workflow
    --------
    1. Connect to the PostGIS database.
    2. Query the distinct sign types present in ``panoramax_semantics``.
    3. For each sign type:
    a. Fetch all panoramic-photo annotations within the configured bounding box.
    b. Convert each annotation to a bearing line (a ray from the camera toward
        the sign, bounded by estimated min/max distance).
    c. Find all pairwise intersections between bearing lines.
    d. Enrich intersections with distance, size, and angular-deviation metrics.
    e. Filter intersections to retain only geometrically reliable ones.
    f. Cluster the reliable intersections with :func:`tsgf.split_cluster`.
    g. Compute for each cluster a summary records (centroid, size, direction).
    h. Persist results to the database and to local GeoPackage files.
    4. Optionally append George (NDW wegkenmerken) reference signs for comparison.
    
    Dependencies
    ------------
        psycopg2, geopandas, shapely, scipy, numpy, geopy, haversine, hdbscan,
        scikit-learn, json
        tisgrade_functons (local module, imported as tsgf)
    """
    
    # ===========================================================================
    # Configuration
    # ===========================================================================
    cur = pg_conn.cursor()
    

    # ===========================================================================
    # Create temp data set
    # ===========================================================================
    logger.info(f"Regex types to process: {sign_parameters.sign_regex}")

    tsg_dl.create_temp_dataset(cursor=cur,
                        longitude_min = geo_bounds.longitude_min,
                        longitude_max = geo_bounds.longitude_max,
                        latitude_min = geo_bounds.latitude_min,
                        latitude_max = geo_bounds.latitude_max, 
                        start_date = run_parameters.run_start_date_time,
                        end_date = run_parameters.run_end_date_time,                                             
                        sign_regex = sign_parameters.sign_regex)

    pg_conn.commit()


    # ===========================================================================
    # Main processing loop
    # ===========================================================================
    logger.info("Get sign list")
    arr_sign_types = []

    arr_sign_types = tsg_dl.get_signs_list(cursor=cur)

    logger.info(f"Sign types to process: {arr_sign_types}")
    
    for row in arr_sign_types:
        sign_type       = row[0]
        sign_linecount  = row[1]
        logger.info(f"Processing sign_type: {sign_type}  (observations: {sign_linecount})")
    
        # ---------------------------------------------------------------------- #
        # Step 1: Fetch annotations and build bearing lines
        # ---------------------------------------------------------------------- #
        records = []
        records = tsg_dl.get_signs(cursor=cur,  
                                        sign_type=sign_type)

        logger.info(f'start data preprocessing')
    
        logger.info(f'1 create objects')

        records_input = []
        for record in records:
            sing_min_size = sign_parameters.sign_size_min * (1 - sign_parameters.sign_size_margin)
            sing_max_size = sign_parameters.sign_size_max * (1 + sign_parameters.sign_size_margin)

            rec_inp = {                
                "origin"                    : record["origin"],
                "picture_field_of_view"     : record["picture_field_of_view"],
                "picture_azimuth"           : record["picture_azimuth"],
                "picture_size_horizontal_px": record["picture_px_horizontal"],
                "picture_size_vertical_px"  : record["picture_px_vertical"],
                "bbox"                      : record["bbox_clean"],
                "object_minimum_size_meters": sing_min_size,
                "object_maximum_size_meters": sing_max_size
            }
            records_input.append(rec_inp)

        logger.info(f'records: {len(records)}, object annatotions: {len(records_input)}')

        labels, centriods_gpkg, clusters_gpkg, intersections_reliable_gpkg, intersections_gpkg, lines_gpkg  = lib_locate_object(records_input, quality)

        print(f'lables: {labels}')
    

        centriods_gpkg["object_code"] = sign_type
        clusters_gpkg["object_code"] = sign_type
        intersections_reliable_gpkg["object_code"] = sign_type
        intersections_gpkg["object_code"] = sign_type
        lines_gpkg["object_code"] = sign_type


        if lines_gpkg.empty:
            logger.info(f"No lines for sign_type {sign_type}")
            continue

        id_to_annotation_id         = {i: rec.get("annotation_id")          for i, rec in enumerate(records)}
        id_to_picture_id            = {i: rec.get("picture_id")             for i, rec in enumerate(records)}
        id_to_picture_location      = {i: rec.get("picture_coordinates")    for i, rec in enumerate(records)}
        id_to_picture_timestamptz   = {i: rec.get("picture_timestamptz")    for i, rec in enumerate(records)}

        lines_gpkg["id"] = lines_gpkg["id"].astype("int64")
        lines_gpkg["annotation_id"] = lines_gpkg["id"].map(id_to_annotation_id)
        lines_gpkg["picture_id"] = lines_gpkg["id"].map(id_to_picture_id)
        lines_gpkg["picture_location"] = lines_gpkg["id"].map(id_to_picture_location)
        lines_gpkg["picture_timestamptz"] = lines_gpkg["id"].map(id_to_picture_timestamptz)
        lines_gpkg["link"] = (
            f"{tsgcf.PANORAMAX_END_POINT}?annot=" + lines_gpkg["annotation_id"].astype(str)
            + "&pic=" + lines_gpkg["picture_id"].astype(str)
        )
        lines_gpkg["sign_type"] = sign_type



        # output_path = Path(tsgcf.OUTPUT_DIR_GEO_PACK) / "centriods_02.gpkg"
        tsg_ds.store_gpkg(gpkg_data = centriods_gpkg, type="Centroid")
        tsg_ds.store_gpkg(gpkg_data = clusters_gpkg, type="Cluster")
        tsg_ds.store_gpkg(gpkg_data = intersections_reliable_gpkg, type="Intersections reliable")
        tsg_ds.store_gpkg(gpkg_data = intersections_gpkg, type="Intersections")
        tsg_ds.store_gpkg(gpkg_data = lines_gpkg, type="Line")
        # all_lines.append(lines_gpkg)

        tsg_ds.store_centriod_db(
            centriods_gpkg= centriods_gpkg,
            lines_gpkg= lines_gpkg,
            cur=cur,
            start_timestamp= run_parameters.run_start_date_time,
            end_timestamp= run_parameters.run_end_date_time,
            run_name= run_parameters.run_name
        )
    
        logger.info(f'Done — {sign_linecount} observations processed for sign type: {sign_type}')
    
    logger.info(f'\nAll sign types processed. Pipeline complete.')


"""
main.py
-------
Main entry point for the traffic-sign geolocation pipeline.
"""
if __name__ == "__main__":

    logger = tsgcf.setup_logging(debug=True)

    try:
        logger.info("Application started")

        # define area
        # LONGITUDE_MIN = 4.730066 # x min
        # LONGITUDE_MAX = 5.108092 # x max
        # LATITUDE_MIN = 52.282888 # y min
        # LATITUDE_MAX = 52.432766 # y max

        LONGITUDE_MIN = 4.907730 # x min
        LONGITUDE_MAX = 4.910374 # x max
        LATITUDE_MIN = 52.358927 # y min
        LATITUDE_MAX = 52.361559 # y max        

        # sign details
        # SIGN_REX = 'NL:L02'
        SIGN_REX = 'NL:J14'
        SIGN_REX = 'zone:other'
        SIGN_MIN = 0.4
        SIGN_MAX = 0.8
        SIGN_MARGIN = 0.25

        # run details
        # RUN_START = '2024-01-01 00:00:00'
        # RUN_END = '2026-09-01 00:00:00'
        RUN_START = '2025-01-01 00:00:00'
        RUN_END = '2026-01-01 00:00:00'
        RUN_NAME = 'Joost test db store'

        # quality setting
        INTERS_MIN = 10         # minimum angle betwee to lines to form a intersection
        INTERS_MAX = 150        # maximum angle between to lines to form a intersection
        SIGN_SIZE_SCORE = 0.8   # minimum score for compaing the size of the same sing for two lines based on there intersection.
        MAX_CLUSTER_RADIUS_M = 3 #distance for all the intersections to be from the middel of the cluster.
        MIN_CLUSTER_SIZE = 1    # Minimum amount of intersections in a cluster
        MAX_RECURSIONS = 20     # Maximum dept of recursion of cluster algoritm
        EPSILON_START = 1.5     # Start value of Epsilon
        EPSION_DECREASE = .75   # The factor with epsilon gets smaller each time a recursion occurs. Must be smaller than 1 and bigger than 0
        

        geo_bounds=tsgdc.GeoBounds(
            longitude_min=LONGITUDE_MIN,
            longitude_max=LONGITUDE_MAX,
            latitude_min=LATITUDE_MIN,
            latitude_max=LATITUDE_MAX
        )

        logger.info(f"geo settings, longitude: {geo_bounds.longitude_min} - {geo_bounds.longitude_max}, latitude: {geo_bounds.latitude_min} - {geo_bounds.latitude_max}")

        sign_param=tsgdc.SignParameters(
            sign_regex = SIGN_REX,
            sign_size_min = SIGN_MIN,
            sign_size_max = SIGN_MAX,
            sign_size_margin = SIGN_MARGIN
        )

        logger.info(f"sign settings, regex: {sign_param.sign_regex}, sign sizes: {sign_param.sign_size_min} - {sign_param.sign_size_max}, margin: {sign_param.sign_size_margin}")

        run_param=tsgdc.RunParameters(
            run_start_date_time = datetime.strptime(RUN_START, '%Y-%m-%d %H:%M:%S').astimezone(),
            run_end_date_time = datetime.strptime(RUN_END, '%Y-%m-%d %H:%M:%S').astimezone(),
            run_name = RUN_NAME
        )

        ql_setings=tsgdc.QualitySettings(
            intersection_angle_deg_min = INTERS_MIN,
            intersection_angle_deg_max = INTERS_MAX,
            sign_size_score = SIGN_SIZE_SCORE,
            cluster_radius_m_max = MAX_CLUSTER_RADIUS_M,
            cluster_min_size = MIN_CLUSTER_SIZE,
            cluster_max_depth = MAX_RECURSIONS,
            cluster_epsilon_start = EPSILON_START,
            cluster_epsilon_decrease = EPSION_DECREASE
        )

        pg_conn = tsgcf.get_db_connection()

        locate_panoramax_objects(sign_param, geo_bounds, run_param, ql_setings, pg_conn)  

        logger.info("Application completed successfully")

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        raise
