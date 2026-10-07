# Standard library
from datetime import datetime

# Third-party
import psycopg2

# Local
import tisgrade_config as tsgcf
import tisgrade_data_classes as tsgdc
import tisgrade_data_load as tsg_dl
import tisgrade_data_store as tsg_ds
from triangulation import locate_object



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

        # labels, centriods_gpkg, clusters_gpkg, intersections_reliable_gpkg, intersections_gpkg, lines_gpkg  = lib_locate_object(records_input, quality)
        labels, centriods_gpkg, clusters_gpkg, intersections_reliable_gpkg, intersections_gpkg, lines_gpkg  = locate_object(
            records_input,
            intersection_angle_deg_min = quality.intersection_angle_deg_min,
            intersection_angle_deg_max = quality.intersection_angle_deg_max,
            object_size_score = quality.sign_size_score,
            cluster_radius_m_max = quality.cluster_radius_m_max,
            cluster_min_size = quality.cluster_min_size,
            cluster_max_depth = quality.cluster_max_depth,
            cluster_epsilon_start = quality.cluster_epsilon_start,
            cluster_epsilon_decrease = quality.cluster_epsilon_decrease
        )    

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
        if tsgcf.STORE_GEO_PACK:
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
