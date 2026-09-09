def get_db_connection () -> "psycopg2.extensions.connection":
    _pgpass_path = TOKEN_PATH
    with open(_pgpass_path, "r") as _f:
        _token = _f.read().strip().split(":")[-1]
    
    _username = USERNAME
    _host     = HOST
    _port     = PORT
    _database = DB
    
    _conn_string = (
        f"postgresql://{quote_plus(_username)}:{_token}"
        f"@{_host}:{_port}/{_database}"
    )

    print(f"_conn_string: {_conn_string}")

    logger.info("Connecting to database")
    pg_connection = psycopg2.connect(
        _conn_string,
        keepalives=KEEPALIVES,
        keepalives_idle=KEEPALIVES_IDLE,            # start probing after xs idle
        keepalives_interval=KEEPALIVES_INTERVAL,    # retry every xs
        keepalives_count=KEEPALIVES_COUNT,          # give up (and let the OS/driver notice) after x failed probes
    )
    return pg_connection


def setup_logging(debug=False):
    """Centralized logging configuration for the entire application."""

    # Create log directory
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}.log"

    # Get the named logger
    logger = logging.getLogger("tisgrade")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear any existing handlers
    for h in logger.handlers:
        h.close()
    logger.handlers.clear()

    logger.propagate = False


    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create console handler if in debug mode
    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Suppress noisy third-party logs
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('pyogrio').setLevel(logging.WARNING)

    logger.info("Logging initialized. Log file: %s", log_file)

    return logger


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
    f. Cluster the reliable intersections with :func:`tsgf.funct_cluster`.
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

    logger.info("Sign types to process: {arr_sign_types}")

    
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

        start_dt = datetime.now()
        logger.info(f'start data preprocessing')

        lst_obj_anna = []
    
        logger.info(f'1 create objects')
        for record in records:
            annObj = tsgc.ObjectAnnotation(
                    object_code = sign_type, 
                    bbox = record["bbox_clean"], 
                    origin = record["origin"], 
                    pic_size_hor_px = record["picture_px_horizontal"], 
                    pic_size_ver_px = record["picture_px_vertical"],
                    pic_field_of_view = record["picture_field_of_view"],
                    pic_azimuth = record["picture_azimuth"],
                    picture_id = record["picture_id"],
                    annotation_id = record["annotation_id"]
            )

            sing_min_size = sign_parameters.sign_size_min * (1 - sign_parameters.sign_size_margin)
            sing_max_size = sign_parameters.sign_size_max * (1 + sign_parameters.sign_size_margin)

            annObj.set_min_object_size(sing_min_size)
            annObj.set_max_object_size(sing_max_size)

            lst_obj_anna.append(annObj)
    
        # ---------------------------------------------------------------------- #
        # Step 2: Find pairwise intersections and enrich them
        # ---------------------------------------------------------------------- #
        logger.info(f'2 find intersections')
        lst_intersection = tsgc.Intersection.find_intersections(lst_obj_anna)

    
        # ---------------------------------------------------------------------- #
        # Step 3: Filter to geometrically reliable intersections
        #
        # Criteria:
        #   - Angle between lines ≥ MIN_INTERSECTION_ANGLE_DEG  (location precision)
        #   - Angle between lines ≤ MAX_INTERSECTION_ANGLE_DEG  (both cameras can
        #     plausibly see the same sign within a ±75° viewing cone)
        #   - Relative size discrepancy ≤ MIN_SIZE_SCORE         (same sign, not two)
        # ---------------------------------------------------------------------- #

        logger.info(f'3 find reliable intersections')
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
            continue

        logger.info(f'4 create base cluster')
        cluster = tsgc.Cluster()

        for inter_sec in lst_inter_reliable:
            cluster.add_intersection(inter_sec)


        logger.info(f'{datetime.now()-start_dt} time pre processing')   
        # ---------------------------------------------------------------------- #
        # Step 4: Cluster reliable intersections → sign locations
        # ---------------------------------------------------------------------- #
        start_dt = datetime.now()
        logger.info(f'start clustering')
        lst_cluster  = tsgc.Cluster.funct_cluster(
            cluster=cluster,
            max_radius_m=quality.cluster_radius_m_max,
            epsilon=quality.cluster_epsilon_start,
            epsilon_decrease=quality.cluster_epsilon_decrease,
            max_dept=quality.cluster_max_dept,
            min_samples=quality.cluster_min_size
        )

        logger.info(f'{datetime.now()-start_dt} time clustering')
        # ghost busters and save the orpins part
        # ghost cluster will be eliminated
        # orpin lines close to a cluster will be connected to a cluster

        
        start_dt = datetime.now()
        logger.info(f'start save the orphans and ghostbuster')
        lst_centroid: list[tsgc.Centroid] = []

        lst_obj_anna = [
            obj_anna
            for obj_anna in lst_obj_anna
            if obj_anna.get_line() is not None and obj_anna.get_line().is_valid
        ]



        for cluster in lst_cluster:
            centroid = tsgc.Centroid(cluster)                    
            lst_centroid.append(centroid)
        tsgc.Centroid.add_valid_object_annotations_all_centroids(lst_centroids=lst_centroid, list_object_annotation=lst_obj_anna, MAX_CLUSTER_RADIUS_M= quality.cluster_radius_m_max)

        lst_clean_centroid = tsgc.Centroid.resolve_centroid_assignments(lst_centroid, quality.cluster_radius_m_max)

        logger.info(f'{datetime.now()-start_dt} time post processing')
        logger.info(f'done')

        logger.info(f'records: {len(records)}, object annatotions: {len(lst_obj_anna)}')
        logger.info(f'len cluster: {len(lst_cluster)}, len centroid: {len(lst_centroid)} len clean cent: {len(lst_clean_centroid)}')
            
    
        # ---------------------------------------------------------------------- #
        # Step 5: Persist to database and GeoPackage files
        # ---------------------------------------------------------------------- #
        if STORE_GEO_PACK:
            if lst_obj_anna:
                tsg_ds.write_to_gpkg_lines(OUTPUT_DIR_GEO_PACK, lst_obj_anna)
            if lst_intersection:
                tsg_ds.write_to_gpkg_intersection(OUTPUT_DIR_GEO_PACK, lst_intersection, sign_type)
            if lst_inter_reliable:
                tsg_ds.write_to_gpkg_intersection_reliable(OUTPUT_DIR_GEO_PACK, lst_inter_reliable, sign_type)
            if lst_cluster:
                tsg_ds.write_to_gpkg_clusters(OUTPUT_DIR_GEO_PACK, lst_cluster, sign_type)
            if lst_clean_centroid:
                tsg_ds.write_to_gpkg_centriods(OUTPUT_DIR_GEO_PACK, lst_clean_centroid, sign_type)
        
        tsg_ds.write_centriod_to_db(
            cur, 
            lst_clean_centroid, 
            sign_type, 
            start_timestamp=run_parameters.run_start_date_time, 
            end_timestamp=run_parameters.run_end_date_time, 
            run_name=run_parameters.run_name
        )
    
        logger.info(f'Done — {sign_linecount} observations processed for sign type: {sign_type}')
    
    logger.info(f'\nAll sign types processed. Pipeline complete.')


"""
main.py
-------
Main entry point for the traffic-sign geolocation pipeline.
"""
if __name__ == "__main__":

    # Standard library
    from urllib.parse import quote_plus
    from datetime import datetime
    import os
    from pathlib import Path
    import logging
    from dotenv import load_dotenv
    
    
    # Third-party
    import psycopg2

    
    # Local
    import tisgrade_classes as tsgc
    import tisgrade_data_classes as tsgdc
    import tisgrade_data_load as tsg_dl
    import tisgrade_data_store as tsg_ds 

    # config information
    # from tisgrade_config import STORE_GEO_PACK, OUTPUT_DIR_GEO_PACK
    # from tisgrade_config import USERNAME, HOST, PORT, DB, TOKEN_PATH
    # from tisgrade_config import KEEPALIVES, KEEPALIVES_IDLE, KEEPALIVES_INTERVAL, KEEPALIVES_COUNT
    # from tisgrade_config import setup_logging

    # For development (shows debug and info messages in console)
    # logger = setup_logging(debug=True)
    
    # For production (only writes to log file)

    load_dotenv()

    STORE_GEO_PACK          = STORE_GEO_PACK = os.environ["STORE_GEO_PACK"].lower() == "true"
    OUTPUT_DIR_GEO_PACK     = Path(os.environ["OUTPUT_DIR_GEO_PACK"])

    USERNAME                = os.environ["DB_USERNAME"]
    HOST                    = os.environ["HOST"]
    PORT                    = os.environ["PORT"]
    DB                      = os.environ["DB"]
    TOKEN_PATH              = Path(os.environ["TOKEN_PATH"])

    KEEPALIVES              = int(os.environ["KEEPALIVES"])
    KEEPALIVES_IDLE         = int(os.environ["KEEPALIVES_IDLE"])
    KEEPALIVES_INTERVAL     = int(os.environ["KEEPALIVES_INTERVAL"])
    KEEPALIVES_COUNT        = int(os.environ["KEEPALIVES_COUNT"])


    logger = setup_logging(debug=True)

    try:
        logger.info("Application started")

        # define area
        LONGITUDE_MIN = 4.730066 # x min
        LONGITUDE_MAX = 5.108092 # x max
        LATITUDE_MIN = 52.282888 # y min
        LATITUDE_MAX = 52.432766 # y max

        # sign details
        SIGN_REX = 'NL:C21'
        SIGN_MIN = 0.4
        SIGN_MAX = 0.8
        SIGN_MARGIN = 0.25

        # run details
        RUN_START = '2025-01-01 00:00:00'
        RUN_END = '2026-01-01 00:00:00'
        RUN_NAME = 'Joost test run'

        # quality setting
        INTERS_MIN = 10 # minimum angle betwee to lines to form a intersection
        INTERS_MAX = 150 # maximum angle between to lines to form a intersection
        SIGN_SIZE_SCORE = 0.8 # minimum score for compaing the size of the same sing for two lines based on there intersection.
        MAX_CLUSTER_RADIUS_M = 3 #distance for all the intersections to be from the middel of the cluster.
        MIN_CLUSTER_SIZE = 1 # Minimum amount of intersections in a cluster
        MAX_RECURSIONS = 20 # Maximum dept of recursion of cluster algoritm
        EPSILON_START = 1.5 # Start value of Epsilon
        EPSION_DECREASE = .75 # The factor with epsilon gets smaller each time a recursion occurs. Must be smaller than 1 and bigger than 0
        

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
            sign_size_score = SIGN_MARGIN,
            cluster_radius_m_max = MAX_CLUSTER_RADIUS_M,
            cluster_min_size = MIN_CLUSTER_SIZE,
            cluster_max_dept = MAX_RECURSIONS,
            cluster_epsilon_start = EPSILON_START,
            cluster_epsilon_decrease = EPSION_DECREASE
        )

        pg_conn = get_db_connection()

        locate_panoramax_objects(sign_param, geo_bounds, run_param, ql_setings, pg_conn)  

        logger.info("Application completed successfully")

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        raise
