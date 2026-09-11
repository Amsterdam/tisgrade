# Standard library
from datetime import datetime
from pathlib import Path
import logging

# Third-party
import geopandas as gpd
from psycopg2 import sql
from psycopg2.extras import execute_values

# Local
import tisgrade_config as tsgcf
import tisgrade_classes as tsgc


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_data_store
logger = logging.getLogger(f"tisgrade.{__name__}")
 

# ===========================================================================
# Database persistence
# ===========================================================================
 
def write_centriod_to_db(cur, lst_centroid: list[tsgc.Centroid], sign_code: str, start_timestamp: str, end_timestamp: str, run_name: str = '') -> None:

    """
    Persist calculated centroid locations and their source bearing lines to PostGIS.

    The function writes two types of records:
    1. Object locations / centroids.
    2. Bearing lines that contributed to those centroids.

    All database writes are wrapped in a single transaction.
    If an error occurs, the transaction is rolled back and the exception is raised.

    Parameters
    ----------
    cur:
        Open psycopg2 cursor.

    lst_centroid:
        List of Centroid objects to write.

    sign_code:
        Traffic sign code used to tag the inserted records.

    start_timestamp:
        Start timestamp of the input data period.

    end_timestamp:
        End timestamp of the input data period.

    run_name:
        Optional name for this run.
    """

    # If cur is explicitly False, do nothing.
    if cur==False:
        return
    
    # If there are no centroids to write, do nothing.
    if not lst_centroid:
        return
 
    # Timestamp for this write/run.
    timestamp = datetime.now().astimezone()
 
    try:
        # Start an explicit transaction.
        cur.execute("BEGIN")
 
        # ------------------------------------------------------------------ #
        # Insert sign locations
        # ------------------------------------------------------------------ #

        # Build rows for the centroid/output location table.
        # Each tuple matches the column order in the INSERT statement below.
        location_values = [
            (
                sign_code,
                centriod.get_center().y, # latitude
                centriod.get_center().x, # longitude
                centriod.get_object_size(),
                centriod.get_perpendicular(),
                timestamp,
                centriod.get_center().wkt,
                start_timestamp,
                end_timestamp,
                run_name,
                tsgcf.USERNAME
            )
            for centriod in lst_centroid
        ]

        # Build INSERT query for centroid locations.
        # Schema and table names are safely inserted as SQL identifiers.
        insert_query = sql.SQL (
            """INSERT INTO {schema}.{table}
                    (object, latitude, longitude, size, direction, run_timestamp, location, period_start, period_end, run_name, run_user)
                VALUES %s
            """).format(
                schema = sql.Identifier(tsgcf.DB_SCHEMA),
                table = sql.Identifier(tsgcf.OUT_TABLE_CENTRIOD)
        )

        # Bulk insert all centroid location rows.
        execute_values(
            cur,
            insert_query,
            location_values
        )  
 
        # Retrieve the auto-generated IDs in insertion order.
        # These IDs are needed to link bearing lines to their centroid location.
        query = sql.SQL (
            """
                SELECT id
                FROM {schema}.{table}
                WHERE object = %s AND run_timestamp = %s
                ORDER BY id
            """).format(
                schema = sql.Identifier(tsgcf.DB_SCHEMA),
                table = sql.Identifier(tsgcf.OUT_TABLE_CENTRIOD)
        )

        # Fetch IDs of the centroids just inserted.
        cur.execute(query, (sign_code, timestamp))
        location_ids = [row[0] for row in cur.fetchall()]
 
        # Check that every inserted centroid has a matching generated ID.
        if len(location_ids) != len(lst_centroid):
            raise ValueError(
                f"Expected {len(lst_centroid)} location IDs, got {len(location_ids)}."
            )
 
        # ------------------------------------------------------------------ #
        # Insert bearing lines
        # ------------------------------------------------------------------ #

        # Build rows for the line output table.
        # Each line belongs to one centroid location through object_location_id.
        line_values = [
            (
                location_ids[cluster_idx],
                -1,
                obj_ann.get_picture_id(),                
                -1,
                -1,
                -1,
                -1,
                obj_ann.get_object_annotation_id(),
                obj_ann.get_object_code(),
                -1,
                -1,
                obj_ann.direction_deg(),
                obj_ann.get_line().wkt,
                obj_ann.distance_m(centriod.get_center()),
                obj_ann.object_size_m(centriod.get_center()),
                obj_ann.get_origin().wkt,
            )
            for cluster_idx, centriod in enumerate(lst_centroid)
            for obj_ann in centriod.get_lst_object_annotation()
        ]

        # Build INSERT query for source bearing lines.
        # Schema and table names are safely inserted as SQL identifiers.
        insert_query = sql.SQL ("""
            INSERT INTO {schema}.{table} (
                object_location_id,
                collection_id,
                picture_id,
                picture_size_px_horizontal, 
                picture_size_px_vertical,
                picture_azimuth, 
                picture_field_of_view, 
                annotation_id, 
                sign_code,
                sign_size_px_horizontal, 
                sign_size_px_vertical,
                object_direction, 
                line, 
                object_distance, 
                object_size,
                scan_location
            )
            VALUES %s
            """).format(
                schema = sql.Identifier(tsgcf.DB_SCHEMA),
                table = sql.Identifier(tsgcf.OUT_TABLE_LINE)
        )


        # Execute bulk insert for all bearing line rows.
        execute_values(
            cur,
            insert_query,
            line_values
        )     
 
        # Commit the transaction when both inserts succeeded.
        cur.connection.commit()
 
    except Exception as exc:
        # Roll back the full transaction if any part fails.
        cur.connection.rollback()

        # Raise a clearer error while keeping the original exception context.
        raise RuntimeError(f"Database write failed: {exc}") from exc
 
 
# ===========================================================================
# GeoPackage export helpers
# ===========================================================================
 
def write_to_gpkg_lines(lst_object_annotation: list[tsgc.ObjectAnnotation]) -> None:
    """
    Append bearing lines to lines_01.gpkg.

    Each ObjectAnnotation contributes one line geometry and related metadata.

    Parameters
    ----------
    lst_object_annotation:
        List of ObjectAnnotation objects whose lines should be written.
    """

    # Create a GeoDataFrame from all object annotation lines.
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": object_annotation.get_line(),
                "sign_code": object_annotation.get_object_code(),
                "line_index": object_annotation.get_id(),
                "scan_direction": object_annotation.direction_deg(),
                "link": (
                    f'{tsgcf.PANORAMAX_END_POINT}'
                    f"?annot={object_annotation.get_object_annotation_id()}&pic={object_annotation.get_picture_id()}"
                ),
            }
            for object_annotation in lst_object_annotation
        ],
        crs="EPSG:4326",
    )  

    # Write to the configured GeoPackage output directory.
    output_path = Path(tsgcf.OUTPUT_DIR_GEO_PACK) / "lines_01.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path.exists() else "w"

    # Write the GeoDataFrame to the "lines" layer.
    gdf.to_file(output_path, layer="lines", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_intersection(lst_intersection: list[tsgc.Intersection], sign_type: str) -> None:
    """
    Append all intersection points to intersections_03.gpkg.

    Parameters
    ----------
    lst_intersection:
        List of Intersection objects to write.

    sign_type:
        Traffic sign code used to tag the records.
    """

    # Create a GeoDataFrame from all intersection points.
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": intersection.intersection(),
                "sign_type": sign_type,
                "angle_between_lines": intersection.angle_between_lines(),
                "sign_size_average_m": intersection.object_size(),
                "sign_size_score": intersection.object_size_score(),
            }
            for intersection in lst_intersection
        ],
        crs="EPSG:4326",
    )

    # Write to the configured GeoPackage output directory.
    output_path = tsgcf.OUTPUT_DIR_GEO_PACK / "intersections_03.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path.exists() else "w"

    # Write the GeoDataFrame to the "intersections" layer.
    gdf.to_file(output_path, layer="intersections", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_intersection_reliable(lst_intersection: list[tsgc.Intersection], sign_type: str) -> None:
    """
    Append high-confidence or filtered intersection points to intersections_reliable_03.gpkg.

    Parameters
    ----------
    lst_intersection:
        List of Intersection objects to write.

    sign_type:
        Traffic sign code used to tag the records.
    """

    # Create a GeoDataFrame from the reliable intersection points.
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": intersection.intersection(),
                "sign_type": sign_type,
                "angle_between_lines": intersection.angle_between_lines(),
                "sign_size_average_m": intersection.object_size(),
                "sign_size_score": intersection.object_size_score(),
            }
            for intersection in lst_intersection
        ],
        crs="EPSG:4326",
    )

    # Write to the configured GeoPackage output directory.
    output_path = tsgcf.OUTPUT_DIR_GEO_PACK / "intersections_reliable_03.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path.exists() else "w"

    # Write the GeoDataFrame to the "intersections_reliable" layer.
    gdf.to_file(output_path, layer="intersections_reliable", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_clusters(lst_cluster: list[tsgc.Cluster], sign_type: str) -> None:
    """
    Append cluster centre points to clusters.gpkg.

    This writes one point per Cluster, using the calculated cluster centre.

    Parameters
    ----------
    lst_cluster:
        List of Cluster objects to write.

    sign_type:
        Traffic sign code used to tag the records.
    """

    # Build rows for the GeoDataFrame.
    rows = [
        {
            "geometry": cluster.get_center(),
            "sign_type": sign_type,
            "cluster_label": cluster.get_id()
        }
        for cluster in lst_cluster
        # for item in items
    ]

    # Create a GeoDataFrame with WGS84 coordinates.
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Write to the configured GeoPackage output directory.
    output_path = tsgcf.OUTPUT_DIR_GEO_PACK / "clusters.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path.exists() else "w"

    # Write the GeoDataFrame to the "cluster" layer.
    gdf.to_file(output_path, layer="cluster", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_centriods(
    lst_centroid: list[tsgc.Centroid],
    object_code: str
) -> None:
    """
    Write centroids and their related lines to GeoPackage files.

    This function writes two files:
    1. centriods.gpkg:
       Contains the centroid point locations.

    2. centriod_lines.gpkg:
       Contains the source ObjectAnnotation lines assigned to the centroids.

    Parameters
    ----------
    lst_centroid:
        List of Centroid objects to be written.

    object_code:
        Traffic sign code to include in the output.
    """

    # Build rows for centroid point output.
    data = [
        {
            "geometry": centroid.get_center(),
            "sign_code": object_code,
            "sign_size": centroid.get_object_size(),
            "scan_direction": centroid.get_perpendicular(),
            "link": (
                f'{tsgcf.PANORAMAX_END_POINT}'
                f"?annot={centroid.get_object_annotation_id()}&pic={centroid.get_picture_id()}"
            ),
        }
        for centroid in lst_centroid
    ]

    # Write centroid points to GeoPackage.
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    output_path_gpkg = tsgcf.OUTPUT_DIR_GEO_PACK / "centriods.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path_gpkg.exists() else "w"

    # Write the GeoDataFrame to the "centroid" layer.
    gdf.to_file(output_path_gpkg, layer="centroid", driver="GPKG", mode=mode)    

    # Build rows for the lines assigned to each centroid.
    data = [
        {
            "geometry": anno.get_line(),
            "sign_code": anno.get_object_code(),
            "centriod_id": centroid.get_id(),
            "scan_direction": anno.direction_deg(),
            "link": (
                f'{tsgcf.PANORAMAX_END_POINT}'
                f"?annot={anno.get_object_annotation_id()}&pic={anno.get_picture_id()}"
            ),
        } 
        for centroid in lst_centroid
        for anno in centroid.get_lst_object_annotation()
    ]

    # Write centroid source lines to GeoPackage.
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    output_path_gpkg = tsgcf.OUTPUT_DIR_GEO_PACK / "centriod_lines.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path_gpkg.exists() else "w"

    # Write the GeoDataFrame to the "centroid_lines" layer.
    gdf.to_file(output_path_gpkg, layer="centroid_lines", driver="GPKG", mode=mode)    



def old_write_to_gpkg_clean_centriod(centroid_lines: list[dict]=None, sign_code:str=None) -> None:
    """
    Append older-style centroid records to clean_centriods.gpkg.

    This function appears to support an older data structure where centroids are
    stored as dictionaries instead of Centroid objects.

    Parameters
    ----------
    centroid_lines:
        List of centroid dictionaries.

    sign_code:
        Traffic sign code used to tag the records.
    """

    # Create a GeoDataFrame from dictionary-based centroid data.
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": cluster["point"],
                "sign_code": sign_code,
                "count": len(cluster["nearby_lines"]),
                "cluster_label": cluster["cluster_label"],
                "direction": cluster["direction"],
                "sign_size": cluster["size"],
                "size_sd": cluster["size_sd"],
                "score" : cluster["score"],              
                "link": cluster["link"],
            }
            for cluster in centroid_lines
        ],
        crs="EPSG:4326",
    )

    # Write to the configured GeoPackage output directory.
    output_path = tsgcf.OUTPUT_DIR_GEO_PACK / "clean_centriods.gpkg"

    # Append if the file already exists, otherwise create a new file.
    mode = "a" if output_path.exists() else "w"

    # Write the GeoDataFrame to the "cluster_center" layer.
    gdf.to_file(output_path, layer="cluster_center", driver="GPKG", mode=mode)

    # Convert geometry coordinates to separate x and y columns.
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
