
"""
sign_detection.py
-----------------
Utilities for detecting and geolocating traffic signs from panoramic street-level
imagery (e.g. Panoramax).
 
High-level pipeline
-------------------
1. For each annotated sign in a panoramic image, compute a *bearing line*: a
   ray cast from the camera position in the direction of the sign.
2. Intersect bearing lines from different viewpoints. Each intersection is a
   candidate sign location.
3. Cluster intersections with DBSCAN (recursive, with adaptive epsilon) to
   separate distinct signs from noise.
4. Score each cluster by spatial density,the ratio of observed intersections
   to the theoretical maximum and the amount of points in te cluster
   Then resolve line-ownership conflicts bases on this score so that
   each bearing line is assigned to at most one sign.
5. Compute a centroid, estimated physical sign size, and viewing direction for every
   accepted cluster.
6. Persist results to a PostGIS database and/or GeoPackage files.
 
Dependencies
------------
    numpy, pandas, geopandas, shapely, scipy, scikit-learn,
    geopy, haversine, psycopg2
"""
 
# Standard library
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import pandas as pd
 
# Third-party
from psycopg2 import sql
from psycopg2.extras import execute_values
# from shapely.geometry import Point
import geopandas as gpd

import tisgrade_classes as tsgc
from tisgrade_config import DB_SCHEMA, OUT_TABLE_CENTRIOD, OUT_TABLE_LINE
from tisgrade_config import PANORAMAX_END_POINT, USERNAME

import logging
logger = logging.getLogger(f"tisgrade.{__name__}")
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
# EARTH_RADIUS_M = 6_367_445  # Mean Earth radius in metres (used for unit conversions)
 
# Output directory for local GeoPackage exports.
# OUTPUT_DIR = Path(
#     r"C:\Users\postma032\OneDrive - Gemeente Amsterdam"
#     r"\VOR - KenK - Onderzoek en Kennis-Projecten 2021-2025 - Projecten 2021-2025"
#     r"\2026\269999 Team OMA\2026-02-19 Tisgrade GOM-846"
# )

# OUTPUT_DIR = Path(
#     r"C:\Users\joost\Documents\amsterdam\tisgrade\2026-05-01 solve line problem"
# )

# ===========================================================================
# Database persistence
# ===========================================================================
 
# def write_cluster_to_db(cur, clusters: list[dict], sign_code: str) -> None:
def write_centriod_to_db(cur, lst_centroid: list[tsgc.Centroid], sign_code: str, start_timestamp: str, end_timestamp: str, run_name: str = '') -> None:

    """Persist sign location clusters (and their source lines) to PostGIS.
 
    All inserts are wrapped in a single transaction; on any error the
    transaction is rolled back and the exception is re-raised.
 
    Parameters
    ----------
    cur:
        An open psycopg2 cursor.
    clusters:
        List of centroid dicts as returned by :func:`cluster_data`.
    sign_code:
        RVV sign code used to tag the inserted records (e.g. ``"B6"``).
    """
    if cur==False:
        return
    
    if not lst_centroid:
        return
 
    # timestamp = datetime.now(timezone.utc)
    timestamp = datetime.now().astimezone()
 
    try:
        cur.execute("BEGIN")
 
        # ------------------------------------------------------------------ #
        # Insert sign locations
        # ------------------------------------------------------------------ #
        location_values = [
            (
                sign_code,
                centriod.get_center().y, #latitude
                centriod.get_center().x, #longitude
                centriod.get_object_size(),
                centriod.get_perpendicular(),
                timestamp,
                centriod.get_center().wkt,
                start_timestamp,
                end_timestamp,
                run_name,
                USERNAME
            )
            for centriod in lst_centroid
        ]

        insert_query = sql.SQL (
            """INSERT INTO {schema}.{table}
                    (object, latitude, longitude, size, direction, run_timestamp, location, period_start, period_end, run_name, run_user)
                VALUES %s
            """).format(
                schema = sql.Identifier(DB_SCHEMA),
                table = sql.Identifier(OUT_TABLE_CENTRIOD)
        )
 
        # insert_query(
        #     cur,
        #     """
        #     INSERT INTO "20260220_jp_tisgrade".out_object_location_panoramax
        #         (object, latitude, longitude, size, direction, run_timestamp, location, period_start, period_end, run_name)
        #     VALUES %s
        #     """,
        #     location_values,
        # )

        execute_values(
            cur,
            insert_query,
            location_values
        )  
 
        # Retrieve the auto-generated IDs in insertion order
        query = sql.SQL (
            """
                SELECT id
                FROM {schema}.{table}
                WHERE object = %s AND run_timestamp = %s
                ORDER BY id
            """).format(
                schema = sql.Identifier(DB_SCHEMA),
                table = sql.Identifier(OUT_TABLE_CENTRIOD)
        )
        

        # cur.execute(
        #     """
        #     SELECT id
        #     FROM "20260220_jp_tisgrade".out_object_location_panoramax
        #     WHERE object = %s AND run_timestamp = %s
        #     ORDER BY id
        #     """,
        #     (sign_code, timestamp),
        # )
        cur.execute(query, (sign_code, timestamp))
        location_ids = [row[0] for row in cur.fetchall()]
 
        if len(location_ids) != len(lst_centroid):
            raise ValueError(
                f"Expected {len(lst_centroid)} location IDs, got {len(location_ids)}."
            )
 
        # ------------------------------------------------------------------ #
        # Insert bearing lines
        # ------------------------------------------------------------------ #

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
                schema = sql.Identifier(DB_SCHEMA),
                table = sql.Identifier(OUT_TABLE_LINE)
        )

        # Execute bulk insert
        execute_values(
            cur,
            insert_query,
            line_values
        )     
 
        cur.connection.commit()
 
    except Exception as exc:
        cur.connection.rollback()
        raise RuntimeError(f"Database write failed: {exc}") from exc
 
 
# ===========================================================================
# GeoPackage export helpers
# ===========================================================================
 
def write_to_gpkg_lines(OUTPUT_DIR, lst_object_annotation: list[tsgc.ObjectAnnotation]) -> None:
    """Append bearing lines to ``lines_01.gpkg`` in :data:`OUTPUT_DIR`.
 
    Parameters
    ----------
    lines:
        List of bearing-line dicts containing ``"line"`` (Shapely LineString),
        ``"sign_code"``, ``"scan_direction"``, ``"annotation_id"``,
        and ``"picture_id"``.
    """
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": object_annotation.get_line(),
                "sign_code": object_annotation.get_object_code(),
                "line_index": object_annotation.get_id(),
                "scan_direction": object_annotation.direction_deg(),
                # f"?annot={line['annotation_id']}&pic={line['picture_id']}"
                "link": (
                    f"{PANORAMAX_END_POINT}"                    
                    f"?annot={object_annotation.get_object_annotation_id()}&pic={object_annotation.get_picture_id()}"
                ),
            }
            for object_annotation in lst_object_annotation
        ],
        crs="EPSG:4326",
    )
    output_path = OUTPUT_DIR / "lines_01.gpkg"
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer="lines", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_intersection(OUTPUT_DIR, lst_intersection: list[tsgc.Intersection], sign_type: str) -> None:
    """Append all intersection points to ``intersections_03.gpkg``.
 
    Parameters
    ----------
    intersection:
        List of intersection dicts with keys ``"point"``,
        ``"angle_scan_directions"``, ``"sign_size_average_m"``,
        and ``"sign_size_score"``.
    sign_type:
        Sign code used to tag the records.
    """
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
    output_path = OUTPUT_DIR / "intersections_03.gpkg"
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer="intersections", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_intersection_reliable(OUTPUT_DIR, lst_intersection: list[tsgc.Intersection], sign_type: str) -> None:
    """Append high-confidence intersection points to ``intersections_reliable_03.gpkg``.
 
    Parameters
    ----------
    intersection:
        Same structure as the *intersection* parameter in
        :func:`write_to_gpkg_intersection`.
    sign_type:
        Sign code used to tag the records.
    """
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
    output_path = OUTPUT_DIR / "intersections_reliable_03.gpkg"
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer="intersections_reliable", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_clusters(OUTPUT_DIR, lst_cluster: list[tsgc.Cluster], sign_type: str) -> None:
    """Append all cluster points (before filtering) to ``all_clusters_all_points_all_01.gpkg``.
 
    Parameters
    ----------
    OUTPUT_DIR:
        Directory where the GeoPackage will be written.
    lst_centroid:
        List of Centroid objects to be written.
    object_code:
        Sign code to be included in the output.
    csv_file_loc:
        Optional directory where the CSV file will be written.
        If None, no CSV will be written.
    """
    rows = [
        {
            "geometry": cluster.get_center(),
            "sign_type": sign_type,
            "cluster_label": cluster.get_id()
        }
        for cluster in lst_cluster
        # for item in items
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    output_path = OUTPUT_DIR / "clusters.gpkg"
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer="cluster", driver="GPKG", mode=mode)
 
 
def write_to_gpkg_centriods(
    OUTPUT_DIR,
    lst_centroid: list[tsgc.Centroid],
    object_code: str,
    csv_file_loc: Path | None = None
) -> None:
    """Write centroids to a GeoPackage and optionally to a CSV file.


    Parameters
    ----------
    lst_centroid:
        List of Centroid objects to be written.
    object_code:
        Sign code to be included in the output.
    """

    data = [
        {
            "geometry": centroid.get_center(),
            "sign_code": object_code,
            "sign_size": centroid.get_object_size(),
            # "line_index": centroid.get_id(),
            "scan_direction": centroid.get_perpendicular(),
            "link": (
                # f"https://nl.panoramax.xyz/"
                f"{PANORAMAX_END_POINT}"
                f"?annot={centroid.get_object_annotation_id()}&pic={centroid.get_picture_id()}"
            ),
        }
        for centroid in lst_centroid
    ]

    # Write to GeoPackage
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    output_path_gpkg = OUTPUT_DIR / "centriods.gpkg"
    mode = "a" if output_path_gpkg.exists() else "w"
    gdf.to_file(output_path_gpkg, layer="centroid", driver="GPKG", mode=mode)    

    data = [
        {
            "geometry": anno.get_line(),
            "sign_code": anno.get_object_code(),
            "centriod_id": centroid.get_id(),
            # "line_index": centroid.get_id(),
            "scan_direction": anno.direction_deg(),
            "link": (
                # f"https://nl.panoramax.xyz/"
                f"{PANORAMAX_END_POINT}"
                f"?annot={anno.get_object_annotation_id()}&pic={anno.get_picture_id()}"
            ),
        } 
        for centroid in lst_centroid
        for anno in centroid.get_lst_object_annotation()
    ]

    # Write to GeoPackage
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    output_path_gpkg = OUTPUT_DIR / "centriod_lines.gpkg"
    mode = "a" if output_path_gpkg.exists() else "w"
    gdf.to_file(output_path_gpkg, layer="centroid_lines", driver="GPKG", mode=mode)    



    # Selecteer alle kolommen behalve de originele geometrie
    
    # Bepaal CSV-bestandspad
    if csv_file_loc is not None:
        # output_path_csv =csv_file_loc / "clean_centriods_test.csv"
        output_path_csv = OUTPUT_DIR / "centriods.csv"
        # Converteer geometrie naar losse kolommen
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y
        df = gdf.drop(columns='geometry')

        # Schrijf naar CSV
        mode = "a" if output_path_csv.exists() else "w"
        header = not output_path_csv.exists()
        df.to_csv(output_path_csv, mode=mode, header=header, index=False)


def old_write_to_gpkg_clean_centriod(OUTPUT_DIR=None, centroid_lines: list[dict]=None, sign_code:str=None, csv_file_loc: Optional[Union[str, Path]] = None) -> None:
# def write_to_gpkg_clusters_center(OUTPUT_DIR, cluster_center: list[dict], sign_type: str) -> None:
    """Append cluster centroid records to ``clean_centriod.gpkg``.
 
    Parameters
    ----------
    cluster_center:
        List of centroid dicts as returned by :func:`cluster_data`.
    sign_type:
        Sign code used to tag the records.
    """
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": cluster["point"],
                "sign_code": sign_code,
                # "count": row["count"],
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
    output_path = OUTPUT_DIR / "clean_centriods.gpkg"
    mode = "a" if output_path.exists() else "w"
    gdf.to_file(output_path, layer="cluster_center", driver="GPKG", mode=mode)

    # Converteer geometrie naar losse kolommen
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    # Selecteer alle kolommen behalve de originele geometrie
    
    # Bepaal CSV-bestandspad
    if csv_file_loc:
        output_path_csv =csv_file_loc / "clean_centriods_test.csv"
        df = gdf.drop(columns='geometry')
        # Schrijf naar CSV
        df.to_csv(output_path_csv, index=False)

