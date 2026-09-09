"""
tisgrade_data_laod.py

Dependencies
------------
    psycopg2
"""
 
# Standard library
import json
import re
from typing import List, Tuple, Optional, Union, Dict, Any
import os

 
# Third-party
import psycopg2.sql as sql

from shapely import wkb, wkt

# from tisgrade_config import DB_SCHEMA, IN_TABLE_PHOTO, IN_TABLE_SIGNS, IN_TABLE_SEMANTICS, TEMP_TABLE
from dotenv import load_dotenv
import logging

load_dotenv()

DB_SCHEMA           = os.environ["DB_SCHEMA"]
IN_TABLE_PHOTO      = os.environ["IN_TABLE_PHOTO"]
IN_TABLE_SIGNS      = os.environ["IN_TABLE_SIGNS"]
IN_TABLE_SEMANTICS  = os.environ["IN_TABLE_SEMANTICS"]
TEMP_TABLE          = os.environ["TEMP_TABLE"]


logger = logging.getLogger(f"tisgrade.{__name__}")

 
# ===========================================================================
# Database queries
# ===========================================================================
 

# ===========================================================================
# tem data set
# ===========================================================================

def create_temp_dataset(cursor: str, 
                longitude_min: float, longitude_max: float, latitude_min: float, latitude_max: float,
                start_date: str, end_date: str,
                sign_regex: str) -> None:

    # temp_tbl_name = temp_photo_signs

    query = sql.SQL("""
        DROP TABLE IF EXISTS {temp_tbl_name};
        CREATE TEMP TABLE {temp_tbl_name} AS
        SELECT
            photo.id AS picture_id,
            photo.collection AS collection_id,
            ((photo.datetimetz::timestamp with time zone AT TIME ZONE 'UTC')
                AT TIME ZONE 'Europe/Amsterdam') AS datetimetz,
            photo.location AS location_coordinates,
            ST_GeomFromText(photo.location, 4326) AS geom,
            photo.field_of_view,
            photo.azimuth,
            photo.px_horizontal,
            photo.px_vertical,
            sem.annotation_id,
            sem.traffic_sign_code,
            sem.detection_confidence,
            sem.classification_confidence,
            sign.bbox
        FROM {schema}.{tbl_photo} photo
        JOIN {schema}.{tbl_signs} sign
            ON photo.id::text = sign.foto_id::text
        JOIN {schema}.{tbl_sematics} sem
            ON sign.id::text = sem.annotation_id::text
            AND sign.bbox IS NOT NULL
        WHERE sem.traffic_sign_code ~ {sign_regex}
        AND ST_X(ST_GeomFromText(photo.location, 4326)) BETWEEN {longitude_min} AND {longitude_max}
        AND ST_Y(ST_GeomFromText(photo.location, 4326)) BETWEEN {latitude_min} AND {latitude_max}
        AND {start_date} <= photo.datetimetz::timestamptz AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Amsterdam'
        AND photo.datetimetz::timestamptz AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Amsterdam' < {end_date};

        CREATE INDEX ON temp_photo_signs USING GIST(geom);
        CREATE INDEX ON temp_photo_signs (traffic_sign_code);
    """).format(
        sign_regex=sql.Literal(sign_regex),
        longitude_min=sql.Literal(longitude_min),
        longitude_max=sql.Literal(longitude_max),
        latitude_min=sql.Literal(latitude_min),
        latitude_max=sql.Literal(latitude_max),
        start_date=sql.Literal(start_date),
        end_date=sql.Literal(end_date),
        schema = sql.Identifier(DB_SCHEMA),
        tbl_photo = sql.Identifier(IN_TABLE_PHOTO),
        tbl_signs = sql.Identifier(IN_TABLE_SIGNS),
        tbl_sematics = sql.Identifier(IN_TABLE_SEMANTICS),
        temp_tbl_name = sql.Identifier(TEMP_TABLE)
    )

    if cursor:
        try:
            cursor.execute(query)

        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
    else:
        logger.warning("No cursor provided")

    logger.info("Temporary table created")
    return None


def get_signs_list(cursor=None) -> List[Tuple[str, int]]:

    """Return all distinct sign types (and their observation counts) from the DB or CSV file.

    Parameters
    ----------
    cursor : database cursor, optional
        Database cursor for executing queries
    file_loc : str or Path, optional
        Path to directory where CSV file should be read/written

    Returns
    -------
    list of (traffic_sign_code, count) tuples

    Raises
    ------
    FileNotFoundError
        When CSV file doesn't exist and no cursor is provided
    csv.Error
        When there are issues reading/writing the CSV file
    Exception
        For database-related errors
    """

    query = sql.SQL("""
        SELECT traffic_sign_code, COUNT(*)
        FROM {temp_tbl_name}
        GROUP BY traffic_sign_code
        ORDER BY traffic_sign_code
    """).format(
        temp_tbl_name = sql.Identifier(TEMP_TABLE)
    )

    rows = []
    if cursor:
        try:
            cursor.execute(query)
            rows = cursor.fetchall()

        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
    else:
        logger.warning("No cursor provided")
        return []

    return rows
 
def get_signs(cursor=None, 
                sign_type: str=None) -> List[Dict[str, Any]]:
    """Fetch all panoramic-photo annotations for a given sign type.

    The result set includes the photo metadata (location, azimuth, resolution,
    field of view) and the bounding box of the detected sign.

    Parameters
    ----------
    sign_type : str
        RVV sign code to filter on (e.g. ``"NL:B06"``)
    cursor : database cursor, optional
        An open psycopg2 cursor
    file_path : str or Path, optional
        Path to directory where CSV file should be read/written

    Returns
    -------
    list of dicts
        Each dict contains the fields from FIELD_NAMES plus parsed bbox data

    Raises
    ------
    ValueError
        When sign_type is empty or invalid
    FileNotFoundError
        When CSV file doesn't exist and no cursor is provided
    csv.Error
        When there are issues reading/writing the CSV file
    Exception
        For database-related errors
    """
    # ===========================================================================
    # Column names for database result sets
    # ===========================================================================

    FIELD_NAMES = [
        "picture_id",
        "picture_collection_id",
        "picture_timestamp",
        "picture_coordinates",
        "picture_field_of_view",
        "picture_azimuth",
        "picture_px_horizontal",
        "picture_px_vertical",
        "annotation_id",
        "traffic_sign_code",
        "traffic_sign_confidence",
        "bbox",
    ]


    query = sql.SQL("""
        SELECT
            picture_id,
            collection_id,
            datetimetz,
            location_coordinates,
            field_of_view,
            azimuth,
            px_horizontal,
            px_vertical,
            annotation_id,
            traffic_sign_code,
            COALESCE(classification_confidence, detection_confidence) AS traffic_sign_confidence,
            bbox
        FROM {temp_tbl_name}
        WHERE traffic_sign_code = {sign}
    """).format(
        sign=sql.Literal(sign_type),
        temp_tbl_name = sql.Identifier(TEMP_TABLE)
    )

    rows = []
    try:
        # cursor.execute(query, (sign_type,))
        cursor.execute(query)
        rows = cursor.fetchall()
        named_records = [dict(zip(FIELD_NAMES, r)) for r in rows]

        for record in named_records:
            record.update(parse_bbox(record["bbox"]))
            record["origin"] = wkt.loads(record["picture_coordinates"])
            record["bbox_clean"] = flatten_coords(json.loads(record["bbox"]))

        return named_records

    except Exception as e:
        logger.error(f"Error in get_signs for sign type {sign_type}: {e}")
        raise

 
# ===========================================================================
# Support functions
# ===========================================================================

def flatten_coords(data: list) -> list:
    """Recursively flatten nested coordinate lists to a list of [x, y] pairs.
 
    Parameters
    ----------
    data:
        Arbitrarily nested list of numbers or sub-lists, as returned by
        ``json.loads`` on a GeoJSON bbox field.
 
    Returns
    -------
    list of [x, y] pairs
    """
    if isinstance(data[0], (int, float)):
        return [data]
    return [point for item in data for point in flatten_coords(item)]
 
 
def parse_bbox(bbox_json: str | None) -> dict:
    """Parse a GeoJSON bbox string into min/max pixel coordinates.
 
    Parameters
    ----------
    bbox_json:
        Raw JSON string from the database, or ``None``.
 
    Returns
    -------
    dict
        ``{"bbox_xmin", "bbox_xmax", "bbox_ymin", "bbox_ymax"}``
        All values are ``None`` when *bbox_json* is ``None``.
    """
    if bbox_json is None:
        return {"bbox_xmin": None, "bbox_xmax": None, "bbox_ymin": None, "bbox_ymax": None}
 
    coords = flatten_coords(json.loads(bbox_json))
    return {
        "bbox_xmin": min(p[0] for p in coords),
        "bbox_xmax": max(p[0] for p in coords),
        "bbox_ymin": min(p[1] for p in coords),
        "bbox_ymax": max(p[1] for p in coords),
    }
