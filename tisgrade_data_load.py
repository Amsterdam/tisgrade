# Standard library
import json
import logging
from typing import List, Tuple, Dict, Any

# Third-party
import psycopg2.sql as sql
from shapely import wkt

# Local
import tisgrade_config as tsgcf


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_data_load
logger = logging.getLogger(f"tisgrade.{__name__}")

 
# ===========================================================================
# Create temporary data set
# ===========================================================================

def create_temp_dataset(cursor: str, 
                longitude_min: float, longitude_max: float, latitude_min: float, latitude_max: float,
                start_date: str, end_date: str,
                sign_regex: str) -> None:
    """
    Create a temporary database table with photo, sign and semantic data.

    The temporary table contains only the records that match:
    - the requested traffic sign regex;
    - the requested longitude and latitude bounding box;
    - the requested start and end date.

    The temporary table is used by the rest of the pipeline to avoid repeating
    the same filtering query.

    Parameters
    ----------
    cursor:
        Database cursor used to execute the SQL query.

    longitude_min:
        Minimum longitude of the area filter.

    longitude_max:
        Maximum longitude of the area filter.

    latitude_min:
        Minimum latitude of the area filter.

    latitude_max:
        Maximum latitude of the area filter.

    start_date:
        Start date/time filter.

    end_date:
        End date/time filter.

    sign_regex:
        Regular expression used to select traffic sign codes.

    Returns
    -------
    None
    """

    # Build a SQL query safely with psycopg2.sql.
    # Identifiers, such as schema and table names, are inserted with sql.Identifier.
    # Values, such as coordinates and dates, are inserted with sql.Literal.
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
        # Filter value for the traffic sign code regex.
        sign_regex=sql.Literal(sign_regex),

        # Spatial filter values.
        longitude_min=sql.Literal(longitude_min),
        longitude_max=sql.Literal(longitude_max),
        latitude_min=sql.Literal(latitude_min),
        latitude_max=sql.Literal(latitude_max),

        # Date/time filter values.
        start_date=sql.Literal(start_date),
        end_date=sql.Literal(end_date),

        # Schema and table names from configuration.
        schema = sql.Identifier(tsgcf.DB_SCHEMA),
        tbl_photo = sql.Identifier(tsgcf.IN_TABLE_PHOTO),
        tbl_signs = sql.Identifier(tsgcf.IN_TABLE_SIGNS),
        tbl_sematics = sql.Identifier(tsgcf.IN_TABLE_SEMANTICS),
        temp_tbl_name = sql.Identifier(tsgcf.TEMP_TABLE)
    )



    # Execute the query if a cursor was provided.
    if cursor:
        try:
            cursor.execute(query)

        except Exception as e:
            # Log database errors and raise them again so the caller can handle them.
            logger.error(f"Database error: {e}")
            raise
    else:
        # If no cursor is available, the temporary table cannot be created.
        logger.warning("No cursor provided")

    logger.info("Temporary table created")
    return None


def get_signs_list(cursor=None) -> List[Tuple[str, int]]:

    """
    Return all distinct sign types and their observation counts.

    The function reads from the temporary table created by create_temp_dataset().
    It groups records by traffic sign code and counts how many observations are
    available per code.

    Parameters
    ----------
    cursor:
        Database cursor for executing the query.

    Returns
    -------
    List[Tuple[str, int]]
        List of tuples:
        - traffic sign code;
        - number of observations.

    Raises
    ------
    Exception
        For database-related errors.
    """

    # Query the temporary table and count records per traffic sign code.
    query = sql.SQL("""
        SELECT traffic_sign_code, COUNT(*)
        FROM {temp_tbl_name}
        GROUP BY traffic_sign_code
        ORDER BY traffic_sign_code
    """).format(
        temp_tbl_name = sql.Identifier(tsgcf.TEMP_TABLE)
    )


    rows = []

    # Execute query if a cursor was provided.
    if cursor:
        try:
            cursor.execute(query)
            rows = cursor.fetchall()

        except Exception as e:
            # Log database errors and raise them again so the caller can handle them.
            logger.error(f"Database error: {e}")
            raise
    else:
        # Without a cursor, the database cannot be queried.
        logger.warning("No cursor provided")
        return []

    return rows
 
def get_signs(cursor=None, 
                sign_type: str=None) -> List[Dict[str, Any]]:
    """
    Fetch all panoramic-photo annotations for a given sign type.

    The result set includes:
    - photo metadata;
    - photo location;
    - camera azimuth;
    - image resolution;
    - field of view;
    - annotation ID;
    - traffic sign code;
    - confidence score;
    - bounding box.

    The function also adds parsed and cleaned geometry fields to each record.

    Parameters
    ----------
    cursor:
        Database cursor for executing the query.

    sign_type:
        Traffic sign code to filter on, for example "NL:B06".

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries.
        Each dictionary contains the selected database fields plus:
        - parsed bbox min/max values;
        - origin as a Shapely geometry;
        - flattened bbox coordinates.

    Raises
    ------
    Exception
        For database-related errors or parsing errors.
    """

    # ===========================================================================
    # Column names for database result sets
    # ===========================================================================

    # These names are used to convert database rows into dictionaries.
    # The order must match the SELECT statement below.
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


    # Query records for one specific traffic sign type from the temporary table.
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
        temp_tbl_name = sql.Identifier(tsgcf.TEMP_TABLE)
    )


    rows = []

    try:
        # Execute the query.
        # The query already contains the safely formatted sign_type value.
        # cursor.execute(query, (sign_type,))
        cursor.execute(query)

        # Fetch all rows from the database cursor.
        rows = cursor.fetchall()

        # Convert each database row to a dictionary using FIELD_NAMES.
        named_records = [dict(zip(FIELD_NAMES, r)) for r in rows]

        # Add parsed fields to every record.
        for record in named_records:
            # Add bbox min/max values to the record.
            record.update(parse_bbox(record["bbox"]))

            # Parse photo coordinates from WKT into a Shapely geometry.
            record["origin"] = wkt.loads(record["picture_coordinates"])

            # Parse bbox JSON and flatten it to a list of [x, y] coordinate pairs.
            record["bbox_clean"] = flatten_coords(json.loads(record["bbox"]))

        return named_records

    except Exception as e:
        # Log the sign type for easier debugging.
        logger.error(f"Error in get_signs for sign type {sign_type}: {e}")
        raise

 
# ===========================================================================
# Support functions
# ===========================================================================

def flatten_coords(data: list) -> list:
    """
    Recursively flatten nested coordinate lists to a list of [x, y] pairs.

    Bounding box data can be nested, for example when it comes from GeoJSON-like
    structures. This function walks through that nested structure until it finds
    coordinate pairs.

    Parameters
    ----------
    data:
        Arbitrarily nested list of numbers or sub-lists, as returned by
        json.loads() on a bbox field.

    Returns
    -------
    list
        List of [x, y] coordinate pairs.
    """

    # Base case:
    # if the first item is a number, this level is already one coordinate pair.
    if isinstance(data[0], (int, float)):
        return [data]

    # Recursive case:
    # flatten each nested item and combine all resulting coordinate pairs.
    return [point for item in data for point in flatten_coords(item)]
 
 
def parse_bbox(bbox_json: str | None) -> dict:
    """
    Parse a JSON bbox string into minimum and maximum pixel coordinates.

    Parameters
    ----------
    bbox_json:
        Raw JSON string from the database, or None.

    Returns
    -------
    dict
        Dictionary with:
        - bbox_xmin;
        - bbox_xmax;
        - bbox_ymin;
        - bbox_ymax.

        All values are None when bbox_json is None.
    """

    # If no bbox is available, return empty bbox values.
    if bbox_json is None:
        return {"bbox_xmin": None, "bbox_xmax": None, "bbox_ymin": None, "bbox_ymax": None}

    # Parse JSON and flatten possible nested coordinates.
    coords = flatten_coords(json.loads(bbox_json))

    # Calculate min/max pixel coordinates.
    return {
        "bbox_xmin": min(p[0] for p in coords),
        "bbox_xmax": max(p[0] for p in coords),
        "bbox_ymin": min(p[1] for p in coords),
        "bbox_ymax": max(p[1] for p in coords),
    }
