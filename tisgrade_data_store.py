# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Literal
import logging

# Third-party
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import cursor as psycopg_cursor

# Local
import tisgrade_config as tsgcf
# import tisgrade_classes as tsgc


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_data_store
logger = logging.getLogger(f"tisgrade.{__name__}")



def store_gpkg(    gpkg_data: GeoDataFrame,
                    type: Literal["Centroid", "Cluster", "Intersections reliable", "Intersections", "Line"] ) -> None:
        file_name =""
        layer = ""
        if type == "Centroid":
            file_name = "centriods.gpkg"
            layer = "Centroid"
        elif type == "Cluster":
            file_name = "cluster.gpkg"
            layer = "Cluster"
        elif type == "Intersections reliable":
            file_name = "intersections_reliable.gpkg"
            layer = "Intersections reliable"
        elif type == "Intersections":
            file_name = "intersections.gpkg"
            layer = "Intersections"
        elif type == "Line":
            file_name = "lines.gpkg"
            layer = "Line"

        path = Path(tsgcf.OUTPUT_DIR_GEO_PACK) / file_name
        print(f'path: {path}')
        # Append if the file already exists, otherwise create a new file.
        mode = "a" if path.exists() else "w"

        # Write the GeoDataFrame to the "lines" layer.
        gpkg_data.to_file(path, layer=layer, driver="GPKG", mode=mode)         


# ===========================================================================
# Database persistence
# ===========================================================================

def _py(value: Any) -> Any:
    """Convert NaN/NaT/None to None and numpy scalars to plain Python values."""
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _wkt(value: Any) -> Optional[str]:
    """Return WKT for a shapely geometry, pass strings through, None stays None."""
    if value is None:
        return None
    return value.wkt if hasattr(value, "wkt") else value


def _to_wgs84(gdf: GeoDataFrame) -> GeoDataFrame:
    """Make sure the active geometry is in lon/lat (EPSG:4326)."""
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(4326)
    return gdf




def store_centriod_db(
    centriods_gpkg: GeoDataFrame,
    lines_gpkg: GeoDataFrame,
    cur: Optional[psycopg_cursor] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """
    Persist centroid locations and their source bearing lines to PostGIS.

    Both tables are written in one transaction. On any error the transaction
    is rolled back and a RuntimeError is raised.

    Parameters
    ----------
  centriods_gpkg:
      Required columns: id, object_code, object_size_meter, object_azimuth, geometry.
  lines_gpkg:
      Required columns: centroid_id (matches centriods_gpkg["id"]), picture_id,
      annotation_id, object_code, line_azimuth, object_distance_meter,
      object_size_meter, picture_location, picture_timestamptz, geometry.

    cur:
        Open psycopg2 cursor. If None, nothing is written.

    start_timestamp, end_timestamp:
        Period of the input data.

    run_name:
        Optional name for this run.
    """
    if cur is None or centriods_gpkg.empty:
        return

    centriods_gpkg = _to_wgs84(centriods_gpkg)
    lines_gpkg = _to_wgs84(lines_gpkg)

    timestamp = datetime.now().astimezone()

    try:
        # psycopg2 opens the transaction implicitly on the first execute,
        # so no explicit BEGIN is needed.

        # ------------------------------------------------------------------ #
        # Insert centroid locations
        # ------------------------------------------------------------------ #
        centriod_values = [
            (
                row.object_code,
                row.geometry.y,  # latitude
                row.geometry.x,  # longitude
                _py(row.object_size_meter),
                _py(row.object_azimuth),
                timestamp,
                row.geometry.wkt,
                start_timestamp,
                end_timestamp,
                run_name,
                tsgcf.USERNAME,
            )
            for row in centriods_gpkg.itertuples(index=False)
        ]

        insert_query = sql.SQL(
            """INSERT INTO {schema}.{table}
                    (object, latitude, longitude, size, azimuth, run_timestamp,
                     location, period_start, period_end, run_name, run_user)
               VALUES %s
               RETURNING id
            """
        ).format(
            schema=sql.Identifier(tsgcf.DB_SCHEMA),
            table=sql.Identifier(tsgcf.OUT_TABLE_CENTRIOD),
        )

        # One page = one INSERT statement, so the RETURNING ids come back in
        # the same order as the rows that were sent.
        returned = execute_values(
            cur,
            insert_query,
            centriod_values,
            page_size=len(centriod_values),
            fetch=True,
        )
        location_ids = [r[0] for r in returned]

        if len(location_ids) != len(centriod_values):
            raise ValueError(
                f"Expected {len(centriod_values)} location IDs, got {len(location_ids)}."
            )

        # Map the centroid's index label in the GeoDataFrame to its database id.
        db_id_by_label = dict(zip(centriods_gpkg.id, location_ids))

        # ------------------------------------------------------------------ #
        # Insert bearing lines
        # ------------------------------------------------------------------ #
        def opt(row, name: str, default: Any = -1) -> Any:
            value = _py(getattr(row, name, default))
            return default if value is None else value

        line_values = []
        for row in lines_gpkg.itertuples(index=False):
            if pd.isna(row.centroid_id):
                continue
            if row.centroid_id not in db_id_by_label:
                raise ValueError(
                    f"Line refers to unknown centroid_id {row.centroid_id!r}."
                )
            line_values.append(
                (
                    db_id_by_label[row.centroid_id],
                    opt(row, "collection_id"),
                    _py(row.picture_id),
                    opt(row, "picture_size_px_horizontal"),
                    opt(row, "picture_size_px_vertical"),
                    opt(row, "picture_azimuth"),
                    opt(row, "picture_field_of_view"),
                    _py(row.annotation_id),
                    row.object_code,
                    opt(row, "sign_size_px_horizontal"),
                    opt(row, "sign_size_px_vertical"),
                    _py(row.line_azimuth),
                    row.geometry.wkt,
                    _py(row.object_distance_meter),
                    _py(row.object_size_meter),
                    _wkt(row.picture_location),
                    _py(row.picture_timestamptz),
                )
            )

        if line_values:
            insert_query = sql.SQL(
                """
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
                    picture_location,
                    picture_datetimetz
                )
                VALUES %s
                """
            ).format(
                schema=sql.Identifier(tsgcf.DB_SCHEMA),
                table=sql.Identifier(tsgcf.OUT_TABLE_LINE),
            )
            execute_values(cur, insert_query, line_values)

        # Commit only when both inserts succeeded.
        cur.connection.commit()

    except Exception as exc:
        cur.connection.rollback()
        raise RuntimeError(f"Database write failed: {exc}") from exc
