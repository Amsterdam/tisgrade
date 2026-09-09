from pathlib import Path
import logging
from datetime import datetime
import os


EARTH_RADIUS_M = 6_371_000

# optional output file parameters
STORE_GEO_PACK = True
OUTPUT_DIR_GEO_PACK = Path(
    r"C:\Users\postma032\OneDrive - Gemeente Amsterdam"
    r"\VOR - KenK - Onderzoek en Kennis-Projecten 2021-2025 - Projecten 2021-2025"
    r"\2026\269999 Team OMA\2026-02-19 Tisgrade GOM-846\with_classes\data_out"
    )

# DB parameters
USERNAME    = os.environ.get('PostgresAzureUsername', None)
HOST        = os.environ.get('PostgresAzureHost', None)
PORT        = os.environ.get('PostgresAzurePort', None)
DB          = os.environ.get('PostgresAzureDatabaseName', None)
TOKEN_PATH  = Path(os.environ.get('PostgresAzureTokenPath', None))


KEEPALIVES          = 1
KEEPALIVES_IDLE     = 30
KEEPALIVES_INTERVAL = 10
KEEPALIVES_COUNT    = 3

DB_SCHEMA = "20260220_jp_tisgrade"
IN_TABLE_PHOTO = "photos_new"
IN_TABLE_SIGNS = "signs_new"
IN_TABLE_SEMANTICS = "semantics_new"
TEMP_TABLE = "temp_photo_signs"
OUT_TABLE_LINE = "out_object_line_panoramax"
OUT_TABLE_CENTRIOD = "out_object_location_panoramax"

PANORAMAX_END_POINT = "https://panoramax.ndw.nu"


