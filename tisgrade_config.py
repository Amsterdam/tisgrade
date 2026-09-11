# Standard library
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
from urllib.parse import quote_plus

# Third-party
import psycopg2


# Load environment variables from a .env file into os.environ.
# By default, load_dotenv() looks for a .env file in the current directory
# or parent directories.
load_dotenv()


# Module-level logger.
# This creates a logger name like: tisgrade.tisgrade_config
logger = logging.getLogger(f"tisgrade.{__name__}")


def get_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean value from an environment variable.

    Environment variables are always strings. This function converts common
    true-like string values to True. All other values become False.

    Parameters
    ----------
    name:
        Name of the environment variable.

    default:
        Value to return if the environment variable does not exist.

    Returns
    -------
    bool
        True if the environment variable value is one of:
        "true", "1", "yes", "y", "on", "ja", "oui".
        False otherwise.
    """

    # Read the environment variable.
    value = os.getenv(name)

    # If the variable is missing, return the default value.
    if value is None:
        return default

    # Normalize the value and compare it with accepted true-like values.
    return value.strip().lower() in ("true", "1", "yes", "y", "on", "ja", "oui")


# ===========================================================================
# Environment configuration
# ===========================================================================

# Boolean switches.
# Values are read from environment variables and converted to bool.
DEBUG                   = get_bool("DEBUG")
STORE_GEO_PACK          = get_bool("STORE_GEO_PACK")

# Output folder for GeoPackage files.
# Converted to Path for safer path handling in Python.
OUTPUT_DIR_GEO_PACK     = Path(os.environ["OUTPUT_DIR_GEO_PACK"])

# Database connection parameters.
# DB_USERNAME is used instead of USERNAME to avoid conflicts with the Windows
# USERNAME environment variable.
USERNAME                = os.environ["DB_USERNAME"]
HOST                    = os.environ["HOST"]
PORT                    = os.environ["PORT"]
DB                      = os.environ["DB"]

# Path to the local token/password file.
TOKEN_PATH              = Path(os.environ["TOKEN_PATH"])

# PostgreSQL keepalive settings.
# These values are converted to integers because psycopg2 expects numeric values.
KEEPALIVES              = int(os.environ["KEEPALIVES"])
KEEPALIVES_IDLE         = int(os.environ["KEEPALIVES_IDLE"])
KEEPALIVES_INTERVAL     = int(os.environ["KEEPALIVES_INTERVAL"])
KEEPALIVES_COUNT        = int(os.environ["KEEPALIVES_COUNT"])

# Database schema and table names.
DB_SCHEMA               = os.environ["DB_SCHEMA"]
IN_TABLE_PHOTO          = os.environ["IN_TABLE_PHOTO"]
IN_TABLE_SIGNS          = os.environ["IN_TABLE_SIGNS"]
IN_TABLE_SEMANTICS      = os.environ["IN_TABLE_SEMANTICS"]
TEMP_TABLE              = os.environ["TEMP_TABLE"]
OUT_TABLE_LINE          = os.environ["OUT_TABLE_LINE"]
OUT_TABLE_CENTRIOD      = os.environ["OUT_TABLE_CENTRIOD"]

# Panoramax endpoint used to build links.
PANORAMAX_END_POINT     = os.environ["PANORAMAX_END_POINT"]

# Earth radius in metres.
# Used for distance calculations, for example converting radians to metres.
EARTH_RADIUS_M          = int(os.environ["EARTH_RADIUS_M"])


def setup_logging(debug=False):
    """
    Configure logging for the entire application.

    This function:
    - creates a log directory if it does not exist;
    - creates a timestamped log file;
    - configures the main "tisgrade" logger;
    - writes logs to a file;
    - also writes logs to the console when debug=True;
    - suppresses noisy logs from selected third-party packages.

    Parameters
    ----------
    debug:
        If True, set log level to DEBUG and add console logging.
        If False, set log level to INFO and log only to file.

    Returns
    -------
    logging.Logger
        Configured application logger.
    """

    # Create the log directory next to this Python file.
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)

    # Create a unique log filename with a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}.log"

    # Get the application-level logger.
    # Other modules can use child loggers such as "tisgrade.module_name".
    logger = logging.getLogger("tisgrade")

    # Use DEBUG level in debug mode, otherwise INFO.
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear any existing handlers.
    # This prevents duplicate log messages when setup_logging() is called again.
    for h in logger.handlers:
        h.close()
    logger.handlers.clear()

    # Prevent messages from also being passed to the root logger.
    logger.propagate = False


    # Define the format used for all log messages.
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a file handler so logs are written to the timestamped log file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # In debug mode, also write log messages to the console.
    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Reduce noise from third-party libraries.
    # These libraries can otherwise produce many INFO or DEBUG messages.
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('pyogrio').setLevel(logging.WARNING)

    # Log where the log file is stored.
    logger.info("Logging initialized. Log file: %s", log_file)

    return logger


def get_db_connection () -> "psycopg2.extensions.connection":
    """
    Create and return a PostgreSQL database connection.

    The connection uses:
    - username, host, port and database name from environment variables;
    - a token/password read from TOKEN_PATH;
    - keepalive settings from environment variables.

    The token file is expected to contain colon-separated values, where the last
    part is the token/password. This matches the current parsing logic.

    Returns
    -------
    psycopg2.extensions.connection
        Open PostgreSQL connection.
    """

    # Path to the local token/password file.
    _pgpass_path = TOKEN_PATH

    # Read the token/password from the file.
    # The current logic takes the last colon-separated part of the file content.
    with open(_pgpass_path, "r") as _f:
        _token = _f.read().strip().split(":")[-1]
    
    # Copy database settings into local variables.
    _username = USERNAME
    _host     = HOST
    _port     = PORT
    _database = DB
    
    # Build the PostgreSQL connection string.
    # quote_plus() makes the username safe for use in a URL.
    _conn_string = (
        f"postgresql://{quote_plus(_username)}:{_token}"
        f"@{_host}:{_port}/{_database}"
    )

    # Do not print the connection string because it contains the token/password.
    # print(f"_conn_string: {_conn_string}")

    # Log that a database connection is being opened.
    logger.info("Connecting to database")

    # Open the PostgreSQL connection.
    pg_connection = psycopg2.connect(
        _conn_string,

        # Keepalive settings help detect broken network/database connections.
        keepalives=KEEPALIVES,
        keepalives_idle=KEEPALIVES_IDLE,            # start probing after xs idle
        keepalives_interval=KEEPALIVES_INTERVAL,    # retry every xs
        keepalives_count=KEEPALIVES_COUNT,          # give up after x failed probes
    )

    return pg_connection