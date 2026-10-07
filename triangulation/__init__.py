import logging
from .locate import locate_object

# Library rule: attach a NullHandler to the package's top-level logger.
# This silences "no handlers found" warnings if the caller configures nothing,
# and leaves all real handler/level setup to the application.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["locate_object"]