import logging
from .arrays import DesignUnitArray, DesignUnitDtype, MoleculeArray, MoleculeDtype, DisplayDtype, DisplayArray
from .pandas_extensions import (
    read_sdf,
    read_smi,
    read_oedb,
    read_molecule_csv,
    read_oeb,
    read_oedu
)
from .exception import FileError, UnsupportedFileFormat

__version__ = '2.1.0'

__all__ = [
    "exception",
    "util",
    "DesignUnitArray",
    "DesignUnitDtype",
    "MoleculeDtype",
    "MoleculeArray",
    "DisplayDtype",
    "DisplayArray",
    "read_sdf",
    "read_oeb",
    "read_smi",
    "read_molecule_csv",
    "read_oedb",
    "read_oedu",
]


########################################################################################################################
# Configure logging
########################################################################################################################
log = logging.getLogger("oepandas")


class LevelSpecificFormatter(logging.Formatter):
    """
    A logging formatter
    """
    NORMAL_FORMAT = "%(message)s"
    LEVEL_FORMAT = "%(levelname)s: %(message)s"

    def __init__(self):
        super().__init__(fmt=self.NORMAL_FORMAT, datefmt=None, style='%')

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record for printing
        :param record: Record to format
        :return: Formatted record
        """
        # Set log style
        self._style._fmt = self.LEVEL_FORMAT if record.levelno != logging.INFO else self.NORMAL_FORMAT

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        return result


############################
# Configure the logger
############################

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(LevelSpecificFormatter())
log.addHandler(ch)

log.setLevel(logging.INFO)
