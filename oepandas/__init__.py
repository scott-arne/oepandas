import logging
from .molecule import MoleculeArray, MoleculeDtype
from .exception import FileError, UnsupportedFileFormat

__version__ = '0.1.4'


########################################################################################################################
# Configure logging
########################################################################################################################
log = logging.getLogger("oepandas")


class LevelSpecificFormatter(logging.Formatter):
    """
    A logging formatter
    """
    NORMAL_FORMAT = "%(message)s"
    DEBUG_FORMAT = "%(levelname)s: %(message)s"

    def __init__(self):
        super().__init__(fmt=self.NORMAL_FORMAT, datefmt=None, style='%')

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record for printing
        :param record: Record to format
        :return: Formatted record
        """
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.DEBUG_FORMAT
        else:
            self._style._fmt = self.NORMAL_FORMAT

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
