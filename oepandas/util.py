from openeye import oechem
from .exception import UnsupportedFileFormat


def get_oeformat(ext_or_oeformat: int | str) -> int:
    """
    Get the OpenEye file format (from OEFormat)
    :param ext_or_oeformat: Extension or value from the OEFormat namespace
    :return: OEFormat value
    """
    if isinstance(ext_or_oeformat, str):
        if (oeformat := oechem.GetFileFormat(ext_or_oeformat)) != oechem.OEFormat_UNDEFINED:
            return oeformat
        raise UnsupportedFileFormat(f'Unsupported file format: {ext_or_oeformat}')
    elif isinstance(ext_or_oeformat, int):
        return ext_or_oeformat
    raise TypeError(f'Cannot get OEFormat from {ext_or_oeformat} of type {type(ext_or_oeformat).__name__}')
