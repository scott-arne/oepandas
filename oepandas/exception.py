class FileError(Exception):
    """
    General file errors
    """
    pass


class UnsupportedFileFormat(FileError):
    """
    Unsupported file formats
    """
    pass
