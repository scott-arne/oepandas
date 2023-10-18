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


class InvalidSyntax(Exception):
    """
    Invalid syntax for chemical file formats
    """
    pass


class InvalidSMARTS(InvalidSyntax):
    """
    Invalid SMARTS pattern
    """
    pass
