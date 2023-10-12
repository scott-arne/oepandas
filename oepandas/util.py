import logging
from openeye import oechem
from dataclasses import dataclass
from base64 import b64encode, b64decode
from .exception import UnsupportedFileFormat

log = logging.getLogger("oepandas")


@dataclass(frozen=True)
class FileFormat:
    ext: str
    oeformat: int
    name: str
    gzip: bool

    @property
    def is_binary_format(self):
        return self.oeformat in {oechem.OEFormat_OEB, oechem.OEFormat_OEZ}


def get_oeformat(ext_or_oeformat: int | str, gzip: bool = False) -> FileFormat:
    """
    Get the OpenEye file format (from OEFormat)
    :param ext_or_oeformat: Extension or value from the OEFormat namespace
    :param gzip: Override for gzip (useful if ext_or_oeformat is an OEFormat type)
    :return: OEFormat value
    """
    if isinstance(ext_or_oeformat, str):
        if (oeformat := oechem.GetFileFormat(ext_or_oeformat)) != oechem.OEFormat_UNDEFINED:
            return FileFormat(
                ext=oechem.OEGetFormatExtension(oeformat).split(',')[0],
                oeformat=oeformat,
                name=oechem.OEGetFormatString(oeformat),
                gzip=gzip or ext_or_oeformat.endswith(".gz")
            )
        raise UnsupportedFileFormat(f'Unsupported file format: {ext_or_oeformat}')
    elif isinstance(ext_or_oeformat, int):
        return FileFormat(
            ext=oechem.OEGetFormatExtension(ext_or_oeformat).split(',')[0],
            oeformat=ext_or_oeformat,
            name=oechem.OEGetFormatString(ext_or_oeformat),
            gzip=gzip
        )
    raise TypeError(f'Cannot get OEFormat from {ext_or_oeformat} of type {type(ext_or_oeformat).__name__}')


def molecule_to_string(mol: oechem.OEMolBase, fmt: FileFormat) -> str:
    """
    Convert a molecule to a string representation
    :param mol: OpenEye molecule
    :param fmt: File format
    :return: Molecule as string
    """
    b = oechem.OEWriteMolToBytes(mol, fmt.oeformat, fmt.gzip)  # type: bytes

    if fmt.is_binary_format or fmt.gzip:
        return b64encode(b).decode("utf-8")

    return b.decode("utf-8")


def molecule_from_string(mol: oechem.OEMolBase, string_or_bytes: str | bytes, fmt: FileFormat) -> bool:
    """
    Convert a molecule from a string representation
    :param mol: OpenEye molecule
    :param string_or_bytes: String or bytes that represent the molecule
    :param fmt: File format
    :return: Molecule
    """
    mol.Clear()

    if isinstance(string_or_bytes, str):
        string_or_bytes = string_or_bytes.encode("utf-8")

    if fmt.is_binary_format or fmt.gzip:
        string_or_bytes = b64decode(string_or_bytes)

    return oechem.OEReadMolFromBytes(mol, fmt.oeformat, fmt.gzip, string_or_bytes)
