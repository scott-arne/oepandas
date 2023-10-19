import base64
import logging
import base64 as b64
import gzip as python_gzip
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Callable
from openeye import oechem
from dataclasses import dataclass
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


def get_oeformat(ext_or_oeformat: int | str | Path, gzip: bool = False) -> FileFormat:
    """
    Get the OpenEye file format (from OEFormat)
    :param ext_or_oeformat: Extension or value from the OEFormat namespace
    :param gzip: Override for gzip (useful if ext_or_oeformat is an OEFormat type)
    :return: File format
    """
    # Just look at the file name if this was a path object
    if isinstance(ext_or_oeformat, Path):
        ext_or_oeformat = ext_or_oeformat.name

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


def get_oeformat_from_ext(p: str | Path) -> FileFormat:
    """
    Get the OpenEye file format from the extension of a particular file
    :param p: Path of the file to write
    :return: File format
    """
    # Ensure input is a path
    p = Path(p)
    suffix = ''.join(p.suffixes)
    return get_oeformat(suffix)


def is_gz(p: str | Path) -> bool:
    """
    Get whether a file is gzipped based on the extension
    :param p: File path
    :return: True if the file is gzipped
    """
    # Ensure input is a path
    p = Path(p)
    suffix = ''.join(p.suffixes)
    return suffix.endswith("gz")


def create_molecule_to_bytes_writer(
    fmt: str | int | FileFormat = "smiles",
    flavor: int = None,
    gzip: bool = False
):
    """
    Create a writer that takes a molecule and writes it to bytes.

    Note that "smiles", "canonical smiles" and "canonical_smiles" all have a special meaning and use the
    oechem.OEMolToSmiles function to create a canonical isomeric SMILES that does not contain the title. If using
    fmt=oechem.OEFormatSMI or fmt=".smi" or fmt=".ism" then you will get SMILES with titles.

    :param fmt: Molecule file format (string, oehem.OEFormat or oepandas FileFormat)
    :param flavor: Molecule flavor
    :param gzip: Gzip the molecule (forces b64 encoding)
    :return: Writer that takes a molecule and returns bytes
    """
    if isinstance(fmt, str):
        if fmt in ("smiles", "canonical smiles", "canonical_smiles"):

            def molecule_to_bytes(m: oechem.OEMolBase):
                """
                Provides access to a simpler version of a SMILES writer that does not add the title
                """
                retval = oechem.OEMolToSmiles(m).encode('utf-8')
                if gzip:
                    return base64.b64encode(python_gzip.compress(retval.encode('utf-8')))
                return retval

            return molecule_to_bytes

    if isinstance(fmt, (str, int)):
        # Get the molecule format
        fmt = get_oeformat(fmt)

    if isinstance(fmt, FileFormat):

        def molecule_to_bytes(m: oechem.OEMolBase):
            """
            Write to standard molecule formats
            """
            return oechem.OEWriteMolToBytes(
                fmt.oeformat,
                flavor or oechem.OEGetDefaultOFlavor(fmt.oeformat),
                fmt.gzip or gzip,
                m
            )

        return molecule_to_bytes

    raise TypeError(f'Cannot create a molecule_to_bytes writer from: {fmt}')


def create_molecule_to_string_writer(
        fmt: str | int | FileFormat = "smiles",
        flavor: int = None,
        gzip: bool = False,
        b64encode: bool = False,
        strip: bool = True
) -> Callable[[oechem.OEMolBase], str]:
    """
    Create a writer that takes a molecule and writes it to a string.

    Note that "smiles", "canonical smiles" and "canonical_smiles" all have a special meaning and use the
    oechem.OEMolToSmiles function to create a canonical isomeric SMILES that does not contain the title. If using
    fmt=oechem.OEFormatSMI or fmt=".smi" or fmt=".ism" then you will get SMILES with titles.

    :param fmt: Molecule file format (string, oehem.OEFormat or oepandas FileFormat)
    :param flavor: Molecule flavor
    :param gzip: Gzip the molecule (forces b64 encoding)
    :param b64encode: Force b64 encoding of the molecule
    :param strip: Strip newlines
    :return: Writer that takes a molecule and returns a string
    """
    if isinstance(fmt, str):
        if fmt in ("smiles", "canonical smiles", "canonical_smiles"):

            def molecule_to_string(m: oechem.OEMolBase):
                """
                Provides access to a simpler version of a SMILES writer that does not add the title
                """
                retval = oechem.OEMolToSmiles(m)
                if gzip:
                    retval = base64.b64encode(python_gzip.compress(retval.encode('utf-8'))).decode('utf-8')
                if b64encode:
                    retval = base64.b64encode(retval.encode('utf-8')).decode('utf-8')
                return retval

            return molecule_to_string

    if isinstance(fmt, (str, int)):
        # Get the molecule format
        fmt = get_oeformat(fmt)

    if isinstance(fmt, FileFormat):

        # Create the molecule to bytes writer (this creates a closure with the function below)
        molecule_to_bytes = create_molecule_to_bytes_writer(fmt, flavor, gzip)

        def molecule_to_string(m: oechem.OEMolBase):
            """
            Write to standard molecule formats
            """
            # First write to bytes
            mol_bytes = molecule_to_bytes(m)

            # Convert gzip or binary formats to base64
            if fmt.gzip or gzip or fmt.is_binary_format or b64encode:
                retval = b64.b64encode(mol_bytes).decode('utf-8')
            else:
                retval = mol_bytes.decode('utf-8')

            return retval.strip() if strip else retval

        return molecule_to_string

    raise TypeError(f'Cannot create a molecule_to_string writer from: {fmt}')


def molecule_from_string(
        mol: oechem.OEMolBase,
        string_or_bytes: str | bytes,
        fmt: FileFormat,
        b64decode: bool = False
) -> bool:
    """
    Convert a molecule from a string representation
    :param mol: OpenEye molecule
    :param string_or_bytes: String or bytes that represent the molecule
    :param fmt: File format
    :param b64decode: Force base64 decoding of string
    :return: Molecule
    """
    mol.Clear()

    if isinstance(string_or_bytes, str):
        string_or_bytes = string_or_bytes.encode("utf-8")

    if fmt.is_binary_format or fmt.gzip or b64decode:
        string_or_bytes = b64.b64decode(string_or_bytes)

    return oechem.OEReadMolFromBytes(mol, fmt.oeformat, fmt.gzip, string_or_bytes)


def predominant_type(series: pd.Series, sample_size: int = 25) -> None | type:
    """
    Look at a random subset off no empty rows and test if they are molecules
    :param series: Pandas series
    :param sample_size: Inspect at most this many rows
    :return: Predominant class found in the DataFrame sample
    """
    # Take a random sample of non-empty rows and test if these are molecules
    members = [type(x) for x in series[series.notnull()].sample(n=min(sample_size, len(series)))]
    if len(members) > 0:
        counts = Counter(members)
        return max(counts, key=counts.get)
    return None
