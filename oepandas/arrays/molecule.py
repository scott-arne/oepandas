import logging
import numpy as np
import pandas as pd
from openeye import oechem
from typing import Any, Generator, Literal
from collections.abc import Iterable, Sequence
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.api.extensions import register_extension_dtype
# noinspection PyProtectedMember
from pandas._typing import FilePath, Dtype
from ..util import (
    get_oeformat,
    is_gz,
    molecule_from_string,
    create_molecule_to_string_writer,
    create_molecule_to_bytes_writer,
)
from .base import OEExtensionArray
from ..exception import InvalidSMARTS

log = logging.getLogger("oepandas")


########################################################################################################################
# Helpers
########################################################################################################################

def _read_molecules(
        fp: FilePath,
        file_format: int | str,
        *,
        flavor: int | None = None,
        conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
        gzip: bool = False
) -> Generator[oechem.OEMolBase, None, None]:
    """
    Generator over flavored reading of molecules in a specific file format

    Use conformer_test to combine single conformers into multi-conformer molecules:
        - "default":
                No conformer testing.
        - "absolute":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same title.
        - "absolute_canonical":
                Combine conformers if they have the same canonical SMILES
        - "isomeric":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same atom and bond
                stereochemistry, (4) have the same title.
        - "omega":
                Equivalent to "isomeric" except that invertible nitrogen stereochemistry is also taken into account.

    :param fp: File path
    :param file_format: File format (oechem.OEFormat)
    :param flavor: Optional flavor (oechem.OEIFlavor)
    :param conformer_test: OpenEye conformer testing method
    :param gzip: File is gzipped
    :return: Generator over molecules
    """
    fmt = get_oeformat(file_format, gzip=gzip or is_gz(fp))

    with oechem.oemolistream(str(fp)) as ifs:

        if ifs.GetFormat() != oechem.OEFormat_UNDEFINED and ifs.GetFormat() != fmt.oeformat:
            actual_fmt = get_oeformat(ifs.GetFormat())
            log.warning(
                "Reader expects '%s' format but file was opened with '%s' format. Forcing expected '%s' format, "
                "which may have undesirable consequences. This may not be the correct reader for this format.",
                fmt.name, actual_fmt.name, fmt.name
            )

        ifs.SetFormat(fmt.oeformat)

        # Conformer testing
        if conformer_test == "default":
            ifs.SetConfTest(oechem.OEDefaultConfTest())

        elif conformer_test == "absolute":
            ifs.SetConfTest(oechem.OEAbsoluteConfTest())

        elif conformer_test == "absolute_canonical":
            ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())

        elif conformer_test == "isomeric":
            ifs.SetConfTest(oechem.OEIsomericConfTest())

        elif conformer_test == "omega":
            ifs.SetConfTest(oechem.OEOmegaConfTest())

        # Set flavor if requested
        if flavor is not None:
            ifs.SetFlavor(file_format, flavor)

        # Read gzipped formats
        ifs.Setgz(fmt.gzip)

        for mol in ifs.GetOEMols():
            yield mol.CreateCopy()


########################################################################################################################
# Molecule Array (for oechem.OEMol objects)
########################################################################################################################

class MoleculeArray(OEExtensionArray[oechem.OEMol]):

    # For type checking in methods defined in OEExtensionArray
    _base_openeye_type = oechem.OEMol

    """
    Custom extension for an array of molecules
    """
    def __init__(
            self,
            mols: None | oechem.OEMolBase | Iterable[oechem.OEMolBase | None],
            copy: bool = False
    ):
        """
        Initialize
        :param mols: Molecule or an iterable of molecules
        :param copy: Create copy of the molecules if True
        """
        # Handle singleton mols
        if isinstance(mols, oechem.OEMolBase):
            mols = (mols,)

        # Handle OEGraphMol types
        mols = [oechem.OEMol(mol) if isinstance(mol, oechem.OEGraphMol) else mol for mol in mols]

        # Superclass initialization
        super().__init__(mols, copy=copy)

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            molecule_format: str | int | None = None,
            gzip: bool = False
    ) -> 'MoleculeArray':
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param copy: Copy the molecules (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        # Molecules
        mols = []

        # Default format is SMILES if none was specified
        fmt = get_oeformat(oechem.OEFormat_SMI) if molecule_format is None else get_oeformat(molecule_format, gzip)

        for i, obj in enumerate(scalars):

            # Nones are OK
            if obj is None or pd.isna(obj):
                mols.append(None)

            # Molecule subclasses
            elif isinstance(obj, oechem.OEMol):
                mols.append(obj)

            elif isinstance(obj, oechem.OEGraphMol):
                mols.append(oechem.OEMol(obj))

            elif isinstance(obj, bytes):
                mol = oechem.OEMol()
                if not oechem.OEReadMolFromBytes(mol, fmt.oeformat, fmt.gzip, obj):
                    log.warning("Could read molecule %i from bytes using format '%s'", i + 1, fmt.name)
                mols.append(mol)

            # Read from string
            elif isinstance(obj, str):
                mol = oechem.OEMol()
                if not molecule_from_string(mol, obj, fmt):
                    log.warning("Could read molecule %i from string using format '%s'", i + 1, fmt.name)
                mols.append(mol)

            else:
                raise TypeError(f'Cannot create a molecule from {type(obj).__name__}')

        return cls(mols, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            molecule_format: int | None = None,
            b64decode: bool = False) -> 'MoleculeArray':
        """
        Read molecules form a sequence of strings (this is an optimization of _from_sequence, which does more
        type checking)
        :param strings: Sequence of strings
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :param b64decode: Force base64 decoding of molecule strings
        :return: Array of molecules
        """
        # Default format is SMILES
        molecule_format = molecule_format or oechem.OEFormat_SMI

        # Standardize the format
        molecule_format = get_oeformat(molecule_format)

        mols = []
        for i, s in enumerate(strings):  # type: int, str
            mol = oechem.OEMol()

            if not (isinstance(s, str) and molecule_from_string(mol, s.strip(), molecule_format)):
                log.warning("Could not convert molecule %d from '%s': %s", i + 1, molecule_format.name, s)

            mols.append(mol)

        return cls(mols, copy=False)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return MoleculeDtype()

    # ------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------

    @classmethod
    def read_smi(
            cls,
            fp: FilePath,
            flavor: int | None = None
    ) -> 'MoleculeArray':
        """
        Read molecules from an SMILES file and return an array
        :param fp: Path to the SMILES file
        :param flavor: OpenEye input flavor
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_SMI,
                flavor=flavor
            )
        )

    @classmethod
    def read_sdf(
            cls,
            fp: FilePath,
            flavor: int | None = None,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default"
    ) -> 'MoleculeArray':
        """
        Read molecules from an SD file and return an array

        Use conformer_test to combine single conformers into multi-conformer molecules:
        - "default":
                No conformer testing.
        - "absolute":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same title.
        - "absolute_canonical":
                Combine conformers if they have the same canonical SMILES
        - "isomeric":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same atom and bond
                stereochemistry, (4) have the same title.
        - "omega":
                Equivalent to "isomeric" except that invertible nitrogen stereochemistry is also taken into account.

        :param fp: Path to the SD file
        :param flavor: OpenEye input flavor
        :param conformer_test: Conformer testing
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_SDF,
                flavor=flavor,
                conformer_test=conformer_test
            )
        )

    @classmethod
    def read_oeb(
            cls,
            fp: FilePath,
            flavor: int | None = None,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default"
    ) -> 'MoleculeArray':
        """
        Read molecules from an OEB file and return an array

        Use conformer_test to combine single conformers into multi-conformer molecules:
        - "default":
                No conformer testing.
        - "absolute":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same title.
        - "absolute_canonical":
                Combine conformers if they have the same canonical SMILES
        - "isomeric":
                Combine conformers if they (1) have the same number of atoms and bonds in the same order, (2)
                each atom and bond have identical properties in the connection table, (3) have the same atom and bond
                stereochemistry, (4) have the same title.
        - "omega":
                Equivalent to "isomeric" except that invertible nitrogen stereochemistry is also taken into account.

        :param fp: Path to the OEB file
        :param flavor: OpenEye input flavor
        :param conformer_test: Conformer testing
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_OEB,
                flavor=flavor,
                conformer_test=conformer_test
            )
        )

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    # noinspection PyPep8Naming
    def subsearch(self, pattern: str | oechem.OESubSearch, adjustH: bool = False) -> np.ndarray:
        """
        Return a boolean array of whether molecules are a substructure match to a pattern
        :param pattern: SMARTS pattern or OpenEye subsearch object
        :param adjustH: Match implicit/explicit hydrogen state between query and target molecule
        :return: Boolean array
        """
        ss = oechem.OESubSearch(pattern) if isinstance(pattern, str) else pattern

        if not ss.IsValid():
            if isinstance(pattern, str):
                raise InvalidSMARTS(f'Invalid SMARTS pattern: {pattern}')
            else:
                raise InvalidSMARTS("Invalid oechem.OESubSearch object provided to match")

        matches = []
        for mol in self:
            oechem.OEPrepareSearch(mol, ss, adjustH)
            matches.append(ss.SingleMatch(mol))
        return np.array(matches, dtype=bool)

    # --------------------------------------------------------
    # Conversions
    # --------------------------------------------------------

    def to_molecule_strings(
            self,
            molecule_format: str | int = "smiles",
            flavor: int | None = None,
            gzip: bool = False,
            b64encode: bool = False
    ) -> np.ndarray:
        """
        Write molecules to an array of strings.

        Missing or invalid molecules are represented by empty strings. Binary or gzipped formats are automatically
        base64 encoded so for valid string representations. Note that gzip is automatically inferred if you provide
        an extension ending in gz.

        Note that excess newlines are stripped form the strings.

        :param molecule_format: Molecule format (extension or oechem.OEFormat)
        :param flavor: Output flavor (None will give the default)
        :param gzip: Gzip the molecule string (will be base64 encoded)
        :param b64encode: Force base64 encoding for all molecules
        :return: Array of molecule strings
        """
        # Create the function that will convert molecules to strings
        molecule_to_string = create_molecule_to_string_writer(
            fmt=molecule_format,
            flavor=flavor,
            gzip=gzip,
            b64encode=b64encode,
            strip=True
        )

        molecule_strings = []
        for mol in self:
            if mol is not None and mol.IsValid():
                molecule_strings.append(molecule_to_string(mol))
            else:
                molecule_strings.append('')

        return np.array(molecule_strings, dtype=str)

    def to_molecule_bytes(
            self,
            molecule_format: str | int = oechem.OEFormat_SMI,
            flavor: int | None = None,
            gzip: bool = False
    ) -> np.ndarray:
        """
        Write molecules to an array of bytes.

        Invalid or empty molecules are added as an empty bytes string.

        :param molecule_format: Molecule format (extension or oechem.OEFormat)
        :param flavor: Output flavor (None will give the default flavor)
        :param gzip: Gzip the molecule bytes
        :return: Array of molecule bytes
        """

        # Get the molecule format
        to_molecule_bytes = create_molecule_to_bytes_writer(molecule_format, flavor, gzip)

        molecule_bytes = []
        for mol in self:
            if mol is not None and mol.IsValid():
                molecule_bytes.append(to_molecule_bytes(mol))

            # None for invalid molecules
            else:
                molecule_bytes.append(b'')

        return np.array(molecule_bytes, dtype=bytes)

    def to_smiles(self, flavor: int | None = None) -> np.ndarray:
        """
        Convert array to SMILES.

        This is implemented using a more efficient method for SMILES than to_molecule_string. Invalid or missing
        molecules are represented by empty strings.

        :return: Array of molecule strings.
        """
        # Default flavor is canonical isomeric SMILES
        flavor = flavor or oechem.OESMILESFlag_ISOMERIC

        smiles = []
        for mol in self:
            if mol is not None and mol.IsValid():
                smiles.append(oechem.OECreateSmiString(mol, flavor))
            else:
                smiles.append('')
        return np.array(smiles, dtype=str)


@register_extension_dtype
class MoleculeDtype(PandasExtensionDtype):
    """
    OpenEye molecule datatype for Pandas
    """

    type: type = oechem.OEMol
    name: str = "molecule"
    kind: str = "O"
    base = np.dtype("O")

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return MoleculeArray

    # Required
    def __hash__(self) -> int:
        return hash(self.name)

    # Required
    def __eq__(self, other: str | type) -> bool:
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, type(self))

    # Required
    def __str__(self):
        return self.name

    # Required
    def __repr__(self):
        return self.__str__()
