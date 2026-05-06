import base64
import logging
from collections.abc import Generator, Iterable, Sequence
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self, TypeAlias

import numpy as np
import pandas as pd
from openeye import oechem

# noinspection PyProtectedMember
from pandas._typing import Dtype, FilePath
from pandas.api.extensions import register_extension_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype

from ..exception import InvalidSMARTS
from ..util import (
    create_molecule_to_bytes_writer,
    create_molecule_to_string_writer,
    get_oeformat,
    is_gz,
    molecule_from_string,
)
from .base import OEExtensionArray

log = logging.getLogger("oepandas")


########################################################################################################################
# Helpers
########################################################################################################################

class MoleculeType(StrEnum):
    """
    Molecule implementation to use when constructing molecule arrays.
    """

    DEFAULT = "default"
    OEMOL = "oemol"
    OEGRAPHMOL = "oegraphmol"
    OEQMOL = "oeqmol"


MoleculeTypeInput: TypeAlias = MoleculeType | str | type[oechem.OEMolBase] | None

_MOLECULE_CLASS_BY_TYPE: dict[MoleculeType, type[oechem.OEMolBase] | None] = {
    MoleculeType.DEFAULT: None,
    MoleculeType.OEMOL: oechem.OEMol,
    MoleculeType.OEGRAPHMOL: oechem.OEGraphMol,
    MoleculeType.OEQMOL: oechem.OEQMol,
}

_MOLECULE_TYPE_BY_CLASS: dict[type[oechem.OEMolBase], MoleculeType] = {
    oechem.OEMol: MoleculeType.OEMOL,
    oechem.OEGraphMol: MoleculeType.OEGRAPHMOL,
    oechem.OEQMol: MoleculeType.OEQMOL,
}

_MOLECULE_TYPE_BY_STRING: dict[str, MoleculeType] = {
    molecule_type.value: molecule_type for molecule_type in MoleculeType
}


def _normalize_molecule_type(molecule_type: MoleculeTypeInput) -> MoleculeType:
    """
    Normalize user molecule type input.

    :param molecule_type: Molecule type selector.
    :returns: Normalized molecule type.
    :raises ValueError: When molecule type is unsupported.
    """
    if molecule_type is None:
        return MoleculeType.DEFAULT

    if isinstance(molecule_type, MoleculeType):
        return molecule_type

    if isinstance(molecule_type, str):
        key = molecule_type.casefold().replace("-", "").replace("_", "").replace(" ", "")
        if key in _MOLECULE_TYPE_BY_STRING:
            return _MOLECULE_TYPE_BY_STRING[key]

    if isinstance(molecule_type, type) and molecule_type in _MOLECULE_TYPE_BY_CLASS:
        return _MOLECULE_TYPE_BY_CLASS[molecule_type]

    raise ValueError(f"Unsupported molecule_type: {molecule_type!r}")


def _molecule_class_for_type(molecule_type: MoleculeTypeInput) -> type[oechem.OEMolBase] | None:
    """
    Return the OpenEye molecule class for a molecule type selector.

    :param molecule_type: Molecule type selector.
    :returns: Molecule class, or ``None`` for the default preservation mode.
    """
    return _MOLECULE_CLASS_BY_TYPE[_normalize_molecule_type(molecule_type)]


def _new_molecule(molecule_type: MoleculeTypeInput) -> oechem.OEMolBase:
    """
    Create a new molecule for the requested molecule type.

    :param molecule_type: Molecule type selector.
    :returns: New molecule instance.
    """
    molecule_cls = _molecule_class_for_type(molecule_type) or oechem.OEMol
    return molecule_cls()


def _coerce_molecule(mol: oechem.OEMolBase, molecule_type: MoleculeTypeInput) -> oechem.OEMolBase:
    """
    Coerce a molecule to the requested molecule type.

    :param mol: Molecule to coerce.
    :param molecule_type: Molecule type selector.
    :returns: Coerced molecule, or the original molecule in default mode.
    """
    molecule_cls = _molecule_class_for_type(molecule_type)
    if molecule_cls is None:
        return mol
    return molecule_cls(mol)


def _has_data_and_is_not_blank(mol: oechem.OEMolBase, tag: str) -> bool:
    """
    Check if a molecue has data (SD or generic) and that it is not blank/None
    :param mol: Molecule
    :param tag: Data tag
    :return: True if the molecule has that data tag, and it is not blank/None
    """
    # noinspection PyBroadException
    try:
        if oechem.OEHasSDData(mol, tag) and oechem.OEGetSDData(mol, tag) not in (None, ""):
            return True

        if mol.HasData(tag) and mol.GetData(tag) not in (None, ""):
            return True
    except Exception:
        pass

    return False

def _read_molecules(
        fp: FilePath,
        file_format: int | str,
        *,
        flavor: int | None = None,
        conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
        gzip: bool = False,
        no_title: bool = False,
        molecule_type: MoleculeTypeInput = None
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
    :param no_title: Remove titles
    :param molecule_type: Molecule implementation for returned molecules
    :return: Generator over molecules
    """
    # noinspection PyTypeChecker
    fmt = get_oeformat(file_format, gzip=gzip or is_gz(fp))
    normalized_molecule_type = _normalize_molecule_type(molecule_type)
    read_molecule_type = (
        MoleculeType.OEMOL
        if normalized_molecule_type == MoleculeType.DEFAULT
        else normalized_molecule_type
    )

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
            if no_title:
                mol.SetTitle("")
                mol.GetActive().SetTitle("")
            yield _coerce_molecule(mol, read_molecule_type)


########################################################################################################################
# Molecule Array (for oechem.OEMol objects)
########################################################################################################################

class MoleculeArray(OEExtensionArray[oechem.OEMolBase]):

    # For type checking in methods defined in OEExtensionArray
    _base_openeye_type = oechem.OEMolBase

    """
    Custom extension for an array of molecules
    """
    def __init__(
            self,
            mols: None | oechem.OEMolBase | Iterable[oechem.OEMolBase | None],
            copy: bool = False,
            metadata: dict | None = None,
            deepcopy: bool = False,
            molecule_type: MoleculeTypeInput = None
    ):
        """
        Initialize
        :param mols: Molecule or an iterable of molecules
        :param copy: Create copy of the molecules if True
        """
        normalized_molecule_type = _normalize_molecule_type(molecule_type)
        processed = []

        if mols is not None:

            # Handle singleton mols
            if isinstance(mols, oechem.OEMolBase):
                mols = (mols,)

            # noinspection PyTypeChecker
            for i, mol in enumerate(mols):

                # Molecules
                if isinstance(mol, oechem.OEMolBase):

                    if normalized_molecule_type == MoleculeType.DEFAULT:
                        # Preserve the concrete OpenEye molecule class unless an explicit type is requested.
                        if deepcopy:
                            processed.append(mol.CreateCopy())
                        else:
                            processed.append(mol)

                    else:
                        processed.append(_coerce_molecule(mol, normalized_molecule_type))

                # None/NaN values are allowed
                elif pd.isna(mol):
                    processed.append(None)

                # Anything else is invalid
                else:
                    raise TypeError(
                        f"Cannot create MoleculeArray containing object of type {type(mol).__name__} "
                        f"at index {i}. All elements must be OEMolBase or None/NaN."
                    )

        # Superclass initialization
        super().__init__(processed, copy=copy, metadata=metadata)

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,  # noqa
            copy: bool = False,
            molecule_format: str | int | None = None,
            gzip: bool = False,
            molecule_type: MoleculeTypeInput = None
    ) -> Self:
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param copy: Copy the molecules (otherwise stores pointers)
        :param molecule_type: Molecule implementation for parsed and existing molecules
        :return: New instance of Molecule Array
        """
        normalized_molecule_type = _normalize_molecule_type(molecule_type)

        # Molecules
        mols = []

        # Default format is SMILES if none was specified
        fmt = get_oeformat(oechem.OEFormat_SMI) if molecule_format is None else get_oeformat(molecule_format, gzip)

        for i, obj in enumerate(scalars):

            # Nones are OK
            if obj is None or pd.isna(obj):
                mols.append(None)

            # Molecule subclasses
            elif isinstance(obj, oechem.OEMolBase):
                mols.append(_coerce_molecule(obj, normalized_molecule_type))

            elif isinstance(obj, bytes):
                mol = _new_molecule(normalized_molecule_type)
                if not oechem.OEReadMolFromBytes(mol, fmt.oeformat, fmt.gzip, obj):
                    log.warning("Could read molecule %i from bytes using format '%s'", i + 1, fmt.name)
                mols.append(mol)

            # Read from string
            elif isinstance(obj, str):
                mol = _new_molecule(normalized_molecule_type)
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
            dtype: Dtype | None = None,  # noqa
            copy: bool = False,  # noqa
            molecule_format: int | str | None = None,
            b64decode: bool = False,
            molecule_type: MoleculeTypeInput = None) -> Self:
        """
        Read molecules form a sequence of strings (this is an optimization of _from_sequence, which does more
        type checking)
        :param strings: Sequence of strings
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :param molecule_format: Molecule file format
        :param b64decode: Force base64 decoding of molecule strings
        :param molecule_type: Molecule implementation for parsed molecules
        :return: Array of molecules
        """
        normalized_molecule_type = _normalize_molecule_type(molecule_type)

        # Default format is SMILES
        molecule_format = molecule_format or oechem.OEFormat_SMI

        # Standardize the format
        molecule_format_ = get_oeformat(molecule_format)

        mols = []
        for i, s in enumerate(strings):
            mol = _new_molecule(normalized_molecule_type)

            if isinstance(s, str):

                # If we need to base64 decode
                if b64decode:
                    s = base64.b64decode(s).decode('utf-8')

                if not molecule_from_string(mol, s.strip(), molecule_format_):
                    log.warning("Could not convert molecule %d from '%s': %s", i + 1, molecule_format_.name, s)

            mols.append(mol)

        return cls(mols, copy=False)

    @classmethod
    def from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            molecule_format: int | str | None = None,
            b64decode: bool = False,
            molecule_type: MoleculeTypeInput = None
    ) -> Self:
        """
        Public alias of _from_sequence_of_strings
        :param strings: Sequence of strings
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :param molecule_format: Molecule file format
        :param b64decode: Force base64 decoding of molecule strings
        :param molecule_type: Molecule implementation for parsed molecules
        :return: Array of molecules
        """
        return cls._from_sequence_of_strings(
            strings,
            dtype=dtype,
            copy=copy,
            molecule_format=molecule_format,
            b64decode=b64decode,
            molecule_type=molecule_type
        )

    @classmethod
    def from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            molecule_format: str | int | None = None,
            gzip: bool = False,
            molecule_type: MoleculeTypeInput = None
    ) -> Self:
        """
        Public alias of _from_sequence
        :param scalars: Sequence of objects
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :param molecule_format: Molecule file format
        :param gzip: Whether the objects are gzipped
        :param molecule_type: Molecule implementation for parsed and existing molecules
        :return: Array of molecules
        """
        return cls._from_sequence(
            scalars,
            dtype=dtype,
            copy=copy,
            molecule_format=molecule_format,
            gzip=gzip,
            molecule_type=molecule_type
        )

    @property
    def dtype(self) -> PandasExtensionDtype:
        return MoleculeDtype()

    def deepcopy(self, metadata: bool | dict | None = True):
        return MoleculeArray(self._objs, metadata=self._resolve_metadata(metadata), deepcopy=True)

    # ------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------

    @classmethod
    def read_smi(
            cls,
            fp: FilePath,
            flavor: int | None = None,
            no_title: bool = False,
            molecule_type: MoleculeTypeInput = None,
            **_
    ) -> Self:
        """
        Read molecules from an SMILES file and return an array
        :param fp: Path to the SMILES file
        :param flavor: OpenEye input flavor
        :param no_title: If true, do not include title
        :param molecule_type: Molecule implementation for parsed molecules
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_SMI,
                flavor=flavor,
                no_title=no_title,
                molecule_type=molecule_type
            )
        )

    @classmethod
    def read_sdf(
            cls,
            fp: FilePath,
            flavor: int | None = None,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
            no_title: bool = False,
            molecule_type: MoleculeTypeInput = None
    ) -> Self:
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
        :param no_title: If true, do not include title
        :param molecule_type: Molecule implementation for parsed molecules
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_SDF,
                flavor=flavor,
                conformer_test=conformer_test,
                no_title=no_title,
                molecule_type=molecule_type
            )
        )

    @classmethod
    def read_oeb(
            cls,
            fp: FilePath,
            flavor: int | None = None,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
            no_title: bool = False,
            molecule_type: MoleculeTypeInput = None
    ) -> Self:
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
        :param no_title: If true, do not include title
        :param molecule_type: Molecule implementation for parsed molecules
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecules(
                fp,
                oechem.OEFormat_OEB,
                flavor=flavor,
                conformer_test=conformer_test,
                no_title=no_title,
                molecule_type=molecule_type
            )
        )

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    # noinspection PyPep8Naming
    def substructure_search(
            self,
            pattern: str | oechem.OEQMol | oechem.OESubSearch,
            adjustH: bool = False
    ) -> np.ndarray:
        """
        Return a boolean array of whether molecules are a substructure match to a pattern.

        :param pattern: SMARTS pattern, OEQMol query, or OESubSearch object.
        :param adjustH: Match implicit/explicit hydrogen state between query and target molecule.
        :param mapidx: Annotate SMARTS map indexes on matches
        :returns: Boolean array.
        """
        if isinstance(pattern, str | oechem.OEQMol):
            ss = oechem.OESubSearch(pattern)
        elif isinstance(pattern, oechem.OESubSearch):
            ss = pattern
        else:
            raise InvalidSMARTS(
                f"Invalid substructure search pattern of type {type(pattern).__name__}; "
                "expected SMARTS string, oechem.OEQMol, or oechem.OESubSearch."
            )

        if not ss.IsValid():
            if isinstance(pattern, str):
                raise InvalidSMARTS(f'Invalid SMARTS pattern: {pattern}')
            else:
                raise InvalidSMARTS("Invalid oechem.OESubSearch object provided to match")

        matches = []
        for mol in self:
            if mol is None or not mol.IsValid():
                matches.append(False)

            # noinspection PyBroadException
            try:
                oechem.OEPrepareSearch(mol, ss, adjustH)
                matches.append(ss.SingleMatch(mol))

            except Exception:
                matches.append(False)

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

        def _smiles(mol):
            if mol is not None and mol.IsValid():
                return oechem.OECreateSmiString(mol, flavor)
            return ''

        vectorized = np.frompyfunc(_smiles, 1, 1)(self._objs)
        return np.asarray(vectorized, dtype=str)

    def structural_eq(self, other, flavor=None):
        s1 = self.to_smiles(flavor=flavor)
        s2 = other.to_smiles(flavor=flavor)
        return s1 == s2

    def to_fingerprints(
            self,
            fptype: str,
            num_bits: int,
            min_distance: int,
            max_distance: int,
            atom_type: str | int,
            bond_type: str | int,
            desalt: bool = True,
            quiet: bool = True
    ):
        """
        Generate fingerprints from molecules in this array.

        :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
        :param num_bits: Number of bits in the fingerprint.
        :param min_distance: Minimum distance (path, tree) or radius (circular).
        :param max_distance: Maximum distance (path, tree) or radius (circular).
        :param atom_type: "|" separated OEFPAtomType members or bitmask.
        :param bond_type: "|" separated OEFPBondType members or bitmask.
        :param desalt: Remove salts by keeping only the largest component.
        :param quiet: Silence the progress bar.
        :returns: FingerprintArray containing the generated fingerprints.
        """
        # Late import to avoid circular dependency
        from .fingerprint import FingerprintArray

        return FingerprintArray.from_molecules(
            self,
            fptype=fptype,
            num_bits=num_bits,
            min_distance=min_distance,
            max_distance=max_distance,
            atom_type=atom_type,
            bond_type=bond_type,
            desalt=desalt,
            quiet=quiet
        )


@register_extension_dtype
class MoleculeDtype(PandasExtensionDtype):
    """
    OpenEye molecule datatype for Pandas
    """

    type: ClassVar[type] = oechem.OEMolBase
    name: ClassVar[str] = "molecule"
    kind: ClassVar[str] = "O"
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
