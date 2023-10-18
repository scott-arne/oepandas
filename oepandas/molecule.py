import logging
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Literal, Any, Mapping, Protocol
from collections.abc import Iterable, Hashable, Sequence, Generator
from pandas.core.ops import unpack_zerodim_and_defer
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.core.algorithms import take as pandas_take
# noinspection PyProtectedMember
from pandas.io.parsers.readers import _c_parser_defaults
# noinspection PyProtectedMember
from pandas._libs import lib
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
    register_dataframe_accessor,
    register_series_accessor
)
# noinspection PyProtectedMember
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    Dtype,
    DtypeArg,
    DtypeBackend,
    StorageOptions,
)
from openeye import oechem
from copy import copy as shallow_copy
from .util import get_oeformat, molecule_from_string, create_molecule_to_string_writer, create_molecule_to_bytes_writer
from .exception import FileError, InvalidSMARTS

if sys.version_info >= (3, 11):
    from typing import Self  # pyright: ignore[reportUnusedImport]
else:
    from typing_extensions import Self  # pyright: ignore[reportUnusedImport]

# noinspection PyProtectedMember
from pandas._typing import Shape, FilePath, IndexLabel, ReadBuffer, HashableT, TakeIndexer, ArrayLike, FillnaOptions

log = logging.getLogger("oepandas")


########################################################################################################################
# Helpers
########################################################################################################################

def _read_molecule_file(
        fp: FilePath,
        file_format: int | str,
        *,
        flavor: int | None = None,
        astype: type[oechem.OEMolBase] = oechem.OEMol,
        conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default"
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
    :param astype: OpenEye molecule type to read (oechem.OEMolBase or oechem.OEGraphMol)
    :return: Generator over molecules
    """
    fmt = get_oeformat(file_format)

    # Conformer test forces OEMol
    if conformer_test != "default":
        astype = oechem.OEMol

    with oechem.oemolistream(str(fp)) as ifs:
        ifs.SetFormat(fmt.oeformat)

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

        iterator = ifs.GetOEMols if astype is oechem.OEMol else ifs.GetOEGraphMols
        for mol in iterator():
            yield mol.CreateCopy()


########################################################################################################################
# Pandas Extensions
#
# Great resources for this:
#   - https://itnext.io/guide-to-pandas-extension-types-and-how-to-create-your-own-3b213d689c86
#   - https://stackoverflow.com/questions/68893521/simple-example-of-pandas-extensionarray
#
# Pandas Documentation:
#   - https://github.com/pandas-dev/pandas/blob/e7e7b40722e421ef7e519c645d851452c70a7b7c/pandas/core/arrays/base.py
#   - https://pandas.pydata.org/docs/reference/api/pandas.api.extensions.ExtensionDtype.html
########################################################################################################################


class MoleculeArray(ExtensionScalarOpsMixin, ExtensionArray):
    """
    Custom extension for an array of molecules
    """
    def __init__(self, mols: oechem.OEMolBase | Iterable[oechem.OEMolBase], copy=False):
        """
        Initialize
        :param mols: Sequence/array of molecules
        :param copy: Create copy of the molecules if True
        """
        if isinstance(mols, Iterable):
            self.mols = np.array([mol.CreateCopy() if copy else mol for mol in mols])
        elif isinstance(mols, oechem.OEMolBase):
            self.mols = np.array([mols.CreateCopy()] if copy else [mols])
        else:
            raise TypeError(f'Cannot create MoleculeArray from {type(mols).__name__}')

    @classmethod
    def _from_sequence(cls, scalars: Iterable[Any], *, dtype=None, copy=False,
                       fmt: str | int = oechem.OEFormat_SMI) -> Self:
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param dtype: Coerce to this datatype (must be a subclass of oechem.OEMolBase)
        :param copy: Copy the molecules (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        mols = []
        fmt = oechem.OEGetFormatString(fmt)

        for i, obj in enumerate(scalars):

            # Nones are OK
            if obj is None:
                mols.append(None)

            # Molecule subclasses
            elif isinstance(obj, oechem.OEMolBase):
                mols.append(mol.CreateCopy() if copy else mol for mol in scalars)

            elif isinstance(obj, (str, bytes)):
                mol = oechem.OEGraphMol()
                if not molecule_from_string(mol, obj, fmt):
                    log.warning("Could not convert molecule %d from '%s' %s",
                                i + 1, fmt.name, type(obj).__name__)

            # Else who knows
            else:
                raise TypeError(f'Cannot create a molecule from {type(obj).__name__}')
        return cls([mol.CreateCopy() if copy else mol for mol in scalars])

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            astype: type[oechem.OEMolBase] = oechem.OEGraphMol,
            copy: bool = False,
            fmt: int = oechem.OEFormat_SMI,
            b64decode: bool = False) -> Self:
        """
        Read molecules form a sequence of strings
        :param strings: Sequence of strings
        :param astype: Data type for molecules (must be oechem.OEMolBase)
        :param copy: Not used (here for API compatibility)
        :param b64decode: Force base64 decoding of molecule strings
        :return: Array of molecules
        """
        if not issubclass(astype, oechem.OEMolBase):
            raise TypeError("Can only read molecules from string as an oechem.OEMolBase type")

        # Standardize the format
        fmt = get_oeformat(fmt)

        mols = []
        for i, s in enumerate(strings):  # type: int, str
            mol = astype()

            if not (isinstance(s, str) and molecule_from_string(mol, s.strip(), fmt)):
                log.warning("Could not convert molecule %d from '%s': %s", i + 1, fmt.name, s)

            mols.append(mol)

        return cls(mols)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return MoleculeDtype()

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill N/A values and invalid molecules
        :param value: Fill value
        :param method: Method (does not do anything here)
        :param limit: Maximum number of entries to fill
        :param copy: Whether to copy the data
        :return: Filled extension array
        """
        # Sanity check
        if limit is not None:
            limit = max(0, limit)

        # Data to fill
        data = np.array([(obj.CreateCopy() if isinstance(obj, oechem.OEMolBase) else obj) for obj in self.mols]) \
            if copy else self.mols

        # Filled data
        filled = []

        for i, obj in enumerate(data):
            # Termination condition with limit
            if limit is not None and i >= limit:
                filled.extend(data[i:])
                break

            # NaN evaluation
            if pd.isna(obj):
                filled.append(value)
            elif isinstance(obj, oechem.OEMolBase):
                filled.append((obj.CreateCopy() if copy else obj) if obj.IsValid() else value)
            else:
                raise TypeError(f'MoleculeArray cannot determine of object of type {type(obj).__name__} is Na')

        return MoleculeArray(filled)

    def dropna(self) -> Self:
        """
        Drop all NA and invalid molecules
        :return: MoleculeArray with no missing or invalid molecules
        """
        non_missing = []

        for obj in self.mols:
            if not pd.isna(obj):
                if isinstance(obj, oechem.OEMolBase):
                    if obj.IsValid():
                        non_missing.append(obj)
                else:
                    raise TypeError(f'MoleculeArray cannot determine of object of type {type(obj).__name__} is Na')

        return MoleculeArray(non_missing)

    @property
    def shape(self) -> Shape:
        return self.mols.shape

    @property
    def na_value(self):
        return None

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:
        """
        Take elements from the array
        :param indices:
        :param allow_fill:
        :param fill_value:
        :return:
        """
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = pandas_take(np.array(self.mols), indices, allow_fill=allow_fill, fill_value=fill_value)
        return self._from_sequence(result)

    # --------------------------------------------------------
    # I/O
    # --------------------------------------------------------

    @classmethod
    def read_sdf(
            cls,
            fp,
            flavor=None,
            astype: type[oechem.OEMolBase] = oechem.OEMol,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default"
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
        :param astype: Type of molecule to read
        :param conformer_test: Conformer testing (will override astype to oechem.OEMol)
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecule_file(
                fp,
                oechem.OEFormat_SDF,
                flavor=flavor,
                astype=astype,
                conformer_test=conformer_test
            )
        )

    @classmethod
    def read_oeb(
            cls,
            fp: FilePath,
            flavor=None,
            astype: type[oechem.OEMolBase] = oechem.OEMol,
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default"
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
        :param astype: Type of molecule to read
        :param conformer_test: Conformer testing (will override astype to oechem.OEMol)
        :return: Molecule array populated by the molecules in the file
        """
        return cls(
            _read_molecule_file(
                fp,
                oechem.OEFormat_OEB,
                flavor=flavor,
                astype=astype,
                conformer_test=conformer_test
            )
        )

    @classmethod
    def read_smi(cls, fp, flavor=None, astype: type[oechem.OEMolBase] = oechem.OEMol) -> Self:
        """
        Read molecules from an SMILES file and return an array
        :param fp: Path to the SMILES file
        :param flavor: OpenEye input flavor
        :param astype: Type of molecule to read
        :return: Molecule array populated by the molecules in the file
        """
        return cls(_read_molecule_file(fp, oechem.OEFormat_SMI, flavor=flavor, astype=astype))

    def tolist(self, copy=False) -> list[oechem.OEMolBase]:
        """
        Convert to a list
        :param copy: Whether to copy the molecules or return pointers
        :return: List of molecules
        """
        if copy:
            return [mol.CreateCopy() for mol in self.mols]
        return self.mols.tolist()

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """
        Concatenate objects with the same datatype
        :param to_concat: Objects to concatenate
        :return: Concatenated object
        """
        # NOTE: tuple(...) was very important below to prevent a really strange concatenation error
        return MoleculeArray(np.concatenate(tuple(arr.mols for arr in to_concat)))

    @classmethod
    def _from_factorized(cls, values, original):
        """
        NOT IMPLEMENTED: Reconstruct an MoleculeArray after factorization
        """
        raise NotImplemented("Factorization not implemented for MoleculeArray")

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        """
        Formatter to used
        :param boxed: Whether this object is boxed in a series
        :return: Formatter to use for rendering this object
        """
        return str

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def copy(self):
        """
        Create a shallow copy of the array (molecules objects are not copied)
        :return: Shallow copy of the array
        :rtype: list[oechem.OEMolBase]
        """
        return MoleculeArray(shallow_copy(self.mols))

    def deepcopy(self):
        """
        Create a deep copy of the array (molecule objects are copied)
        :return: Deep copy of the array
        :rtype: list[oechem.OEMolBase]
        """
        return MoleculeArray([mol.CreateCopy() for mol in self.mols])

    def isna(self):
        """
        Return a boolean array of whether elements in the array are None
        :return: Boolean array
        """
        return np.array([mol is None for mol in self.mols])

    def valid(self) -> np.ndarray:
        """
        Return a boolean array of whether molecules are valid or invalid
        :return: Boolean array
        """
        return np.array([mol.IsValid() for mol in self.mols], dtype=bool)

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
        for mol in self.mols:
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
        for mol in self.mols:
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
        for mol in self.mols:
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
        for mol in self.mols:
            if mol is not None and mol.IsValid():
                smiles.append(oechem.OECreateSmiString(mol, flavor))
            else:
                smiles.append('')
        return np.array(smiles, dtype=str)

    # --------------------------------------------------------
    # Operators
    # --------------------------------------------------------

    def append(self, mol: oechem.OEMolBase):
        """
        Append a molecule to the array
        :param mol: Molecule to append
        """
        if not isinstance(mol, oechem.OEMolBase):
            raise TypeError("Can only append oechem.OEMolBase types to a MoleculeArray")

        # noinspection PyTypeChecker
        self.mols = np.append(self.mols, mol, axis=None)

    def extend(self, mols: Iterable[oechem.OEMolBase]):
        """
        Extend the molecule array
        :param mols: Molecules to extend array with
        :type mols: list[oechem.OEMolBase
        """
        if not all(isinstance(mol, oechem.OEMolBase) for mol in mols):
            raise TypeError("Can only extend a MoleculeArray with a list of oechem.OEMolBase objects")

        self.mols = np.concatenate((self.mols, mols), axis=0)

    @property
    def nbytes(self) -> int:
        """
        Number of bytes in this object
        Note: This is nowhere near accurate because sys.getsizeof does not return the correct size of the
              underlying OpenEye molecule objects.
        :return: Size of the contents of this array
        """
        return sum(sys.getsizeof(mol) for mol in self.mols)

    def equals(self, other: object) -> bool:
        """
        Test equality with other objects
        :param other: Another MoleculeArray or list of molecules
        :return: True if the objects are equal
        """
        if isinstance(other, MoleculeArray):
            return self.mols == other.mols
        elif isinstance(other, list):
            return self.mols == other
        else:
            raise TypeError(f'Cannot compare equality between MoleculeArray and {type(other).__name__}')

    @unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other):
        return self.equals(other)

    @unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other: 'MoleculeArray') -> bool:
        return self.mols != other.mols

    @unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other: 'MoleculeArray') -> bool:
        return len(self.mols) < len(other.mols)

    @unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other) -> bool:
        return len(self.mols) > len(other.mols)

    @unpack_zerodim_and_defer('__le__')
    def __le__(self, other: 'MoleculeArray') -> bool:
        return len(self.mols) <= len(other.mols)

    @unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other: 'MoleculeArray') -> bool:
        return len(self.mols) >= len(other.mols)

    @pd.core.ops.unpack_zerodim_and_defer('__add__')
    def __add__(self, other: Sequence[Self]) -> 'MoleculeArray':
        if isinstance(other, pd.DataFrame):
            raise NotImplemented("Adding MoleculeArray and pandas DataFrame is not implemented")
        elif isinstance(other, pd.Series):
            return MoleculeArray(np.concatenate((self.mols, other.array)))
        elif isinstance(other, MoleculeArray):
            return MoleculeArray(np.concatenate((self.mols, other.mols)))
        elif isinstance(other, Iterable):
            return MoleculeArray(np.concatenate((self.mols, other)))
        else:
            raise TypeError(f'Do not know how to add MoleculeArray and {type(other).__name__}')

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other: 'MoleculeArray') -> 'MoleculeArray':
        raise NotImplemented("Subtraction not implemented for MoleculeArray")

    def __reversed__(self):
        return MoleculeArray(reversed(self.mols))

    def __iter__(self):
        return iter(self.mols)

    def __len__(self):
        """
        Get the number of molecules in this array
        :return: Number of molecules in array
        :rtype: int
        """
        return self.mols.size

    # noinspection PyDefaultArgument
    def __deepcopy__(self, memodict={}):
        return self.deepcopy()

    def __copy__(self):
        return self.copy()

    def __hash__(self):
        return hash(self.mols)

    def __setitem__(self, index, value):
        """
        Set an item in the array
        :param index: Item index
        :type index: int
        :param value: Item to set
        :type value: oechem.OEMolBase
        :return:
        """
        self.mols[index] = value

    def __getitem__(self, index):
        """
        Get an item in the array
        :param index: Item index
        :return: Item at index
        :rtype: oechem.OEMolBase
        """
        if isinstance(index, int):
            return self.mols[index]
        return MoleculeArray(self.mols[index])

    def __str__(self):
        return f'<MoleculeArray size={len(self)}>'

    def __repr__(self):
        return self.__str__()


@register_extension_dtype
class MoleculeDtype(PandasExtensionDtype):
    """
    OpenEye molecule datatype for Pandas
    """

    type = oechem.OEMol
    name: str = "molecule"
    kind: str = "O"
    base = np.dtype("O")
    isbuiltin = 0
    isnative = 0

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return MoleculeArray

    # Required
    def __hash__(self) -> int:
        return hash(str(self))

    # Required
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, type(self))

    # Required
    def __str__(self):
        return self.name

    # Required
    def __repr__(self):
        return self.__str__()


########################################################################################################################
# Pandas: Global Readers
########################################################################################################################

class Column:
    """
    Column to create a Pandas DataFrame

    The index attribute
    """
    def __init__(
            self,
            name: str,
            data: Iterable[Any] = None,
            dtype: Dtype = object,
            index: list[Any] | None = None
    ):
        self.name = name
        self.data = data or []
        self.dtype = dtype
        self.index = index or []

    def to_series_tuple(self) -> tuple[str, pd.Series]:
        """
        Convert to a tuple with the name and Pandas series
        :return: Name, series tuple
        """
        # If we did not track custom indexes
        if self.index is None or len(self.index) == 0:
            return self.name, pd.Series(self.data, dtype=self.dtype)

        if len(self.index) != len(self.data):
            raise ValueError(
                "Number of data values ({}) != number of index values ({}) for column {}".format(
                    len(self.data),
                    len(self.index),
                    self.name
                )
            )

        return self.name, pd.Series(self.data, index=self.index, dtype=self.dtype)


def _add_smiles_columns(
        df: pd.DataFrame,
        molecule_columns: str | Iterable[str] | dict[str, int] | dict[str, str],
        add_smiles: bool | str | Iterable[str] | dict[str, str]):
    """
    Helper function to add SMILES column(s) to a DataFrame.

    The add_smiles parameter can be a number of things:
    - bool => Add SMILES for the first molecule column
    - str => Add SMILES for a specific molecule column
    - Iterable[str] => Add SMILES for one or more molecule columns
    - dict[str, str] => Add SMILES for one or more molecule columns with custom column names

    :param df: Dataframe to add SMILES columns to
    :param molecule_columns: Column(s) that the user requested
    :param add_smiles: Column definition(s) for adding SMILES
    """
    # Map of molecule column -> SMILES column
    add_smiles_map = {}

    if not isinstance(add_smiles, dict):

        if isinstance(add_smiles, bool) and add_smiles:
            # We only add SMILES to the first molecule column with the suffix " SMILES"
            col = molecule_columns if isinstance(molecule_columns, str) else next(iter(molecule_columns))
            add_smiles_map[col] = f'{col} SMILES'

        elif isinstance(add_smiles, str):
            if add_smiles in df.columns:
                if isinstance(df.dtypes[add_smiles], MoleculeDtype):
                    add_smiles_map[add_smiles] = f'{add_smiles} SMILES'
                else:
                    log.warning(f'Column {add_smiles} is not a MoleculeDtype')
            else:
                log.warning(f'Column {add_smiles} not found in DataFrame')

        elif isinstance(add_smiles, Iterable):
            for col in add_smiles:
                if col in df.columns:
                    if isinstance(df.dtypes[col], MoleculeDtype):
                        add_smiles_map[col] = f'{col} SMILES'
                    else:
                        log.warning(f'Column {col} is not a MoleculeDtype')
                else:
                    log.warning(f'Column {col} not found in DataFrame')

    # isinstance(add_smiles, dict)
    else:
        for col, smiles_col in add_smiles.items():
            if col in df.columns:
                if isinstance(df.dtypes[col], MoleculeDtype):
                    add_smiles_map[col] = smiles_col
                else:
                    log.warning(f'Column {col} is not a MoleculeDtype')
            else:
                log.warning(f'Column {col} not found in DataFrame')

    # Add the SMILES column(s)
    # We always add canonical isomeric SMILES
    for col, smiles_col in add_smiles_map.items():
        df[smiles_col] = df[col].to_smiles(flavor=oechem.OESMILESFlag_ISOMERIC)


# Types
class MoleculeArrayReaderProtocol(Protocol):
    def __call__(
            self,
            fp: FilePath,
            flavor: int | None,
            astype: type[oechem.OEMolBase],
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"]
    ) -> MoleculeArray: ...


def _read_file_with_data(
    reader: MoleculeArrayReaderProtocol,
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    flavor: int | None = oechem.OEIFlavor_SDF_Default,
    molecule_column_name: str = "Molecule",
    title_column_name: str | None = "Title",
    add_smiles: None | bool | str | Iterable[str] = None,
    molecule_columns: None | str | Iterable[str] = None,
    read_generic_data=True,
    read_sd_data=True,
    usecols: None | str | Iterable[str] = None,
    numeric: None | str | dict[str, Literal["integer", "signed", "unsigned", "float"] | None] | Iterable[str] = None,
    conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
    combine_tags: Literal["prefix", "prefer_sd", "prefer_generic"] = "prefix",
    sd_prefix: str = "SD Tag: ",
    generic_prefix: str = "Generic Tag: ",
    astype=oechem.OEGraphMol
) -> pd.DataFrame:
    """
    Read a molecule file with SD data and/or generic data
    :param reader: MoleculeArray method to use to read molecule
    :param filepath_or_buffer: File path or buffer
    :param flavor: SMILES flavor (part of oechem.OEIFlavor namespace)
    :param molecule_column_name: Name of the molecule column in the dataframe
    :param title_column_name: Name of the column with molecule titles in the dataframe
    :param add_smiles: Include a SMILES column in the dataframe (SMILES will be re-canonicalized)
    :param molecule_columns: Additional columns to convert to molecules
    :param read_generic_data: If True, read generic data (default is True)
    :param read_sd_data: If True, read SD data (default is True)
    :param usecols: List of data tags to read (default is all read all generic and SD data)
    :param numeric: Data column(s) to make numeric
    :param conformer_test: Combine single conformer molecules into multiconformer
    :param combine_tags: Strategy for combining identical SD and generic data tags
    :param sd_prefix: Prefix for SD data with corresponding generic data columns (for combine_tags='prefix')
    :param generic_prefix: Prefix for generic data with corresponding SD data columns (for combine_tags='prefix')
    :param astype: Type of OpenEye molecule to read
    :return:
    """
    # Read the molecules themselves
    if not isinstance(filepath_or_buffer, (str, Path)):
        raise NotImplementedError("Reading from buffers is not yet supported for raed_oeb")

    # --------------------------------------------------
    # Preprocess usecols and numeric
    # If we have a subset of columns that we are reading
    # or are converting columns to numeric types
    # --------------------------------------------------
    if usecols is not None:
        usecols = frozenset((usecols,)) if isinstance(usecols, str) else frozenset(usecols)

    # Make sure numeric is a dict of columns and type strings (None = let Pandas figure it out
    if numeric is not None:
        if isinstance(numeric, str):
            numeric = {numeric: None}
        elif isinstance(numeric, Iterable) and not isinstance(numeric, dict):
            numeric = {col: None for col in numeric}

        # Sanity check the molecule column
        if molecule_column_name in numeric:
            raise KeyError(f'Cannot make molecule column {molecule_column_name} numeric')

    # Read the molecules into a molecule array
    mols = reader(filepath_or_buffer, flavor=flavor, astype=astype, conformer_test=conformer_test)

    # --------------------------------------------------
    # Start building our DataFrame with the molecules
    # --------------------------------------------------
    data = {molecule_column_name: Column(molecule_column_name, mols, dtype=MoleculeDtype())}

    # Titles
    if title_column_name is not None:
        data[title_column_name] = Column(
            title_column_name,
            [mol.GetTitle() if isinstance(mol, oechem.OEMolBase) else None for mol in mols],
            dtype=str
        )

    # ----------------------------------------------------------------------
    # Index the data tags and create the columns
    # We need to track both generic data and SD data separately in case
    # we have identical tags. We will apply the combine_tags rule in order
    # to differentiate between identical tags at the end.
    # ----------------------------------------------------------------------

    sd_data = {}
    generic_data = {}

    for mol in mols:
        if read_sd_data:
            for dp in oechem.OEGetSDDataPairs(mol):
                if (usecols is None or dp.GetTag() in usecols) and (dp.GetTag() not in sd_data):
                    # SD data is always type object, which we'll use until string support is no longer experimental
                    # in Pandas
                    sd_data[dp.GetTag()] = Column(dp.GetTag(), dtype=object)

            if read_generic_data:
                for diter in mol.GetDataIter():
                    tag = oechem.OEGetTag(diter.GetTag())
                    if (usecols is None or tag in usecols) and (tag not in generic_data) and (tag != "SDTagData"):
                        try:
                            value = diter.GetData()

                            # Can value ever be None/null in the toolkits?
                            if value is not None:
                                generic_data[tag] = Column(tag, dtype=type(value))

                        except Exception as ex:  # noqa
                            continue

    # ----------------------------------------------------------------------
    # Get the data off the molecules
    # ----------------------------------------------------------------------

    for idx, mol in enumerate(mols):

        # Differentiate between SD data and generic data
        sd_data_found = {}
        generic_data_found = {}

        if read_sd_data:
            for dp in oechem.OEGetSDDataPairs(mol):

                # Only read SD data indexed above
                if dp.GetTag() in sd_data:
                    sd_data_found[dp.GetTag()] = dp.GetValue()

            if read_generic_data:
                for diter in mol.GetDataIter():

                    # Only read generic data indexed above
                    tag = oechem.OEGetTag(diter.GetTag())
                    if tag in generic_data:
                        try:
                            val = diter.GetData()
                            generic_data_found[tag] = val
                        except:  # noqa
                            # Use NaN for floats
                            if generic_data[tag].dtype == float or np.issubdtype(generic_data[tag].dtype, np.floating):
                                generic_data_found[tag] = np.nan
                            # TODO: Customize integer NaN value
                            elif generic_data[tag].dtype == int or np.issubdtype(generic_data[tag].dtype, np.integer):
                                generic_data_found[tag] = -1
                            elif generic_data[tag].dtype == str or np.issubdtype(generic_data[tag].dtype, np.str_):
                                generic_data[tag] = ''
                            # And None for everything else
                            else:
                                generic_data_found[tag] = None

        # Add all the found SD data
        for k, v in sd_data_found.items():
            sd_data[k].data.append(v)
            sd_data[k].index.append(idx)

        # Add all the found generic data
        for k, v in generic_data_found.items():
            generic_data[k].data.append(v)
            generic_data[k].index.append(idx)

    # Resolve overlapping column names between SD data and generic data
    for col in set(sd_data.keys()).intersection(set(generic_data.keys())):
        if combine_tags == "prefix":

            new_sd_name = f'{sd_prefix}{col}'
            new_generic_name = f'{generic_prefix}{col}'

            sd_data[new_sd_name] = sd_data.pop(col)
            sd_data[new_sd_name].name = new_sd_name

            generic_data[new_generic_name] = generic_data.pop(col)
            generic_data[new_generic_name].name = new_generic_name

        elif combine_tags == "prefer_sd":
            del generic_data[col]

        elif combine_tags == "prefer_generic":
            del sd_data[col]

        else:
            raise KeyError(f'Unknown combine_tags strategy: {combine_tags}')

    # Combine the tags:
    data = {
        **data,
        **sd_data,
        **generic_data
    }

    # Create the DataFrame
    df = pd.DataFrame(dict(col.to_series_tuple() for col in data.values()))

    # Post-process the dataframe only if we have data
    if len(df) > 0:

        if molecule_columns is not None:
            if isinstance(molecule_columns, str) and molecule_columns != molecule_column_name:
                if molecule_columns in df.columns:
                    df.as_molecule(molecule_column_name, inplace=True)
                else:
                    log.warning(f'Column not found in DataFrame: {molecule_columns}')

            elif isinstance(molecule_columns, Iterable):
                for col in molecule_columns:

                    # Check if we have been asked to make this column numeric later
                    if col in numeric:
                        raise KeyError(f'Cannot make molecule column {col} numeric')

                    if col in df.columns:
                        if col != molecule_column_name:
                            df.as_molecule(col, inplace=True)
                    else:
                        log.warning(f'Column not found in DataFrame: {col}')

        # Cast numeric columns
        if numeric is not None:
            for col, dtype in numeric.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="ignore", downcast=dtype)
                else:
                    log.warning(f'Column not found in DataFrame: {col}')

        # Add SMILES column(s)
        if add_smiles is not None:

            if molecule_columns is None:
                molecule_columns = [molecule_column_name]

            _add_smiles_columns(df, molecule_columns, add_smiles)

    return df


def read_molecule_csv(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    molecule_columns: str | dict[str, int] | dict[str, str],
    *,
    add_smiles: None | bool | str | Iterable[str] = None,
    astype=oechem.OEGraphMol,
    # pd.read_csv options here for type completion
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: list[HashableT] | Callable[[Hashable], bool] | None = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool = False,
    date_parser: Callable | lib.NoDefault = lib.no_default,  # noqa
    date_format: str | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool = False,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: Literal["high", "legacy"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default
) -> pd.DataFrame:
    """
    Read a delimited text file with molecules. Note that this wraps the standard Pandas CSV reader and then converts
    the molecule column(s).
    """
    # Deletegate the CSV reading to pandas
    # noinspection PyTypeChecker
    df: pd.DataFrame = pd.read_csv(
        filepath_or_buffer,
        sep=sep,
        delimiter=delimiter,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        engine=engine,
        converters=converters,
        true_values=true_values,
        false_values=false_values,
        skipinitialspace=skipinitialspace,
        skiprows=skiprows,
        skipfooter=skipfooter,
        nrows=nrows,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        verbose=verbose,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        infer_datetime_format=infer_datetime_format,
        keep_date_col=keep_date_col,
        date_parser=date_parser,
        date_format=date_format,
        dayfirst=dayfirst,
        cache_dates=cache_dates,
        iterator=iterator,
        chunksize=chunksize,
        compression=compression,
        thousands=thousands,
        decimal=decimal,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        escapechar=escapechar,
        comment=comment,
        encoding=encoding,
        encoding_errors=encoding_errors,
        dialect=dialect,
        on_bad_lines=on_bad_lines,
        delim_whitespace=delim_whitespace,
        low_memory=low_memory,
        memory_map=memory_map,
        float_precision=float_precision,
        storage_options=storage_options,
        dtype_backend=dtype_backend
    )

    # Convert molecule columns if we have data
    if len(df) > 0:
        df.as_molecule(molecule_columns, astype=astype, inplace=True)

    # Process 'add_smiles' by first standardizing it to a dictionary
    if add_smiles is not None:
        _add_smiles_columns(df, molecule_columns, add_smiles)

    return df


def read_smi(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    cx: bool = False,
    flavor: int | None = None,
    add_smiles: bool = False,
    add_inchi_key: bool = False,
    molecule_column_name: str = "Molecule",
    title_column_name: str = "Title",
    smiles_column_name: str = "SMILES",
    inchi_key_column_name: str = "InChI Key",
    astype=oechem.OEGraphMol
) -> pd.DataFrame:
    """
    Read structures from a SMILES file to a dataframe
    :param filepath_or_buffer: File path or buffer
    :param cx: Read CX SMILES format (versus default SMILES)
    :param flavor: SMILES flavor (part of oechem.OEIFlavor namespace)
    :param add_smiles: Include a SMILES column in the dataframe (SMILES will be re-canonicalized)
    :param add_inchi_key: Include an InChI key column in the dataframe
    :param molecule_column_name: Name of the molecule column in the dataframe
    :param title_column_name: Name of the column with molecule titles in the dataframe
    :param smiles_column_name: Name of the SMILES column (if smiles is True)
    :param inchi_key_column_name: Name of the InChI key column (if inchi_key is True)
    :param astype: Type of OpenEye molecule to read
    :return: Dataframe with molecules
    """
    if not isinstance(filepath_or_buffer, (Path, str)):
        raise NotImplemented("Only reading from molecule paths is implemented")

    data = []

    # Configure the column headers and order
    columns = [title_column_name, molecule_column_name]

    if add_smiles:
        columns.append(smiles_column_name)

    if add_inchi_key:
        columns.append(inchi_key_column_name)

    # -----------------------------------
    # Read a file
    # -----------------------------------

    fp = Path(filepath_or_buffer)

    if not fp.exists():
        raise FileNotFoundError(f'File does not exist: {fp}')

    for mol in _read_molecule_file(
            filepath_or_buffer,
            file_format=oechem.OEFormat_CXSMILES if cx else oechem.OEFormat_SMI,
            flavor=flavor,
            astype=astype
    ):
        row_data = {title_column_name: mol.GetTitle(), molecule_column_name: mol.CreateCopy()}

        # If adding smiles
        if add_smiles:
            row_data[smiles_column_name] = oechem.OEMolToSmiles(mol)

        # If adding InChI keys
        if add_inchi_key:
            row_data[inchi_key_column_name] = oechem.OEMolToInChIKey(mol)

        data.append(row_data)

    df = pd.DataFrame(data, columns=columns)

    # Convert only if the dataframe is not empty
    if len(df) > 0:
        df.as_molecule(molecule_column_name, inplace=True)

    return df


def read_sdf(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    flavor: int | None = oechem.OEIFlavor_SDF_Default,
    molecule_column_name: str = "Molecule",
    title_column_name: str | None = "Title",
    add_smiles: None | bool | str | Iterable[str] = None,
    molecule_columns: None | str | Iterable[str] = None,
    usecols: None | str | Iterable[str] = None,
    numeric: None | str | dict[str, Literal["integer", "signed", "unsigned", "float"] | None] | Iterable[str] = None,
    conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
    sd_data: bool = True,
    astype=oechem.OEGraphMol
) -> pd.DataFrame:
    """
    Read structures from an SD file into a DataFrame.

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

    Use numeric to cast data columns to numeric types, which is specifically useful for SD data, which is always stored
    as a string:
        1. Single column name (default numeric cast)
        2. List of column names (default numeric cast)
        3. Dictionary of column names and specific numeric types to downcast to

    :param filepath_or_buffer: File path or buffer
    :param flavor: SMILES flavor (part of oechem.OEIFlavor namespace)
    :param add_smiles: Include a SMILES column in the dataframe (SMILES will be re-canonicalized)
    :param molecule_column_name: Name of the molecule column in the dataframe
    :param molecule_columns: Additional columns to convert to molecules
    :param title_column_name: Name of the column with molecule titles in the dataframe
    :param usecols: List of SD tags to read (default is all SD data is read)
    :param numeric: Data column(s) to make numeric
    :param conformer_test: Combine single conformer molecules into multiconformer
    :param sd_data: Read SD data
    :param astype: Type of OpenEye molecule to read
    :return: Pandas DataFrame
    """
    return _read_file_with_data(
        MoleculeArray.read_sdf,
        filepath_or_buffer,
        flavor=flavor,
        molecule_column_name=molecule_column_name,
        title_column_name=title_column_name,
        add_smiles=add_smiles,
        molecule_columns=molecule_columns,
        read_generic_data=False,
        read_sd_data=sd_data,
        usecols=usecols,
        numeric=numeric,
        conformer_test=conformer_test,
        astype=astype
    )


def read_oeb(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    flavor: int | None = oechem.OEIFlavor_SDF_Default,
    molecule_column_name: str = "Molecule",
    title_column_name: str | None = "Title",
    add_smiles: None | bool | str | Iterable[str] = None,
    molecule_columns: None | str | Iterable[str] = None,
    read_generic_data: bool = True,
    read_sd_data: bool = True,
    usecols: None | str | Iterable[str] = None,
    numeric: None | str | dict[str, Literal["integer", "signed", "unsigned", "float"] | None] | Iterable[str] = None,
    conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
    combine_tags: Literal["prefix", "prefer_sd", "prefer_generic"] = "prefix",
    sd_prefix: str = "SD Tag: ",
    generic_prefix: str = "Generic Tag: ",
    astype=oechem.OEGraphMol
) -> pd.DataFrame:
    """
    Read structures OpenEye binary molecule files.

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

    Use numeric to cast data columns to numeric types, which is specifically useful for SD data, which is always stored
    as a string:
        1. Single column name (default numeric cast)
        2. List of column names (default numeric cast)
        3. Dictionary of column names and specific numeric types to downcast to

    :param filepath_or_buffer: File path or buffer
    :param flavor: SMILES flavor (part of oechem.OEIFlavor namespace)
    :param add_smiles: Include a SMILES column in the dataframe (SMILES will be re-canonicalized)
    :param molecule_column_name: Name of the molecule column in the dataframe
    :param molecule_columns: Additional columns to convert to molecules
    :param title_column_name: Name of the column with molecule titles in the dataframe
    :param read_generic_data: If True, read generic data (default is True)
    :param read_sd_data: If True, read SD data (default is True)
    :param usecols: List of data tags to read (default is all read all generic and SD data)
    :param numeric: Data column(s) to make numeric
    :param conformer_test: Combine single conformer molecules into multiconformer
    :param combine_tags: Strategy for combining identical SD and generic data tags
    :param sd_prefix: Prefix for SD data with corresponding generic data columns (for combine_tags='prefix')
    :param generic_prefix: Prefix for generic data with corresponding SD data columns (for combine_tags='prefix')
    :param astype: Type of OpenEye molecule to read
    :return: Pandas DataFrame
    """
    return _read_file_with_data(
        MoleculeArray.read_oeb,
        filepath_or_buffer,
        flavor=flavor,
        molecule_column_name=molecule_column_name,
        title_column_name=title_column_name,
        add_smiles=add_smiles,
        molecule_columns=molecule_columns,
        read_generic_data=read_generic_data,
        read_sd_data=read_sd_data,
        usecols=usecols,
        numeric=numeric,
        conformer_test=conformer_test,
        combine_tags=combine_tags,
        sd_prefix=sd_prefix,
        generic_prefix=generic_prefix,
        astype=astype
    )


def read_oedb(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    usecols: None | str | Iterable[str] = None,
    int_na: int | None = None,
) -> pd.DataFrame:
    data = {}

    # Open the record file
    ifs = oechem.oeifstream()
    if not ifs.open(str(filepath_or_buffer)):
        raise FileError(f'Could not open file for reading: {filepath_or_buffer}')

    # Read the records
    for idx, record in enumerate(oechem.OEReadRecords(ifs)):  # type: oechem.OERecord
        for field in record.get_fields():  # type: oechem.OEFieldBase
            name = field.get_name()
            # noinspection PyUnresolvedReferences
            dtype = field.get_type()

            # Skip columns we weren't asked to read (if provided)
            if usecols is not None and name not in usecols:
                continue

            # ------------------------------
            # Float field
            # ------------------------------
            if dtype == oechem.Types.Float:
                if name not in data:
                    data[name] = Column(name, dtype=float)

                val = record.get_value(field)
                data[name].data.append(val if isinstance(val, float) else np.nan)
                data[name].index.append(idx)

            # ------------------------------
            # Integer field
            # ------------------------------
            elif dtype == oechem.Types.Int:
                if name not in data:
                    data[name] = Column(name, dtype=int)

                val = record.get_value(field)
                if not isinstance(val, int):
                    if int_na is None:
                        data[name].dtype = object
                        data[name].data.append(None)
                    else:
                        data[name].data.append(int_na)
                else:
                    data[name].data.append(val)

                data[name].index.append(idx)

            # ------------------------------
            # Boolean field
            # ------------------------------
            elif dtype == oechem.Types.Bool:
                if name not in data:
                    data[name] = Column(name, dtype=bool)

                val = record.get_value(field)
                if not isinstance(val, bool):
                    data[name].dtype = object
                    data[name].data.append(None)
                else:
                    data[name].data.append(val)

                data[name].index.append(idx)

            # ------------------------------
            # Molecule field
            # ------------------------------
            elif dtype == oechem.Types.Chem.Mol:
                if name not in data:
                    data[name] = Column(name, dtype=MoleculeDtype())

                val = record.get_value(field)
                data[name].data.append(val if isinstance(val, oechem.OEMolBase) else None)
                data[name].index.append(idx)

            # ------------------------------
            # Everything else
            # ------------------------------
            else:
                if name not in data:
                    data[name] = Column(name, dtype=object)

                val = record.get_value(field)
                data[name].data.append(val)
                data[name].index.append(idx)

    # Close the record file
    ifs.close()

    # Create the DataFrame
    df = pd.DataFrame(dict(col.to_series_tuple() for col in data.values()))
    return df


# ----------------------------------------------------------------------
# Monkey patch these into Pandas, which makes them discoverable to
# people looking there and not in the oepandas package
# ----------------------------------------------------------------------

pd.read_molecule_csv = read_molecule_csv
pd.read_smi = read_smi
pd.read_sdf = read_sdf
pd.read_oeb = read_oeb
pd.read_oedb = read_oedb


########################################################################################################################
# Pandas DataFrame: Writers
########################################################################################################################

@register_dataframe_accessor("to_sdf")
class WriteToSDFAccessor:
    """
    Write to SD file
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
            self,
            fp: FilePath,
            primary_molecule_column,
            *,
            title_column: str | None = None,
            columns: str | Iterable[str] | None = None,
            index: bool = True,
            index_tag: str = "index",
            secondary_molecules_as: int | str = "smiles",
            secondary_molecule_flavor: int | str | None = None
    ) -> None:
        """
        Write DataFrame to an SD file
        Note: Writing conformers not yet supported
        :param primary_molecule_column: Primary molecule column
        :param columns: Optional column(s) to include as SD tags
        :param index: Write index
        :param index_tag: SD tag for writing index
        :param secondary_molecules_as: Encoding for secondary molecules (default: SMILES)
        :return:
        """
        # Make sure we're working with a list of columns
        if columns is None:
            columns = list(self._obj.columns)
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)

        # Validate primary molecule column
        if primary_molecule_column not in self._obj.columns:
            raise KeyError(f'Primary molecule column {primary_molecule_column} not found in DataFrame')
        if not isinstance(self._obj[primary_molecule_column].dtype, MoleculeDtype):
            raise TypeError(f'Primary molecule column {primary_molecule_column} is not a MoleculeDtype')

        # Validate title column
        if title_column is not None and title_column not in self._obj.columns:
            raise KeyError(f'Title column {title_column} not found in DataFrame')

        # Set of secondary molecule columns
        secondary_molecule_cols = set()

        # Create the secondary molecule writer
        secondary_molecule_to_string = create_molecule_to_string_writer(
            fmt=secondary_molecules_as,
            flavor=secondary_molecule_flavor,
            gzip=False,
            b64encode=False,
            strip=True
        )

        for col in columns:
            if col not in self._obj.columns:
                raise KeyError(f'Column {col} not found in DataFrame')

            if col != primary_molecule_column and isinstance(self._obj[col].dtype, MoleculeDtype):
                secondary_molecule_cols.add(col)

        # Process the molecules
        with oechem.oemolostream(str(fp)) as ofs:
            for idx, row in self._obj.iterrows():
                mol = row[primary_molecule_column].CreateCopy()

                # Set the title
                if title_column is not None:
                    mol.SetTitle(str(row[title_column]))

                for col in columns:

                    # Secondary molecule column
                    if col in secondary_molecule_cols:
                        oechem.OESetSDData(
                            mol,
                            col,
                            secondary_molecule_to_string(row[col])
                        )

                    # Everything else (except our primary molecule column)
                    elif col != primary_molecule_column:
                        oechem.OESetSDData(
                            mol,
                            col,
                            str(row[col])
                        )

                # Write out the molecule
                oechem.OEWriteMolecule(ofs, mol)


@register_dataframe_accessor("to_molecule_csv")
class WriteToMoleculeCSVAccessor:
    """
    Write to a CSV file containing molecules
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj.copy()

    def __call__(
            self,
            fp: FilePath,
            *,
            molecule_format: str | int = "smiles",
            flavor: int | None = None,
            gzip: bool = False,
            b64encode: bool = False,
            columns: str | Iterable[str] | None = None,
            index: bool = True,
            sep=',',
            na_rep='',
            float_format=None,
            header=True,
            encoding=None,
            lineterminator=None,
            date_format=None,
            quoting=None,
            quotechar='"',
            doublequote=True,
            escapechar=None,
            decimal='.',
            index_label: str = "index",
    ) -> None:
        """
        Write to a CSV file with molecules
        :param fp: File path
        :param molecule_format: Molecule file format
        :param columns: Columns to include in the output CSV
        :param index: Whether to write the index
        :param sep: String of length 1. Field delimiter for the output file
        :param na_rep: Missing data representation
        :param float_format: Format string for floating point numbers
        :param header: Write out the column names. A list of strings is assumed to be aliases for the column names.
        :param encoding: A string representing the encoding to use in the output file ('utf-8' is default)
        :param lineterminator: Newline character or character sequence to use. Default is system dependent.
        :param date_format: Format string for datetime objects.
        :param quoting: Defaults to csv.QUOTE_MINIMAL
        :param quotechar: String of length 1. Character used to quote fields.
        :param doublequote: Control quoting of quotechar inside a field.
        :param escapechar: String of length 1. Character used to escape sep and quotechar when appropriate.
        :param decimal: Character recognized as decimal separator. E.g., use ‘,’ for European data.
        :param index_label: Column label to use for the index
        """
        # Convert all the molecule columns
        for col in self._obj.columns:
            if isinstance(self._obj.dtypes[col], MoleculeDtype):
                self._obj[col] = self._obj[col].array.to_molecule_strings(
                    molecule_format=molecule_format,
                    flavor=flavor,
                    gzip=gzip,
                    b64encode=b64encode
                )

        # Write to CSV
        self._obj.to_csv(
            fp,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            lineterminator=lineterminator,
            encoding=encoding,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal
        )

# def to_sdf(
#         self: pd.DataFrame,
#         x):
#     print(self, x)
#
#
# def to_oeb(
#         self: pd.DataFrame,
#         x):
#     print(self, x)
#
#
# def to_oez(
#         self: pd.DataFrame,
#         x):
#     print(self, x)
#
#
# def to_oecsv(
#         sefl: pd.DataFrame,
#         x):
#     pass
#
#
# pd.DataFrame.to_sdf = to_sdf
# pd.DataFrame.to_oeb = to_oeb
# pd.DataFrame.to_oez = to_oez
# pd.DataFrame.to_oecsv = to_oecsv


########################################################################################################################
# Pandas DataFrame: Utilities
########################################################################################################################

@register_dataframe_accessor("as_molecule")
class DataFrameAsMoleculeAccessor:
    """
    Accessor that adds the as_molecule method to a Pandas DataFrame
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
            self,
            columns: str | Iterable[str],
            *,
            fmt: dict[str, str] | dict[str, int] | str | int = oechem.OEFormat_SMI,
            astype=oechem.OEGraphMol,
            inplace=False
    ):
        # Make sure we're working with a list of columns
        columns = [columns] if isinstance(columns, str) else list(columns)

        # Validate column names
        for name in columns:
            if name not in self._obj.columns:
                raise KeyError(f'Column {name} not found in DataFrame: {", ".join(self._obj.columns)}')

        # Whether we are working inplace or on a copy
        df = self._obj if inplace else self._obj.copy()

        # Convert the columns
        for col in columns:

            # Get the underlying array
            arr = self._obj[col].array

            # Peek at the first non-null value and see if it looks like an OpenEye object
            _type = None
            for obj in arr:
                if not pd.isna(obj):
                    _type = type(obj)
                    break

            # If we have strings then use the optimized parser for sequences of strings
            if issubclass(_type, str):

                # File format of the column (as an OEFormat)
                if fmt is None:
                    col_fmt = oechem.OEFormat_SMI
                elif isinstance(fmt, dict):
                    col_fmt = get_oeformat(fmt.get(col, oechem.OEFormat_SMI))
                else:
                    col_fmt = get_oeformat(fmt)

                # noinspection PyProtectedMember
                mol_array = MoleculeArray._from_sequence_of_strings(arr, astype=astype, fmt=col_fmt.oeformat)

            # Otherwise use the more general sequence parser
            else:
                # noinspection PyProtectedMember
                mol_array = MoleculeArray._from_sequence(arr)

            # Replace the column
            df[col] = pd.Series(mol_array, index=self._obj.index)

        return df


@register_dataframe_accessor("filter_invalid_molecules")
class FilterInvalidMoleculesAccessor:
    """
    Filter invalid molecules in one or more columns
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
            self,
            columns: str | Iterable[str],
            *,
            inplace=False
    ):

        # Make sure we're working with a list of columns
        columns = [columns] if isinstance(columns, str) else list(columns)

        # Validate column names
        for name in columns:
            if name not in self._obj.columns:
                raise KeyError(f'Column {name} not found in DataFrame: {", ".join(self._obj.columns)}')

        # Compute a bitmask of rows that we want to keep over all the columns
        mask = np.array([True] * len(self._obj))
        for col in columns:
            mask &= self._obj[col].array.valid()

        if inplace:
            self._obj.drop(self._obj[~mask].index, inplace=True)
            return self._obj

        return self._obj.drop(self._obj[~mask].index)

########################################################################################################################
# Pandas Series: Utilities
########################################################################################################################


@register_series_accessor("as_molecule")
class SeriesAsMoleculeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
            self,
            *,
            fmt: str | int = oechem.OEFormat_SMI,
            astype=oechem.OEGraphMol):
        """
        Convert a series to molecules
        :param fmt: File format of column to convert to molecules (extension or from oechem.OEFormat namespace)
        :param astype: oechem.OEGraphMol (default) or oechem.OEMol
        :return: Series as molecule
        """
        # Column OEFormat
        _fmt = get_oeformat(fmt)

        # noinspection PyProtectedMember
        arr = MoleculeArray._from_sequence_of_strings(self._obj, astype=astype, fmt=_fmt.oeformat)
        return pd.Series(arr, index=self._obj.index, dtype=MoleculeDtype())


@register_series_accessor("to_molecule_bytes")
class SeriesToMoleculeBytesAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, MoleculeDtype):
            raise TypeError(
                "to_molecule_bytes only works on molecule columns (oepandas.MoleculeDtype). If this column has "
                "molecules, use pd.Series.as_molecule to convert to a molecule column first."
            )

        self._obj = pandas_obj

    def __call__(
            self,
            *,
            molecule_format: str | int = oechem.OEFormat_SMI,
            flavor: int | None = None,
            gzip: bool = False,
    ) -> pd.Series:
        """
        Convert a series to molecule bytes
        :param molecule_format: Molecule format extension or oechem.OEFormat
        :param flavor: Flavor for generating SMILES (bitmask from oechem.OESMILESFlag)
        :param gzip: Gzip the molecule bytes
        :return: Series of molecules as SMILES
        """
        arr = self._obj.array.to_molecule_bytes(
            molecule_format=molecule_format,
            flavor=flavor,
            gzip=gzip
        )

        return pd.Series(arr, index=self._obj.index, dtype=object)


@register_series_accessor("to_molecule_strings")
class SeriesToMoleculeStringsAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, MoleculeDtype):
            raise TypeError(
                "to_molecule_string only works on molecule columns (oepandas.MoleculeDtype). If this column has "
                "molecules, use pd.Series.as_molecule to convert to a molecule column first."
            )

        self._obj = pandas_obj

    def __call__(
            self,
            *,
            molecule_format: str | int = "smiles",
            flavor: int | None = None,
            gzip: bool = False,
            b64encode: bool = False
    ) -> pd.Series:
        """
        Convert a series to molecule strings
        :param molecule_format: Molecule format extension or oechem.OEFormat
        :param flavor: Flavor for generating SMILES (bitmask from oechem.OESMILESFlag)
        :param gzip: Gzip the molecule strings (will be base64 encoded)
        :param b64encode: Force base64 encoding for all molecules
        :return: Series of molecules as SMILES
        """
        arr = self._obj.array.to_molecule_strings(
            molecule_format=molecule_format,
            flavor=flavor,
            gzip=gzip,
            b64encode=b64encode
        )

        return pd.Series(arr, index=self._obj.index, dtype=object)


@register_series_accessor("to_smiles")
class SeriesToSmilesAccessor:
    def __init__(self, pandas_obj: pd.Series):
        if not isinstance(pandas_obj.dtype, MoleculeDtype):
            raise TypeError(
                "to_smiles only works on molecule columns (oepandas.MoleculeDtype). If this column has "
                "molecules, use pd.Series.as_molecule to convert to a molecule column first."
            )

        self._obj = pandas_obj

    def __call__(
            self,
            *,
            flavor: int = oechem.OESMILESFlag_ISOMERIC,
            astype=oechem.OEGraphMol):
        """
        Convert a series to SMILES
        :param flavor: Flavor for generating SMILES (bitmask from oechem.OESMILESFlag)
        :return: Series of molecules as SMILES
        """
        # noinspection PyUnresolvedReferences
        arr = self._obj.array.to_smiles(flavor)
        return pd.Series(arr, index=self._obj.index, dtype=object)


@register_series_accessor("subsearch")
class SeriesSubsearchAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, MoleculeDtype):
            raise TypeError(
                "subsearch only works on molecule columns (oepandas.MoleculeDtype). If this column has "
                "molecules, use pd.Series.as_molecule to convert to a molecule column first."
            )

        self._obj = pandas_obj

    def __call__(
            self,
            pattern: str | oechem.OESubSearch,
            *,
            adjustH: bool = False  # noqa
    ):
        """
        Perform a substructure search
        :param pattern: SMARTS pattern or OESubSearch object
        :param adjustH: Adjust implicit / explicit hydrogens to match query
        :return: Series as molecule
        """
        return pd.Series(self._obj.array.subsearch(pattern, adjustH=adjustH), dtype=bool)
