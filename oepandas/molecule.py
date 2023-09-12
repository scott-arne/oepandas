import logging
import sys
import numpy as np
import pandas as pd
from typing import Callable, Literal, Any
from collections.abc import Iterable, Hashable, Sequence, Generator
from pandas.core.ops import unpack_zerodim_and_defer
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.core.algorithms import take as pandas_take
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    register_extension_dtype,
    register_dataframe_accessor,
    register_series_accessor
)
from openeye import oechem
from copy import copy as shallow_copy
from .util import get_oeformat
from .exception import UnsupportedFileFormat

if sys.version_info >= (3, 11):
    from typing import Self  # pyright: ignore[reportUnusedImport]
else:
    from typing_extensions import Self  # pyright: ignore[reportUnusedImport]

# noinspection PyProtectedMember
from pandas._typing import Shape, FilePath, IndexLabel, ReadBuffer, HashableT, TakeIndexer

log = logging.getLogger("oepandas")


class FileError(Exception):
    """
    File-related errors
    """
    pass


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
    def __init__(self, mols, copy=False):
        """
        Initialize
        :param mols: Sequence/array of molecules
        :type mols: Iterable[oechem.OEMolBase]
        :param copy: Create copy of the molecules if True
        """
        self.mols = np.array([mol.CreateCopy() if copy else mol for mol in mols])

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False) -> Self:
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param dtype: Coerce to this datatype (must be a subclass of oechem.OEMolBase)
        :param copy: Copy the molecules (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        return cls([mol.CreateCopy() if copy else mol for mol in scalars])

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            astype: type[oechem.OEMolBase] = oechem.OEGraphMol,
            copy: bool = False,
            fmt: int = oechem.OEFormat_SMI) -> Self:
        """
        Read molecules form a sequence of strings
        :param strings: Sequence of strings
        :param astype: Data type for molecules (must be oechem.OEMolBase)
        :param copy: Not used (here for API compatibility)
        :return:
        """
        if not issubclass(astype, oechem.OEMolBase):
            raise TypeError("Can only read molecules from string as an oechem.OEMolBase type")

        # Standardize the format
        fmt = get_oeformat(fmt)

        mols = []
        for i, s in enumerate(strings):
            mol = astype()

            if not oechem.OEReadMolFromString(mol, fmt, False, s.strip()):
                log.warning("Could not convert molecule %d from '%s': %s", i + 1, oechem.OEGetFormatString(fmt), s)

            mols.append(mol)

        return cls(mols)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return MoleculeDtype()

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

    @staticmethod
    def _read_molecule_file(
            fp: FilePath,
            file_format: int | str,
            *,
            flavor: int | None = None,
            astype: type[oechem.OEMolBase] =
            oechem.OEMol) -> Generator[oechem.OEMolBase, None, None]:
        """
        Generator over flavored reading of molecules in a specific file format
        :param fp: File path
        :param file_format: File format (oechem.OEFormat)
        :param flavor: Optional flavor (oechem.OEIFlavor)
        :param astype: OpenEye molecule type to read (oechem.OEMolBase or oechem.OEGraphMol)
        :return: Generator over molecules
        """
        with oechem.oemolistream(str(fp)) as ifs:
            ifs.SetFormat(get_oeformat(file_format))

            # Set flavor if requested
            if flavor is not None:
                ifs.SetFlavor(file_format, flavor)

            iterator = ifs.GetOEMols if astype is oechem.OEMol else ifs.GetOEGraphMols
            for mol in iterator():
                yield mol.CreateCopy()

    @classmethod
    def read_sdf(cls, fp, flavor=None, astype=oechem.OEMol) -> Self:
        """
        Read molecules from an SD file and return an array
        :param fp: Path to the SD file
        :param flavor: OpenEye input flavor
        :param astype: Type of molecule to read
        :return: Molecule array populated by the molecules in the file
        """
        return cls(cls._read_molecule_file(fp, oechem.OEFormat_SDF, flavor=flavor, astype=astype))

    @classmethod
    def read_oeb(cls, fp, flavor=None, astype=oechem.OEMol) -> Self:
        """
        Read molecules from an OEB file and return an array
        :param fp: Path to the OEB file
        :param flavor: OpenEye input flavor
        :param astype: Type of molecule to read
        :return: Molecule array populated by the molecules in the file
        """
        return cls(cls._read_molecule_file(fp, oechem.OEFormat_OEB, flavor=flavor, astype=astype))

    @classmethod
    def read_smi(cls, fp, flavor=None, astype=oechem.OEMol) -> Self:
        """
        Read molecules from an SMILES file and return an array
        :param fp: Path to the SMILES file
        :param flavor: OpenEye input flavor
        :param astype: Type of molecule to read
        :return: Molecule array populated by the molecules in the file
        """
        return cls(cls._read_molecule_file(fp, oechem.OEFormat_SMI, flavor=flavor, astype=astype))

    # TODO: from_oez?

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

    def valid(self):
        """
        Return a boolean array of whether molecules are valid or invalid
        :return: Boolean array
        """
        return np.array([mol.IsValid() for mol in self.mols])

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

def read_sdf(filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
             *,
             index_tag: IndexLabel | Literal[False] | None = None,
             usecols: list[HashableT] | Callable[[Hashable], bool] = None,
             skipmols: list[int] | int | Callable[[Hashable], bool] | None = None,
             conf_test: str | oechem.OEConfTestBase | None = None,
             astype=oechem.OEMol,
             expand_confs=False,
             include_first_conf_data=False,
             flavor: int | None = None,
             nmols: int | None = None) -> pd.DataFrame:
    """
    Read molecules from an SD file into a DataFrame

    The conf_test parameter can either be a string or one of the OpenEye conformer testing types. Refer to the following
    list to see the valid string values and the corresponding Openeye conformer testing type::

        - isomeric: oechem.OEIsomericConfTest()
        - absolute: oechem.OEAbsoluteConfTest()
        - omega: oechem.OEOmegaConfTest()

    :param filepath_or_buffer:
    :param index_tag:
    :param usecols:
    :param skipmols:
    :param flavor: Flavored input (OEIFlavor)
    :param nmols:
    :param conf_test: Optional testing strategy for conformers in the SD file
    :return:
    """
    # Parse the conf test
    if conf_test is not None:
        if isinstance(conf_test, str):
            if conf_test == "isomeric":
                conf_test = oechem.OEIsomericConfTest()
            elif conf_test == "absolute":
                conf_test = oechem.OEAbsoluteConfTest()
            elif conf_test == "omega":
                conf_test = oechem.OEOmegaConfTest()
            else:
                raise KeyError(f'Invalid OpenEye conformer test type: {conf_test} (valid: isomeric, absolute, omega)')

    if isinstance(filepath_or_buffer, FilePath):
        with oechem.oemolistream(str(filepath_or_buffer)) as ifs:

            # Check if SD file
            if not ifs.GetFormat() == oechem.OEFormat_SDF:
                raise FileError(f'{filepath_or_buffer} is not an SD file')

            # Flavored I/O
            if flavor is not None:
                ifs.SetFlavor(oechem.OEFormat_SDF, flavor)

            # Conformer testing
            if conf_test is not None:
                ifs.SetConfTest(conf_test)

            # Iterate molecules
            iterator = ifs.GetOEMols if astype is oechem.OEMol else ifs.GetOEGraphMols
            for mol in iterator():
                print(mol)

    print(filepath_or_buffer)


def read_oeb(
        filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        *,
        index_tag: IndexLabel | Literal[False] | None = None,
        usecols: list[HashableT] | Callable[[Hashable], bool] = None,
        skipmols: list[int] | int | Callable[[Hashable], bool] | None = None,
        astype=oechem.OEMol,
        expand_confs=False,
        include_first_conf_data=False,
        flavor: int | None = None,
        nmols: int | None = None,
        sd_data: bool = False) -> pd.DataFrame:
    """

    :param filepath_or_buffer:
    :param index_tag:
    :param usecols:
    :param skipmols:
    :param flavor: Flavored input (OEIFlavor)
    :type flavor: int or None
    :param nmols:
    :param sd_data:
    :return:
    """
    print(filepath_or_buffer)


def read_oez(
        filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        *,
        index_field: IndexLabel | Literal[False] | None = None,
        usecols: list[HashableT] | Callable[[Hashable], bool] = None,
        skipmols: list[int] | int | Callable[[Hashable], bool] | None = None,
        nmols: int | None = None) -> pd.DataFrame:
    """

    :param filepath_or_buffer:
    :param index_field:
    :param usecols:
    :param skipmols:
    :param nmols:
    :return:
    """
    print(filepath_or_buffer)


def read_oecsv(
        filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        *,
        index_col: IndexLabel | Literal[False] | None = None,
        usecols: list[HashableT] | Callable[[Hashable], bool] = None,
        skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
        nrows: int | None = None) -> pd.DataFrame:
    pass


# pd.read_sdf = read_sdf
# pd.read_oeb = read_oeb
# pd.read_oez = read_oez
# pd.read_oecsv = read_oecsv


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
            secondary_molecules_as: int | str = oechem.OEFormat_SMI
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

        for col in columns:
            if col not in self._obj.columns:
                raise KeyError(f'Column {col} not found in DataFrame')

            if col != primary_molecule_column and isinstance(self._obj[col].dtype, MoleculeDtype):
                secondary_molecule_cols.add(col)

        # Get the secondary molecule format
        secondary_fmt = secondary_molecules_as if isinstance(secondary_molecules_as, int) \
            else oechem.GetFileFormat(secondary_molecules_as)

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
                            oechem.OEWriteMolToBytes(secondary_fmt, False, row[col]).decode("utf-8")
                        )

                    # Everything else
                    else:
                        oechem.OESetSDData(
                            mol,
                            col,
                            str(row[col])
                        )

                # Write out the molecule
                oechem.OEWriteMolecule(ofs, mol)

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

            # Column OEFormat
            if fmt is None:
                col_fmt = oechem.OEFormat_SMI
            elif isinstance(fmt, dict):
                col_fmt = get_oeformat(fmt.get(col, oechem.OEFormat_SMI))
            else:
                col_fmt = get_oeformat(fmt)

            # noinspection PyProtectedMember
            array = MoleculeArray._from_sequence_of_strings(df[col].array, astype=astype, fmt=col_fmt)

            df[col] = pd.Series(array, index=self._obj.index)

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

        # Whether we are working inplace or on a copy
        df = self._obj if inplace else self._obj.copy()

        # Compute a bitmask of rows that we want to keep over all the columns
        mask = np.array([True] * len(df))
        for col in columns:
            mask &= df[col].array.valid()

        # We copy the DataFrame so that we're not operating on a view (truly filtering out the values)
        df = pd.DataFrame(df[mask])

        return df

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
        :param fmt:
        :param astype:
        :return:
        """
        # Column OEFormat
        col_fmt = get_oeformat(fmt)

        # noinspection PyProtectedMember
        array = MoleculeArray._from_sequence_of_strings(self._obj, astype=astype, fmt=col_fmt)
        return pd.Series(array, index=self._obj.index)
