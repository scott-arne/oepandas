import logging
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from openeye import oechem
from typing import Callable, Literal, Mapping, Protocol
from collections.abc import Iterable, Sequence, Hashable
# noinspection PyProtectedMember
from pandas.api.types import is_numeric_dtype, is_float, is_integer
from pandas.core.dtypes.dtypes import PandasExtensionDtype
# noinspection PyProtectedMember
from pandas.io.parsers.readers import _c_parser_defaults
# noinspection PyProtectedMember
from pandas._libs import lib
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor
# noinspection PyProtectedMember
from pandas._typing import (
    FilePath,
    IndexLabel,
    ReadBuffer,
    HashableT,
    CompressionOptions,
    CSVEngine,
    Dtype,
    DtypeArg,
    DtypeBackend,
    StorageOptions,
)
from .util import (
    get_oeformat,
    is_gz,
    create_molecule_to_string_writer,
    predominant_type
)
from .arrays import MoleculeArray, MoleculeDtype, DesignUnitArray, DesignUnitDtype
# noinspection PyProtectedMember
from .arrays.molecule import _read_molecules
from .exception import FileError

log = logging.getLogger("oepandas")


########################################################################################################################
# Helpers
########################################################################################################################

class Column:
    """
    Column to create in a Pandas DataFrame from any iterable of data. This more importantly allows us to create or
    preserve an index associated with the data elements.
    """
    def __init__(
            self,
            name: str,
            data: Iterable | None = None,
            dtype: Dtype = object,
            index: Iterable | None = None
    ):
        self.name = name
        self.data = [] if data is None else list(data)
        self.dtype = dtype
        self.index = [] if index is None else list(index)

    def __len__(self):
        return len(self.data)


class Dataset:
    """
    Dataset read off of objects to be put into a DataFrame
    """
    TYPES = {
        str: str,
        float: float,
        int: int,
        bool: bool,
        oechem.OEMolBase: MoleculeDtype(),
        oechem.OEDesignUnit: DesignUnitDtype()
    }

    # Sentinel for cases where the type has not been determined
    TYPE_NOT_DETERMINED = object()

    def __init__(self, usecols: Iterable[str] | None = None):
        self.columns: dict[str, Column] = {}
        self.usecols = None if usecols is None else set(usecols)

    def add(
            self,
            col: str,
            val: str | float | int | bool | oechem.OEMolBase | oechem.OEDesignUnit,
            idx: int | None = None,
            force_type: type | PandasExtensionDtype | None = None
    ):
        """
        Add data to the dataset
        :param col: Column name
        :param val: Value to add
        :param idx: Index of the object
        :param force_type: Value type to force
        """
        # If we are using on certain columns
        if self.usecols is None or col in self.usecols:

            t = self.TYPES.get(type(val), object) if force_type is None else force_type

            # If this is a new column
            if col not in self.columns:

                self.columns[col] = Column(
                    col,
                    dtype=self.TYPE_NOT_DETERMINED if pd.isna(val) else t
                )

            # Check if we have resolved a data type
            if self.columns[col].dtype is self.TYPE_NOT_DETERMINED and not pd.isna(val):
                self.columns[col].dtype = t

            else:
                # Check if we have to modify the datatype
                if self.columns[col].dtype != t:

                    # "Downgrade" numeric types to float
                    if is_numeric_dtype(t) and is_numeric_dtype(self.columns[col].dtype):
                        self.columns[col].dtype = float

                    # Everything else is downgraded to object
                    else:
                        self.columns[col].dtype = object

            # Add the value
            self.columns[col].data.append(val)

            if idx is not None:
                self.columns[col].index.append(idx)

    def keys(self):
        """
        Get the column names
        :return: Column keys
        """
        return self.columns.keys()

    def pop(self, key: str):
        """
        Pop a column
        :return: Popped column
        """
        return self.columns.pop(key)

    def to_series_dict(self) -> dict[str, pd.Series]:
        """
        Convert to a dictionary with names and Pandas series
        :return: Dictionary
        """
        return {
            name: pd.Series(
                data=column.data,
                index=None if len(column.index) == 0 else column.index,
                dtype=column.dtype
            ) for name, column in self.columns.items()
        }

    def __getitem__(self, item):
        return self.columns[item]

    def __setitem__(self, key, value):
        self.columns[key] = value

    def __delitem__(self, key):
        del self.columns[key]


########################################################################################################################
# Molecule Array: I/O
########################################################################################################################

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

    :param df: DataFrame to add SMILES columns to
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
            conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"]
    ) -> MoleculeArray: ...


def _read_molecules_to_dataframe(
    reader: MoleculeArrayReaderProtocol,
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    flavor: int | None = oechem.OEIFlavor_SDF_Default,
    molecule_column_name: str = "Molecule",
    title_column_name: str | None = "Title",
    add_smiles: None | bool | str | Iterable[str] = None,
    molecule_columns: None | str | Iterable[str] = None,
    expand_confs: bool = False,
    generic_data=True,
    sd_data=True,
    usecols: None | str | Iterable[str] = None,
    numeric_columns: None | str | dict[str, Literal["integer", "signed", "unsigned", "float"] | None] | Iterable[str] = None,  # noqa
    conformer_test: Literal["default", "absolute", "absolute_canonical", "isomeric", "omega"] = "default",
    combine_tags: Literal["prefix", "prefer_sd", "prefer_generic"] = "prefix",
    conf_index_column_name: str = "ConfIdx",
    sd_prefix: str = "SD Tag: ",
    generic_prefix: str = "Generic Tag: "
) -> pd.DataFrame:
    """
    Read a molecule file with SD data and/or generic data
    :param reader: MoleculeArray method to use to read molecule
    :param filepath_or_buffer: File path or buffer
    :param flavor: SMILES flavor (part of oechem.OEIFlavor namespace)
    :param molecule_column_name: Name of the molecule column in the dataframe
    :param title_column_name: Name of the column with molecule titles in the dataframe
    :param add_smiles: Include a SMILES column in the dataframe (SMILES will be re-canonicalized)
    :param expand_confs: Expand conformers (i.e., create a new molecule for each conformer)
    :param molecule_columns: Additional columns to convert to molecules
    :param generic_data: If True, read generic data (default is True)
    :param sd_data: If True, read SD data (default is True)
    :param usecols: List of data tags to read (default is all read all generic and SD data)
    :param numeric_columns: Data column(s) to make numeric
    :param conformer_test: Combine single conformer molecules into multiconformer
    :param combine_tags: Strategy for combining identical SD and generic data tags
    :param sd_prefix: Prefix for SD data with corresponding generic data columns (for combine_tags='prefix')
    :param generic_prefix: Prefix for generic data with corresponding SD data columns (for combine_tags='prefix')
    :return: Pandas DataFrame
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
    if numeric_columns is not None:
        if isinstance(numeric_columns, str):
            numeric_columns = {numeric_columns: None}
        elif isinstance(numeric_columns, Iterable) and not isinstance(numeric_columns, dict):
            numeric_columns = {col: None for col in numeric_columns}

        # Sanity check the molecule column
        if molecule_column_name in numeric_columns:
            raise KeyError(f'Cannot make molecule column {molecule_column_name} numeric')

    # --------------------------------------------------
    # Start building our DataFrame with the molecules
    # --------------------------------------------------

    # Read the molecules into a molecule array
    mols = reader(filepath_or_buffer, flavor=flavor, conformer_test=conformer_test)

    # Our initial dataframe is built from the molecules themselves
    if expand_confs:
        confs = []
        conf_idx = []
        for mol in mols:  # type: oechem.OEMol
            for conf in mol.GetConfs():  # type: oechem.OEConfBase
                confs.append(oechem.OEMol(conf))
                conf_idx.append(conf.GetIdx())

        data = {
            molecule_column_name: pd.Series(data=confs, dtype=MoleculeDtype()),
            conf_index_column_name: pd.Series(data=conf_idx, dtype=str)
        }
    else:
        data = {molecule_column_name: pd.Series(data=mols, dtype=MoleculeDtype())}

    # Add titles to the dataframe
    if title_column_name is not None:
        data[title_column_name] = pd.Series(
            [mol.GetTitle() if isinstance(mol, oechem.OEMolBase) else None for mol in mols],
            dtype=str
        )

    # ----------------------------------------------------------------------
    # Get the data off the molecules
    # ----------------------------------------------------------------------

    if sd_data or generic_data:
        # Differentiate between SD data and generic data, so that we can prefix overlapping column names
        sd_dataset = Dataset(usecols=usecols)
        generic_dataset = Dataset(usecols=usecols)

        for idx, mol in enumerate(mols):  # type: int, oechem.OEMol

            if sd_data:

                for dp in oechem.OEGetSDDataPairs(mol.GetActive()):
                    sd_dataset.add(dp.GetTag(), dp.GetValue(), idx)

            if generic_data:

                for diter in mol.GetDataIter():

                    try:
                        tag = oechem.OEGetTag(diter.GetTag())
                        val = diter.GetData()
                        generic_dataset.add(tag, val, idx)

                    except ValueError:
                        continue

        # Resolve overlapping column names between SD data and generic data
        for col in set(sd_dataset.keys()).intersection(set(generic_dataset.keys())):
            if combine_tags == "prefix":

                new_sd_name = f'{sd_prefix}{col}'
                new_generic_name = f'{generic_prefix}{col}'

                sd_dataset[new_sd_name] = sd_dataset.pop(col)
                sd_dataset[new_sd_name].name = new_sd_name

                generic_dataset[new_generic_name] = generic_dataset.pop(col)
                generic_dataset[new_generic_name].name = new_generic_name

            elif combine_tags == "prefer_sd":
                del generic_dataset[col]

            elif combine_tags == "prefer_generic":
                del sd_dataset[col]

            else:
                raise KeyError(f'Unknown combine_tags strategy: {combine_tags}')

        # Combine the tags:
        data = {
            **data,
            **sd_dataset.to_series_dict(),
            **generic_dataset.to_series_dict()
        }

    # Create the DataFrame
    df = pd.DataFrame(data)

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
                    if col in numeric_columns:
                        raise KeyError(f'Cannot make molecule column {col} numeric')

                    if col in df.columns:

                        # We don't need to convert the primary molecule column
                        if col != molecule_column_name:
                            df.as_molecule(col, inplace=True)

                    else:
                        log.warning(f'Column not found in DataFrame: {col}')

        # Cast numeric columns
        if numeric_columns is not None:
            for col, dtype in numeric_columns.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="ignore", downcast=dtype)
                else:
                    log.warning(f'Column not found in DataFrame: {col}')

        # Add SMILES column(s)
        if add_smiles is not None:

            # Only adding SMILES to the primary molecule column
            if molecule_columns is None:
                molecule_columns = [molecule_column_name]

            _add_smiles_columns(df, molecule_columns, add_smiles)

    return df


def read_molecule_csv(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    molecule_columns: str | dict[str, int] | dict[str, str] | Literal["detect"],
    *,
    add_smiles: None | bool | str | Iterable[str] = None,
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
        if molecule_columns == "detect":
            df.detect_molecule_columns()
        else:
            df.as_molecule(molecule_columns, inplace=True)

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
    inchi_key_column_name: str = "InChI Key"
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
    :return: DataFrame with molecules
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

    for mol in _read_molecules(
            filepath_or_buffer,
            file_format=oechem.OEFormat_CXSMILES if cx else oechem.OEFormat_SMI,
            flavor=flavor,
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
    read_sd_data: bool = True
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
    :param read_sd_data: Read SD data
    :return: Pandas DataFrame
    """
    return _read_molecules_to_dataframe(
        MoleculeArray.read_sdf,
        filepath_or_buffer,
        flavor=flavor,
        molecule_column_name=molecule_column_name,
        title_column_name=title_column_name,
        add_smiles=add_smiles,
        molecule_columns=molecule_columns,
        generic_data=False,
        sd_data=read_sd_data,
        usecols=usecols,
        numeric_columns=numeric,
        conformer_test=conformer_test
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
    generic_prefix: str = "Generic Tag: "
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
    :return: Pandas DataFrame
    """
    return _read_molecules_to_dataframe(
        MoleculeArray.read_oeb,
        filepath_or_buffer,
        flavor=flavor,
        molecule_column_name=molecule_column_name,
        title_column_name=title_column_name,
        add_smiles=add_smiles,
        molecule_columns=molecule_columns,
        generic_data=read_generic_data,
        sd_data=read_sd_data,
        usecols=usecols,
        numeric_columns=numeric,
        conformer_test=conformer_test,
        combine_tags=combine_tags,
        sd_prefix=sd_prefix,
        generic_prefix=generic_prefix
    )


def read_oedb(
    fp: FilePath,
    *,
    usecols: None | str | Iterable[str] = None,
    int_na: int | None = None,
) -> pd.DataFrame:
    """
    Read an OEDB file
    :param fp: Path to OEDB file
    :param usecols: Optional columns to use
    :param int_na: Value to use in place of NaN for integers
    :return: DataFrame
    """
    data = Dataset(usecols=usecols)

    # Open the record file
    filename = str(fp) if isinstance(fp, Path) else fp

    if filename.startswith("."):
        raise FileError("Reading OERecords from STDIN not yet supported")
    else:
        ifs = oechem.oeifstream()
        if not ifs.open(filename):
            raise FileError(f'Could not open file for reading: {fp}')

    # Read the records
    for idx, record in enumerate(oechem.OEReadRecords(ifs)):  # type: int, oechem.OERecord
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
                val = record.get_value(field)

                if pd.isna(val):
                    data.add(name, np.NaN, idx, force_type=float)
                else:
                    data.add(name, val, idx, force_type=float)

            # ------------------------------
            # Integer field
            # ------------------------------
            elif dtype == oechem.Types.Int:
                val = record.get_value(field)

                # Value is NaN - we try to preserve the integer data type, but int_na could mess that up
                if pd.isna(val):

                    # If our NaN value is None, then we'll convert this type float
                    if pd.isna(int_na):
                        data.add(name, np.NaN, idx, force_type=float)

                    elif is_float(int_na):
                        data.add(name, int_na, idx, force_type=float)

                    elif is_integer(int_na):
                        data.add(name, int_na, idx, force_type=int)

                    else:
                        data.add(name, int_na, idx, force_type=object)

                # Else we have an integer value
                else:
                    data.add(name, val, idx, force_type=int)

            # ------------------------------
            # Boolean field
            # ------------------------------
            elif dtype == oechem.Types.Bool:
                val = record.get_value(field)
                data.add(name, val, idx, force_type=bool)

            # ------------------------------
            # Molecule field
            # ------------------------------
            elif dtype == oechem.Types.Chem.Mol:
                val = record.get_value(field)
                data.add(name, val, idx, force_type=MoleculeDtype())

            # ------------------------------
            # Design unit field
            # ------------------------------
            elif dtype == oechem.Types.Chem.DesignUnit:
                val = record.get_value(field)
                data.add(name, val, idx, force_type=DesignUnitDtype())

            # ------------------------------
            # Everything else
            # ------------------------------
            else:
                val = record.get_value(field)
                data.add(name, val, idx, force_type=object)

    # Close the record file
    ifs.close()

    # Create the DataFrame
    return pd.DataFrame(data.to_series_dict())


########################################################################################################################
# Molecule Array: DataFrame Extensions
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
            molecule_format: dict[str, str] | dict[str, int] | str | int | None = None,
            inplace=False
    ):
        # Default format is SMILES if none is specified
        molecule_format = molecule_format or oechem.OEFormat_SMI

        # Make sure we're working with a list of columns
        columns = [columns] if isinstance(columns, str) else list(columns)

        # Validate column names
        for name in columns:
            if name not in self._obj.columns:
                raise KeyError(f'Column {name} not found in DataFrame: {", ".join(self._obj.columns)}')

        # Whether we are working inplace or on a copy
        df = self._obj if inplace else self._obj.copy()

        # Convert the columns using the as_molecule series accessor
        for col in columns:

            df[col] = df[col].as_molecule(molecule_format=molecule_format or oechem.OEFormat_SMI)

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


@register_dataframe_accessor("detect_molecule_columns")
class DataFrameDetectMoleculeColumnsAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def __call__(self, *, sample_size: int = 25) -> None:
        """
        Detects molecule columns based on their predominant type and convert them to MoleculeDtype. This works if
        the columns primarily contain objects that derive from oechem.OEMolBase (all OpenEye molecule objects).
        :param sample_size: Maximum number of non-null values to sample to determine column type
        """
        molecule_columns = []

        for col in self._obj.columns:

            # Skip columns that are already molecule columns
            if self._obj[col].dtype != MoleculeDtype():

                t = predominant_type(self._obj[col], sample_size=sample_size)

                if t is not None and issubclass(t, oechem.OEMolBase):
                    molecule_columns.append(col)

        # Convert to molecule columns
        self._obj.as_molecule(molecule_columns, inplace=True)


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
            secondary_molecule_flavor: int | str | None = None,
            gzip: bool = False
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
        # Convert to file path
        fp = Path(fp)

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

            # Force SD format
            ofs.SetFormat(oechem.OEFormat_SDF)
            ofs.Setgz(gzip or is_gz(fp))

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


@register_dataframe_accessor("to_smi")
class WriteToSmilesAccessor:
    """
    Write to SD file
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
            self,
            fp: FilePath,
            primary_molecule_column: str,
            *,
            flavor: int | None = None,
            molecule_format: str | int = oechem.OEFormat_SMI,
            title_column: str | None = None,
            gzip: bool = False
    ) -> None:
        """
        Write DataFrame to an SD file
        Note: Writing conformers not yet supported
        :param primary_molecule_column: Primary molecule column
        :param title_column: Optional column to get molecule titles
        """
        # Convert to a path
        fp = Path(fp)

        # Validate molecule format
        fmt = get_oeformat(
            molecule_format,
            gzip or is_gz(fp)
        )

        if fmt.oeformat not in (oechem.OEFormat_SMI, oechem.OEFormat_ISM, oechem.OEFormat_CXSMILES,
                                oechem.OEFormat_USM):
            raise ValueError("to_smi can only take SMILES formats as a molecule_format")

        # Validate title column
        if title_column is not None and title_column not in self._obj.columns:
            raise KeyError(f'Title column {title_column} not found in DataFrame')

        # Validate primary molecule column
        if primary_molecule_column not in self._obj.columns:
            raise KeyError(f'Primary molecule column {primary_molecule_column} not found in DataFrame')
        if not isinstance(self._obj[primary_molecule_column].dtype, MoleculeDtype):
            raise TypeError(f'Primary molecule column {primary_molecule_column} is not a MoleculeDtype')

        # Process the molecules
        with oechem.oemolostream(str(fp)) as ofs:

            # Set the output file stream attributes
            ofs.SetFormat(fmt.oeformat)
            ofs.Setgz(fmt.gzip)

            for idx, row in self._obj.iterrows():
                mol = row[primary_molecule_column].CreateCopy()

                # Set the title
                if title_column is not None:
                    mol.SetTitle(str(row[title_column]))

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


_FLOAT_TYPES = (
    pd.Float32Dtype,
    pd.Float64Dtype,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    float,
    np.floating
)

_INTEGER_TYPES = (
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
    np.dtypes.IntDType,
    np.dtypes.Int8DType,
    np.dtypes.Int16DType,
    np.dtypes.Int32DType,
    np.dtypes.Int64DType,
    np.dtypes.UInt8DType,
    np.dtypes.UInt16DType,
    np.dtypes.UInt32DType,
    np.dtypes.UInt64DType,
    np.dtypes.ShortDType,
    np.dtypes.UShortDType,
    np.integer,
    int
)

_BOOLEAN_TYPES = (
    pd.BooleanDtype,
    np.dtypes.BoolDType,
    bool
)

_STRING_TYPES = (
    pd.StringDtype,
    np.dtypes.StrDType,
    str
)

_BYTES_TYPES = (
    np.dtypes.ByteDType,
    np.dtypes.BytesDType,
    np.dtypes.UByteDType,
    bytes
)


@register_dataframe_accessor("to_oedb")
class WriteToOEDB:
    """
    Write to a CSV file containing molecules
    TODO: Add compress_confs argument to compress conformers into single multi-conformer molecules
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj.copy()

    def __call__(
            self,
            fp: FilePath,
            primary_molecule_column: str | None = None,
            *,
            title_column: str | None = None,
            columns: str | Iterable[str] | None = None,
            index: bool = True,
            index_label: str = "index",
            sample_size: int = 25,
            safe: bool = True
    ):
        """
        Write OERecords

        This will write OEMolRecords if primary_molecule_column is not None. The title_column can be used to
        optionally add a title to the primary molecule. If include_title is False, then that column will be
        excluded from the output.

        :param fp: Path to the record file
        :param columns: Optional column(s) to use
        :param index: Write an index field if True
        :param index_label: Name of the index field
        :param sample_size: Sample size of non-null values if we need to determine a column's type
        :param safe: Check the type of each value to ensure OEField compatibility before writing
        :return:
        """
        # Validate primary molecule column
        if primary_molecule_column is not None:
            if primary_molecule_column not in self._obj.columns:
                raise KeyError(f'Primary molecule column {primary_molecule_column} not found in DataFrame')
            if not isinstance(self._obj[primary_molecule_column].dtype, MoleculeDtype):
                raise TypeError(f'Primary molecule column {primary_molecule_column} is not a MoleculeDtype')

        # Validate title column
        if title_column is not None and title_column not in self._obj.columns:
            raise KeyError(f'Title column {title_column} not found in DataFrame')

        # Get the valid field names
        valid_cols = set(self._obj.columns) if columns is None else set(self._obj.columns).intersection(set(columns))
        if len(valid_cols) == 0 and primary_molecule_column is None:
            raise FileError("No data columns to write to output file")

        # Get the correct ordering of those columns
        cols = [col for col in self._obj.columns if col in valid_cols]

        # Make these fields
        # We check for field compatibility with field_types as we write the oedb
        fields = {}
        field_types = {}
        for col in cols:

            # Float field
            if isinstance(self._obj.dtypes[col], _FLOAT_TYPES):
                fields[col] = oechem.OEField(col, oechem.Types.Float)
                field_types[col] = _FLOAT_TYPES

            # Integer field
            elif isinstance(self._obj.dtypes[col], _INTEGER_TYPES):
                fields[col] = oechem.OEField(col, oechem.Types.Int)
                field_types[col] = _INTEGER_TYPES

            # Boolean field
            elif isinstance(self._obj.dtypes[col], _BOOLEAN_TYPES):
                fields[col] = oechem.OEField(col, oechem.Types.Bool)
                field_types[col] = _BOOLEAN_TYPES

            # Molecule field
            elif isinstance(self._obj.dtypes[col], MoleculeDtype):
                fields[col] = oechem.OEField(col, oechem.Types.Chem.Mol)
                field_types[col] = oechem.OEMolBase

            # Else we need to look closer before we assign a type
            else:

                # Get the predominant type in from non-null values
                t = predominant_type(self._obj[col], sample_size)

                # String field
                if issubclass(t, _STRING_TYPES):
                    fields[col] = oechem.OEField(col, oechem.Types.String)
                    field_types[col] = _STRING_TYPES

                # Bytes field
                elif issubclass(t, _BYTES_TYPES):
                    fields[col] = oechem.OEField(col, oechem.Types.Blob)
                    field_types[col] = _BYTES_TYPES

                # Design unit field
                elif issubclass(t, oechem.OEDesignUnit):
                    fields[col] = oechem.OEField(col, oechem.Types.Chem.DesignUnit)
                    field_types[col] = oechem.OEDesignUnit

                else:
                    log.warning("Do not know the OEField type that maps to dtype {} in column {}".format(
                        t.__name__, col))

        # Our record type is based on whether we have a primary molecule
        record_type = oechem.OEMolRecord if primary_molecule_column is not None else oechem.OERecord

        ofs = oechem.oeofstream()
        if not ofs.open(str(fp)):
            raise FileError(f'Could not open {fp} for writing')

        for idx, row in self._obj.iterrows():
            # Create a new record
            record = record_type()

            # If this is an OEMolRecord
            if primary_molecule_column is not None:
                record.set_mol(row[primary_molecule_column])

            for col in cols:
                f = fields[col]  # Field on record
                v = row[col]     # Value for field on current row

                record.add_field(f)

                # If this is not the expected type then do not add it
                if (not safe) or isinstance(v, field_types[col]):
                    record.set_value(f, v)

            # Write out the record
            oechem.OEWriteRecord(ofs, record)

        ofs.close()


########################################################################################################################
# Molecule Array: Series Extensions
########################################################################################################################

@register_series_accessor("copy_molecules")
class SeriesCopyMoleculeAccessor:
    def __init__(self, pandas_obj: pd.Series):
        if not isinstance(pandas_obj.dtype, MoleculeDtype):
            raise TypeError(
                "copy_molecules only works on molecule columns (oepandas.MoleculeDtype). If this column has "
                "molecules, use pd.Series.as_molecule to convert to a molecule column first."
            )

        self._obj = pandas_obj

    def __call__(self) -> pd.Series:
        # noinspection PyUnresolvedReferences
        return pd.Series(self._obj.array.deepcopy(), dtype=MoleculeDtype())


@register_series_accessor("as_molecule")
class SeriesAsMoleculeAccessor:
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def __call__(
            self,
            *,
            molecule_format: str | int | None = None,
    ) -> pd.Series:
        """
        Convert a series to molecules
        :param molecule_format: File format of column to convert to molecules (extension or from oechem.OEFormat)
        :return: Series as molecule
        """
        # Column OEFormat
        _fmt = get_oeformat(oechem.OEFormat_SMI) if molecule_format is None else get_oeformat(molecule_format)

        # noinspection PyProtectedMember
        arr = MoleculeArray._from_sequence(self._obj, molecule_format=_fmt.oeformat)
        return pd.Series(arr, index=self._obj.index, dtype=MoleculeDtype())


@register_series_accessor("to_molecule_bytes")
class SeriesToMoleculeBytesAccessor:
    def __init__(self, pandas_obj: pd.Series):
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
        # noinspection PyUnresolvedReferences
        arr = self._obj.array.to_molecule_bytes(
            molecule_format=molecule_format,
            flavor=flavor,
            gzip=gzip
        )

        return pd.Series(arr, index=self._obj.index, dtype=object)


@register_series_accessor("to_molecule_strings")
class SeriesToMoleculeStringsAccessor:
    def __init__(self, pandas_obj: pd.Series):
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
        # noinspection PyUnresolvedReferences
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
    ) -> pd.Series:
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
    def __init__(self, pandas_obj: pd.Series):
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
        # noinspection PyUnresolvedReferences
        return pd.Series(self._obj.array.subsearch(pattern, adjustH=adjustH), dtype=bool)


########################################################################################################################
# Design Unit Array: I/O
########################################################################################################################

def read_oedu(
    filepath_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    design_unit_column_name: str = "Design_Unit",
    design_unit_title_column_name: str = "Title",
    generic_data: bool = True
) -> pd.DataFrame:
    """
    Read an OEDB file into a Pandas DataFrame
    :param filepath_or_buffer: File path or buffer
    :param design_unit_column_name: Column name for the design units
    :param design_unit_title_column_name: Column name for the design unit title
    :param generic_data: Read data from the design unit into columns
    :return: Pandas DataFrame
    """
    # Cannot yet read from STDIN or other buffers
    if not isinstance(filepath_or_buffer, (str, Path)):
        raise NotImplementedError("Reading from buffers is not yet supported for read_oedu")

    # Read design units
    du_array = DesignUnitArray.read_oedu(filepath_or_buffer)

    # Read data
    data = Dataset()
    if generic_data:
        # Get the data and use indexes to assign data
        for idx, du in enumerate(du_array):
            for diter in du.GetDataIter():
                try:
                    tag = oechem.OEGetTag(diter.GetTag())
                    value = diter.GetValue()
                    data.add(tag, value, idx)

                except Exception as ex:  # noqa
                    continue

    return pd.DataFrame({
        design_unit_column_name: pd.Series(du_array, dtype=DesignUnitDtype()),
        design_unit_title_column_name: pd.Series([du.GetTitle() for du in du_array], dtype=str),
        **data.to_series_dict()
    })


########################################################################################################################
# Design Unit Array: DataFrame Extensions
########################################################################################################################

@register_dataframe_accessor("as_design_unit")
class DataFrameAsDesignUnitAccessor:
    """
    Accessor that adds the as_molecule method to a Pandas DataFrame
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

        # Convert the columns using the as_molecule series accessor
        for col in columns:

            df[col] = df[col].as_design_unit()

        return df


########################################################################################################################
# Design Unit Array: Series Extensions
########################################################################################################################

@register_series_accessor("copy_design_units")
class SeriesCopyDesignUnitsAccessor:
    def __init__(self, pandas_obj: pd.Series):
        if not isinstance(pandas_obj.dtype, DesignUnitDtype):
            raise TypeError(
                "copy_design_units only works on design unit columns (oepandas.DesignUnitDtype). If this column has "
                "design units, use pd.Series.as_design_unit to convert to a design unit column first."
            )

        self._obj = pandas_obj

    def __call__(self) -> pd.Series:
        # noinspection PyUnresolvedReferences
        return pd.Series(self._obj.array.deepcopy(), dtype=DesignUnitDtype())


@register_series_accessor("get_ligands")
class SeriesGetLigandAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, DesignUnitDtype):
            raise TypeError(
                "get_ligand only works on design unit columns (oepandas.DesignUnitDtype). If this column has "
                "design units, use pd.Series.as_design_unit to convert to a design unit column first."
            )

        self._obj = pandas_obj

    def __call__(self):
        """
        Get ligands from design units
        :return: Molecule series with ligands
        """
        return pd.Series(self._obj.array.get_ligands(), dtype=MoleculeDtype())


@register_series_accessor("get_proteins")
class SeriesGetProteinAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, DesignUnitDtype):
            raise TypeError(
                "get_protein only works on design unit columns (oepandas.DesignUnitDtype). If this column has "
                "design units, use pd.Series.as_design_unit to convert to a design unit column first."
            )

        self._obj = pandas_obj

    def __call__(self):
        """
        Get ligands from design units
        :return: Molecule series with ligands
        """
        return pd.Series(self._obj.array.get_proteins(), dtype=MoleculeDtype())


@register_series_accessor("get_components")
class SeriesGetProteinAccessor:
    def __init__(self, pandas_obj):
        if not isinstance(pandas_obj.dtype, DesignUnitDtype):
            raise TypeError(
                "get_components only works on design unit columns (oepandas.DesignUnitDtype). If this column has "
                "design units, use pd.Series.as_design_unit to convert to a design unit column first."
            )

        self._obj = pandas_obj

    def __call__(self, mask: int):
        """
        Get ligands from design units
        :param mask: Component mask
        :return: Molecule series with ligands
        """
        return pd.Series(self._obj.array.get_components(mask), dtype=MoleculeDtype())


@register_series_accessor("as_design_unit")
class SeriesAsDesignUnitAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self) -> pd.Series:
        """
        Convert a series to design units
        :return: Series as design units
        """
        # noinspection PyProtectedMember
        arr = DesignUnitArray._from_sequence(self._obj)
        return pd.Series(arr, index=self._obj.index, dtype=DesignUnitDtype())


########################################################################################################################
# Pandas monkeypatching
########################################################################################################################

# Molecules
pd.read_molecule_csv = read_molecule_csv
pd.read_smi = read_smi
pd.read_sdf = read_sdf
pd.read_oeb = read_oeb
pd.read_oedb = read_oedb

# Design units
pd.read_oedu = read_oedu
