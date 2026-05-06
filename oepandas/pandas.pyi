# Type stubs for pandas extensions
# This file tells PyCharm about dynamically added accessor methods

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
from openeye import oechem

from .arrays.molecule import MoleculeTypeInput
from .arrays.query import QueryFormatInput

# Monkey patch pandas classes with type hints for our accessors
module = pd


class Series(pd.Series[Any]):
    # Molecule-related accessors
    def copy_molecules(self) -> pd.Series: ...
    def is_valid(self) -> pd.Series: ...

    def as_molecule(
        self,
        *,
        molecule_format: str | int | None = None,
        molecule_type: MoleculeTypeInput = None,
        no_title: bool = False,
    ) -> pd.Series: ...

    def as_query(
        self,
        *,
        query_format: QueryFormatInput = ...,
        no_title: bool = False,
    ) -> pd.Series: ...

    def to_molecule(
        self,
        *,
        molecule_format: str | int | None = None,
    ) -> pd.Series: ...

    def to_molecule_strings(
        self,
        *,
        molecule_format: str | int = "smiles",
        flavor: int | None = None,
        gzip: bool = False,
        b64encode: bool = False
    ) -> pd.Series: ...

    def to_molecule_bytes(
        self,
        *,
        molecule_format: str | int = ...,
        flavor: int | None = None,
        gzip: bool = False
    ) -> pd.Series: ...

    def to_smiles(self, *, flavor: int | None = None) -> pd.Series: ...

    # noinspection PyPep8Naming
    def substructure_search(
        self,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False
    ) -> pd.Series: ...

    # noinspection PyPep8Naming
    def substructure_filter(
        self,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False
    ) -> pd.Series: ...

    # Design Unit-related accessors
    def copy_design_units(self) -> pd.Series: ...
    def get_ligands(self, *, no_title: bool = False, clear_titles: bool | None = None) -> pd.Series: ...
    def get_proteins(self, *, no_title: bool = False, clear_titles: bool | None = None) -> pd.Series: ...
    def get_components(self, mask: int) -> pd.Series: ...
    def as_design_unit(self) -> pd.Series: ...


class DataFrame(pd.DataFrame):
    def as_molecule(
        self,
        columns: str | list[str],
        *,
        inplace: bool = False,
        molecule_format: str | int | None = None,
        molecule_type: MoleculeTypeInput = None,
        no_title: bool = False,
    ) -> pd.DataFrame: ...

    def as_query(
        self,
        columns: str | list[str],
        *,
        inplace: bool = False,
        query_format: QueryFormatInput = ...,
        no_title: bool = False,
    ) -> pd.DataFrame: ...

    def filter_valid(
        self,
        columns: str | list[str],
        *,
        inplace: bool = False,
    ) -> pd.DataFrame: ...

    def detect_molecule_columns(
        self,
        *,
        inplace: bool = False
    ) -> pd.DataFrame: ...

    def to_sdf(
        self,
        fp: str | Path,
        primary_molecule_column: str,
        *,
        title_column: str | None = None,
        columns: str | list[str] | None = None,
        index: bool = True,
        index_tag: str = "index",
        secondary_molecules_as: str | int = "smiles",
        secondary_molecule_flavor: str | int | None = None,
        gzip: bool = False,
    ) -> None: ...

    def to_smi(
        self,
        fp: str | Path,
        primary_molecule_column: str,
        *,
        flavor: int | None = None,
        molecule_format: str | int = ...,
        title_column: str | None = None,
        gzip: bool = False,
    ) -> None: ...

    def to_molecule_csv(
        self,
        fp: str | Path,
        *,
        molecule_format: str | int = "smiles",
        flavor: int | None = None,
        gzip: bool = False,
        b64encode: bool = False,
        columns: list[Hashable] | None = None,
        index: bool = True,
        **kwargs: Any
    ) -> None: ...

    def as_design_unit(
        self,
        columns: str | list[str],
        *,
        inplace: bool = False
    ) -> pd.DataFrame: ...

    # noinspection PyPep8Naming
    def substructure_search(
        self,
        column: str,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False
    ) -> pd.DataFrame: ...

    # noinspection PyPep8Naming
    def substructure_filter(
        self,
        column: str,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False
    ) -> pd.DataFrame: ...
