# Type stubs for pandas extensions
# This file tells PyCharm about dynamically added accessor methods

from typing import Optional, Union, List, Any
import pandas as pd
import numpy as np
from pathlib import Path
from openeye import oechem

# Monkey patch pandas classes with type hints for our accessors
module = pd

class Series(pd.Series[Any]):
    # Molecule-related accessors
    def copy_molecules(self) -> pd.Series: ...
    
    def as_molecule(
        self, 
        *, 
        molecule_format: Optional[Union[str, int]] = None
    ) -> pd.Series: ...
    
    def to_molecule_strings(
        self,
        molecule_format: Union[str, int] = "smiles",
        flavor: Optional[int] = None,
        gzip: bool = False,
        b64encode: bool = False
    ) -> np.ndarray: ...
    
    def to_molecule_bytes(
        self,
        molecule_format: Union[str, int] = ...,
        flavor: Optional[int] = None,
        gzip: bool = False
    ) -> np.ndarray: ...
    
    def to_smiles(self, flavor: Optional[int] = None) -> np.ndarray: ...

    # noinspection PyPep8Naming
    def subsearch(
        self, 
        pattern: Union[str, oechem.OESubSearch], 
        adjustH: bool = False
    ) -> np.ndarray: ...
    
    # Design Unit-related accessors
    def copy_design_units(self) -> pd.Series: ...
    def get_ligands(self) -> pd.Series: ...
    def get_proteins(self) -> pd.Series: ...
    def get_components(self) -> pd.Series: ...
    def as_design_unit(self) -> pd.Series: ...

class DataFrame(pd.DataFrame):
    def as_molecule(
        self,
        columns: Union[str, List[str]],
        *,
        inplace: bool = False,
        molecule_format: Optional[Union[str, int]] = None,
    ) -> Optional[pd.DataFrame]: ...
    
    def filter_invalid_molecules(
        self,
        columns: Union[str, List[str]]
    ) -> pd.DataFrame: ...
    
    def detect_molecule_columns(
        self,
        *,
        inplace: bool = False
    ) -> Optional[pd.DataFrame]: ...
    
    def to_sdf(
        self,
        path: Union[str, Path],
        *,
        molecule_column: str = "Molecule",
        flavor: Optional[int] = None,
        conformer_test: str = "default",
        exclude_columns: Optional[List[str]] = None
    ) -> None: ...
    
    def to_smi(
        self,
        path: Union[str, Path],
        *,
        molecule_column: str = "Molecule",
        flavor: Optional[int] = None
    ) -> None: ...
    
    def to_molecule_csv(
        self,
        path: Union[str, Path],
        *,
        molecule_columns: Optional[Union[str, List[str]]] = None,
        molecule_format: Union[str, int] = "smiles",
        flavor: Optional[int] = None,
        gzip: bool = False,
        b64encode: bool = False,
        **kwargs: Any
    ) -> None: ...
    
    def as_design_unit(
        self,
        columns: Union[str, List[str]],
        *,
        inplace: bool = False
    ) -> Optional[pd.DataFrame]: ...