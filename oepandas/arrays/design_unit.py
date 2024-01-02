import logging
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from openeye import oechem
from typing import Any
from collections.abc import Iterable, Sequence
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.api.extensions import register_extension_dtype
# noinspection PyProtectedMember
from pandas._typing import Dtype
from .base import OEExtensionArray
from .molecule import MoleculeArray

log = logging.getLogger("oepandas")


########################################################################################################################
# Design Unit Array
########################################################################################################################

class DesignUnitArray(OEExtensionArray[oechem.OEDesignUnit]):

    # For type checking in methods defined in OEExtensionArray
    _base_openeye_type = oechem.OEDesignUnit

    """
    Custom extension for an array of design units
    """
    def __init__(
            self,
            design_units: oechem.OEDesignUnit | Iterable[oechem.OEDesignUnit],
            copy: bool = False,
            metadata: dict | None = None
    ):
        """
        Initialize
        :param design_units: Design unit or an iterable of design units
        :param copy: Create copy of the molecules if True
        """
        # Handle singleton design units
        if isinstance(design_units, oechem.OEDesignUnit):
            design_units = (design_units,)

        super().__init__(design_units, copy=copy, metadata=metadata)

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[oechem.OEDesignUnit | bytes | str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
    ) -> 'DesignUnitArray':
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Copy the design units (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        design_units = []

        for i, obj in enumerate(scalars):

            # Nones are OK
            if obj is None or pd.isna(obj):
                design_units.append(None)

            # Design units
            elif isinstance(obj, oechem.OEDesignUnit):
                design_units.append(obj)

            # Design unit bytes
            elif isinstance(obj, bytes):
                du = oechem.OEDesignUnit()
                oechem.OEReadDesignUnitFromBytes(du, obj)
                design_units.append(du)

            # Base64-encoded design unit bytes
            elif isinstance(obj, str):
                du = oechem.OEDesignUnit()
                design_unit_bytes = base64.b64decode(obj.encode('utf-8'))
                oechem.OEReadDesignUnitFromBytes(du, design_unit_bytes)
                design_units.append(du)

            # Else who knows
            else:
                raise TypeError(f'Cannot create a molecule from {type(obj).__name__}')

        return cls(design_units, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
    ) -> 'DesignUnitArray':
        """
        Read molecules form a sequence of base64-encoded design unit strings
        :param strings: Sequence of strings
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :return: Array of molecules
        """
        design_units = []
        for i, s in enumerate(strings):  # type: int, str
            du = oechem.OEDesignUnit()
            design_unit_bytes = base64.b64decode(s.encode('utf-8'))
            oechem.OEReadDesignUnitFromBytes(du, design_unit_bytes)
            design_units.append(du)

        return cls(design_units, copy=False)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return DesignUnitDtype()

    # --------------------------------------------------------
    # I/O
    # --------------------------------------------------------

    @classmethod
    def read_oedu(
            cls,
            fp: Path | str,
            astype: type[oechem.OEDesignUnit] = oechem.OEDesignUnit,
    ) -> 'DesignUnitArray':
        """
        Read molecules from a design unit file
        :param fp: Path to the SD file
        :param astype: Type of OEDesignUnit to read (really just here for API consistency)
        :return: Design unit array
        """
        if astype is not oechem.OEDesignUnit:
            raise TypeError("OEDesignUnit is the only design unit type currently available")

        design_units = []
        du = oechem.OEDesignUnit()

        ifs = oechem.oeifstream(str(fp))

        while oechem.OEReadDesignUnit(ifs, du):
            design_units.append(du.CreateCopy())

        ifs.close()

        return cls(design_units, copy=False, metadata={"source": str(fp)})

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def get_components(self, mask: int) -> MoleculeArray:
        """
        Get a molecule array of components
        :param mask: Component mask
        :return: Molecule array of components
        """
        mols = []
        for du in self:
            mol = oechem.OEMol()
            du.GetComponents(mol, mask)
            mols.append(mol)
        return MoleculeArray(mols)

    def get_ligands(self) -> MoleculeArray:
        """
        Get a molecule array of just the ligands
        :return:  Molecule array of just ligands
        """
        ligs = []
        for du in self:
            lig = oechem.OEMol()
            du.GetLigand(lig)
            ligs.append(lig)
        return MoleculeArray(ligs)

    def get_proteins(self) -> MoleculeArray:
        """
        Get a molecule array of just the proteins
        :return: Molecule array of just proteins
        """
        prots = []
        for du in self:
            prot = oechem.OEMol()
            du.GetProtein(prot)
            prots.append(prot)
        return MoleculeArray(prots)


@register_extension_dtype
class DesignUnitDtype(PandasExtensionDtype):
    """
    OpenEye design unit datatype for Pandas
    """

    type: type = oechem.OEDesignUnit
    name: str = "design_unit"
    kind: str = "O"
    base = np.dtype("O")
    isbuiltin = 0
    isnative = 0

    _is_numeric = False
    _is_boolean = False

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return DesignUnitArray

    # Required
    def __hash__(self) -> int:
        return hash(self.name)

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
