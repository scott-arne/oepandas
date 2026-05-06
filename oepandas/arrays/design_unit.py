import base64
import logging
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from openeye import oechem

# noinspection PyProtectedMember
from pandas._typing import Dtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype

from .base import OEExtensionArray
from .molecule import MoleculeArray

log = logging.getLogger("oepandas")


########################################################################################################################
# Design Unit Array
########################################################################################################################

def _read_oedus_from_file(fp: Path) -> Generator[oechem.OEDesignUnit, None, None]:
    """
    Generator over OEDesignUnit objects from a file
    :param fp: Path to file
    :return: Generator over design units
    """
    du = oechem.OEDesignUnit()
    ifs = oechem.oeifstream(str(fp))
    while oechem.OEReadDesignUnit(ifs, du):
        yield du
    ifs.close()


def _read_oedus_from_directory(directory: Path) -> Generator[oechem.OEDesignUnit, None, None]:
    """
    Generator over OEDesignUnit objects in multiple .oedu files in a directory
    :param directory: Path to directory
    :return: Generator over design units
    """
    for fp in directory.glob("*.oedu"):
        yield from _read_oedus_from_file(fp)


def _resolve_no_title(no_title: bool, clear_titles: bool | None) -> bool:
    """
    Resolve the canonical ``no_title`` option and the legacy ``clear_titles`` alias.

    :param no_title: Canonical title-clearing option.
    :param clear_titles: Backward-compatible alias for ``no_title``.
    :returns: Whether titles should be cleared.
    :raises ValueError: When both options are supplied with conflicting values.
    """
    if clear_titles is None:
        return no_title

    if no_title and not clear_titles:
        raise ValueError("clear_titles conflicts with no_title=True")

    return clear_titles


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

    # noinspection PyUnusedLocal
    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[oechem.OEDesignUnit | bytes | str | None],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
    ) -> Self:
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Copy the design units (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        design_units = []

        for obj in scalars:

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

    # noinspection PyUnusedLocal
    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
    ) -> Self:
        """
        Read molecules form a sequence of base64-encoded design unit strings
        :param strings: Sequence of strings
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Not used (here for API compatibility with Pandas)
        :return: Array of molecules
        """
        design_units = []
        for s in strings:
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
    ) -> Self:
        """
        Read molecules from a design unit file
        :param fp: Path to the SD file
        :param astype: Type of OEDesignUnit to read (really just here for API consistency)
        :return: Design unit array
        """
        if astype is not oechem.OEDesignUnit:
            raise TypeError("OEDesignUnit is the only design unit type currently available")

        design_units = []

        filepath = Path(fp)
        reader = _read_oedus_from_file if filepath.is_file() else _read_oedus_from_directory

        for du in reader(filepath):
            design_units.append(du.CreateCopy())

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
            if du is None:
                mols.append(None)
                continue

            mol = oechem.OEMol()
            du.GetComponents(mol, mask)
            mols.append(mol)
        return MoleculeArray(mols)

    def get_ligands(self, *, no_title: bool = False, clear_titles: bool | None = None) -> MoleculeArray:
        """
        Get a molecule array of just the ligands
        :param no_title: Clear ligand titles
        :param clear_titles: Backward-compatible alias for no_title
        :return:  Molecule array of just ligands
        """
        clear_title_values = _resolve_no_title(no_title, clear_titles)
        ligs = []
        for du in self:
            if du is None:
                ligs.append(None)
                continue

            lig = oechem.OEMol()
            du.GetLigand(lig)

            if clear_title_values:
                lig.SetTitle('')
                lig.GetActive().SetTitle('')

            ligs.append(lig)
        return MoleculeArray(ligs)

    def get_proteins(self, *, no_title: bool = False, clear_titles: bool | None = None) -> MoleculeArray:
        """
        Get a molecule array of just the proteins
        :param no_title: Clear protein titles
        :param clear_titles: Backward-compatible alias for no_title
        :return: Molecule array of just proteins
        """
        clear_title_values = _resolve_no_title(no_title, clear_titles)
        prots = []
        for du in self:
            if du is None:
                prots.append(None)
                continue

            prot = oechem.OEMol()
            du.GetProtein(prot)

            if clear_title_values:
                prot.SetTitle('')
                prot.GetActive().SetTitle('')

            prots.append(prot)
        return MoleculeArray(prots)


@register_extension_dtype
class DesignUnitDtype(PandasExtensionDtype):
    """
    OpenEye design unit datatype for Pandas
    """

    type: ClassVar[type] = oechem.OEDesignUnit
    name: ClassVar[str] = "design_unit"
    kind: ClassVar[str] = "O"
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
