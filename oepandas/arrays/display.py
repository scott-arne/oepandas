import numpy as np
import pandas as pd
from openeye import oedepict
from collections.abc import Iterable
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.api.extensions import register_extension_dtype
# noinspection PyProtectedMember
from pandas._typing import Dtype
from .base import OEExtensionArray


class DisplayArray(OEExtensionArray[oedepict.OE2DMolDisplay]):

    # For type checking in methods defined in OEExtensionArray
    _base_openeye_type = oedepict.OE2DMolDisplay

    def __init__(
            self,
            objs: Iterable[oedepict.OE2DMolDisplay | None],
            copy: bool = False,
            metadata: dict | None = None
    ) -> None:
        """
        Initialize with objects, ensuring NaN values are stored as None. This also type checks all elements for safety.
        :param objs: Objects that belong to this extension array
        :param copy: Whether to copy the objects to this extension array
        """
        displays = []

        for obj in objs:

            if isinstance(obj, self._base_openeye_type):
                # This is overloaded because CreateCopy does not work properly for an OE2DMolDisplay object, you have
                # to use the copy constructor instead
                displays.append(oedepict.OE2DMolDisplay(obj) if copy else obj)

            elif pd.isna(obj):
                displays.append(None)

            else:
                raise TypeError(
                    "Cannot create {} containing object of type {}. All elements must derive from {}.".format(
                        type(self).__name__, type(obj).__name__, self._base_openeye_type.__name__
                    )
                )

        super().__init__(objs=displays, copy=False, metadata=metadata)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return DisplayDtype()

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[oedepict.OE2DMolDisplay | None],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
    ) -> 'DisplayArray':
        """
        Iniitialize from a sequence of scalar values
        :param scalars: Scalars
        :param dtype: Not used (here for API compatibility with Pandas)
        :param copy: Copy the design units (otherwise stores pointers)
        :return: New instance of Molecule Array
        """
        displays = []

        for i, obj in enumerate(scalars):

            # Nones are OK
            if obj is None or pd.isna(obj):
                displays.append(None)

            # Design units
            elif isinstance(obj, oedepict.OE2DMolDisplay):
                displays.append(obj)

            # Else who knows
            else:
                raise TypeError(f'Only OE2DMolDisplay and None are valid in a DisplayArray, not {type(obj).__name__}')

        return cls(displays, copy=copy)

    def deepcopy(self, metadata: bool | dict | None = True):
        """
        Make a deep copy of this object using OE2DMolDisplay copy constructor
        :param metadata: Metadata for copied object (if dict), or whether to copy the metadata, or None (same as False)
        :return: Deep copy of object
        """
        from copy import copy as shallow_copy
        
        if isinstance(metadata, bool) and metadata:
            metadata = shallow_copy(self.metadata)

        # Use DisplayArray's special copy constructor approach for deep copying
        new_obj = self.__class__(
            [oedepict.OE2DMolDisplay(obj) if isinstance(obj, self._base_openeye_type) else obj
             for obj in self._objs],
             copy=False,  # We've already done the copying above
             metadata=metadata
        )

        return new_obj


@register_extension_dtype
class DisplayDtype(PandasExtensionDtype):
    """
    OpenEye molecule datatype for Pandas
    """

    type: type = oedepict.OE2DMolDisplay
    name: str = "display"
    kind: str = "O"
    base = np.dtype("O")

    @property
    def na_value(self):
        """Return the missing value for this dtype"""
        return None

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype
        """
        return DisplayArray

    @classmethod
    def construct_from_string(cls, string: str):
        """
        Construct dtype from string representation
        """
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a '{cls}' from '{string}'")

    @classmethod  
    def _is_dtype(cls, dtype) -> bool:
        """
        Check if dtype is an instance of this dtype
        """
        if isinstance(dtype, cls):
            return True
        if isinstance(dtype, str):
            return dtype == cls.name
        return False

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
