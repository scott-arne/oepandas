import sys
import logging
import numpy as np
import pandas as pd
from copy import copy as shallow_copy
from itertools import chain
from openeye import oechem, oedepict
from abc import ABCMeta
from typing import Generic, TypeVar, Any, Callable, Self
from collections.abc import Sized, Iterable, Sequence, Iterator
from pandas.api.extensions import ExtensionArray
from pandas.core.algorithms import take as pandas_take
# noinspection PyProtectedMember
from pandas._typing import Shape, TakeIndexer, ArrayLike, FillnaOptions

log = logging.getLogger("oepandas")

# Sentinel for no fill value given
NotSet = object()

########################################################################################################################
# Base ExtensionArray definition for OpenEye objects
########################################################################################################################

T = TypeVar('T', bound=oechem.OEMolBase | oechem.OEDesignUnit | oedepict.OE2DMolDisplay)


class OEExtensionArray(ExtensionArray, Iterable, Generic[T], metaclass=ABCMeta):

    # Each subclass must define this
    _base_openeye_type = type(None)

    def __init__(
            self,
            objs: Iterable[T | None],
            copy: bool = False,
            metadata: dict | None = None
    ) -> None:
        """
        Initialize with objects, ensuring NaN values are stored as None. This also type checks all elements for safety.
        :param objs: Objects that belong to this extension array
        :param copy: Whether to copy the objects to this extension array
        """
        self._objs = []
        self.metadata = metadata if isinstance(metadata, dict) else {}

        for obj in objs:

            if pd.isna(obj):
                self._objs.append(None)

            elif isinstance(obj, self._base_openeye_type):
                self._objs.append(obj.CreateCopy() if copy else obj)  # noqa

            else:
                raise TypeError(
                    "Cannot create {} containing object of type {} at index {}. "
                    "All elements must derive from {} or be None/NaN.".format(
                        type(self).__name__, type(obj).__name__, len(self._objs),
                        self._base_openeye_type.__name__
                    )
                )

    def append(self, item: T | None) -> None:
        """
        Append to this molecule array
        Note: This only checks for NaN/None values, but does not type check other inputs for performance
        :param item: Item to append
        """
        self._objs.append(None if pd.isna(item) else item)

    def extend(self, items: Iterable[T]) -> None:
        """
        Extend this molecule array
        Note: This only checks for NaN/None values, but does not type check other inputs for performance
        :param items: Items to add to array
        """
        # Optimization for other extension arrays, where we're confidence about None safety
        if isinstance(items, OEExtensionArray):
            self._objs.extend(items._objs)
        else:
            self._objs.extend(map(lambda x: None if pd.isna(x) else x, items))

    def copy(self, metadata: bool | dict | None = True) -> Self:
        """
        Make a shallow copy of this object
        :param metadata: Metadata for copied object (if dict), or whether to copy the metadata, or None (same as False)
        :return: Shallow copy of this object
        """
        if isinstance(metadata, bool) and metadata:
            metadata = shallow_copy(self.metadata)

        new_obj = self.__class__(
            shallow_copy(self._objs),
            metadata=metadata
        )
        return new_obj

    def deepcopy(self, metadata: bool | dict | None = True) -> Self:
        """
        Make a deep copy of this object
        :param metadata: Metadata for copied object (if dict), or whether to copy the metadata, or None (same as False)
        :return: Deep copy of object
        """
        if isinstance(metadata, bool) and metadata:
            metadata = shallow_copy(self.metadata)

        new_obj = self.__class__(
            [obj.CreateCopy() if isinstance(obj, self._base_openeye_type) else obj
             for obj in self._objs],
            metadata=metadata
        )

        return new_obj

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ):
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

        # Filled data
        filled = []
        fills_done = 0

        for obj in self._objs:

            if limit is not None and fills_done >= limit:

                filled.append(
                    obj if not copy else (obj.CreateCopy() if isinstance(obj, self._base_openeye_type) else obj))

                continue

            if pd.isna(obj):
                filled.append(value)
                fills_done += 1

            elif isinstance(obj, self._base_openeye_type):
                if not obj.IsValid():
                    filled.append(value)
                    fills_done += 1

                else:
                    filled.append(obj.CreateCopy() if copy else obj)
            else:
                raise TypeError(
                    f'{type(self).__name__} cannot determine if object of type '
                    f'{type(obj).__name__} is NaN. Only {self._base_openeye_type.__name__} '
                    f'or None values are supported.'
                )

        return self.__class__(filled, metadata=shallow_copy(self.metadata))

    def dropna(self):
        """
        Drop all NA and invalid molecules
        :return: MoleculeArray with no missing or invalid molecules
        """
        non_missing = []

        for obj in self._objs:
            if obj is not None:
                if isinstance(obj, self._base_openeye_type):
                    if obj.IsValid():
                        non_missing.append(obj)
                else:
                    raise TypeError(
                        f'{type(self).__name__} cannot determine if object of type '
                        f'{type(obj).__name__} is NaN. Only {self._base_openeye_type.__name__} '
                        f'or None values are supported.'
                    )

        return self.__class__(non_missing, metadata=shallow_copy(self.metadata))

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = NotSet,
    ) -> Self:
        """
        Take elements from the array
        :param indices:
        :param allow_fill:
        :param fill_value:
        :return:
        """
        if allow_fill and fill_value is NotSet:
            fill_value = self.dtype.na_value

        raw = self._objs if isinstance(self._objs, np.ndarray) else np.array(self._objs, dtype=object)
        result = pandas_take(raw, indices, allow_fill=allow_fill, fill_value=fill_value)
        return self.__class__(result, metadata=shallow_copy(self.metadata))

    def tolist(self, copy=False) -> list[oechem.OEMolBase]:
        """
        Convert to a list
        :param copy: Whether to copy the molecules or return pointers
        :return: List of molecules
        """
        if copy:
            return [obj.CreateCopy() if isinstance(obj, self._base_openeye_type) else obj for obj in self._objs]
        return shallow_copy(self._objs)

    @property
    def shape(self) -> Shape:
        return (len(self._objs),)

    @property
    def na_value(self):
        return None

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence['OEExtensionArray[T]']) -> Self:
        """
        Concatenate objects with the same datatype
        :param to_concat: Objects to concatenate
        :return: Concatenated object
        """
        # Optimized: Use itertools.chain for better performance with large concatenations
        return cls(chain.from_iterable(arr._objs for arr in to_concat))

    @classmethod
    def _from_factorized(cls, values, original):
        # Optimized: Use list comprehension for better performance
        # noinspection PyProtectedMember
        objs = [None if idx == -1 else original._objs[idx] for idx in values]
        return cls(objs, metadata=shallow_copy(original.metadata))

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        """
        Formatter to used
        :param boxed: Whether this object is boxed in a series
        :return: Formatter to use for rendering this object
        """
        return str

    def isna(self) -> np.ndarray:
        """
        Return a boolean array of whether elements in the array are None
        :return: Boolean array
        """
        # Optimized: Use numpy fromiter for better performance
        return np.fromiter((obj is None for obj in self._objs), dtype=bool, count=len(self._objs))

    def valid(self) -> np.ndarray:
        """
        Return a boolean array of whether molecules are valid or invalid
        :return: Boolean array
        """
        def _is_valid(obj):
            if obj is None:
                return False
            # noinspection PyBroadException
            try:
                return bool(obj.IsValid())
            except Exception:
                return False

        # Vectorized: Use numpy ufunc for better performance on large arrays
        ufunc_is_valid = np.frompyfunc(_is_valid, 1, 1)
        return ufunc_is_valid(self._objs).astype(bool)

    # noinspection PyDefaultArgument
    def __deepcopy__(self, memodict=None):
        return self.deepcopy(metadata=True)

    def __copy__(self):
        return self.copy(metadata=True)

    @property
    def nbytes(self) -> int:
        """
        Number of bytes required to hold the pointers to all the underlying objects (whose memory is managed by
        OpenEye, so we cannot interrogate the exact total size of this array)
        :return: Size of the contents of this array
        """
        return sum(map(sys.getsizeof, self._objs))

    def __len__(self) -> int:
        """
        Get number of objects in this extension array
        :return: Number of objects in extension array
        """
        return len(self._objs)

    def __iter__(self) -> Iterator[T]:
        """
        Iterate this object
        :return: Iterator over objects in this extension array
        """
        return iter(self._objs)

    def __eq__(self, other):
        if isinstance(other, OEExtensionArray):
            return np.array(self._objs) == np.array(other._objs)
        elif isinstance(other, Iterable):
            return np.array(self._objs) == np.array(other)
        else:
            raise TypeError(f'Cannot compare equality of {type(self).__name__} and {type(other).__name__}')

    def __ne__(self, other):
        if isinstance(other, OEExtensionArray):
            if len(self) != len(other):
                return np.zeros(len(self), dtype=bool)
            return np.array([a is b for a, b in zip(self._objs, other._objs)], dtype=bool)

        elif isinstance(other, Iterable):
            other_list = list(other)
            if len(self) != len(other_list):
                return np.zeros(len(self), dtype=bool)
            return np.array([a is b for a, b in zip(self._objs, other_list)], dtype=bool)

        else:
            raise TypeError(f'Cannot compare non-equality of {type(self).__name__} and {type(other).__name__}')

    def __lt__(self, other: Sized) -> bool:
        return len(self) < len(other)

    def __gt__(self, other: Sized) -> bool:
        return len(self) > len(other)

    def __le__(self, other: Sized) -> bool:
        return len(self) <= len(other)

    def __ge__(self, other: Sized) -> bool:
        return len(self) >= len(other)

    def __add__(self, other: 'Iterable[T | None] | T | None | OEExtensionArray[T]') -> Self:
        """
        Create a shallow copy of this extension array with added element(s)
        :param other: Value(s) to add
        :return: Shallow copy of this extension array with added element(s)
        """
        new_obj = self.copy()

        if isinstance(other, Iterable):
            new_obj.extend(other)
        else:
            new_obj.append(other)

        return new_obj

    def __sub__(self, other: 'Iterable[T | None] | T | None | OEExtensionArray[T]') -> Self:
        raise NotImplemented(f'Subtraction not implemented for {type(self).__name__}')

    def __contains__(self, item: object) -> bool:
        # Check for NA values (None, np.nan, pd.NA, etc.)
        is_na = item is None or item is pd.NA or (isinstance(item, float) and np.isnan(item))
        if is_na:
            return any(obj is None for obj in self._objs)
        return any(obj is item for obj in self._objs)

    def __setitem__(self, index, value) -> None:
        """
        Set an item in the array
        :param index: Item index
        :type index: int
        :param value: Item to set
        :type value: oechem.OEMolBase
        :return:
        """
        # Handle NaN values consistently with constructor
        is_na = value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))
        if is_na:
            self._objs[index] = None
        else:
            self._objs[index] = value

    def __getitem__(self, index) -> T | Self:
        """
        Get an item in the array
        :param index: Item index
        :return: Item at index
        """
        if isinstance(index, int):
            return self._objs[index]
        return self.__class__(self._objs[index], metadata=shallow_copy(self.metadata))

    def __hash__(self):
        return hash(tuple(self._objs))

    def __reversed__(self) -> Self:
        return self.__class__(reversed(self._objs), metadata=shallow_copy(self.metadata))

    def __array__(self, dtype=None, copy=None):
        arr = np.empty(len(self._objs), dtype=object)
        arr[:] = self._objs
        if dtype:
            arr = arr.astype(dtype)
        if copy is False:
            return arr
        elif copy is True:
            return arr.copy()
        else:
            # copy=None means use default behavior (no explicit copy)
            return arr

    def __str__(self):
        return f'{type(self).__name__}(len={len(self._objs)})'

    def __repr__(self):
        return self.__str__()
