import sys
import logging
import numpy as np
import pandas as pd
from more_itertools import flatten
from copy import copy as shallow_copy
from openeye import oechem
from abc import ABCMeta
from typing import Generic, TypeVar, Any, Callable
from collections.abc import Sized, Iterable, Sequence, Iterator
from pandas.api.extensions import ExtensionArray
from pandas.core.algorithms import take as pandas_take
# noinspection PyProtectedMember
from pandas._typing import Shape, TakeIndexer, ArrayLike, FillnaOptions

log = logging.getLogger("oepandas")


########################################################################################################################
# Base ExtensionArray definition for OpenEye objects
########################################################################################################################

T = TypeVar('T', bound=oechem.OEMolBase | oechem.OEDesignUnit)


class OEExtensionArray(ExtensionArray, Iterable, Generic[T], metaclass=ABCMeta):

    # Each subclass must define this
    _base_openeye_type = None

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
        self.metadata = metadata or {}

        for obj in objs:

            if isinstance(obj, self._base_openeye_type):
                self._objs.append(obj.CreateCopy() if copy else obj)

            elif pd.isna(obj):
                self._objs.append(None)

            else:
                raise TypeError(
                    "Cannot create {} containing object of type {}. All elements must derive from {}.".format(
                        type(self).__name__, type(obj).__name__, self._base_openeye_type.__name__
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

    def copy(self, metadata: bool | dict | None = True):
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

    def deepcopy(self, metadata: bool | dict | None = True):
        """
        Make a deep copy of this object
        :param metadata: Metadata for copied object (if dict), or whether to copy the metadata, or None (same as False)
        :return: Deep copy of object
        """
        if isinstance(metadata, bool) and metadata:
            metadata = shallow_copy(self.metadata)

        new_obj = self.__class__(
            [obj.CreateCopy() for obj in self._objs],
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

        # Data to fill
        if copy:
            data = np.array(
                [(obj.CreateCopy() if isinstance(obj, self._base_openeye_type) else obj) for obj in self._objs]
            )
        else:
            data = self._objs

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
            elif isinstance(obj, self._base_openeye_type):
                filled.append((obj.CreateCopy() if copy else obj) if obj.IsValid() else value)
            else:
                raise TypeError(f'MoleculeArray cannot determine of object of type {type(obj).__name__} is NaN')

        return self.__class__(filled, metadata=shallow_copy(self.metadata))

    def dropna(self):
        """
        Drop all NA and invalid molecules
        :return: MoleculeArray with no missing or invalid molecules
        """
        non_missing = []

        for obj in self._objs:
            if not pd.isna(obj):
                if isinstance(obj, self._base_openeye_type):
                    if obj.IsValid():
                        non_missing.append(obj)
                else:
                    raise TypeError(f'MoleculeArray cannot determine of object of type {type(obj).__name__} is NaN')

        return self.__class__(non_missing, metadata=shallow_copy(self.metadata))

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> 'OEExtensionArray[T]':
        """
        Take elements from the array
        :param indices:
        :param allow_fill:
        :param fill_value:
        :return:
        """
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = pandas_take(np.array(self._objs), indices, allow_fill=allow_fill, fill_value=fill_value)
        return self.__class__(result)

    def tolist(self, copy=False) -> list[oechem.OEMolBase]:
        """
        Convert to a list
        :param copy: Whether to copy the molecules or return pointers
        :return: List of molecules
        """
        if copy:
            return [obj.CreateCopy() for obj in self._objs]
        return shallow_copy(self._objs)

    @property
    def shape(self) -> Shape:
        return (len(self._objs),)

    @property
    def na_value(self):
        return None

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence['OEExtensionArray[T]']) -> 'OEExtensionArray[T]':
        """
        Concatenate objects with the same datatype
        :param to_concat: Objects to concatenate
        :return: Concatenated object
        """
        return cls(flatten([obj._objs for obj in to_concat]))

    @classmethod
    def _from_factorized(cls, values, original):
        """
        NOT IMPLEMENTED: Reconstruct an MoleculeArray after factorization
        """
        raise NotImplemented(f'Factorization not implemented for {cls.__name__}')

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
        return np.array([obj is None for obj in self._objs], dtype=bool)

    def valid(self) -> np.ndarray:
        """
        Return a boolean array of whether molecules are valid or invalid
        :return: Boolean array
        """
        return np.array([obj.IsValid() for obj in self._objs], dtype=bool)

    # noinspection PyDefaultArgument
    def __deepcopy__(self, memodict={}):
        return self.deepcopy(metadata=True)

    def __copy__(self):
        print("COPY!")
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

    def __ne__(self, other) -> bool:
        if isinstance(other, OEExtensionArray):
            return np.array(self._objs) != np.array(other._objs)
        elif isinstance(other, Iterable):
            return np.array(self._objs) != np.array(other)
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

    def __add__(self, other: Iterable[T | None] | T | None | 'OEExtensionArray[T]') -> 'OEExtensionArray[T]':
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

    def __sub__(self, other: Iterable[T | None] | T | None | 'OEExtensionArray[T]') -> 'OEExtensionArray[T]':
        raise NotImplemented(f'Subtraction not implemented for {type(self).__name__}')

    def __contains__(self, item: object) -> bool:
        # Handle None/NaN
        if pd.isna(item):
            return any(map(lambda obj: obj is None, self._objs))
        else:
            # Delegate to list of objects
            return item in self._objs

    def __setitem__(self, index, value) -> None:
        """
        Set an item in the array
        :param index: Item index
        :type index: int
        :param value: Item to set
        :type value: oechem.OEMolBase
        :return:
        """
        self._objs[index] = value

    def __getitem__(self, index) -> T | 'OEExtensionArray[T]':
        """
        Get an item in the array
        :param index: Item index
        :return: Item at index
        """
        if isinstance(index, int):
            return self._objs[index]
        return self.__class__(self._objs[index], metadata=shallow_copy(self.metadata))

    def __hash__(self):
        return hash(self._objs)

    def __reversed__(self):
        return self.__class__(reversed(self._objs), metadata=shallow_copy(self.metadata))

    def __str__(self):
        return f'{type(self).__name__}(len={len(self._objs)})'

    def __repr__(self):
        return self.__str__()
