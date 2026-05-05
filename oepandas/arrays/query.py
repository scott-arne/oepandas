import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, Self, TypeAlias, cast

import numpy as np
import pandas as pd
from openeye import oechem

# noinspection PyProtectedMember
from pandas._typing import Dtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype

from .base import OEExtensionArray

log = logging.getLogger("oepandas")


class QueryFormat(StrEnum):
    """
    Query string format to use when constructing query arrays.
    """

    SMARTS = "smarts"
    SMIRKS = "smirks"


QueryFormatInput: TypeAlias = QueryFormat | str

_QUERY_FORMAT_BY_STRING: dict[str, QueryFormat] = {
    query_format.value: query_format for query_format in QueryFormat
}


def _normalize_query_format(query_format: QueryFormatInput) -> QueryFormat:
    """
    Normalize user query format input.

    :param query_format: Query format selector.
    :returns: Normalized query format.
    :raises ValueError: When query format is unsupported.
    """
    if isinstance(query_format, QueryFormat):
        return query_format

    if isinstance(query_format, str):
        key = query_format.casefold()
        if key in _QUERY_FORMAT_BY_STRING:
            return _QUERY_FORMAT_BY_STRING[key]

    raise ValueError(f"Unsupported query_format: {query_format!r}")


def _parse_query_string(
        value: str,
        query_format: QueryFormatInput = QueryFormat.SMARTS,
) -> oechem.OEQMol | None:
    """
    Parse a query string into an OpenEye query molecule.

    :param value: Query string to parse.
    :param query_format: Query string format.
    :returns: Parsed query molecule, or ``None`` when parsing fails.
    """
    normalized_query_format = _normalize_query_format(query_format)
    query = oechem.OEQMol()

    if normalized_query_format == QueryFormat.SMARTS:
        parsed = oechem.OEParseSmarts(query, value)
    else:
        parsed = oechem.OEParseSmirks(query, value)

    if not parsed:
        log.warning(
            "Could not convert query from '%s': %s",
            normalized_query_format.value,
            value,
        )
        return None

    return query


def _is_missing_query_value(value: object) -> bool:
    """
    Check whether a value should be treated as a missing query value.

    :param value: Value to check.
    :returns: ``True`` when the value is a scalar missing value.
    """
    if isinstance(value, oechem.OEMolBase):
        return False

    try:
        is_missing = pd.isna(cast(Any, value))
    except (TypeError, ValueError):
        return False

    if isinstance(is_missing, (bool, np.bool_)):
        return bool(is_missing)

    return False


class QueryArray(OEExtensionArray[oechem.OEQMol]):
    """
    Custom extension array for OpenEye query molecules.
    """

    _base_openeye_type = oechem.OEQMol

    def __init__(
            self,
            queries: None | oechem.OEMolBase | Iterable[oechem.OEMolBase | None],
            copy: bool = False,
            metadata: dict | None = None,
            deepcopy: bool = False,
    ):
        """
        Initialize a query array.

        :param queries: Query molecule, molecule, or iterable of molecules.
        :param copy: Create copies of query molecules if ``True``.
        :param metadata: Optional array metadata.
        :param deepcopy: Deep-copy existing query molecules before storage.
        """
        processed: list[oechem.OEQMol | None] = []

        if queries is not None:
            if isinstance(queries, oechem.OEMolBase):
                queries = (queries,)

            for i, query in enumerate(queries):
                if isinstance(query, oechem.OEQMol):
                    processed.append(query.CreateCopy() if deepcopy else query)
                elif isinstance(query, oechem.OEMolBase):
                    processed.append(oechem.OEQMol(query))
                elif _is_missing_query_value(query):
                    processed.append(None)
                else:
                    raise TypeError(
                        f"Cannot create QueryArray containing object of type {type(query).__name__} "
                        f"at index {i}. All elements must be OEMolBase or None/NaN."
                    )

        super().__init__(processed, copy=copy, metadata=metadata)

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,  # noqa
            copy: bool = False,
            query_format: QueryFormatInput = QueryFormat.SMARTS,
    ) -> Self:
        """
        Initialize from a sequence of scalar values.

        :param scalars: Scalars to convert to query molecules.
        :param dtype: Not used; present for Pandas API compatibility.
        :param copy: Copy query molecules before storage.
        :param query_format: Query string format for string scalars.
        :returns: New query array.
        """
        normalized_query_format = _normalize_query_format(query_format)
        queries: list[oechem.OEQMol | None] = []

        for obj in scalars:
            if isinstance(obj, str):
                queries.append(_parse_query_string(obj.strip(), normalized_query_format))
            elif isinstance(obj, oechem.OEQMol):
                queries.append(obj)
            elif isinstance(obj, oechem.OEMolBase):
                queries.append(oechem.OEQMol(obj))
            elif _is_missing_query_value(obj):
                queries.append(None)
            else:
                raise TypeError(f"Cannot create a query from {type(obj).__name__}")

        return cls(queries, copy=copy)

    @classmethod
    def from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            query_format: QueryFormatInput = QueryFormat.SMARTS,
    ) -> Self:
        """
        Public alias of ``_from_sequence``.

        :param scalars: Scalars to convert to query molecules.
        :param dtype: Not used; present for Pandas API compatibility.
        :param copy: Copy query molecules before storage.
        :param query_format: Query string format for string scalars.
        :returns: New query array.
        """
        return cls._from_sequence(
            scalars,
            dtype=dtype,
            copy=copy,
            query_format=query_format,
        )

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            query_format: QueryFormatInput = QueryFormat.SMARTS,
    ) -> Self:
        """
        Read query molecules from a sequence of string-like input.

        :param strings: Query strings, molecules, or missing values to convert.
        :param dtype: Not used; present for Pandas API compatibility.
        :param copy: Copy query molecules before storage.
        :param query_format: Query string format.
        :returns: New query array.
        """
        return cls._from_sequence(
            strings,
            dtype=dtype,
            copy=copy,
            query_format=query_format,
        )

    @classmethod
    def from_sequence_of_strings(
            cls,
            strings: Sequence[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False,
            query_format: QueryFormatInput = QueryFormat.SMARTS,
    ) -> Self:
        """
        Public alias of ``_from_sequence_of_strings``.

        :param strings: Query strings, molecules, or missing values to convert.
        :param dtype: Not used; present for Pandas API compatibility.
        :param copy: Copy query molecules before storage.
        :param query_format: Query string format.
        :returns: New query array.
        """
        return cls._from_sequence_of_strings(
            strings,
            dtype=dtype,
            copy=copy,
            query_format=query_format,
        )

    @property
    def dtype(self) -> PandasExtensionDtype:
        return QueryDtype()

    def deepcopy(self, metadata: bool | dict | None = True):
        """
        Return a deep copy of this query array.

        :param metadata: Metadata copy behavior.
        :returns: Deep copy of this query array.
        """
        return QueryArray(self._objs, metadata=self._resolve_metadata(metadata), deepcopy=True)


@register_extension_dtype
class QueryDtype(PandasExtensionDtype):
    """
    OpenEye query molecule datatype for Pandas.
    """

    type: type = oechem.OEQMol  # noqa
    name: str = "query"  # noqa
    kind: str = "O"
    base = np.dtype("O")

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return QueryArray

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: str | type) -> bool:
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, type(self))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
