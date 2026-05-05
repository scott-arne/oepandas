import pandas as pd
import pytest
from openeye import oechem

import oepandas as oepd
from oepandas import QueryArray, QueryDtype


def test_query_array_from_smarts_strings():
    arr = QueryArray.from_sequence_of_strings(["[#6]-[#8]"], query_format="smarts")

    assert isinstance(arr.dtype, QueryDtype)
    assert isinstance(arr[0], oechem.OEQMol)
    assert arr[0].IsValid()
    assert arr[0].NumAtoms() == 2


def test_query_array_from_smirks_strings():
    arr = QueryArray.from_sequence_of_strings(
        ["[#6:1]-[#8:2]>>[#6:1]=[#8:2]"],
        query_format=oepd.QueryFormat.SMIRKS,
    )

    assert isinstance(arr[0], oechem.OEQMol)
    assert arr[0].IsValid()
    assert arr[0].NumAtoms() == 4


def test_query_array_casts_existing_molecule_to_qmol():
    mol = oechem.OEMol()
    assert oechem.OEParseSmiles(mol, "CCO")

    arr = QueryArray([mol])

    assert isinstance(arr[0], oechem.OEQMol)
    assert oechem.OEMolToSmiles(arr[0]) == "CCO"


def test_query_array_from_sequence_of_strings_casts_existing_molecule_to_qmol():
    mol = oechem.OEMol()
    assert oechem.OEParseSmiles(mol, "CCO")

    arr = QueryArray.from_sequence_of_strings([mol])

    assert isinstance(arr[0], oechem.OEQMol)
    assert oechem.OEMolToSmiles(arr[0]) == "CCO"


def test_query_array_from_sequence_preserves_existing_qmol_identity_without_copy():
    query = oechem.OEQMol()
    assert oechem.OEParseSmarts(query, "[#6]")

    arr = QueryArray.from_sequence([query])

    assert arr[0] is query


def test_query_array_from_sequence_copies_existing_qmol_when_requested():
    query = oechem.OEQMol()
    assert oechem.OEParseSmarts(query, "[#6]")

    arr = QueryArray.from_sequence([query], copy=True)

    assert isinstance(arr[0], oechem.OEQMol)
    assert arr[0] is not query


def test_query_array_invalid_parse_becomes_missing():
    arr = QueryArray.from_sequence_of_strings(["not smarts"], query_format="smarts")

    assert arr[0] is None
    assert arr.isna().tolist() == [True]


def test_query_array_from_sequence_of_strings_rejects_invalid_objects():
    with pytest.raises(TypeError, match="Cannot create a query from object"):
        QueryArray.from_sequence_of_strings([object()])


def test_query_array_rejects_list_like_invalid_objects():
    with pytest.raises(TypeError, match="Cannot create QueryArray containing object of type list"):
        QueryArray([[1, 2]])


def test_query_array_from_sequence_rejects_list_like_invalid_objects():
    with pytest.raises(TypeError, match="Cannot create a query from list"):
        QueryArray.from_sequence([[1, 2]])


def test_query_array_rejects_invalid_query_format():
    with pytest.raises(ValueError, match="Unsupported query_format"):
        QueryArray.from_sequence_of_strings(["[#6]"], query_format="smi")


def test_query_dtype_round_trips_through_series():
    arr = QueryArray.from_sequence_of_strings(["[#6]"])
    series = pd.Series(arr, dtype=QueryDtype())

    assert isinstance(series.dtype, QueryDtype)
    assert isinstance(series.iloc[0], oechem.OEQMol)
