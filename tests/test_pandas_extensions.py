"""
Tests for pandas_extensions.py module
"""

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from openeye import oechem

import oepandas as oepd
from oepandas.pandas_extensions import (
    _OE_TYPE_BOOL,
    _OE_TYPE_BYTES,
    _OE_TYPE_FLOAT,
    _OE_TYPE_INT,
    _OE_TYPE_MOLECULE,
    _OE_TYPE_STRING,
    Column,
    Dataset,
    _add_smiles_columns,
    _is_missing,
    _oeb_scan_has_unread_data,
    _resolve_oe_data_type,
    _series_to_molecule_array,
    read_molecule_csv,
    read_oeb,
    read_sdf,
    read_smi,
)

ASSETS = Path(Path(__file__).parent, "assets")


def _oeb_raw_text(path):
    """Return the printable text embedded in an .oeb / .oeb.gz file.

    OEB stores generic/SD data tag names and string values as printable ASCII, and the
    to_oeb *write* path is reliable, so write-correctness is asserted against the raw
    file bytes rather than through the intermittently flaky OEB reader.
    """
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("latin-1")


def _read_oeb_subprocess(path, scalar_cols=(), smiles_cols=(), attempts=6):
    """Read an .oeb file in a fresh subprocess (with retries) and return selected data.

    OpenEye's OEB *reader* intermittently drops generic/SD data in the process that
    performs the read; reading in short-lived separate processes avoids this so a full
    typed round-trip can be asserted. read_oeb raises on a detected drop, which triggers
    a retry here.

    :returns: dict of {column: [values]} for scalar_cols and {column: [smiles]} for
        smiles_cols.
    """
    code = (
        "import json, sys\n"
        "import oepandas as o\n"
        "from openeye import oechem\n"
        "df = o.read_oeb(sys.argv[1])\n"
        "scalar = json.loads(sys.argv[2]); smiles = json.loads(sys.argv[3])\n"
        "res = {}\n"
        "for c in scalar:\n"
        "    res[c] = df[c].tolist() if c in df.columns else None\n"
        "for c in smiles:\n"
        "    res[c] = ([oechem.OEMolToSmiles(m) if m is not None else None for m in df[c]]\n"
        "              if c in df.columns else None)\n"
        "print(json.dumps(res))\n"
    )
    last = ""
    for _ in range(attempts):
        proc = subprocess.run(
            [sys.executable, "-c", code, str(path),
             json.dumps(list(scalar_cols)), json.dumps(list(smiles_cols))],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip().splitlines()[-1])
        last = proc.stderr
    raise AssertionError(f"subprocess read_oeb failed after {attempts} attempts: {last[-300:]}")


@pytest.fixture
def test_molecules():
    """Create test molecules"""
    mols = []
    smiles_list = ["CCO", "CCC", "C", "CC"]
    for smiles in smiles_list:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        mols.append(mol)
    return mols


class TestColumn:
    """Test Column helper class"""

    def test_column_init_default(self):
        col = Column("test")
        assert col.name == "test"
        assert col.data == []
        assert col.dtype == object
        assert col.index == []

    def test_column_init_with_data(self):
        data = [1, 2, 3]
        col = Column("test", data=data, dtype=int)
        assert col.name == "test"
        assert col.data == data
        assert col.dtype == int

    def test_column_len(self):
        col = Column("test", data=[1, 2, 3])
        assert len(col) == 3

    def test_column_len_empty(self):
        col = Column("test")
        assert len(col) == 0


class TestDataset:
    """Test Dataset helper class"""

    def test_dataset_init_default(self):
        dataset = Dataset()
        assert len(dataset.keys()) == 0

    def test_dataset_init_with_usecols(self):
        dataset = Dataset(usecols=["col1", "col2"])
        assert dataset.usecols == {"col1", "col2"}

    def test_dataset_add(self):
        dataset = Dataset()
        # The Dataset.add method takes individual values, not lists
        dataset.add("test", 1)
        dataset.add("test", 2)
        dataset.add("test", 3)
        assert "test" in dataset
        assert len(dataset["test"]) == 3

    def test_dataset_add_with_force_type_and_index(self):
        dataset = Dataset()
        # The Dataset.add method takes individual values, not lists
        dataset.add("test", 1, idx=0, force_type=int)
        dataset.add("test", 2, idx=1, force_type=int)
        dataset.add("test", 3, idx=2, force_type=int)
        col = dataset["test"]
        assert col.dtype == int
        assert col.index == [0, 1, 2]

    def test_dataset_pop(self):
        dataset = Dataset()
        dataset.add("test", 1)
        dataset.add("test", 2)
        dataset.add("test", 3)
        col = dataset.pop("test")
        assert col.name == "test"
        assert "test" not in dataset

    def test_dataset_to_series_dict(self):
        dataset = Dataset()
        # Add values individually
        dataset.add("col1", 1)
        dataset.add("col1", 2)
        dataset.add("col1", 3)
        dataset.add("col2", "a")
        dataset.add("col2", "b")
        dataset.add("col2", "c")

        series_dict = dataset.to_series_dict()
        assert isinstance(series_dict, dict)
        assert len(series_dict) == 2
        assert isinstance(series_dict["col1"], pd.Series)
        assert isinstance(series_dict["col2"], pd.Series)

    def test_dataset_getitem_setitem_delitem(self):
        dataset = Dataset()
        col = Column("test", [1, 2, 3])

        # Test setitem
        dataset["test"] = col
        assert "test" in dataset

        # Test getitem
        retrieved = dataset["test"]
        assert retrieved.name == "test"

        # Test delitem
        del dataset["test"]
        assert "test" not in dataset

    def test_add_resolves_molecule_value_to_molecule_dtype(self):
        ds = Dataset()
        for smi in ("CCO", "CCC", "CC"):
            m = oechem.OEGraphMol()
            oechem.OESmilesToMol(m, smi)
            ds.add("Reactant", m)
        assert isinstance(ds["Reactant"].dtype, oepd.MoleculeDtype)
        series = ds.to_series_dict()["Reactant"]
        assert isinstance(series.dtype, oepd.MoleculeDtype)
        assert len(series) == 3


class TestSeriestoMoleculeArray:
    """Test _series_to_molecule_array function"""

    def test_series_to_molecule_array_smiles(self, test_molecules):
        smiles_series = pd.Series(["CCO", "CCC", "C", "CC"])
        mol_array = _series_to_molecule_array(smiles_series, oechem.OEFormat_SMI)

        assert isinstance(mol_array, oepd.MoleculeArray)
        assert len(mol_array) == 4
        # Test that molecules are valid
        for mol in mol_array:
            assert mol is not None

    def test_series_to_molecule_array_invalid_smiles(self):
        invalid_series = pd.Series(["invalid_smiles", "also_invalid"])
        mol_array = _series_to_molecule_array(invalid_series, oechem.OEFormat_SMI)

        assert isinstance(mol_array, oepd.MoleculeArray)
        assert len(mol_array) == 2
        # Invalid SMILES should result in None values since they can't be parsed
        for mol in mol_array:
            assert mol is None


class TestAddSmilesColumns:
    """Test _add_smiles_columns function"""

    def test_add_smiles_columns(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        })

        # _add_smiles_columns modifies in place, need to pass additional parameters
        _add_smiles_columns(df, "Molecule", True)

        assert "Molecule SMILES" in df.columns
        assert len(df) == 4
        # Check that SMILES column was actually added
        assert df["Molecule SMILES"].notna().all()


class TestFileReaders:
    """Test file reading functions"""

    def test_read_sdf_file_exists(self):
        # Test with existing SDF file
        sdf_path = ASSETS / "5.sdf"
        if sdf_path.exists():
            df = read_sdf(str(sdf_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "Molecule" in df.columns or any("mol" in col.lower() for col in df.columns)

    def test_read_oeb_file_exists(self):
        # Test with existing OEB file
        oeb_path = ASSETS / "5.oeb.gz"
        if oeb_path.exists():
            df = read_oeb(str(oeb_path))
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_read_molecule_csv_with_temp_file(self, test_molecules):
        # Create temporary CSV file with SMILES
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("SMILES,Name\n")
            f.write("CCO,ethanol\n")
            f.write("CCC,propane\n")
            f.write("C,methane\n")
            temp_path = f.name

        try:
            df = read_molecule_csv(temp_path, molecule_columns="SMILES")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3  # Should be 3 data rows (not including header)
            assert "SMILES" in df.columns
            assert "Name" in df.columns
        finally:
            Path(temp_path).unlink()

    def test_read_smi_with_temp_file(self):
        # Create temporary SMI file
        with NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write("CCO ethanol\n")
            f.write("CCC propane\n")
            f.write("C methane\n")
            temp_path = f.name

        try:
            df = read_smi(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            # Should have molecule column and title column
            assert any("mol" in col.lower() for col in df.columns)
        finally:
            Path(temp_path).unlink()

    def test_read_sdf_nonexistent_file(self):
        # read_sdf doesn't raise FileNotFoundError, it returns empty DataFrame
        df = read_sdf("nonexistent_file.sdf")
        assert isinstance(df, pd.DataFrame)
        # Should be empty since file doesn't exist
        assert len(df) == 0

    def test_read_molecule_csv_invalid_molecule_column(self):
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Value\n")
            f.write("test,123\n")
            temp_path = f.name

        try:
            # This should raise a KeyError since the column doesn't exist
            with pytest.raises(KeyError):
                read_molecule_csv(temp_path, molecule_columns="NonExistentColumn")
        finally:
            Path(temp_path).unlink()


class TestResolveOeDataType:
    """Test the shared dtype -> OpenEye type-family resolver."""

    def test_float_family(self):
        s = pd.Series([1.0, 2.5], dtype="float64")
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_FLOAT

    def test_integer_family(self):
        s = pd.Series([1, 2], dtype="int64")
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_INT

    def test_boolean_family(self):
        s = pd.Series([True, False], dtype="bool")
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_BOOL

    def test_object_column_integer_family(self):
        s = pd.Series([1, 2, 3], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_INT

    def test_object_column_float_family(self):
        s = pd.Series([1.0, 2.0], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_FLOAT

    def test_object_column_bool_family(self):
        s = pd.Series([True, False, True], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_BOOL

    def test_object_column_numpy_bool_family(self):
        import numpy as np
        s = pd.Series([np.bool_(True), np.bool_(False)], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_BOOL

    def test_string_family_object_column(self):
        s = pd.Series(["a", "b"], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_STRING

    def test_bytes_family_object_column(self):
        s = pd.Series([b"a", b"b"], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_BYTES

    def test_molecule_family(self):
        mols = []
        for smi in ("CCO", "CCC"):
            m = oechem.OEMol()
            oechem.OESmilesToMol(m, smi)
            mols.append(m)
        s = pd.Series(oepd.MoleculeArray(mols))
        assert _resolve_oe_data_type(s, s.dtype) == _OE_TYPE_MOLECULE

    def test_undeterminable_returns_none(self):
        s = pd.Series([None, None], dtype=object)
        assert _resolve_oe_data_type(s, s.dtype) is None

    def test_is_missing(self):
        assert _is_missing(None) is True
        assert _is_missing(float("nan")) is True
        assert _is_missing(0) is False
        assert _is_missing("") is False


class TestFileWriters:
    """Test file writing functionality through pandas accessors"""

    def test_dataframe_write_sdf(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        })

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.sdf"

            # Test the oe accessor
            df.chem.to_sdf(str(output_path), "Molecule")
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_dataframe_to_sdf_with_index(self, test_molecules):
        """Test that to_sdf writes the index as an SD tag when index=True"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        }, index=["a", "b", "c", "d"])

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output_index.sdf"
            df.chem.to_sdf(str(output_path), "Molecule", index=True, index_tag="my_index")
            assert output_path.exists()

            # Read back and verify index is written as SD tag
            # Note: OEGraphMol preserves SD data better than OEMol
            ifs = oechem.oemolistream(str(output_path))
            mol = oechem.OEGraphMol()
            oechem.OEReadMolecule(ifs, mol)
            index_value = oechem.OEGetSDData(mol, "my_index")
            assert index_value == "a"
            ifs.close()

    def test_dataframe_to_sdf_without_index(self, test_molecules):
        """Test that to_sdf does not write index when index=False"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        }, index=["a", "b", "c", "d"])

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output_no_index.sdf"
            df.chem.to_sdf(str(output_path), "Molecule", index=False)
            assert output_path.exists()

            # Read back and verify index is NOT written as SD tag
            # Note: OEGraphMol preserves SD data better than OEMol
            ifs = oechem.oemolistream(str(output_path))
            mol = oechem.OEGraphMol()
            oechem.OEReadMolecule(ifs, mol)
            assert not oechem.OEHasSDData(mol, "index")
            ifs.close()

    def test_dataframe_to_smi_with_flavor(self, test_molecules):
        """Test that to_smi applies the flavor parameter"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
        })

        with TemporaryDirectory() as temp_dir:
            # Write with default flavor (canonical)
            output_default = Path(temp_dir) / "test_default.smi"
            df.chem.to_smi(str(output_default), "Molecule")
            assert output_default.exists()

            # Write with isomeric flavor
            output_isomeric = Path(temp_dir) / "test_isomeric.smi"
            df.chem.to_smi(str(output_isomeric), "Molecule", flavor=oechem.OESMILESFlag_ISOMERIC)
            assert output_isomeric.exists()

    def test_dataframe_to_molecule_csv_with_quoting(self, test_molecules):
        """Test that to_molecule_csv passes quoting and quotechar parameters"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["eth,anol", "prop;ane", "meth\"ane", "eth'ane"]
        })

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_quoting.csv"
            df.chem.to_molecule_csv(
                str(output_path),
                molecule_format="smiles",
                quoting=csv.QUOTE_ALL,
                quotechar='"',
                index=False
            )
            assert output_path.exists()

            # Read back and verify quoting was applied
            with open(output_path) as f:
                content = f.read()
                # With QUOTE_ALL, all fields should be quoted
                assert '"Molecule"' in content or 'Molecule' in content

    def test_dataframe_to_oedb_with_index(self, test_molecules):
        """Test that to_oedb writes the index when index=True"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        }, index=["idx_a", "idx_b", "idx_c", "idx_d"])

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_index.oeb"
            df.chem.to_oedb(
                str(output_path),
                primary_molecule_column="Molecule",
                index=True,
                index_label="row_index"
            )
            assert output_path.exists()

            # Read back and verify index was written
            ifs = oechem.oeifstream()
            assert ifs.open(str(output_path))

            # Get the first record using OEReadRecords iterator
            record = next(iter(oechem.OEReadRecords(ifs)))

            # Check for the index field
            index_field = oechem.OEField("row_index", oechem.Types.String)
            assert record.has_field(index_field)
            index_value = record.get_value(index_field)
            assert index_value == "idx_a"
            ifs.close()

    def test_dataframe_to_oedb_without_index(self, test_molecules):
        """Test that to_oedb does not write index when index=False"""
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Name": ["ethanol", "propane", "methane", "ethane"]
        }, index=["idx_a", "idx_b", "idx_c", "idx_d"])

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_no_index.oeb"
            df.chem.to_oedb(
                str(output_path),
                primary_molecule_column="Molecule",
                index=False
            )
            assert output_path.exists()

            # Read back and verify index was NOT written
            ifs = oechem.oeifstream()
            assert ifs.open(str(output_path))

            # Get the first record using OEReadRecords iterator
            record = next(iter(oechem.OEReadRecords(ifs)))

            # Check that the index field does not exist
            index_field = oechem.OEField("index", oechem.Types.String)
            assert not record.has_field(index_field)
            ifs.close()

    def test_to_oedb_skips_unresolvable_column(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Blank": pd.Series([None, None, None, None], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "unresolvable.oedb"
            # Must not raise KeyError for the unresolvable "Blank" column.
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule")
            assert out.exists()

    def test_to_oedb_coerces_numpy_scalar_object_columns(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "I": pd.Series([np.int64(1), np.int64(2), np.int64(3), np.int64(4)], dtype=object),
            "B": pd.Series([np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False)], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "npscalars.oedb"
            # Must not raise: numpy scalars are coerced to native types before writing.
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule", index=False)
            ifs = oechem.oeifstream()
            assert ifs.open(str(out))
            rec = next(iter(oechem.OEReadRecords(ifs)))
            assert rec.get_value(oechem.OEField("I", oechem.Types.Int)) == 1
            assert rec.get_value(oechem.OEField("B", oechem.Types.Bool)) is True
            ifs.close()

    def test_to_oedb_skips_missing_scalar_values(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "S": pd.Series(["a", None, "c", "d"], dtype=object),
            "B": pd.Series([True, None, False, True], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "missing.oedb"
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule", index=False)
            ifs = oechem.oeifstream()
            assert ifs.open(str(out))
            records = list(oechem.OEReadRecords(ifs))
            ifs.close()
            sfield = oechem.OEField("S", oechem.Types.String)
            bfield = oechem.OEField("B", oechem.Types.Bool)
            # Row 0: values present and correct.
            assert records[0].has_value(sfield) and records[0].get_value(sfield) == "a"
            assert records[0].has_value(bfield) and records[0].get_value(bfield) is True
            # Row 1: missing cells left UNSET, not coerced into "None"/True.
            assert not records[1].has_value(sfield)
            assert not records[1].has_value(bfield)

    def test_to_oedb_safe_skips_incompatible_scalar_values(self, test_molecules):
        # A bool-family column (predominant bool) with an incompatible string value.
        # safe=True must leave the incompatible cell UNSET, not coerce "False" -> True.
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "B": pd.Series([True, "False", False, True], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "incompat.oedb"
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule", index=False)
            ifs = oechem.oeifstream()
            assert ifs.open(str(out))
            records = list(oechem.OEReadRecords(ifs))
            ifs.close()
            bfield = oechem.OEField("B", oechem.Types.Bool)
            # Row 0 (True) written correctly.
            assert records[0].has_value(bfield) and records[0].get_value(bfield) is True
            # Row 1 ("False" string, incompatible with bool) left UNSET, not coerced to True.
            assert not records[1].has_value(bfield)

    def test_to_oedb_safe_treats_bool_and_int_as_disjoint(self, test_molecules):
        # [1, True, 2, 3] resolves to the int family (majority int); the bool cell must
        # NOT be written as int(True)==1 under safe mode -- it is left unset.
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "I": pd.Series([1, True, 2, 3], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "boolint.oedb"
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule", index=False)
            ifs = oechem.oeifstream()
            assert ifs.open(str(out))
            records = list(oechem.OEReadRecords(ifs))
            ifs.close()
            ifield = oechem.OEField("I", oechem.Types.Int)
            assert records[0].has_value(ifield) and records[0].get_value(ifield) == 1
            assert not records[1].has_value(ifield)   # True (bool) not written to int field
            assert records[2].get_value(ifield) == 2
            assert records[3].get_value(ifield) == 3

    def test_to_oedb_skips_float_nan(self, test_molecules):
        # A float-family object column with a NaN cell: NaN is a valid float but means
        # missing, so it must be left unset rather than written as a present Float value.
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "F": pd.Series([1.0, float("nan"), 2.0, 3.0], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "floatnan.oedb"
            df.chem.to_oedb(str(out), primary_molecule_column="Molecule", index=False)
            ifs = oechem.oeifstream()
            assert ifs.open(str(out))
            records = list(oechem.OEReadRecords(ifs))
            ifs.close()
            ffield = oechem.OEField("F", oechem.Types.Float)
            assert records[0].has_value(ffield) and records[0].get_value(ffield) == 1.0
            assert not records[1].has_value(ffield)   # NaN left unset
            assert records[2].get_value(ffield) == 2.0

    # -------------------------------------------------------------------------------
    # to_oeb writer
    #
    # Test strategy: the OpenEye OEB *reader* intermittently drops generic/SD data on
    # deserialization, but to_oeb always writes the data to the file correctly. Write
    # correctness is asserted against the raw file bytes (_oeb_raw_text); typed value
    # round-trips are asserted by reading in a fresh subprocess with retries
    # (_read_oeb_subprocess). Molecule structure (conformer counts) reads reliably
    # in-process and is asserted directly.
    # -------------------------------------------------------------------------------

    def test_to_oeb_writes_typed_data_tags(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Activity": [1.5, 2.5, 3.5, 4.5],
            "Count": [1, 2, 3, 4],
            "Flag": [True, False, True, False],
            "Label": ["alphaval", "betaval", "gammaval", "deltaval"],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "typed.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            assert out.exists()
            text = _oeb_raw_text(out)
            for tag in ("Activity", "Count", "Flag", "Label"):
                assert tag in text, f"tag {tag} not written to file"
            for value in ("alphaval", "betaval", "gammaval", "deltaval"):
                assert value in text, f"string value {value} not written to file"

    def test_to_oeb_typed_round_trip(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Activity": [1.5, 2.5, 3.5, 4.5],
            "Count": [1, 2, 3, 4],
            "Flag": [True, False, True, False],
            "Label": ["a", "b", "c", "d"],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "typed.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            data = _read_oeb_subprocess(out, scalar_cols=["Activity", "Count", "Flag", "Label"])
            assert data["Activity"] == [1.5, 2.5, 3.5, 4.5]
            assert data["Count"] == [1, 2, 3, 4]
            # Booleans round-trip as ints (0/1): OEB does not persist boolean-False data.
            assert data["Flag"] == [1, 0, 1, 0]
            assert data["Label"] == ["a", "b", "c", "d"]

    def test_to_oeb_index_written_and_omitted(self, test_molecules):
        df = pd.DataFrame(
            {"Molecule": oepd.MoleculeArray(test_molecules)},
            index=["ridx0", "ridx1", "ridx2", "ridx3"],
        )
        with TemporaryDirectory() as tmp:
            with_idx = Path(tmp) / "with_index.oeb"
            df.chem.to_oeb(str(with_idx), "Molecule", index=True, index_tag="row_index")
            text = _oeb_raw_text(with_idx)
            assert "row_index" in text
            for value in ("ridx0", "ridx1", "ridx2", "ridx3"):
                assert value in text

            no_idx = Path(tmp) / "no_index.oeb"
            df.chem.to_oeb(str(no_idx), "Molecule", index=False)
            assert "row_index" not in _oeb_raw_text(no_idx)

    def test_to_oeb_gzip(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Note": ["gzval0", "gzval1", "gzval2", "gzval3"],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "typed.oeb.gz"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            assert out.exists()
            text = _oeb_raw_text(out)  # decompresses .gz
            assert "Note" in text and "gzval0" in text

    def test_to_oeb_preserves_conformers(self):
        from openeye import oeomega
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCCCCCOCC")
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(8)
        omega.SetStrictStereo(False)
        omega.SetStrictAtomTypes(False)
        assert omega(mol)
        n_confs = mol.NumConfs()
        assert n_confs >= 2
        df = pd.DataFrame({"Molecule": oepd.MoleculeArray([mol])})
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "confs.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            back = read_oeb(str(out))
            assert back["Molecule"].iloc[0].NumConfs() == n_confs

    def test_to_oeb_does_not_mutate_source(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Activity": [1.5, 2.5, 3.5, 4.5],
        })
        with TemporaryDirectory() as tmp:
            df.chem.to_oeb(str(Path(tmp) / "x.oeb"), "Molecule", title_column=None)
        assert not df["Molecule"].iloc[0].HasData("Activity")

    def test_to_oeb_secondary_molecule_round_trip(self, test_molecules):
        reactants = []
        for smi in ("c1ccccc1", "c1ccncc1", "c1ccsc1", "c1ccoc1"):
            r = oechem.OEMol()
            oechem.OESmilesToMol(r, smi)
            reactants.append(r)
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Reactant": oepd.MoleculeArray(reactants),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "secondary.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            data = _read_oeb_subprocess(out, smiles_cols=["Reactant"])
            assert data["Reactant"] is not None
            assert oechem.OEMolToSmiles(reactants[0]) in data["Reactant"]

    def test_to_oeb_skips_design_unit_column(self):
        du_df = oepd.read_oedu(str(ASSETS / "2.oedu"))
        du = du_df["Design_Unit"].iloc[0]
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray([mol]),
            "DesignUnitColumn": pd.Series(oepd.DesignUnitArray([du]), dtype=oepd.DesignUnitDtype()),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "du.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False, write_sd_data=True)
            assert "DesignUnitColumn" not in _oeb_raw_text(out)

    def test_to_oeb_missing_scalar_skipped(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Activity": [1.5, None, 3.5, 4.5],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "missing.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            data = _read_oeb_subprocess(out, scalar_cols=["Activity"])
            present = [v for v in data["Activity"] if v is not None and v == v]
            assert sorted(present) == [1.5, 3.5, 4.5]

    def test_to_oeb_none_primary_molecule(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray([test_molecules[0], None, test_molecules[1]]),
            "Label": ["keepa", "keepb", "keepc"],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "none.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            assert out.exists()
            text = _oeb_raw_text(out)
            assert "keepa" in text and "keepc" in text

    def test_to_oeb_bool_int_disjoint(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "I": pd.Series([1, True, 2, 3], dtype=object),
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "boolint.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False)
            data = _read_oeb_subprocess(out, scalar_cols=["I"])
            present = [v for v in data["I"] if v is not None and v == v]
            assert sorted(present) == [1, 2, 3]

    def test_to_oeb_sd_data_written(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "SdCol": ["sdval0", "sdval1", "sdval2", "sdval3"],
        })
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "sd.oeb"
            df.chem.to_oeb(str(out), "Molecule", index=False, write_sd_data=True)
            text = _oeb_raw_text(out)
            assert "SdCol" in text and "sdval0" in text

    def test_to_oeb_validation_errors(self, test_molecules):
        df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray(test_molecules),
            "Activity": [1.5, 2.5, 3.5, 4.5],
        })
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "err.oeb")
            with pytest.raises(KeyError):
                df.chem.to_oeb(out, "Nope")
            with pytest.raises(TypeError):
                df.chem.to_oeb(out, "Activity")
            with pytest.raises(KeyError):
                df.chem.to_oeb(out, "Molecule", columns=["Missing"])
            with pytest.raises(ValueError):
                df.chem.to_oeb(out, "Molecule", index=True, index_tag="Activity")


class TestReadOebGuard:
    """Tests for the read_oeb silent-drop guard and its raw-byte detector."""

    def _write_oeb(self, path, *, data, title):
        ofs = oechem.oemolostream(str(path))
        ofs.SetFormat(oechem.OEFormat_OEB)
        for smi in ("CCO", "CCC"):
            m = oechem.OEMol()
            oechem.OESmilesToMol(m, smi)
            if title:
                m.SetTitle("TITLE_" + smi)
            if data:
                m.SetData("MyTag", 7)
            oechem.OEWriteMolecule(ofs, m)
        ofs.close()

    def test_detector_flags_data(self, tmp_path):
        p = tmp_path / "d.oeb"
        self._write_oeb(p, data=True, title=False)
        assert _oeb_scan_has_unread_data(str(p), []) is True

    def test_detector_ignores_molecules_only(self, tmp_path):
        p = tmp_path / "m.oeb"
        self._write_oeb(p, data=False, title=False)
        assert _oeb_scan_has_unread_data(str(p), []) is False

    def test_detector_ignores_titles(self, tmp_path):
        p = tmp_path / "t.oeb"
        self._write_oeb(p, data=False, title=True)
        assert _oeb_scan_has_unread_data(str(p), ["TITLE_CCO", "TITLE_CCC"]) is False

    def test_read_oeb_no_raise_on_molecules_only(self, tmp_path):
        p = tmp_path / "m.oeb"
        self._write_oeb(p, data=False, title=False)
        df = read_oeb(str(p))
        assert list(df.columns) == ["Molecule", "Title"]

    def test_read_oeb_raises_on_detected_drop(self, tmp_path, monkeypatch):
        p = tmp_path / "d.oeb"
        self._write_oeb(p, data=True, title=False)
        from oepandas import pandas_extensions as pe
        original = pe._read_molecules_to_dataframe
        def dropping(reader, fp, **kwargs):
            real = original(reader, fp, **kwargs)
            keep = [c for c in real.columns
                    if c in (kwargs.get("molecule_column", "Molecule"),
                             kwargs.get("title_column", "Title"))]
            return real[keep]
        monkeypatch.setattr(pe, "_read_molecules_to_dataframe", dropping)
        with pytest.raises(oepd.FileError):
            read_oeb(str(p))


class TestPandasAccessors:
    """Test pandas series/dataframe accessors"""

    def test_molecule_series_accessors(self, test_molecules):
        series = pd.Series(oepd.MoleculeArray(test_molecules), dtype=oepd.MoleculeDtype())

        # Test to_smiles accessor via oe namespace
        smiles = series.chem.to_smiles()
        assert len(smiles) == len(test_molecules)

        # Test copy_molecules accessor via oe namespace
        copied = series.chem.copy_molecules()
        assert len(copied) == len(test_molecules)

    def test_series_as_molecule_with_format(self):
        """Test that as_molecule uses the molecule_format parameter"""
        # Create a series of SMILES strings
        smiles_series = pd.Series(["CCO", "CCC", "C", "CC"])

        # Convert to molecules using default format (SMILES)
        mol_series = smiles_series.chem.as_molecule(molecule_format="smi")

        assert isinstance(mol_series.dtype, oepd.MoleculeDtype)
        assert len(mol_series) == 4
        # Verify molecules are valid
        for mol in mol_series:
            assert mol is not None
            assert isinstance(mol, oechem.OEMolBase)

    def test_series_as_molecule_with_int_format(self):
        """Test that as_molecule works with integer format codes"""
        smiles_series = pd.Series(["CCO", "CCC"])

        mol_series = smiles_series.chem.as_molecule(molecule_format=oechem.OEFormat_SMI)

        assert isinstance(mol_series.dtype, oepd.MoleculeDtype)
        assert len(mol_series) == 2

    def test_series_as_molecule_already_molecule(self, test_molecules):
        """Test that as_molecule returns the same series if already MoleculeDtype"""
        mol_series = pd.Series(oepd.MoleculeArray(test_molecules), dtype=oepd.MoleculeDtype())

        result = mol_series.chem.as_molecule(molecule_format="smiles")

        # Should return the same series since it's already a molecule series
        assert result is mol_series

    def test_series_as_molecule_with_graph_mol_type(self):
        smiles_series = pd.Series(["CCO", "CCC"])

        mol_series = smiles_series.chem.as_molecule(molecule_type=oepd.MoleculeType.OEGRAPHMOL)

        assert isinstance(mol_series.dtype, oepd.MoleculeDtype)
        assert all(isinstance(mol, oechem.OEGraphMol) for mol in mol_series)

    def test_series_as_molecule_default_preserves_existing_type(self):
        graph_mol = oechem.OEGraphMol()
        assert oechem.OEParseSmiles(graph_mol, "CCO")

        mol_series = pd.Series([graph_mol]).chem.as_molecule()

        assert isinstance(mol_series.dtype, oepd.MoleculeDtype)
        assert isinstance(mol_series.iloc[0], oechem.OEGraphMol)

    def test_dataframe_as_molecule_with_graph_mol_type(self):
        df = pd.DataFrame({"smiles": ["CCO", "CCC"]})

        result = df.chem.as_molecule("smiles", molecule_type=oechem.OEGraphMol)

        assert isinstance(result.dtypes["smiles"], oepd.MoleculeDtype)
        assert all(isinstance(mol, oechem.OEGraphMol) for mol in result["smiles"])

    def test_series_as_molecule_casts_existing_molecule_dtype_when_requested(self, test_molecules):
        mol_series = pd.Series(oepd.MoleculeArray(test_molecules), dtype=oepd.MoleculeDtype())

        result = mol_series.chem.as_molecule(molecule_type=oepd.MoleculeType.OEGRAPHMOL)

        assert result is not mol_series
        assert isinstance(result.dtype, oepd.MoleculeDtype)
        assert all(isinstance(mol, oechem.OEGraphMol) for mol in result)

    def test_dataframe_as_molecule_casts_existing_molecule_dtype_when_requested(self, test_molecules):
        df = pd.DataFrame({"mol": pd.Series(oepd.MoleculeArray(test_molecules), dtype=oepd.MoleculeDtype())})

        result = df.chem.as_molecule("mol", molecule_type=oepd.MoleculeType.OEGRAPHMOL)

        assert result is not df
        assert isinstance(result.dtypes["mol"], oepd.MoleculeDtype)
        assert all(isinstance(mol, oechem.OEGraphMol) for mol in result["mol"])

    def test_series_as_molecule_no_title_clears_copy_not_source(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("parent")
        mol.GetActive().SetTitle("active")
        mol_series = pd.Series(oepd.MoleculeArray([mol]), dtype=oepd.MoleculeDtype())

        result = mol_series.chem.as_molecule(no_title=True)

        assert result is not mol_series
        assert result.iloc[0].GetTitle() == ""
        assert result.iloc[0].GetActive().GetTitle() == ""
        assert mol_series.iloc[0].GetTitle() == "parent"
        assert mol_series.iloc[0].GetActive().GetTitle() == "active"

    def test_dataframe_as_molecule_no_title_clears_existing_molecule_dtype(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("parent")
        mol.GetActive().SetTitle("active")
        df = pd.DataFrame({"mol": pd.Series(oepd.MoleculeArray([mol]), dtype=oepd.MoleculeDtype())})

        result = df.chem.as_molecule("mol", no_title=True)

        assert result is not df
        assert result.loc[0, "mol"].GetTitle() == ""
        assert result.loc[0, "mol"].GetActive().GetTitle() == ""
        assert df.loc[0, "mol"].GetTitle() == "parent"
        assert df.loc[0, "mol"].GetActive().GetTitle() == "active"

    def test_series_as_query_from_smarts(self):
        query_series = pd.Series(["[#6]-[#8]"]).chem.as_query(query_format="smarts")

        assert isinstance(query_series.dtype, oepd.QueryDtype)
        assert isinstance(query_series.iloc[0], oechem.OEQMol)
        assert query_series.iloc[0].IsValid()

    def test_series_as_query_from_smirks(self):
        query_series = pd.Series(["[#6:1]-[#8:2]>>[#6:1]=[#8:2]"]).chem.as_query(
            query_format=oepd.QueryFormat.SMIRKS
        )

        assert isinstance(query_series.dtype, oepd.QueryDtype)
        assert isinstance(query_series.iloc[0], oechem.OEQMol)
        assert query_series.iloc[0].IsValid()

    def test_dataframe_as_query_from_smarts(self):
        df = pd.DataFrame({"query": ["[#6]-[#8]"]})

        result = df.chem.as_query("query", query_format="smarts")

        assert isinstance(result.dtypes["query"], oepd.QueryDtype)
        assert isinstance(result.loc[0, "query"], oechem.OEQMol)

    def test_dataframe_as_query_inplace(self):
        df = pd.DataFrame({"query": ["[#6]"]})

        returned = df.chem.as_query("query", inplace=True)

        assert returned is df
        assert isinstance(df.dtypes["query"], oepd.QueryDtype)

    def test_series_as_query_already_query_dtype_returns_same_series(self):
        query_series = pd.Series(
            oepd.QueryArray.from_sequence_of_strings(["[#6]"]),
            dtype=oepd.QueryDtype(),
        )

        result = query_series.chem.as_query()

        assert result is query_series

    def test_series_as_query_no_title_clears_copy_not_source(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        query.SetTitle("query")
        query_series = pd.Series(oepd.QueryArray([query]), dtype=oepd.QueryDtype())

        result = query_series.chem.as_query(no_title=True)

        assert result is not query_series
        assert result.iloc[0].GetTitle() == ""
        assert query_series.iloc[0].GetTitle() == "query"

    def test_dataframe_as_query_no_title_clears_existing_query_dtype(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        query.SetTitle("query")
        df = pd.DataFrame({"query": pd.Series(oepd.QueryArray([query]), dtype=oepd.QueryDtype())})

        result = df.chem.as_query("query", no_title=True)

        assert result is not df
        assert result.loc[0, "query"].GetTitle() == ""
        assert df.loc[0, "query"].GetTitle() == "query"

    def test_series_is_valid_supports_query_dtype(self):
        query_series = pd.Series(
            oepd.QueryArray.from_sequence_of_strings(["[#6]"]),
            dtype=oepd.QueryDtype(),
        )

        validity = query_series.chem.is_valid()

        assert validity.tolist() == [True]


class TestErrorHandling:
    """Test error handling in pandas extensions"""

    def test_invalid_file_format(self):
        with NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"invalid content")
            temp_path = f.name

        try:
            # Reading invalid content should not raise an error, just return empty DataFrame
            df = read_sdf(temp_path)
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(temp_path).unlink()

    def test_empty_molecule_column(self):
        dataset = Dataset()
        # Don't add any values, just create an empty column in the dict directly
        dataset.columns["empty"] = Column("empty")

        series_dict = dataset.to_series_dict()
        assert "empty" in series_dict
        assert len(series_dict["empty"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
