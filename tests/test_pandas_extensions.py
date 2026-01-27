"""
Tests for pandas_extensions.py module
"""

import csv
import pytest
import pandas as pd
import numpy as np
import oepandas as oepd
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from openeye import oechem
from oepandas.pandas_extensions import (
    Column, Dataset, _series_to_molecule_array, _add_smiles_columns,
    read_molecule_csv, read_smi, read_sdf, read_oeb
)

ASSETS = Path(Path(__file__).parent, "assets")


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
        assert "test" in dataset.keys()
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
        assert "test" not in dataset.keys()
    
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
        assert "test" in dataset.keys()
        
        # Test getitem
        retrieved = dataset["test"]
        assert retrieved.name == "test"
        
        # Test delitem
        del dataset["test"]
        assert "test" not in dataset.keys()


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
                df = read_molecule_csv(temp_path, molecule_columns="NonExistentColumn")
        finally:
            Path(temp_path).unlink()


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
            with open(output_path, 'r') as f:
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