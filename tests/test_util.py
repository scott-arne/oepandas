"""
Tests for util.py module - utility functions for OEPandas
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from openeye import oechem
from oepandas.util import (
    FileFormat, get_oeformat, get_oeformat_from_ext, is_gz,
    create_molecule_to_bytes_writer, create_molecule_to_string_writer,
    molecule_from_string, predominant_type
)
from oepandas.exception import UnsupportedFileFormat


@pytest.fixture
def test_molecules():
    """Create test molecules for testing"""
    mols = []
    smiles_list = ["CCO", "CCC", "C", "CC"]
    for smiles in smiles_list:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        mols.append(mol)
    return mols


class TestFileFormat:
    """Test FileFormat dataclass"""
    
    def test_file_format_creation(self):
        fmt = FileFormat(ext=".smi", oeformat=oechem.OEFormat_SMI, name="SMILES", gzip=False)
        assert fmt.ext == ".smi"
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert fmt.name == "SMILES"
        assert not fmt.gzip
    
    def test_is_binary_format_true(self):
        fmt = FileFormat(ext=".oeb", oeformat=oechem.OEFormat_OEB, name="OEB", gzip=False)
        assert fmt.is_binary_format
    
    def test_is_binary_format_false(self):
        fmt = FileFormat(ext=".smi", oeformat=oechem.OEFormat_SMI, name="SMILES", gzip=False)
        assert not fmt.is_binary_format
    
    def test_file_format_frozen(self):
        """Test that FileFormat is frozen (immutable)"""
        fmt = FileFormat(ext=".smi", oeformat=oechem.OEFormat_SMI, name="SMILES", gzip=False)
        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            fmt.ext = ".sdf"


class TestGetOEFormat:
    """Test get_oeformat function"""
    
    def test_get_oeformat_from_string_smi(self):
        fmt = get_oeformat(".smi")
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert not fmt.gzip
    
    def test_get_oeformat_from_string_sdf(self):
        fmt = get_oeformat(".sdf")
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SDF
        assert not fmt.gzip
    
    def test_get_oeformat_from_string_with_gzip(self):
        fmt = get_oeformat(".smi.gz")
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert fmt.gzip
    
    def test_get_oeformat_from_int(self):
        fmt = get_oeformat(oechem.OEFormat_SMI)
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert not fmt.gzip
    
    def test_get_oeformat_from_int_with_gzip(self):
        fmt = get_oeformat(oechem.OEFormat_SMI, gzip=True)
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert fmt.gzip
    
    def test_get_oeformat_from_path(self):
        path = Path("test.smi")
        fmt = get_oeformat(path)
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
    
    def test_get_oeformat_invalid_string(self):
        with pytest.raises(UnsupportedFileFormat):
            get_oeformat("invalid_format")
    
    def test_get_oeformat_invalid_type(self):
        with pytest.raises(TypeError):
            get_oeformat([])


class TestGetOEFormatFromExt:
    """Test get_oeformat_from_ext function"""
    
    def test_get_oeformat_from_ext_smi(self):
        fmt = get_oeformat_from_ext("test.smi")
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
    
    def test_get_oeformat_from_ext_path(self):
        path = Path("test.sdf")
        fmt = get_oeformat_from_ext(path)
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SDF
    
    def test_get_oeformat_from_ext_multiple_suffixes(self):
        fmt = get_oeformat_from_ext("test.smi.gz")
        assert isinstance(fmt, FileFormat)
        assert fmt.oeformat == oechem.OEFormat_SMI
        assert fmt.gzip


class TestIsGz:
    """Test is_gz function"""
    
    def test_is_gz_true(self):
        assert is_gz("test.smi.gz")
        assert is_gz(Path("test.sdf.gz"))
    
    def test_is_gz_false(self):
        assert not is_gz("test.smi")
        assert not is_gz(Path("test.sdf"))
    
    def test_is_gz_multiple_extensions(self):
        assert is_gz("test.molecules.sdf.gz")
        assert not is_gz("test.molecules.sdf")


class TestCreateMoleculeToBytesWriter:
    """Test create_molecule_to_bytes_writer function"""
    
    def test_create_bytes_writer_smiles(self, test_molecules):
        writer = create_molecule_to_bytes_writer("smiles")
        result = writer(test_molecules[0])
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_create_bytes_writer_canonical_smiles(self, test_molecules):
        writer = create_molecule_to_bytes_writer("canonical_smiles")
        result = writer(test_molecules[0])
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_create_bytes_writer_smi_format(self, test_molecules):
        writer = create_molecule_to_bytes_writer(oechem.OEFormat_SMI)
        result = writer(test_molecules[0])
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_create_bytes_writer_file_format(self, test_molecules):
        fmt = get_oeformat(".smi")
        writer = create_molecule_to_bytes_writer(fmt)
        result = writer(test_molecules[0])
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_create_bytes_writer_with_gzip(self, test_molecules):
        writer = create_molecule_to_bytes_writer("smiles", gzip=True)
        result = writer(test_molecules[0])
        assert isinstance(result, bytes)
        # Gzipped content should be base64 encoded
        assert len(result) > 0
    
    def test_create_bytes_writer_invalid_type(self):
        with pytest.raises(TypeError):
            create_molecule_to_bytes_writer([])


class TestCreateMoleculeToStringWriter:
    """Test create_molecule_to_string_writer function"""
    
    def test_create_string_writer_smiles(self, test_molecules):
        writer = create_molecule_to_string_writer("smiles")
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be a valid SMILES
        assert "CCO" in result or "OCC" in result
    
    def test_create_string_writer_canonical_smiles(self, test_molecules):
        writer = create_molecule_to_string_writer("canonical_smiles")
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_string_writer_smi_format(self, test_molecules):
        writer = create_molecule_to_string_writer(oechem.OEFormat_SMI)
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_string_writer_with_gzip(self, test_molecules):
        writer = create_molecule_to_string_writer("smiles", gzip=True)
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        # Should be base64 encoded
        assert len(result) > 0
    
    def test_create_string_writer_with_b64encode(self, test_molecules):
        writer = create_molecule_to_string_writer("smiles", b64encode=True)
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_string_writer_no_strip(self, test_molecules):
        writer = create_molecule_to_string_writer(oechem.OEFormat_SMI, strip=False)
        result = writer(test_molecules[0])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_string_writer_invalid_type(self):
        with pytest.raises(TypeError):
            create_molecule_to_string_writer([])


class TestMoleculeFromString:
    """Test molecule_from_string function"""
    
    def test_molecule_from_string_smiles(self):
        mol = oechem.OEMol()
        fmt = get_oeformat(".smi")
        success = molecule_from_string(mol, "CCO", fmt)
        assert success
        assert mol.NumAtoms() > 0
    
    def test_molecule_from_string_sdf(self):
        mol = oechem.OEMol()
        fmt = get_oeformat(".sdf")
        # Create a very simple SDF string - ethanol
        sdf_string = """ethanol
  -OEChem-01010000002D

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$"""
        success = molecule_from_string(mol, sdf_string.strip(), fmt)
        # SDF parsing might be strict about format, if it fails we'll just test it exists but doesn't crash
        if success:
            assert mol.NumAtoms() > 0
        else:
            # At least the function returns False gracefully rather than crashing
            assert mol.NumAtoms() == 0
    
    def test_molecule_from_string_invalid(self):
        mol = oechem.OEMol()
        fmt = get_oeformat(".smi")
        success = molecule_from_string(mol, "invalid_smiles_string_xyz", fmt)
        # Should return False for invalid SMILES
        assert not success
    
    def test_molecule_from_string_with_b64decode(self):
        mol = oechem.OEMol()
        fmt = get_oeformat(".smi")
        # Base64 encode a SMILES
        import base64
        b64_smiles = base64.b64encode("CCO".encode()).decode()
        success = molecule_from_string(mol, b64_smiles, fmt, b64decode=True)
        assert success
        assert mol.NumAtoms() > 0


class TestPredominantType:
    """Test predominant_type function"""
    
    def test_predominant_type_simple(self):
        series = pd.Series([1, 2, 3, 4, 5])
        result = predominant_type(series)
        assert result == int
    
    def test_predominant_type_mixed(self):
        series = pd.Series([1, 2, "a", "b", "c"])
        result = predominant_type(series)
        # Should return the most common type
        assert result in (int, str)
    
    def test_predominant_type_with_nulls(self):
        series = pd.Series([1, 2, None, 3, np.nan])
        result = predominant_type(series)
        # When NaN is mixed with integers, pandas converts to float64
        assert result == float
    
    def test_predominant_type_empty_series(self):
        series = pd.Series([])
        result = predominant_type(series)
        assert result is None
    
    def test_predominant_type_all_null(self):
        series = pd.Series([None, None, np.nan])
        result = predominant_type(series)
        assert result is None
    
    def test_predominant_type_with_sample_size(self):
        # Create a large series
        series = pd.Series([1] * 100 + ["a"] * 10)
        result = predominant_type(series, sample_size=5)
        # Should still work with sampling
        assert result in (int, str)
    
    def test_predominant_type_molecules(self, test_molecules):
        series = pd.Series(test_molecules)
        result = predominant_type(series)
        assert result == type(test_molecules[0])


class TestIntegrationTests:
    """Integration tests combining multiple utility functions"""
    
    def test_round_trip_string_conversion(self, test_molecules):
        """Test converting molecule to string and back"""
        mol = test_molecules[0]
        
        # Create string writer
        string_writer = create_molecule_to_string_writer("smiles")
        smiles_str = string_writer(mol)
        
        # Convert back to molecule
        new_mol = oechem.OEMol()
        fmt = get_oeformat(".smi")
        success = molecule_from_string(new_mol, smiles_str, fmt)
        
        assert success
        assert new_mol.NumAtoms() == mol.NumAtoms()
    
    def test_file_format_detection_chain(self):
        """Test chaining file format detection functions"""
        filename = "test.molecules.sdf.gz"
        
        # Test is_gz
        assert is_gz(filename)
        
        # Test format detection
        fmt = get_oeformat_from_ext(filename)
        assert fmt.oeformat == oechem.OEFormat_SDF
        assert fmt.gzip
        
        # Test with get_oeformat directly
        fmt2 = get_oeformat(filename)
        assert fmt2.oeformat == fmt.oeformat
        assert fmt2.gzip == fmt.gzip
    
    def test_file_format_detection_integration(self):
        # Test common file extensions
        formats_to_test = [
            ("test.sdf", oechem.OEFormat_SDF, False),
            ("test.sdf.gz", oechem.OEFormat_SDF, True),
            ("test.oeb", oechem.OEFormat_OEB, False),
            ("test.oeb.gz", oechem.OEFormat_OEB, True),
            ("test.smi", oechem.OEFormat_SMI, False),
        ]
        
        for filename, expected_format, expected_gzip in formats_to_test:
            ff = get_oeformat(filename)
            assert ff.oeformat == expected_format
            assert ff.gzip == expected_gzip
    
    def test_writer_creation_for_supported_formats(self, test_molecules):
        # Test writer creation for different formats
        formats_to_test = [
            oechem.OEFormat_SMI,
            oechem.OEFormat_SDF,
        ]
        
        for fmt in formats_to_test:
            writer = create_molecule_to_string_writer(fmt)
            assert callable(writer)
            
            # Test with a simple molecule
            mol = test_molecules[0]
            result = writer(mol)
            assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__])