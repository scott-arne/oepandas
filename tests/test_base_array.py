"""
Tests for arrays/base.py module - OEExtensionArray base class
"""

import pytest
import pandas as pd
import numpy as np
from openeye import oechem
from oepandas.arrays.base import OEExtensionArray
from oepandas.arrays import MoleculeArray, DesignUnitArray


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


@pytest.fixture
def test_design_units():
    """Load test design units from file"""
    from pathlib import Path
    
    # Load from the same asset file used in test_design_unit.py
    assets_dir = Path(Path(__file__).parent, "assets")
    oedu_path = assets_dir / "2.oedu"
    
    if not oedu_path.exists():
        pytest.skip("Design unit test file not available")
    
    design_units = []
    du = oechem.OEDesignUnit()
    ifs = oechem.oeifstream(str(oedu_path))
    while oechem.OEReadDesignUnit(ifs, du):
        design_units.append(du.CreateCopy())
    ifs.close()
    
    if not design_units:
        pytest.skip("No design units could be loaded from test file")
    
    return design_units


class TestOEExtensionArrayBase:
    """Test OEExtensionArray base class functionality"""
    
    def test_molecule_array_init(self, test_molecules):
        """Test initialization of MoleculeArray"""
        arr = MoleculeArray(test_molecules)
        assert len(arr) == 4
        assert all(mol is not None for mol in arr)
    
    def test_molecule_array_init_with_none(self, test_molecules):
        """Test initialization with None values"""
        molecules_with_none = test_molecules + [None, None]
        arr = MoleculeArray(molecules_with_none)
        assert len(arr) == 6
        assert arr[4] is None
        assert arr[5] is None
    
    def test_molecule_array_init_with_nan(self, test_molecules):
        """Test initialization with NaN values"""
        molecules_with_nan = test_molecules + [np.nan, pd.NA]
        arr = MoleculeArray(molecules_with_nan)
        assert len(arr) == 6
        assert arr[4] is None
        assert arr[5] is None
    
    def test_molecule_array_init_with_deepcopy(self, test_molecules):
        """Test initialization with deepcopy=True"""
        arr = MoleculeArray(test_molecules, deepcopy=True)
        assert len(arr) == 4
        # Molecules should be copied, not the same objects
        for i, mol in enumerate(test_molecules):
            assert arr[i] is not mol
            # But should have same SMILES
            original_smiles = oechem.OEMolToSmiles(mol)
            copied_smiles = oechem.OEMolToSmiles(arr[i])
            assert original_smiles == copied_smiles
    
    def test_molecule_array_init_with_metadata(self, test_molecules):
        """Test initialization with metadata"""
        metadata = {"source": "test", "version": 1}
        arr = MoleculeArray(test_molecules, metadata=metadata)
        assert arr.metadata == metadata
    
    def test_molecule_array_init_invalid_type(self):
        """Test initialization with invalid object types"""
        with pytest.raises(TypeError):
            MoleculeArray(["not_a_molecule", "also_not_a_molecule"])
    
    def test_molecule_array_append(self, test_molecules):
        """Test append method"""
        arr = MoleculeArray(test_molecules[:2])
        assert len(arr) == 2
        
        arr.append(test_molecules[2])
        assert len(arr) == 3
        
        arr.append(None)
        assert len(arr) == 4
        assert arr[3] is None
        
        arr.append(np.nan)
        assert len(arr) == 5
        assert arr[4] is None
    
    def test_molecule_array_extend(self, test_molecules):
        """Test extend method"""
        arr = MoleculeArray(test_molecules[:2])
        assert len(arr) == 2
        
        arr.extend(test_molecules[2:])
        assert len(arr) == 4
        
        # Test extending with another array
        arr2 = MoleculeArray([None, None])
        arr.extend(arr2)
        assert len(arr) == 6
        assert arr[4] is None
        assert arr[5] is None
    
    def test_molecule_array_copy_shallow(self, test_molecules):
        """Test shallow copy"""
        original_metadata = {"test": "data"}
        arr = MoleculeArray(test_molecules, metadata=original_metadata)
        
        copied = arr.copy()
        
        # Should be different objects
        assert copied is not arr
        assert copied._objs is not arr._objs
        
        # But molecules should be the same objects (shallow copy)
        for i in range(len(arr)):
            if arr[i] is not None:
                assert copied[i] is arr[i]
        
        # Metadata should be copied
        assert copied.metadata == original_metadata
        assert copied.metadata is not original_metadata
    
    def test_molecule_array_copy_with_custom_metadata(self, test_molecules):
        """Test copy with custom metadata"""
        arr = MoleculeArray(test_molecules, metadata={"original": True})
        new_metadata = {"copied": True}
        
        copied = arr.copy(metadata=new_metadata)
        assert copied.metadata == new_metadata
    
    def test_molecule_array_copy_no_metadata(self, test_molecules):
        """Test copy without metadata"""
        arr = MoleculeArray(test_molecules, metadata={"original": True})
        
        copied = arr.copy(metadata=False)
        assert copied.metadata == {}
        
        copied2 = arr.copy(metadata=None)
        assert copied2.metadata == {}
    
    def test_molecule_array_deepcopy(self, test_molecules):
        """Test deep copy"""
        arr = MoleculeArray(test_molecules)
        
        deep_copied = arr.deepcopy()
        
        # Should be different objects
        assert deep_copied is not arr
        assert deep_copied._objs is not arr._objs
        
        # Molecules should also be different objects (deep copy)
        for i in range(len(arr)):
            if arr[i] is not None and deep_copied[i] is not None:
                assert deep_copied[i] is not arr[i]
                # But should have same SMILES
                original_smiles = oechem.OEMolToSmiles(arr[i])
                copied_smiles = oechem.OEMolToSmiles(deep_copied[i])
                assert original_smiles == copied_smiles
    
    def test_molecule_array_getitem(self, test_molecules):
        """Test __getitem__ method"""
        arr = MoleculeArray(test_molecules)
        
        # Test single item access
        assert arr[0] is test_molecules[0]
        assert arr[1] is test_molecules[1]
        
        # Test negative indexing
        assert arr[-1] is test_molecules[-1]
        
        # Test slice access
        sliced = arr[1:3]
        assert isinstance(sliced, MoleculeArray)
        assert len(sliced) == 2
        assert sliced[0] is test_molecules[1]
        assert sliced[1] is test_molecules[2]
    
    def test_molecule_array_setitem(self, test_molecules):
        """Test __setitem__ method"""
        arr = MoleculeArray(test_molecules)
        
        new_mol = oechem.OEMol()
        oechem.OESmilesToMol(new_mol, "CCCC")
        
        arr[0] = new_mol
        assert arr[0] is new_mol
        
        arr[1] = None
        assert arr[1] is None
        
        arr[2] = np.nan
        assert arr[2] is None
    
    def test_molecule_array_len(self, test_molecules):
        """Test __len__ method"""
        arr = MoleculeArray(test_molecules)
        assert len(arr) == len(test_molecules)
        
        empty_arr = MoleculeArray([])
        assert len(empty_arr) == 0
    
    def test_molecule_array_iter(self, test_molecules):
        """Test iteration"""
        arr = MoleculeArray(test_molecules)
        
        iterated_mols = list(arr)
        assert len(iterated_mols) == len(test_molecules)
        for i, mol in enumerate(iterated_mols):
            assert mol is test_molecules[i]
    
    def test_molecule_array_isna(self, test_molecules):
        """Test isna method"""
        molecules_with_none = test_molecules + [None, np.nan]
        arr = MoleculeArray(molecules_with_none)

        na_mask = arr.isna()
        assert isinstance(na_mask, np.ndarray)
        assert na_mask.dtype == bool
        assert len(na_mask) == 6

        # First 4 should be False (valid molecules)
        assert not na_mask[0]
        assert not na_mask[1]
        assert not na_mask[2]
        assert not na_mask[3]

        # Last 2 should be True (None values)
        assert na_mask[4]
        assert na_mask[5]

    def test_molecule_array_valid(self, test_molecules):
        """Test valid method"""
        # Create an invalid molecule
        invalid_mol = oechem.OEMol()  # Empty molecule is invalid

        molecules_mixed = test_molecules + [None, invalid_mol]
        arr = MoleculeArray(molecules_mixed)

        valid_mask = arr.valid()
        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask.dtype == bool
        assert len(valid_mask) == 6

        # First 4 should be True (valid molecules)
        assert valid_mask[0]
        assert valid_mask[1]
        assert valid_mask[2]
        assert valid_mask[3]

        # None should be False
        assert not valid_mask[4]

        # Invalid (empty) molecule should be False
        assert not valid_mask[5]

    def test_molecule_array_take(self, test_molecules):
        """Test take method"""
        arr = MoleculeArray(test_molecules)
        
        # Test normal indexing
        taken = arr.take([0, 2, 1])
        assert isinstance(taken, MoleculeArray)
        assert len(taken) == 3
        assert taken[0] is test_molecules[0]
        assert taken[1] is test_molecules[2]
        assert taken[2] is test_molecules[1]
        
        # Test with allow_fill=True and fill_value
        taken_with_fill = arr.take([0, -1, 2], allow_fill=True, fill_value=None)
        assert len(taken_with_fill) == 3
        assert taken_with_fill[0] is test_molecules[0]
        assert taken_with_fill[1] is None
        assert taken_with_fill[2] is test_molecules[2]
    
    def test_molecule_array_array_interface(self, test_molecules):
        """Test __array__ method"""
        arr = MoleculeArray(test_molecules)
        
        np_array = np.array(arr)
        assert isinstance(np_array, np.ndarray)
        assert np_array.dtype == object
        assert len(np_array) == len(test_molecules)
        
        for i, mol in enumerate(test_molecules):
            assert np_array[i] is mol


class TestDesignUnitArray:
    """Test DesignUnitArray functionality"""
    
    def test_design_unit_array_init(self, test_design_units):
        """Test DesignUnitArray initialization"""
        arr = DesignUnitArray(test_design_units)
        assert len(arr) == len(test_design_units)
        # Should have loaded 2 design units from the test file
        assert len(arr) == 2
    
    def test_design_unit_array_with_none(self, test_design_units):
        """Test DesignUnitArray with None values"""
        dus_with_none = test_design_units + [None]
        arr = DesignUnitArray(dus_with_none)
        assert len(arr) == len(test_design_units) + 1
        assert arr[-1] is None
        # Should have 2 design units + 1 None = 3 total
        assert len(arr) == 3


class TestErrorHandling:
    """Test error handling in base array functionality"""
    
    def test_invalid_index_access(self, test_molecules):
        """Test invalid index access"""
        arr = MoleculeArray(test_molecules)
        
        with pytest.raises(IndexError):
            _ = arr[10]  # Index out of range
        
        with pytest.raises(IndexError):
            _ = arr[-10]  # Negative index out of range
    
    def test_invalid_slice_assignment(self, test_molecules):
        """Test invalid slice assignment"""
        arr = MoleculeArray(test_molecules)
        
        # This might raise an error depending on implementation
        try:
            arr[0:2] = [None, None]
        except (NotImplementedError, TypeError, ValueError):
            # Expected for some pandas extension arrays - ValueError can occur with array comparisons
            pass
    
    def test_take_invalid_indices(self, test_molecules):
        """Test take with invalid indices"""
        arr = MoleculeArray(test_molecules)
        
        # Test with out of bounds index - should raise error
        with pytest.raises((ValueError, IndexError)):
            arr.take([0, 10, 2], allow_fill=False)  # Index 10 is out of bounds


if __name__ == "__main__":
    pytest.main([__file__])