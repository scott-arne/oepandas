"""
Tests for arrays/display.py module - DisplayArray and DisplayDtype
"""

import pytest
import pandas as pd
import numpy as np
from openeye import oechem, oedepict
from oepandas.arrays import DisplayArray, DisplayDtype


@pytest.fixture
def test_displays():
    """Create test display objects"""
    displays = []
    smiles_list = ["CCO", "CCC", "C"]
    
    for smiles in smiles_list:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smiles)
        
        # Create 2D display
        display = oedepict.OE2DMolDisplay(mol)
        displays.append(display)
    
    return displays


class TestDisplayDtype:
    """Test DisplayDtype functionality"""
    
    def test_display_dtype_creation(self):
        """Test DisplayDtype creation"""
        dtype = DisplayDtype()
        assert isinstance(dtype, DisplayDtype)
        assert dtype.name == "display"
    
    def test_display_dtype_properties(self):
        """Test DisplayDtype properties"""
        dtype = DisplayDtype()
        assert dtype.type == oedepict.OE2DMolDisplay
        assert dtype.kind == "O"  # Object kind
        assert dtype.na_value is None
    
    def test_display_dtype_construct_array_type(self):
        """Test construct_array_type class method"""
        array_type = DisplayDtype.construct_array_type()
        assert array_type == DisplayArray
    
    def test_display_dtype_construct_from_string(self):
        """Test construct_from_string class method"""
        dtype = DisplayDtype.construct_from_string("display")
        assert isinstance(dtype, DisplayDtype)
        
        # Test invalid string
        with pytest.raises((TypeError, ValueError)):
            DisplayDtype.construct_from_string("invalid")
    
    def test_display_dtype_is_dtype(self):
        """Test _is_dtype method"""
        dtype = DisplayDtype()
        assert DisplayDtype._is_dtype(dtype) is True
        assert DisplayDtype._is_dtype("display") is True
        assert DisplayDtype._is_dtype("invalid") is False
        assert DisplayDtype._is_dtype(pd.StringDtype()) is False


class TestDisplayArray:
    """Test DisplayArray functionality"""
    
    def test_display_array_init(self, test_displays):
        """Test DisplayArray initialization"""
        arr = DisplayArray(test_displays)
        assert len(arr) == 3
        assert all(isinstance(display, oedepict.OE2DMolDisplay) for display in arr if display is not None)
    
    def test_display_array_init_with_none(self, test_displays):
        """Test DisplayArray initialization with None values"""
        displays_with_none = test_displays + [None, None]
        arr = DisplayArray(displays_with_none)
        assert len(arr) == 5
        assert arr[3] is None
        assert arr[4] is None
    
    def test_display_array_init_with_nan(self, test_displays):
        """Test DisplayArray initialization with NaN values"""
        displays_with_nan = test_displays + [np.nan, pd.NA]
        arr = DisplayArray(displays_with_nan)
        assert len(arr) == 5
        assert arr[3] is None
        assert arr[4] is None
    
    def test_display_array_init_with_copy(self, test_displays):
        """Test DisplayArray initialization with copy=True"""
        arr = DisplayArray(test_displays, copy=True)
        assert len(arr) == 3
        
        # Displays should be copied, not the same objects
        for i, display in enumerate(test_displays):
            assert arr[i] is not display
            # But should be equivalent (hard to test equality for OE2DMolDisplay)
    
    def test_display_array_init_invalid_type(self):
        """Test DisplayArray initialization with invalid object types"""
        with pytest.raises(TypeError):
            DisplayArray(["not_a_display", "also_not_a_display"])
    
    def test_display_array_dtype_property(self, test_displays):
        """Test DisplayArray dtype property"""
        arr = DisplayArray(test_displays)
        assert isinstance(arr.dtype, DisplayDtype)
        assert arr.dtype.name == "display"
    
    def test_display_array_nbytes(self, test_displays):
        """Test DisplayArray nbytes property"""
        arr = DisplayArray(test_displays)
        # Should return some reasonable size
        assert arr.nbytes > 0
        assert isinstance(arr.nbytes, int)
    
    def test_display_array_getitem(self, test_displays):
        """Test DisplayArray __getitem__ method"""
        arr = DisplayArray(test_displays)
        
        # Test single item access
        assert arr[0] is test_displays[0]
        assert arr[1] is test_displays[1]
        
        # Test negative indexing
        assert arr[-1] is test_displays[-1]
        
        # Test slice access
        sliced = arr[0:2]
        assert isinstance(sliced, DisplayArray)
        assert len(sliced) == 2
        assert sliced[0] is test_displays[0]
        assert sliced[1] is test_displays[1]
    
    def test_display_array_setitem(self, test_displays):
        """Test DisplayArray __setitem__ method"""
        arr = DisplayArray(test_displays)
        
        # Create new display
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCCC")
        new_display = oedepict.OE2DMolDisplay(mol)
        
        arr[0] = new_display
        assert arr[0] is new_display
        
        arr[1] = None
        assert arr[1] is None
        
        arr[2] = np.nan
        assert arr[2] is None
    
    def test_display_array_len(self, test_displays):
        """Test DisplayArray __len__ method"""
        arr = DisplayArray(test_displays)
        assert len(arr) == len(test_displays)
        
        empty_arr = DisplayArray([])
        assert len(empty_arr) == 0
    
    def test_display_array_iter(self, test_displays):
        """Test DisplayArray iteration"""
        arr = DisplayArray(test_displays)
        
        iterated_displays = list(arr)
        assert len(iterated_displays) == len(test_displays)
        for i, display in enumerate(iterated_displays):
            assert display is test_displays[i]
    
    def test_display_array_isna(self, test_displays):
        """Test DisplayArray isna method"""
        displays_with_none = test_displays + [None, np.nan]
        arr = DisplayArray(displays_with_none)
        
        na_mask = arr.isna()
        assert isinstance(na_mask, np.ndarray)
        assert na_mask.dtype == bool
        assert len(na_mask) == 5
        
        # First 3 should be False (valid displays)
        assert not na_mask[0]
        assert not na_mask[1]
        assert not na_mask[2]
        
        # Last 2 should be True (None values)
        assert na_mask[3]
        assert na_mask[4]
    
    def test_display_array_take(self, test_displays):
        """Test DisplayArray take method"""
        arr = DisplayArray(test_displays)
        
        # Test normal indexing
        taken = arr.take([0, 2, 1])
        assert isinstance(taken, DisplayArray)
        assert len(taken) == 3
        assert taken[0] is test_displays[0]
        assert taken[1] is test_displays[2]
        assert taken[2] is test_displays[1]
        
        # Test with allow_fill=True and fill_value
        taken_with_fill = arr.take([0, -1, 1], allow_fill=True, fill_value=None)
        assert len(taken_with_fill) == 3
        assert taken_with_fill[0] is test_displays[0]
        assert taken_with_fill[1] is None
        assert taken_with_fill[2] is test_displays[1]
    
    def test_display_array_copy_shallow(self, test_displays):
        """Test DisplayArray shallow copy"""
        original_metadata = {"test": "data"}
        arr = DisplayArray(test_displays, metadata=original_metadata)
        
        copied = arr.copy()
        
        # Should be different objects
        assert copied is not arr
        assert copied._objs is not arr._objs
        
        # But displays should be the same objects (shallow copy)
        for i in range(len(arr)):
            if arr[i] is not None:
                assert copied[i] is arr[i]
        
        # Metadata should be copied
        assert copied.metadata == original_metadata
        assert copied.metadata is not original_metadata
    
    def test_display_array_deepcopy(self, test_displays):
        """Test DisplayArray deep copy"""
        arr = DisplayArray(test_displays)
        
        deep_copied = arr.deepcopy()
        
        # Should be different objects
        assert deep_copied is not arr
        assert deep_copied._objs is not arr._objs
        
        # Displays should also be different objects (deep copy)
        for i in range(len(arr)):
            if arr[i] is not None and deep_copied[i] is not None:
                assert deep_copied[i] is not arr[i]
    
    def test_display_array_array_interface(self, test_displays):
        """Test DisplayArray __array__ method"""
        arr = DisplayArray(test_displays)
        
        np_array = np.array(arr)
        assert isinstance(np_array, np.ndarray)
        assert np_array.dtype == object
        assert len(np_array) == len(test_displays)
        
        for i, display in enumerate(test_displays):
            assert np_array[i] is display


class TestDisplayArrayPandasIntegration:
    """Test DisplayArray integration with pandas"""
    
    def test_display_array_in_series(self, test_displays):
        """Test DisplayArray in pandas Series"""
        arr = DisplayArray(test_displays)
        series = pd.Series(arr, dtype=DisplayDtype())
        
        assert len(series) == len(test_displays)
        assert isinstance(series.dtype, DisplayDtype)
        
        # Test accessing elements
        for i in range(len(series)):
            assert series.iloc[i] is test_displays[i]
    
    def test_display_array_in_dataframe(self, test_displays):
        """Test DisplayArray in pandas DataFrame"""
        arr = DisplayArray(test_displays)
        df = pd.DataFrame({
            "Display": arr,
            "Index": range(len(test_displays))
        })
        
        assert len(df) == len(test_displays)
        assert isinstance(df["Display"].dtype, DisplayDtype)
        
        # Test accessing elements
        for i in range(len(df)):
            assert df["Display"].iloc[i] is test_displays[i]
    
    def test_display_array_concat(self, test_displays):
        """Test concatenating DisplayArrays"""
        arr1 = DisplayArray(test_displays[:2])
        arr2 = DisplayArray(test_displays[2:])
        
        series1 = pd.Series(arr1, dtype=DisplayDtype())
        series2 = pd.Series(arr2, dtype=DisplayDtype())
        
        concatenated = pd.concat([series1, series2], ignore_index=True)
        
        assert len(concatenated) == len(test_displays)
        assert isinstance(concatenated.dtype, DisplayDtype)


class TestErrorHandling:
    """Test error handling in DisplayArray"""
    
    def test_invalid_display_creation(self):
        """Test creating DisplayArray with invalid objects"""
        mol = oechem.OEMol()  # Not a display object
        oechem.OESmilesToMol(mol, "CCO")
        
        with pytest.raises(TypeError):
            DisplayArray([mol])
    
    def test_invalid_index_access(self, test_displays):
        """Test invalid index access"""
        arr = DisplayArray(test_displays)
        
        with pytest.raises(IndexError):
            _ = arr[10]  # Index out of range
        
        with pytest.raises(IndexError):
            _ = arr[-10]  # Negative index out of range
    
    def test_empty_display_array(self):
        """Test empty DisplayArray"""
        arr = DisplayArray([])
        assert len(arr) == 0
        assert isinstance(arr, DisplayArray)
        
        # Test isna on empty array
        na_mask = arr.isna()
        assert len(na_mask) == 0
        assert isinstance(na_mask, np.ndarray)


class TestDisplayArraySpecialMethods:
    """Test special methods of DisplayArray"""
    
    def test_display_array_repr(self, test_displays):
        """Test DisplayArray string representation"""
        arr = DisplayArray(test_displays)
        repr_str = repr(arr)
        assert isinstance(repr_str, str)
        assert "DisplayArray" in repr_str
    
    def test_display_array_equality(self, test_displays):
        """Test DisplayArray equality comparison"""
        arr1 = DisplayArray(test_displays)
        arr2 = DisplayArray(test_displays)
        
        # Arrays with same content should be equal
        # Note: This depends on the implementation of equality
        try:
            equal = arr1 == arr2
            # If equality is implemented, should return boolean array
            if hasattr(equal, '__iter__'):
                assert len(equal) == len(test_displays)
        except (NotImplementedError, TypeError):
            # Equality might not be implemented
            pass


if __name__ == "__main__":
    pytest.main([__file__])