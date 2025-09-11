import os
import pytest
import base64 as b64
import pandas as pd
import numpy as np
import oepandas as oepd
from tempfile import TemporaryDirectory
from oepandas import DisplayArray, DisplayDtype
from pathlib import Path
from openeye import oechem, oedepict

ASSETS = Path(Path(__file__).parent, "assets")


@pytest.fixture
def test_molecules():
    """Create test molecules for display testing"""
    mols = []
    for i in range(4):
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "C" * (i + 1))
        oechem.OEGenerate2DCoordinates(mol)
        mols.append(mol)
    return mols


def copy_mols(test_molecules) -> list[oechem.OEGraphMol]:
    """
    Deep copy of the alkane molecule test set
    :return: Deep copy of molecule test set
    """
    return [m.CreateCopy() for m in test_molecules]


def depictions(test_molecules) -> list[oedepict.OE2DMolDisplay]:
    """Create depictions from test molecules"""
    return [oedepict.OE2DMolDisplay(mol) for mol in copy_mols(test_molecules)]

def test_construct_display_array(test_molecules):
    """
    Create a DisplayArray
    """
    arr = DisplayArray(depictions(test_molecules))
    assert len(arr) == 4
