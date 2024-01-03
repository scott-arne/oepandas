import os
import unittest
import base64 as b64
import pandas as pd
import numpy as np
import oepandas as oepd
from tempfile import TemporaryDirectory
from oepandas import DisplayArray, DisplayDtype
from pathlib import Path
from openeye import oechem, oedepict

ASSETS = Path(Path(__file__).parent, "assets")


class TestDisplayArray(unittest.TestCase):
    def setUp(self) -> None:
        self.mols = []

        for i in range(4):
            mol = oechem.OEMol()
            oechem.OESmilesToMol(mol, "C" * (i + 1))
            oechem.OEGenerate2DCoordinates(mol)
            self.mols.append(mol)

    def copy_mols(self) -> list[oechem.OEGraphMol]:
        """
        Deep copy of the alkane molecule test set
        :return: Deep copy of molecule test set
        """
        return [m.CreateCopy() for m in self.mols]

    def depictions(self) -> list[oedepict.OE2DMolDisplay]:
        return [oedepict.OE2DMolDisplay(mol) for mol in self.copy_mols()]

    def test_construct_display_array(self):
        """
        Create a DisplayArray
        """
        arr = DisplayArray(self.depictions())
        self.assertEqual(4, len(arr))
