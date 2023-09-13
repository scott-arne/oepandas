import unittest
import pandas as pd
from oepandas import MoleculeArray, MoleculeDtype
from pathlib import Path
from openeye import oechem

ASSETS = Path(Path(__file__).parent, "assets")


class TestMoleculeArray(unittest.TestCase):
    def setUp(self) -> None:
        # Some simple alkanes
        self.alkanes_df = pd.DataFrame([
            {"title": "methane", "smiles1": "C", "smiles2": "C"},
            {"title": "ethane", "smiles1": "CC", "smiles2": "CC"},
            {"title": "propane", "smiles1": "CCC", "smiles2": "CCC"},
            {"title": "butane", "smiles1": "CCCC", "smiles2": "CCCC"},
        ])

        self.test_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(self.test_mol, "CCCC")

    def test_from_sdf(self):
        """
        Read an SD file
        """
        x = MoleculeArray.read_sdf(Path(ASSETS, "10.sdf"))
        self.assertEqual(10, len(x))
        self.assertTrue(all(isinstance(mol, oechem.OEMolBase) for mol in x))

    def test_from_smi(self):
        """
        Read a SMILES file
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        self.assertEqual(10, len(x))
        self.assertTrue(all(isinstance(mol, oechem.OEMolBase) for mol in x))

    def test_to_series(self):
        """
        Series conversion
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        s = pd.Series(x)
        self.assertEqual(10, len(s))

    def test_addition(self):
        """
        Adding two molecule arrays
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        y = x + x
        self.assertEqual(20, len(y))

    def test_dataframe_to_molecule(self):
        """
        Convert a string column to a molecule column
        """
        df = self.alkanes_df.copy()
        self.assertIsInstance(df.as_molecule("smiles1").smiles1.dtype, MoleculeDtype)

    def test_dataframe_series_astype(self):
        """
        Convert a column to a molecule dtype
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())
        self.assertIsInstance(df.MOL.dtype, MoleculeDtype)

    def test_accessor_filter_invalid(self):
        """
        Filter rows with invalid molecules (single columns)
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())
        self.assertEqual(2, len(df.filter_invalid_molecules("MOL")))

    def test_regression_as_molecule_formatter_axis_error(self):
        """
        Regression test for formatting large files
        """
        df = pd.read_excel(Path(ASSETS, "phenols.xlsx"))
        df.to_string()

    def test_fillna_simple(self):
        """
        Fill all NA and invalid molecules with None
        """
        x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), self.test_mol.CreateCopy(), None])
        y = x.fillna()

        self.assertIsNone(y[0])
        self.assertIsNone(y[1])
        self.assertIsNotNone(y[2])
        self.assertIsNone(y[3])

    def test_fillna_limit(self):
        """
        Fill at most 1 NA / invalid molecules with None
        """
        x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), self.test_mol.CreateCopy(), None])
        y = x.fillna(limit=1)

        self.assertIsNone(y[0])
        self.assertIsNotNone(y[1])
        self.assertIsNotNone(y[2])
        self.assertIsNone(y[3])

    def test_dropna(self):
        """
        Drop NA and invalid molecules
        """
        x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), self.test_mol.CreateCopy(), None])
        y = x.dropna()
        self.assertEqual(1, len(y))
