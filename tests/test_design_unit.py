import unittest
import pandas as pd
import oepandas as oepd
from oepandas import DesignUnitArray, DesignUnitDtype
from pathlib import Path
from openeye import oechem

oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Fatal)

ASSETS = Path(Path(__file__).parent, "assets")
OEDU_PATH = Path(ASSETS, "2.oedu")


class TestDesignUnitArray(unittest.TestCase):
    def test_read_oedu(self):
        """
        Test read an OEDU file
        """
        arr = DesignUnitArray.read_oedu(OEDU_PATH)
        self.assertEqual(2, len(arr))
        self.assertEqual(arr[0].GetTitle(), "1JFF(AB) > TA1(B-601)")
        self.assertEqual(arr[1].GetTitle(), "1TVK(AB) >  EP(B-1001)")


class TestPandasDesignUnitExtensions(unittest.TestCase):
    def setUp(self):
        # Read some test design units
        self.test_design_units = []
        du = oechem.OEDesignUnit()
        ifs = oechem.oeifstream(str(OEDU_PATH))
        while oechem.OEReadDesignUnit(ifs, du):
            self.test_design_units.append(du.CreateCopy())
        ifs.close()

    def copy_test_design_units(self):
        return [du.CreateCopy() for du in self.test_design_units]

    def test_read_oedu(self):
        """
        Read an OEDU
        """
        df = oepd.read_oedu(OEDU_PATH)
        self.assertIsInstance(df.dtypes["Design_Unit"], DesignUnitDtype)
        self.assertEqual(2, len(df))
        self.assertListEqual(
            ["1JFF(AB) > TA1(B-601)", "1TVK(AB) >  EP(B-1001)"],
            list(df.Title.array)
        )

    def test_get_ligand(self):
        """
        Get ligand from design unit
        """
        df = pd.DataFrame({"Design_Unit": pd.Series(self.copy_test_design_units(), dtype=DesignUnitDtype())})
        df["Ligand"] = df["Design_Unit"].get_ligands()
        df["Ligand_SMILES"] = df.Ligand.apply(oechem.OEMolToSmiles)

        expected = [
            'CC1[C@H](C[C@]2([C@H]([C@H]3[C@@]([C@H](C[C@@H]4[C@]3(CO4)OC(=O)C)O)(C(=O)C(=C1C2(C)C)OC(=O)C)C)OC(=O)c5cc'
            'ccc5)O)OC(=O)[C@@H]([C@H](c6ccccc6)NC(=O)c7ccccc7)O',
            'Cc1nc(cs1)/C=C(\\C)/[C@@H]2C[C@H]3[C@H](O3)CCC[C@@H]([C@@H]([C@H](C(=O)C([C@H](CC(=O)O2)O)(C)C)C)O)C'
        ]

        self.assertListEqual(expected, df.Ligand_SMILES.tolist())

    def test_get_protein(self):
        """
        Get protein from design unit
        """
        df = pd.DataFrame({"Design_Unit": pd.Series(self.copy_test_design_units(), dtype=DesignUnitDtype())})
        df["Protein"] = df["Design_Unit"].get_proteins()
        df["Num_Protein_Atoms"] = df.Protein.apply(lambda mol: mol.NumAtoms())

        expected = [12979, 12961]

        self.assertListEqual(expected, df.Num_Protein_Atoms.tolist())

    def test_series_as_design_unit(self):
        """
        Series as_design_unit
        """
        # Create as object
        df = pd.DataFrame({"Design_Unit": pd.Series(self.copy_test_design_units(), dtype=object)})

        # Convert to design unit
        df["Design_Unit"] = df.Design_Unit.as_design_unit()
        self.assertIsInstance(df.dtypes["Design_Unit"], DesignUnitDtype)

    def test_dataframe_as_design_unit(self):
        """
        DataFrame as_design_unit
        """
        # Create as object
        df = pd.DataFrame({"Design_Unit": pd.Series(self.copy_test_design_units(), dtype=object)})

        # Convert to design unit
        df.as_design_unit(columns=["Design_Unit"], inplace=True)
        self.assertIsInstance(df.dtypes["Design_Unit"], DesignUnitDtype)

    def test_xxxx(self):
        df = oepd.read_oedu("/Users/johnss51/Data/EGFR/2023-12-05_Structures/design_units")
        df["Ligand"] = df.Design_Unit.get_ligands()
        print(len(df))
