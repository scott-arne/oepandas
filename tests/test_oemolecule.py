import unittest
import base64 as b64
import pandas as pd
import numpy as np
import oepandas as oepd
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

    def test_read_molecule_csv(self):
        """
        Read a CSV with molecules
        """
        # noinspection PyUnresolvedReferences
        df = pd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles")
        self.assertTrue(all(isinstance(mol, oechem.OEMolBase) for mol in df.Smiles))

    def test_pandas_readers_monkeypatch(self):
        """
        Readers are monkeypatched
        """
        self.assertTrue(hasattr(pd, "read_molecule_csv"))
        self.assertTrue(hasattr(pd, "read_smi"))
        self.assertTrue(hasattr(pd, "read_sdf"))

    def test_read_molecule_csv_add_smiles(self):
        """
        Adding SMILES columns in different ways when reading a molecule CSV
        """
        with self.subTest("add_smiles=True"):
            df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles=True)
            self.assertIn("Smiles SMILES", df.columns)
            self.assertTrue(all(isinstance(x, str) for x in df["Smiles SMILES"]))

        with self.subTest("add_smiles='Smiles'"):
            df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles="Smiles")
            self.assertIn("Smiles SMILES", df.columns)
            self.assertTrue(all(isinstance(x, str) for x in df["Smiles SMILES"]))

        with self.subTest("add_smiles=['Smiles']"):
            df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles=["Smiles"])
            self.assertIn("Smiles SMILES", df.columns)
            self.assertTrue(all(isinstance(x, str) for x in df["Smiles SMILES"]))

        with self.subTest("add_smiles={'Smiles': 'Testy McTesterson'}"):
            df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles",
                                        add_smiles={"Smiles": "Testy McTesterson"})
            self.assertIn("Testy McTesterson", df.columns)
            self.assertTrue(all(isinstance(x, str) for x in df["Testy McTesterson"]))

    def test_read_smi(self):
        """
        Read a SMILES file into a DataFrame
        """
        # noinspection PyUnresolvedReferences
        df = pd.read_smi(Path(ASSETS, "10.smi"))
        self.assertEqual(10, len(df))
        self.assertTrue(all(isinstance(mol, oechem.OEMolBase) for mol in df.Molecule))

    def test_read_sdf(self):
        """
        Read an SD file with data
        """
        with self.subTest("Read all data as strings (i.e., object)"):
            df = oepd.read_sdf(Path(ASSETS, "10-tagged.sdf"))

            # Check the datatypes
            for col in df.columns:
                if col == "Molecule":
                    self.assertTrue(isinstance(df.dtypes[col], MoleculeDtype))
                else:
                    self.assertTrue(isinstance(df.dtypes[col], object))

        with self.subTest("Read a single column"):
            df = oepd.read_sdf(
                Path(ASSETS, "10-tagged.sdf"),
                usecols="Integer Tag"
            )

            self.assertIn("Integer Tag", df.columns)
            self.assertNotIn("Float Tag", df.columns)
            self.assertNotIn("String Tag", df.columns)

        with self.subTest("Read a single column and cast it to an integer"):
            df = oepd.read_sdf(
                Path(ASSETS, "10-tagged.sdf"),
                usecols="Integer Tag",
                numeric={"Integer Tag": "unsigned"}
            )

            self.assertIn("Integer Tag", df.columns)
            self.assertNotIn("Float Tag", df.columns)
            self.assertNotIn("String Tag", df.columns)

            # Check the dtype
            self.assertEqual(str(df.dtypes["Integer Tag"]), "uint8")

        with self.subTest("Cast multiple numeric columns data"):
            df = oepd.read_sdf(
                Path(ASSETS, "10-tagged.sdf"),
                numeric={"Integer Tag": "unsigned", "Float Tag": "float"}
            )

            self.assertIn("Integer Tag", df.columns)
            self.assertIn("Float Tag", df.columns)
            self.assertIn("String Tag", df.columns)

            # Check the dtype
            self.assertEqual(str(df.dtypes["Integer Tag"]), "uint8")
            self.assertEqual(str(df.dtypes["Float Tag"]), "float32")

    def test_to_smiles(self):
        """
        Convert to a SMILES array
        """
        arr = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        arr.append(oechem.OEMol())

        expected = [
            'CC(=O)Oc1ccccc1C(=O)O',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
            'CC(=O)Nc1ccc(cc1)O',
            'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
            'CN1c2ccc(cc2C(=NCC1=O)c3ccccc3)Cl',
            'Cc1c(c(c(c(n1)C(C)C)C(C)C)C(=O)NCCc2ccc(cc2)OC)c3ccccc3',
            'CN(C)C(=O)N(c1ccccc1)c2ccc(cc2)Cl',
            'CC(C)Cc1ccc(cc1)OC2=CC(=O)c3c(cccc3O)C2=O',
            'Cc1cn(c(=O)o1)[C@@H]2C[C@@H](c3[nH]c4cccc(c4n3)S2(=O)=O)c5ccccc5',
            'CN(C)C1CC[C@H]2C=CC(=C[C@@]2(C1)O)Cl',
            ''
        ]

        self.assertListEqual(expected, arr.to_smiles().tolist())

    def test_series_to_smiles(self):
        """
        Convert a series to SMILES
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())

        expected = [
            "CC(=O)Oc1ccccc1C(=O)O",
            '',
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        ]

        self.assertListEqual(expected, df.MOL.to_smiles().tolist())

    def test_series_to_molecule_string(self):
        """
        Convert a series to various molecule file formats
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())

        # ----------------------------------------------
        # SMILES
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_strings = [
            oechem.OEMolToSmiles(x[0]),
            '',
            oechem.OEMolToSmiles(x[1])
        ]

        with self.subTest("Canonical isomeric SMILES"):
            df["TEST"] = df.MOL.to_molecule_strings(molecule_format="smiles")
            print(df)
            self.assertListEqual(
                expected_strings,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_strings = [
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                False,
                x[0]
            ).decode('utf-8').strip(),
            '',
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                False,
                x[1]
            ).decode('utf-8').strip(),
        ]

        with self.subTest("SDF v3000"):
            df["TEST"] = df.MOL.to_molecule_strings(
                molecule_format=oechem.OEFormat_SDF,
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30
            )
            self.assertListEqual(
                expected_strings,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000 b64 encoded
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_strings = [
            b64.b64encode(
                oechem.OEWriteMolToBytes(
                    oechem.OEFormat_SDF,
                    oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                    False,
                    x[0]
                )
            ).decode('utf-8').strip(),
            '',
            b64.b64encode(
                oechem.OEWriteMolToBytes(
                    oechem.OEFormat_SDF,
                    oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                    False,
                    x[1]
                )
            ).decode('utf-8').strip(),
        ]

        with self.subTest("SDF v3000 with forced b64 encoding"):
            df["TEST"] = df.MOL.to_molecule_strings(
                molecule_format=oechem.OEFormat_SDF,
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                b64encode=True
            )
            self.assertListEqual(
                expected_strings,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000 gzipped
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_strings = [
            b64.b64encode(
                oechem.OEWriteMolToBytes(
                    oechem.OEFormat_SDF,
                    oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                    True,
                    x[0]
                )
            ).decode('utf-8').strip(),
            '',
            b64.b64encode(
                oechem.OEWriteMolToBytes(
                    oechem.OEFormat_SDF,
                    oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                    True,
                    x[1]
                )
            ).decode('utf-8').strip(),
        ]

        with self.subTest("SDF v3000 gzipped"):
            df["TEST"] = df.MOL.to_molecule_strings(
                molecule_format=oechem.OEFormat_SDF,
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                gzip=True
            )
            self.assertListEqual(
                expected_strings,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000 gzipped by extension
        # ----------------------------------------------

        with self.subTest("SDF v3000 gzipped by file format extension"):
            df["TEST"] = df.MOL.to_molecule_strings(
                molecule_format=".sdf.gz",
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
            )
            self.assertListEqual(
                expected_strings,
                df.TEST.tolist()
            )

    def test_series_to_molecule_bytes(self):
        """
        Convert a series to various molecule file formats
        """
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())

        # ----------------------------------------------
        # SMILES
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_bytes = [
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SMI,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SMI),
                False,
                x[0]
            ),
            b'',
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SMI,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SMI),
                False,
                x[1]
            ),
        ]

        with self.subTest("Canonical isomeric SMILES"):
            df["TEST"] = df.MOL.to_molecule_bytes(molecule_format=oechem.OEFormat_SMI)
            self.assertListEqual(
                expected_bytes,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_bytes = [
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                False,
                x[0]
            ),
            b'',
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                False,
                x[1]
            ),
        ]

        with self.subTest("SDF v3000"):
            df["TEST"] = df.MOL.to_molecule_bytes(
                molecule_format=oechem.OEFormat_SDF,
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30
            )
            self.assertListEqual(
                expected_bytes,
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000 gzipped
        # ----------------------------------------------

        # We expect the molecules to have these SMILES strings
        expected_bytes = [
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                True,
                x[0]
            ),
            b'',
            oechem.OEWriteMolToBytes(
                oechem.OEFormat_SDF,
                oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                True,
                x[1]
            ),
        ]

        with self.subTest("SDF v3000 gzipped"):
            df["TEST"] = df.MOL.to_molecule_bytes(
                molecule_format=oechem.OEFormat_SDF,
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
                gzip=True
            )
            self.assertListEqual(
                [b.rstrip(b'\x00') for b in expected_bytes],
                df.TEST.tolist()
            )

        # ----------------------------------------------
        # SDF v3000 gzipped by extension
        # ----------------------------------------------

        with self.subTest("SDF v3000 gzipped by file format extension"):
            df["TEST"] = df.MOL.to_molecule_bytes(
                molecule_format=".sdf.gz",
                flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
            )
            self.assertListEqual(
                [b.rstrip(b'\x00') for b in expected_bytes],
                df.TEST.tolist()
            )

    def test_read_oedb(self):
        """
        Read data records
        """
        df = oepd.read_oedb(Path(ASSETS, "10.oedb"))
        self.assertIn("MolWt Halide Fraction (Calculated)", df.columns)
        self.assertIn("Heavy Atom Count (Calculated)", df.columns)
        self.assertIn("Molecule", df.columns)
        self.assertEqual(df.dtypes["MolWt Halide Fraction (Calculated)"], float)
        self.assertEqual(df.dtypes["Heavy Atom Count (Calculated)"], int)
        self.assertIsInstance(df.dtypes["Molecule"], oepd.MoleculeDtype)

    # def test_to_molecule_csv(self):
    #     x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
    #     df = pd.DataFrame([
    #         {"Title": x[0].GetTitle(), "MOL": x[0]},
    #         {"Title": "Invalid", "MOL": oechem.OEMol()},
    #         {"Title": x[1].GetTitle(), "MOL": x[1]},
    #     ])
    #     df["MOL"] = df.MOL.astype(MoleculeDtype())
    #
    #     df.to_molecule_csv("test-molecule-csv.csv")

    def test_molecule_array_subsearch(self):
        """
        SMARTS matching in a MoleculeArray
        """
        x = MoleculeArray.read_sdf(Path(ASSETS, "10.sdf"))
        sulfones = np.where(x.subsearch('S(=O)=O'))
        self.assertEqual(1, len(sulfones))
        self.assertEqual(8, sulfones[0])
