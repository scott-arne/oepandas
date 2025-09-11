import os
import pytest
import base64 as b64
import pandas as pd
import numpy as np
import oepandas as oepd
from tempfile import TemporaryDirectory
from oepandas import MoleculeArray, MoleculeDtype
from pathlib import Path
from openeye import oechem

ASSETS = Path(Path(__file__).parent, "assets")


@pytest.fixture
def test_mols():
    """Fixture providing test molecules"""
    mols = []
    for i in range(4):
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "C" * (i + 1))
        mols.append(mol)
    return mols


@pytest.fixture
def alkanes_df(test_mols):
    """Fixture providing alkanes dataframe"""
    data = []
    for i, name in enumerate(("methane", "ethane", "propane", "butane")):
        data.append({
            "title": name,
            "smiles1": oechem.OEMolToSmiles(test_mols[i]),
            "smiles2": oechem.OEMolToSmiles(test_mols[i]),
            "molecule": test_mols[i].CreateCopy()
        })
    return pd.DataFrame(data)


def copy_mols(test_mols) -> list[oechem.OEGraphMol]:
    """
    Deep copy of the alkane molecule test set
    
    :param test_mols: Test molecules to copy
    :returns: Deep copy of molecule test set
    """
    return [m.CreateCopy() for m in test_mols]

def test_create_simple(test_mols):
    """
    Create a MoleculeArray from a list of molecules
    """
    arr = MoleculeArray(copy_mols(test_mols))
    assert len(arr) == len(test_mols)

def test_contains(test_mols):
    """
    Test if a molecule is in an array
    """
    mols = copy_mols(test_mols)
    arr = MoleculeArray(mols)
    assert mols[0] in arr

def test_from_smi():
    """
    Read a SMILES file
    """
    x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
    assert len(x) == 10
    assert all(isinstance(mol, oechem.OEMolBase) for mol in x)

def test_to_series():
    """
    Series conversion
    """
    x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
    s = pd.Series(x)
    assert len(s) == 10

def test_from_sdf():
    """
    Read an SD file
    """
    x = MoleculeArray.read_sdf(Path(ASSETS, "10.sdf"))
    assert len(x) == 10
    assert all(isinstance(mol, oechem.OEMolBase) for mol in x)

def test_from_oeb():
    """
    Read an OEB file
    """
    x = MoleculeArray.read_oeb(Path(ASSETS, "10.oeb"))
    assert len(x) == 10
    assert all(isinstance(mol, oechem.OEMolBase) for mol in x)

def test_from_oebgz():
    """
    Read an OEB.gz file
    """
    x = MoleculeArray.read_oeb(Path(ASSETS, "10.oeb.gz"))
    assert len(x) == 10
    assert all(isinstance(mol, oechem.OEMolBase) for mol in x)

def test_addition():
    """
    Adding two molecule arrays
    """
    x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
    
    # Adding two MoleculeArrays
    y = x + x
    assert len(y) == 20
    
    # Adding a molecule to a MoleculeArray
    m = oechem.OEGraphMol()
    y = x + m
    assert len(y) == 11

def test_dataframe_to_molecule(alkanes_df):
    """
    Convert a string column to a molecule column
    """
    df = alkanes_df.copy()
    assert isinstance(df.as_molecule("smiles1").smiles1.dtype, MoleculeDtype)

    # Test that it worked
    df["HACount"] = df.molecule.apply(lambda mol: oechem.OECount(mol, oechem.OEIsHeavy()))
    for i, (_idx, row) in enumerate(df.iterrows()):
        assert row["HACount"] == i + 1

def test_dataframe_series_astype():
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
    assert isinstance(df.MOL.dtype, MoleculeDtype)

def test_accessor_filter_invalid():
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
    assert len(df.filter_invalid_molecules("MOL")) == 2

@pytest.mark.skipif(os.environ.get("OEPANDAS_TEST_LONG", "FALSE").upper() != "TRUE", reason="Skipping long tests")
def test_regression_as_molecule_formatter_axis_error():
    """
    Regression test for formatting large files
    """
    df = pd.read_excel(Path(ASSETS, "phenols.xlsx"))
    df.to_string()

def test_fillna_simple(test_mols):
    """
    Fill all NA and invalid molecules with None
    """
    x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), *copy_mols(test_mols), None])
    y = x.fillna()

    assert y[0] is None
    assert y[1] is None
    assert y[2] is not None
    assert y[3] is not None
    assert y[4] is not None
    assert y[5] is not None
    assert y[6] is None

def test_fillna_limit(test_mols):
    """
    Fill at most 1 NA / invalid molecules with None
    """
    x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), *copy_mols(test_mols), None])
    y = x.fillna(limit=1)

    assert y[0] is None
    assert y[1] is not None
    assert y[2] is not None
    assert y[-1] is None

def test_dropna(test_mols):
    """
    Drop NA and invalid molecules
    """
    mols = copy_mols(test_mols)
    x = MoleculeArray([oechem.OEMol(), oechem.OEGraphMol(), *mols, None])
    y = x.dropna()
    assert mols == y.tolist()

def test_read_molecule_csv():
    """
    Read a CSV with molecules
    """
    # noinspection PyUnresolvedReferences
    df = pd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles")
    assert all(isinstance(mol, oechem.OEMolBase) for mol in df.Smiles)

def test_pandas_readers_monkeypatch():
    """
    Readers are monkeypatched
    """
    assert hasattr(pd, "read_molecule_csv")
    assert hasattr(pd, "read_smi")
    assert hasattr(pd, "read_sdf")

def test_read_molecule_csv_add_smiles():
    """
    Adding SMILES columns in different ways when reading a molecule CSV
    """
    # add_smiles=True
    df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles=True)
    assert "Smiles SMILES" in df.columns
    assert all(isinstance(x, str) for x in df["Smiles SMILES"])

    # add_smiles='Smiles'
    df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles="Smiles")
    assert "Smiles SMILES" in df.columns
    assert all(isinstance(x, str) for x in df["Smiles SMILES"])

    # add_smiles=['Smiles']
    df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles", add_smiles=["Smiles"])
    assert "Smiles SMILES" in df.columns
    assert all(isinstance(x, str) for x in df["Smiles SMILES"])

    # add_smiles={'Smiles': 'Testy McTesterson'}
    df = oepd.read_molecule_csv(Path(ASSETS, "phenols_trunc.csv"), "Smiles",
                                add_smiles={"Smiles": "Testy McTesterson"})
    assert "Testy McTesterson" in df.columns
    assert all(isinstance(x, str) for x in df["Testy McTesterson"])

def test_read_smi():
    """
    Read a SMILES file into a DataFrame
    """
    # noinspection PyUnresolvedReferences
    df = pd.read_smi(Path(ASSETS, "10.smi"))
    assert len(df) == 10
    assert all(isinstance(mol, oechem.OEMolBase) for mol in df.Molecule)

def test_read_sdf():
    """
    Read an SD file with data
    """
    # Read all data as strings (object)
    df = oepd.read_sdf(Path(ASSETS, "10-tagged.sdf"))

    # Check the datatypes
    for col in df.columns:
        if col == "Molecule":
            assert isinstance(df.dtypes[col], MoleculeDtype)
        else:
            assert isinstance(df.dtypes[col], object)

    # Read a single column
    df = oepd.read_sdf(
        Path(ASSETS, "10-tagged.sdf"),
        usecols="Integer Tag"
    )

    assert "Integer Tag" in df.columns
    assert "Float Tag" not in df.columns
    assert "String Tag" not in df.columns

    # Read a single column and cast it to an integer
    df = oepd.read_sdf(
        Path(ASSETS, "10-tagged.sdf"),
        usecols="Integer Tag",
        numeric={"Integer Tag": "unsigned"}
    )

    assert "Integer Tag" in df.columns
    assert "Float Tag" not in df.columns
    assert "String Tag" not in df.columns

    # Check the dtype
    assert str(df.dtypes["Integer Tag"]) == "uint8"

    # Cast multiple numeric columns data
    df = oepd.read_sdf(
        Path(ASSETS, "10-tagged.sdf"),
        numeric={"Integer Tag": "unsigned", "Float Tag": "float"}
    )

    assert "Integer Tag" in df.columns
    assert "Float Tag" in df.columns
    assert "String Tag" in df.columns

    # Check the dtype
    assert str(df.dtypes["Integer Tag"]) == "uint8"
    assert str(df.dtypes["Float Tag"]) == "float32"

def test_to_smiles():
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

    assert expected == arr.to_smiles().tolist()

def test_series_to_smiles():
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

    assert expected == df.MOL.to_smiles().tolist()

def test_series_to_molecule_string():
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

    # Canonical isomeric SMILES
    df["TEST"] = df.MOL.to_molecule_strings(molecule_format="smiles")
    assert expected_strings == df.TEST.tolist()

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

    # SDF v3000
    df["TEST"] = df.MOL.to_molecule_strings(
        molecule_format=oechem.OEFormat_SDF,
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30
    )
    assert expected_strings == df.TEST.tolist()

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

    # SDF v3000 with forced b64 encoding
    df["TEST"] = df.MOL.to_molecule_strings(
        molecule_format=oechem.OEFormat_SDF,
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
        b64encode=True
    )
    assert expected_strings == df.TEST.tolist()

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

    # SDF v3000 gzipped
    df["TEST"] = df.MOL.to_molecule_strings(
        molecule_format=oechem.OEFormat_SDF,
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
        gzip=True
    )
    assert expected_strings == df.TEST.tolist()

    # ----------------------------------------------
    # SDF v3000 gzipped by extension
    # ----------------------------------------------

    # SDF v3000 gzipped by file format extension
    df["TEST"] = df.MOL.to_molecule_strings(
        molecule_format=".sdf.gz",
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
    )
    assert expected_strings == df.TEST.tolist()

def test_series_to_molecule_bytes():
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

    # Canonical isomeric SMILES
    df["TEST"] = df.MOL.to_molecule_bytes(molecule_format=oechem.OEFormat_SMI)
    assert expected_bytes == df.TEST.tolist()

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

    # SDF v3000
    df["TEST"] = df.MOL.to_molecule_bytes(
        molecule_format=oechem.OEFormat_SDF,
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30
    )
    assert expected_bytes == df.TEST.tolist()

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

    # SDF v3000 gzipped
    df["TEST"] = df.MOL.to_molecule_bytes(
        molecule_format=oechem.OEFormat_SDF,
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
        gzip=True
    )
    assert [b.rstrip(b'\x00') for b in expected_bytes] == df.TEST.tolist()

    # ----------------------------------------------
    # SDF v3000 gzipped by extension
    # ----------------------------------------------

    # SDF v3000 gzipped by file format extension
    df["TEST"] = df.MOL.to_molecule_bytes(
        molecule_format=".sdf.gz",
        flavor=oechem.OEGetDefaultOFlavor(oechem.OEFormat_SDF) | oechem.OEOFlavor_SDF_MV30,
    )
    assert [b.rstrip(b'\x00') for b in expected_bytes] == df.TEST.tolist()

def test_copy_molecule(test_mols):
    """
    Copy molecules
    """
    df = pd.DataFrame({"Molecule": pd.Series(MoleculeArray(copy_mols(test_mols)), dtype=MoleculeDtype())})
    df["Molecule2"] = df.Molecule.copy_molecules()

    molecules1 = set(df.Molecule.tolist())
    molecules2 = set(df.Molecule2.tolist())

    assert molecules1 != molecules2

def test_read_oedb():
    """
    Read data records
    """
    df = oepd.read_oedb(Path(ASSETS, "10.oedb"))
    assert "MolWt Halide Fraction (Calculated)" in df.columns
    assert "Heavy Atom Count (Calculated)" in df.columns
    assert "Molecule" in df.columns
    assert df.dtypes["MolWt Halide Fraction (Calculated)"] == float
    assert df.dtypes["Heavy Atom Count (Calculated)"] == int
    assert isinstance(df.dtypes["Molecule"], oepd.MoleculeDtype)

def test_to_molecule_csv():
    """Test exporting to molecule CSV"""
    with TemporaryDirectory() as tempdir:
        x = MoleculeArray.read_smi(Path(ASSETS, "10.smi"))
        df = pd.DataFrame([
            {"Title": x[0].GetTitle(), "MOL": x[0]},
            {"Title": "Invalid", "MOL": oechem.OEMol()},
            {"Title": x[1].GetTitle(), "MOL": x[1]},
        ])
        df["MOL"] = df.MOL.astype(MoleculeDtype())

        outpath = Path(tempdir, "test.csv")
        df.to_molecule_csv(outpath)
        assert outpath.exists()

        # Expected data
        expected_df = pd.read_csv(Path(ASSETS, "test-csv-expected.csv"))

        # Re-read the molecule CSV
        reread_df = pd.read_csv(outpath)
        assert expected_df.equals(reread_df)

def test_molecule_array_subsearch():
    """
    SMARTS matching in a MoleculeArray
    """
    x = MoleculeArray.read_sdf(Path(ASSETS, "10.sdf"))
    sulfones = np.where(x.subsearch('S(=O)=O'))
    assert len(sulfones) == 1
    assert sulfones[0] == 8

def test_series_subsearch():
    """
    SMARTS matching in a Pandas Series
    """
    df = oepd.read_sdf(Path(ASSETS, "10.sdf"))
    view = df[df.Molecule.subsearch("S(=O)=O")]
    assert len(view) == 1
    assert view.iloc[0].Title == 'Omeprazole'

def test_detect_molecule_columns():
    """
    Detect molecule columns in a Pandas DataFrame
    """
    # Create a dataframe ensuring non-molecule dtypes
    df = pd.DataFrame({
        "IsMolecule": pd.Series(MoleculeArray.read_sdf(Path(ASSETS, "10.sdf")).tolist(), dtype=object),
        "NotMolecule": pd.Series(list(range(10)), dtype=int),
        "IsAlsoMolecule": pd.Series(MoleculeArray.read_sdf(Path(ASSETS, "10.sdf")).tolist(), dtype=object)
    })

    # Do the column detection
    df.detect_molecule_columns()

    assert isinstance(df.dtypes["IsMolecule"], MoleculeDtype)
    assert isinstance(df.dtypes["IsAlsoMolecule"], MoleculeDtype)


def test_pandas_molecule_extensions_read_sdf():
    """Test pandas molecule extensions read SDF functionality"""
    df = oepd.read_sdf(Path(ASSETS, "10.sdf"))

    # Calculate molecular weight
    df["MW"] = df.Molecule.apply(oechem.OECalculateMolecularWeight)
    assert np.issubdtype(df.dtypes["MW"], np.floating)

    for _, row in df.iterrows():
        assert row["MW"] > 100, f'MW for {row["Title"]} should be > 100'
