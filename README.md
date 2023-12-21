# OEPandas

**Author:** Scott Arne Johnson ([scott.johnson@bms.com](mailto:scott.johnson@bms.com))

## Description

Deep integration of OpenEye objects into Pandas. This introduces the ```MoleculeArray``` type for holding molecules in
NumPy arrays, and ```MoleculeArrayDtype``` for typing Pandas series.

# Usage

Simply adding ```import oepandas``` at the top of your file registers a number of [Pandas extensions](https://pandas.pydata.org/docs/development/extending.html) that 
support molecule handling. You can also use a number of useful methods directly from OEPandas itself.

## Getting Started with Molecules

First let's experiment with three files in the test directory that all contain the same data. We'll start with the
CSV file, which contains the following data for 5 molecules:

```text
SMILES,TITLE,MolWt,NumAcceptors,NumDonors
CC(=O)Oc1ccccc1C(=O)O,Aspirin,180.15742000000003,2,1
CC(C)Cc1ccc(cc1)C(C)C(=O)O,Ibuprofen,206.28082,1,1
...
```

Because a CSV can contain any arbitrary data, we need to tell OEPandas which columns contain molecules:

```python
import oepandas as oepd

# Read the CSV version of the data
df = oepd.read_molecule_csv("tests/assets/5.csv", molecule_columns="SMILES")

df.head()
```

This will output the following (and if you are using [cnotebook](https://scsgit.rdcloud.bms.com/CADD/cnotebook) in a
Jupyter Notebook, you'll see a pretty formatted table with molecule depictions):

```text
                                              SMILES          TITLE  \
0  <oechem.OEGraphMol; proxy of <Swig Object of t...        Aspirin   
1  <oechem.OEGraphMol; proxy of <Swig Object of t...      Ibuprofen   
2  <oechem.OEGraphMol; proxy of <Swig Object of t...  Acetaminophen   
3  <oechem.OEGraphMol; proxy of <Swig Object of t...       Caffeine   
4  <oechem.OEGraphMol; proxy of <Swig Object of t...       Diazepam   

       MolWt  NumAcceptors  NumDonors  
0  180.15742             2          1  
1  206.28082             1          1  
2  151.16256             1          2  
3  194.19060             3          0  
4  284.74022             2          0  
```

Let's open the exact same data in SD format:

```python
import oepandas as oepd

# Read the SDF version of the data
df = oepd.read_sdf("tests/assets/5.sdf")

df.head()
```

Everything looks exactly the same as above with one exception. Instead of a SMILES column, we now have a Molecule
column. This is because the CSV had a column explicitly called SMILES that we converted to molecules, whereas the
SD file just has a bunch of molecules in it. We can control the name of the molecule (and title) columns with
```molecule_column_name``` (and ```title_column_name```) in ```read_sdf```.

One gotcha with SD data: they are always text! Schrodinger gets around this by having a 
[special naming convention for tags](https://www.schrodinger.com/sites/default/files/s3/public/python_api/2023-3/core_concepts.html#properties)
but that's not yet supported in the package (and would not work with the above tags).

If you want to specify the columns to make numeric:

```python
import oepandas as oepd

# Read the SDF version of the data
df = oepd.read_sdf("tests/assets/5.sdf", numeric=["MolWt", "NumAcceptors", "NumDonors"])

df.head()
```

This uses logic within Pandas to figure out the right type for each column listed, note that it correctly differentiated
between the floating point MolWt column and integral NumAcceptors and NumDonors columns.

```text
Molecule        molecule
Title             object
MolWt            float64
NumAcceptors       int64
NumDonors          int64
dtype: object
```

You can also read OEB files. Note that if data is stored using [OpenEye Generic data](https://docs.eyesopen.com/toolkits/python/oechemtk/genericdata.html),
the type is automatically determined. If your molecules have SD data, they'll all come in as text.

```text
import oepandas as oepd

# Read the SDF version of the data
df = oepd.read_oeb("tests/assets/5.oeb.gz")

df.head()
```

The data and data types are identical to above.

## It's just Pandas

The nice thing is that once the molecules are in the DataFrame, it's all just Pandas. If I wanted to add a column
that counts the number of oxygen atoms in each molecule using standard OpenEye logic:

```python
import oepandas as oepd
from openeye import oechem

# Read the data
df = oepd.read_oeb("tests/assets/5.oeb.gz")

# Use standard OpenEye logic to count the number of oxygens in each molecule
df["OxygenCount"] = df.Molecule.apply(lambda mol: oechem.OECount(mol, oechem.OEIsOxygen()))

df.head()
```

We get the following:

```text
   ...         Title      MolWt  NumAcceptors  NumDonors  OxygenCount
0  ...       Aspirin  180.15742             2          1            4
1  ...     Ibuprofen  206.28082             1          1            2
2  ... Acetaminophen  151.16256             1          2            2
3  ...      Caffeine  194.19060             3          0            2
4  ...      Diazepam  284.74022             2          0            1
```

## Getting Started with Design Units

You can also read design unit files the exact same way. You can see below how we can get the ligand and protein
molecules from the design unit and use them in a few simple calculations.

```python
import oepandas as oepd
from openeye import oechem

# Read the data
df = oepd.read_oedu("tests/assets/2.oedu")

# Use standard OpenEye logic to count the number of oxygens in each molecule
df["Ligand"] = df.Design_Unit.get_ligands()
df["Ligand_SMILES"] = df.Ligand.apply(oechem.OEMolToSmiles)
df["Protein"] = df.Design_Unit.get_proteins()
df["Num_CAlphas"] = df.Protein.apply(lambda mol: oechem.OECount(mol, oechem.OEIsCAlpha()))

df.head()
```

You'll see the following:

```text
                 Design_Unit                   Title              Ligand            Protein  Num_CAlphas   Ligand_SMILES
0  <oechem.OEDesignUnit; ...   1JFF(AB) > TA1(B-601)  <oechem.OEMol; ...  <oechem.OEMol; ...         837  CC1[C@H](C[...
1  <oechem.OEDesignUnit; ...  1TVK(AB) >  EP(B-1001)  <oechem.OEMol; ...  <oechem.OEMol; ...         836  Cc1nc(cs1)/...
```


# Development

This package makes use of [Pandas extensions](https://pandas.pydata.org/docs/development/extending.html) to make working
with OpenEye datatypes easier. Nothing needs to be done other than importing the package. However, feel free to use some
of the utility functions for your own code -- it can be useful beyond Pandas!

# Compatibility Notes

At the moment the requirements are pretty strict on Pandas and the OpenEye Toolkits. This is simply because the
package hasn't been tested for backwards compatibility on previous versions.