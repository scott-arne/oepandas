# OEPandas

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenEye Toolkits](https://img.shields.io/badge/OpenEye-2023.1.0+-green.svg)](https://www.eyesopen.com/toolkits)
[![Pandas 2.1+](https://img.shields.io/badge/pandas-2.1+-orange.svg)](https://pandas.pydata.org/)

**Deep integration of OpenEye objects into Pandas DataFrames with native support for molecules and design units.**

---

## 🚀 Quick Start

```bash
pip install oepandas
```

```python
import oepandas as oepd
from openeye import oechem

# Load molecule data from various formats
df = oepd.read_sdf("molecules.sdf")
df = oepd.read_oeb("molecules.oeb.gz")  
df = oepd.read_molecule_csv("data.csv", molecule_columns="SMILES")

# Use pandas normally with molecules
df["num_oxygens"] = df.Molecule.apply(lambda mol: oechem.OECount(mol, oechem.OEIsOxygen()))
```

## ✨ Features

- **Native OpenEye Integration**: Store `OEGraphMol` and `OEDesignUnit` objects directly in pandas DataFrames
- **Multiple File Formats**: Read SDF, OEB, CSV, SMI, OEDB, and OEDU files seamlessly
- **Pandas Extensions**: Rich accessor methods for molecular operations (`.to_smiles()`, `.depict()`, `.get_mols()`, etc.)
- **Type Safety**: Full type hints and PyCharm IDE support
- **Performance**: Optimized for large molecular datasets

---

## 📖 Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [Reading Molecular Data](#reading-molecular-data)
  - [Working with Molecules](#working-with-molecules)
  - [Design Units](#design-units)
- [Advanced Features](#advanced-features)
  - [Custom Accessors](#custom-accessors)
- [API Reference](#api-reference)
- [Development](#development)

---

## 🔧 Installation

### Requirements
- Python 3.10+
- pandas 2.1.0+
- numpy
- OpenEye Toolkits 2023.1.0+
- more-itertools

### Install from PyPI
```bash
pip install oepandas
```

### Development Installation
```bash
git clone <repository-url>
cd oepandas
pip install -e ".[dev]"
```

---

## 📚 Basic Usage

### Reading Molecular Data

OEPandas provides readers for all major chemical file formats supported by the OpenEye Toolkits, including their
proprietary formats and record files:

```python
import oepandas as oepd

# SDF files - molecules with properties
df = oepd.read_sdf("molecules.sdf")
print(df.head())
#                                   Molecule    Title      MolWt
# 0  <oechem.OEGraphMol; proxy...>   Aspirin   180.157
# 1  <oechem.OEGraphMol; proxy...> Ibuprofen   206.281

# CSV files with SMILES
df = oepd.read_molecule_csv("data.csv", molecule_columns="SMILES")

# OEB files (binary format)
df = oepd.read_oeb("molecules.oeb.gz")

# Design unit files
df = oepd.read_oedu("complexes.oedu")

# OERecord databases
df = oepd.read_oedb("records.oedb")
```

### Working with Molecules

Once loaded, use pandas normally with rich molecular extensions:

```python
from openeye import oechem

# Standard pandas operations work
filtered_df = df[df.MolWt > 200]

# Rich molecular accessors
smiles = df.Molecule.to_smiles()
images = df.Molecule.depict(width=300, height=200)

# Apply OpenEye functions
df["oxygen_count"] = df.Molecule.apply(lambda mol: oechem.OECount(mol, oechem.OEIsOxygen()))
df["has_ring"] = df.Molecule.apply(lambda mol: oechem.OEDetermineRingMembership(mol) > 0)

# Convert to different formats
df["canonical_smiles"] = df.Molecule.to_smiles(flavor=oechem.OESMILESFlag_Canonical)
```

### Design Units

Work with protein-ligand complexes:

```python
# Read design unit file
df = oepd.read_oedu("protein_ligand_complexes.oedu")

# Extract components
df["Ligand"] = df.Design_Unit.get_ligands()
df["Protein"] = df.Design_Unit.get_proteins()

# Analyze components
df["ligand_mw"] = df.Ligand.apply(oechem.OECalculateMolecularWeight)
df["protein_residues"] = df.Protein.apply(lambda mol: oechem.OECount(mol, oechem.OEIsResidue()))
```

---

## 🔥 Advanced Features


### Custom Accessors

OEPandas registers many useful pandas accessors automatically:

```python
# Molecular property accessors
df.Molecule.copy_molecules()          # Deep copy molecules
df.Molecule.to_smiles()              # Generate SMILES strings
df.Molecule.depict()                 # Generate 2D depictions
df.Molecule.as_molecule()            # Convert to different formats

# Design unit accessors  
df.Design_Unit.get_ligands()         # Extract ligand molecules
df.Design_Unit.get_proteins()        # Extract protein molecules
df.Design_Unit.copy_design_units()   # Deep copy design units

# DataFrame-level accessors
df.oechem.write_sdf("output.sdf")    # Write to SDF file
df.oechem.write_oeb("output.oeb")    # Write to OEB file
```

---

## 📋 API Reference

### File Readers
- `read_sdf(filename, **kwargs)` - Read SDF files
- `read_oeb(filename, **kwargs)` - Read OEB files  
- `read_oedu(filename, **kwargs)` - Read OEDU files
- `read_molecule_csv(filename, molecule_columns, **kwargs)` - Read CSV with molecules
- `read_smi(filename, **kwargs)` - Read SMILES files
- `read_oedb(filename, **kwargs)` - Read OEDB files

### Core Classes  
- `MoleculeArray` / `MoleculeDtype` - Pandas extension for molecules
- `DesignUnitArray` / `DesignUnitDtype` - Pandas extension for design units
- `DisplayArray` / `DisplayDtype` - Pandas extension for molecular displays

---


## 🛠 Development

### Running Tests
```bash
invoke test
# or
pytest
```

### Building Package
```bash
invoke build
# or  
python -m build
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under a proprietary license. See the LICENSE file for details.

---

## 👤 Author

**Scott Arne Johnson**
- Email: [scott.arne.johnson@gmail.com](mailto:scott.arne.johnson@gmail.com)

---

## 🔗 Related Projects

- [OpenEye Toolkits](https://www.eyesopen.com/toolkits) - The underlying cheminformatics toolkit
- [Pandas](https://pandas.pydata.org/) - Data analysis library that OEPandas extends

---

*Made with ❤️ for the computational chemistry community*