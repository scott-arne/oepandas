# OEPandas

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEye Toolkits](https://img.shields.io/badge/OpenEye-2025.2.1+-green.svg)](https://www.eyesopen.com/toolkits)
[![Pandas 2.2+](https://img.shields.io/badge/pandas-2.2+-orange.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep integration of OpenEye objects into Pandas DataFrames with native support for molecules and design units.

OEPandas extends Pandas with custom extension arrays that store OpenEye `OEMol` and `OEDesignUnit` objects as first-class DataFrame column types. This enables seamless interoperability between OpenEye's cheminformatics capabilities and Pandas' data analysis workflows.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
  - [Reading Molecular Data](#reading-molecular-data)
  - [Working with Molecules](#working-with-molecules)
  - [Design Units](#design-units)
  - [Data Quality and Filtering](#data-quality-and-filtering)
- [Writing Data](#writing-data)
- [API Reference](#api-reference)
  - [File Readers](#file-readers)
  - [DataFrame Accessor Methods](#dataframe-accessor-methods-dfchem)
  - [Series Accessor Methods](#series-accessor-methods-serieschem)
  - [Extension Arrays and Dtypes](#extension-arrays-and-dtypes)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

---

## Installation

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| pandas | 2.2.0+ |
| numpy | 2.0.0+ |
| OpenEye Toolkits | 2025.2.1+ |

### OpenEye Toolkits License

OpenEye Toolkits requires a commercial license. However, **free licenses are available for academic and non-profit institutions**. Visit [OpenEye Scientific](https://www.eyesopen.com/academic-licensing) to request an academic license.

### Install from PyPI

```bash
pip install oepandas
```

### Development Installation

```bash
git clone https://github.com/scott-arne/oepandas.git
cd oepandas
pip install -e ".[dev]"
```

---

## Quick Start

```python
import oepandas as oepd
from openeye import oechem

# Load molecule data from various formats
df = oepd.read_sdf("molecules.sdf")
df = oepd.read_oeb("molecules.oeb.gz")
df = oepd.read_molecule_csv("data.csv", molecule_columns="SMILES")

# Use pandas normally with molecules
df["num_oxygens"] = df.Molecule.apply(lambda mol: oechem.OECount(mol, oechem.OEIsOxygen()))

# Generate SMILES strings
smiles = df.Molecule.chem.to_smiles()

# Filter invalid molecules
df_valid = df.chem.filter_valid("Molecule")

# Write to file
df.chem.to_sdf("output.sdf", primary_molecule_column="Molecule")
```

---

## Basic Usage

### Reading Molecular Data

OEPandas provides readers for all major chemical file formats supported by the OpenEye Toolkits:

```python
import oepandas as oepd

# SDF files - molecules with SD data as columns
df = oepd.read_sdf("molecules.sdf")

# OEB files (binary format, supports conformers)
df = oepd.read_oeb("molecules.oeb.gz")

# SMILES files
df = oepd.read_smi("molecules.smi")

# CSV files with SMILES column
df = oepd.read_molecule_csv("data.csv", molecule_columns="SMILES")

# OERecord databases
df = oepd.read_oedb("records.oedb")

# Design unit files (protein-ligand complexes)
df = oepd.read_oedu("complexes.oedu")
```

### Working with Molecules

Once loaded, molecules are stored as `MoleculeDtype` columns. Standard pandas operations work seamlessly:

```python
from openeye import oechem

# Standard pandas operations
filtered_df = df[df.MolWt > 200]
sorted_df = df.sort_values("Title")

# Apply OpenEye functions directly
df["MW"] = df.Molecule.apply(oechem.OECalculateMolecularWeight)
df["LogP"] = df.Molecule.apply(oechem.OEGetXLogP)
df["HBD"] = df.Molecule.apply(lambda m: oechem.OECount(m, oechem.OEIsHBondDonor()))

# Use the .chem accessor for molecular operations
df["SMILES"] = df.Molecule.chem.to_smiles()
df["MolCopy"] = df.Molecule.chem.copy_molecules()

# Substructure searching with SMARTS
has_carboxylic_acid = df.Molecule.chem.subsearch("C(=O)O")
df_acids = df[has_carboxylic_acid]
```

### Design Units

Work with protein-ligand complexes stored as `DesignUnitDtype`:

```python
# Read design unit file
df = oepd.read_oedu("protein_ligand_complexes.oedu")

# Extract components using .chem accessor
df["Ligand"] = df.Design_Unit.chem.get_ligands()
df["Protein"] = df.Design_Unit.chem.get_proteins()

# Analyze components
df["ligand_mw"] = df.Ligand.apply(oechem.OECalculateMolecularWeight)

# Deep copy design units
df["DU_copy"] = df.Design_Unit.chem.copy_design_units()
```

### Data Quality and Filtering

OEPandas provides methods to check and filter molecule validity:

```python
# Check which molecules are valid
validity = df.Molecule.chem.is_valid()
print(f"Valid molecules: {validity.sum()}")
print(f"Invalid molecules: {(~validity).sum()}")

# Filter to keep only valid molecules
df_valid = df.chem.filter_valid("Molecule")

# Filter multiple columns at once
df_valid = df.chem.filter_valid(["Molecule", "Product"])

# Add validity as a column for inspection
df["is_valid"] = df.Molecule.chem.is_valid()
```

---

## Writing Data

Export DataFrames to various molecular file formats using the `.chem` accessor:

```python
# Write to SDF (columns become SD tags)
df.chem.to_sdf(
    "output.sdf",
    primary_molecule_column="Molecule",
    title_column="Name",
    columns=["Activity", "MW"]  # Include as SD tags
)

# Write to SMILES file
df.chem.to_smi(
    "output.smi",
    primary_molecule_column="Molecule",
    title_column="Name"
)

# Write to CSV (molecules as SMILES strings)
df.chem.to_molecule_csv(
    "output.csv",
    molecule_format="smiles"
)

# Write to OERecord database
df.chem.to_oedb(
    "output.oedb",
    primary_molecule_column="Molecule"
)
```

---

## API Reference

### File Readers

#### `read_sdf()`

Read SD (Structure Data) files into a DataFrame.

```python
oepd.read_sdf(
    filepath_or_buffer,
    *,
    flavor=oechem.OEIFlavor_SDF_Default,
    molecule_column="Molecule",
    title_column="Title",
    add_smiles=None,
    molecule_columns=None,
    usecols=None,
    numeric=None,
    conformer_test="default",
    read_sd_data=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, Path, buffer | required | Path to SDF file or readable buffer |
| `flavor` | int | `OEIFlavor_SDF_Default` | OpenEye SDF reader flavor |
| `molecule_column` | str | `"Molecule"` | Name of molecule column |
| `title_column` | str, None | `"Title"` | Name of title column (None to skip) |
| `add_smiles` | bool, str, list | `None` | Add SMILES column(s) |
| `molecule_columns` | str, list | `None` | Additional columns to convert to molecules |
| `usecols` | str, list | `None` | SD tags to read (None for all) |
| `numeric` | str, list, dict | `None` | Columns to convert to numeric |
| `conformer_test` | str | `"default"` | Conformer combining strategy: "default", "absolute", "absolute_canonical", "isomeric", "omega" |
| `read_sd_data` | bool | `True` | Read SD data into columns |

#### `read_oeb()`

Read OpenEye Binary (OEB) files into a DataFrame.

```python
oepd.read_oeb(
    filepath_or_buffer,
    *,
    flavor=oechem.OEIFlavor_SDF_Default,
    molecule_column="Molecule",
    title_column="Title",
    add_smiles=None,
    molecule_columns=None,
    read_generic_data=True,
    read_sd_data=True,
    usecols=None,
    numeric=None,
    conformer_test="default",
    combine_tags="prefix",
    sd_prefix="SD Tag: ",
    generic_prefix="Generic Tag: "
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, Path, buffer | required | Path to OEB file or readable buffer |
| `read_generic_data` | bool | `True` | Read generic data |
| `read_sd_data` | bool | `True` | Read SD data |
| `combine_tags` | str | `"prefix"` | Tag conflict resolution: "prefix", "prefer_sd", "prefer_generic" |
| `sd_prefix` | str | `"SD Tag: "` | Prefix for SD data columns |
| `generic_prefix` | str | `"Generic Tag: "` | Prefix for generic data columns |

*Other parameters same as `read_sdf()`*

#### `read_smi()`

Read SMILES files into a DataFrame.

```python
oepd.read_smi(
    filepath_or_buffer,
    *,
    cx=False,
    flavor=None,
    add_smiles=False,
    add_inchi_key=False,
    molecule_column="Molecule",
    title_column="Title",
    smiles_column_name="SMILES",
    inchi_key_column_name="InChI Key"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, Path | required | Path to SMILES file |
| `cx` | bool | `False` | Read CXSMILES format |
| `flavor` | int | `None` | SMILES flavor |
| `add_smiles` | bool | `False` | Include re-canonicalized SMILES column |
| `add_inchi_key` | bool | `False` | Include InChI Key column |
| `molecule_column` | str | `"Molecule"` | Name of molecule column |
| `title_column` | str | `"Title"` | Name of title column |
| `smiles_column_name` | str | `"SMILES"` | Name of SMILES column |
| `inchi_key_column_name` | str | `"InChI Key"` | Name of InChI Key column |

#### `read_molecule_csv()`

Read CSV files with molecule columns.

```python
oepd.read_molecule_csv(
    filepath_or_buffer,
    molecule_columns,
    *,
    add_smiles=None,
    **kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, Path, buffer | required | Path to CSV file |
| `molecule_columns` | str, dict, "detect" | required | Column(s) containing molecules, or "detect" for auto-detection |
| `add_smiles` | bool, str, list | `None` | Add SMILES column(s) |
| `**kwargs` | | | Additional arguments passed to `pd.read_csv()` |

#### `read_oedb()`

Read OpenEye Database (OERecord) files into a DataFrame.

```python
oepd.read_oedb(
    fp,
    *,
    usecols=None,
    int_na=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp` | str, Path | required | Path to OEDB file |
| `usecols` | str, list | `None` | Columns to read (None for all) |
| `int_na` | int | `None` | Value for integer NaN (None uses float NaN) |

#### `read_oedu()`

Read Design Unit files into a DataFrame.

```python
oepd.read_oedu(
    filepath_or_buffer,
    *,
    design_unit_column="Design_Unit",
    title_column="Title",
    generic_data=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath_or_buffer` | str, Path | required | Path to OEDU file |
| `design_unit_column` | str | `"Design_Unit"` | Name of design unit column |
| `title_column` | str | `"Title"` | Name of title column |
| `generic_data` | bool | `True` | Read generic data into columns |

---

### DataFrame Accessor Methods (`df.chem.*`)

Access these methods via `df.chem.<method>()`:

#### `as_molecule()`

Convert column(s) to MoleculeDtype.

```python
df.chem.as_molecule(
    columns,
    *,
    molecule_format=None,
    inplace=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, list | required | Column name(s) to convert |
| `molecule_format` | str, int | `None` | Format for parsing (default: SMILES) |
| `inplace` | bool | `False` | Modify DataFrame in place |

#### `as_design_unit()`

Convert column(s) to DesignUnitDtype.

```python
df.chem.as_design_unit(columns, *, inplace=False)
```

#### `filter_valid()`

Filter rows to keep only those with valid molecules.

```python
df.chem.filter_valid(columns, *, inplace=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, list | required | MoleculeDtype column(s) to check |
| `inplace` | bool | `False` | Modify DataFrame in place |

#### `detect_molecule_columns()`

Auto-detect and convert molecule columns based on predominant type.

```python
df.chem.detect_molecule_columns(*, sample_size=25)
```

#### `to_sdf()`

Write DataFrame to SDF file.

```python
df.chem.to_sdf(
    fp,
    primary_molecule_column,
    *,
    title_column=None,
    columns=None,
    index=True,
    index_tag="index",
    secondary_molecules_as="smiles",
    secondary_molecule_flavor=None,
    gzip=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp` | str, Path | required | Output file path |
| `primary_molecule_column` | str | required | Column with molecules |
| `title_column` | str | `None` | Column for titles |
| `columns` | str, list | `None` | Columns to include as SD tags (None for all) |
| `index` | bool | `True` | Include index as SD tag |
| `index_tag` | str | `"index"` | Name of index SD tag |
| `secondary_molecules_as` | str, int | `"smiles"` | Encoding for other molecule columns |
| `gzip` | bool | `False` | Gzip compress output |

#### `to_smi()`

Write DataFrame to SMILES file.

```python
df.chem.to_smi(
    fp,
    primary_molecule_column,
    *,
    flavor=None,
    molecule_format=oechem.OEFormat_SMI,
    title_column=None,
    gzip=False
)
```

#### `to_molecule_csv()`

Write DataFrame to CSV with molecules as strings.

```python
df.chem.to_molecule_csv(
    fp,
    *,
    molecule_format="smiles",
    flavor=None,
    gzip=False,
    b64encode=False,
    columns=None,
    index=True,
    sep=',',
    **kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `molecule_format` | str, int | `"smiles"` | Output format for molecules |
| `b64encode` | bool | `False` | Base64 encode molecule strings |
| `**kwargs` | | | Additional arguments passed to pandas CSV writer |

#### `to_oedb()`

Write DataFrame to OERecord database.

```python
df.chem.to_oedb(
    fp,
    primary_molecule_column=None,
    *,
    title_column=None,
    columns=None,
    index=True,
    index_label="index",
    sample_size=25,
    safe=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_molecule_column` | str | `None` | Molecule column (None creates OERecord, not OEMolRecord) |
| `sample_size` | int | `25` | Sample size for type detection |
| `safe` | bool | `True` | Type check before writing |

---

### Series Accessor Methods (`series.chem.*`)

Access these methods via `series.chem.<method>()`:

#### Molecule Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `copy_molecules()` | `Series[MoleculeDtype]` | Deep copy all molecules |
| `is_valid()` | `Series[bool]` | Boolean mask of valid molecules |
| `as_molecule(molecule_format=None)` | `Series[MoleculeDtype]` | Convert series to molecules |
| `to_molecule(molecule_format=None)` | `Series[MoleculeDtype]` | Convert from strings to molecules |
| `to_molecule_bytes(molecule_format=OEFormat_SMI, flavor=None, gzip=False)` | `Series[bytes]` | Convert to byte strings |
| `to_molecule_strings(molecule_format="smiles", flavor=None, gzip=False, b64encode=False)` | `Series[str]` | Convert to string representations |
| `to_smiles(flavor=OESMILESFlag_ISOMERIC)` | `Series[str]` | Convert to SMILES strings |
| `subsearch(pattern, adjustH=False)` | `Series[bool]` | Substructure search with SMARTS pattern |

#### Design Unit Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `copy_design_units()` | `Series[DesignUnitDtype]` | Deep copy all design units |
| `get_ligands(clear_titles=False)` | `Series[MoleculeDtype]` | Extract ligand molecules |
| `get_proteins(clear_titles=False)` | `Series[MoleculeDtype]` | Extract protein molecules |
| `get_components(mask)` | `Series[MoleculeDtype]` | Extract components by mask |
| `as_design_unit()` | `Series[DesignUnitDtype]` | Convert series to design units |

---

### Extension Arrays and Dtypes

OEPandas provides three custom Pandas extension types:

| Array Class | Dtype Class | Underlying Type | Description |
|-------------|-------------|-----------------|-------------|
| `MoleculeArray` | `MoleculeDtype` | `oechem.OEMol` | Stores molecular structures |
| `DesignUnitArray` | `DesignUnitDtype` | `oechem.OEDesignUnit` | Stores protein-ligand complexes |
| `DisplayArray` | `DisplayDtype` | `oedepict.OE2DMolDisplay` | Stores 2D molecular depictions |

#### MoleculeArray Class Methods

```python
# Create from file
arr = MoleculeArray.read_sdf("file.sdf")
arr = MoleculeArray.read_oeb("file.oeb")
arr = MoleculeArray.read_smi("file.smi")

# Create from sequences
arr = MoleculeArray._from_sequence(["CCO", "c1ccccc1"])
arr = MoleculeArray._from_sequence_of_strings(["CCO", "c1ccccc1"])

# Conversion methods
smiles = arr.to_smiles(flavor=OESMILESFlag_ISOMERIC)
strings = arr.to_molecule_strings(molecule_format="sdf")
bytes_arr = arr.to_molecule_bytes(molecule_format=OEFormat_OEB)

# Substructure searching
matches = arr.subsearch("c1ccccc1")

# Utility methods
arr.deepcopy()        # Deep copy
arr.valid()           # Boolean mask of valid molecules
arr.isna()            # Boolean mask of None values
arr.dropna()          # Remove None values
arr.fillna(value)     # Fill None values
```

#### DesignUnitArray Class Methods

```python
# Create from file
arr = DesignUnitArray.read_oedu("file.oedu")

# Extract components (returns MoleculeArray)
ligands = arr.get_ligands(clear_titles=False)
proteins = arr.get_proteins(clear_titles=False)
components = arr.get_components(mask)

# Utility methods
arr.deepcopy()        # Deep copy
arr.valid()           # Boolean mask of valid design units
```

---

## Examples

Comprehensive Jupyter notebooks are available in the `examples/` directory:

- **01_getting_started.ipynb** - Basic usage, molecular calculations, data manipulation, validity checking
- **02_advanced_features.ipynb** - File I/O, design units, data quality filtering, performance optimization, ML integration

### Example: Complete Workflow

```python
import oepandas as oepd
from openeye import oechem

# 1. Load data
df = oepd.read_sdf("molecules.sdf", add_smiles=True)

# 2. Filter invalid molecules
df = df.chem.filter_valid("Molecule")

# 3. Calculate properties
df["MW"] = df.Molecule.apply(oechem.OECalculateMolecularWeight)
df["LogP"] = df.Molecule.apply(oechem.OEGetXLogP)
df["HBD"] = df.Molecule.apply(lambda m: oechem.OECount(m, oechem.OEIsHBondDonor()))
df["HBA"] = df.Molecule.apply(lambda m: oechem.OECount(m, oechem.OEIsHBondAcceptor()))

# 4. Filter by Lipinski's Rule of Five
lipinski = (
    (df.MW <= 500) &
    (df.LogP <= 5) &
    (df.HBD <= 5) &
    (df.HBA <= 10)
)
df_druglike = df[lipinski]

# 5. Substructure search for carboxylic acids
has_acid = df_druglike.Molecule.chem.subsearch("C(=O)O")
df_acids = df_druglike[has_acid]

# 6. Export results
df_acids.chem.to_sdf(
    "druglike_acids.sdf",
    primary_molecule_column="Molecule",
    title_column="Title",
    columns=["MW", "LogP"]
)
```

---

## Development

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

### Project Structure

```
oepandas/
├── oepandas/
│   ├── __init__.py              # Public API exports
│   ├── pandas_extensions.py     # Readers, writers, accessors
│   ├── util.py                  # Utility functions
│   ├── exception.py             # Custom exceptions
│   └── arrays/
│       ├── __init__.py
│       ├── base.py              # OEExtensionArray base class
│       ├── molecule.py          # MoleculeArray, MoleculeDtype
│       ├── design_unit.py       # DesignUnitArray, DesignUnitDtype
│       └── display.py           # DisplayArray, DisplayDtype
├── tests/                       # Test suite
├── examples/                    # Jupyter notebooks
└── pyproject.toml              # Project configuration
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Scott Arne Johnson**
- Email: [scott.arne.johnson@gmail.com](mailto:scott.arne.johnson@gmail.com)

---

## Related Projects

- [OpenEye Toolkits](https://www.eyesopen.com/toolkits) - The underlying cheminformatics toolkit
- [Pandas](https://pandas.pydata.org/) - Data analysis library that OEPandas extends
