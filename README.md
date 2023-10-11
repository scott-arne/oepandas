# OEPandas

**Author:** Scott Arne Johnson ([scott.johnson@bms.com](mailto:scott.johnson@bms.com))

## Description

Deep integration of OpenEye objects into Pandas. This introduces the ```MoleculeArray``` type for holding molecules in
NumPy arrays, and ```MoleculeArrayDtype``` for typing Pandas series.

# Usage

Simply adding ```import oepandas``` at the top of your file registers a number of [Pandas extensions](https://pandas.pydata.org/docs/development/extending.html) that 
support molecule handling. You can also use a number of useful methods directly from OEPandas itself.

## Readers

The following readers are available for various molecule formats. You can use these readers either directly from the
```oepandas``` package or from ```pandas``` after executing ```import oepandas```.

For example:

```python
import pandas as pd
import oepandas as oepd

# This works
df = oepd.read_smi("myfile.smi")

# This also works
df = pd.read_smi("myfile.smi")
```

The following table lists all of the readers.

| File Type | Method | Description |

The ```read_oeb``` function for reading .oeb and .oeb.gz files is probably the most complicated, because you have the 
following options:

1. Whether to read SD data
2. Whether to read OpenEye generic data
3. Whether to split conformers

Data handling in the OpenEye Toolkits can be a bit confusing.

Splitting conformers will create a new row in the DataFrame for each conformer of each molecule. The data for each
conformer will be read into the table, but *NOT* the data for multiconformer molecule. Oh boy, that sounds confusing.

# Examples


## Read a CSV with molecules and create a dataframe


```python
import pandas as pd
import oepandas as oepd

df = oepd.read_molecule_csv()

# TODO!
```

# Development

This package makes use of [Pandas extensions](https://pandas.pydata.org/docs/development/extending.html) to make working
with OpenEye datatypes easier. Nothing needs to be done other than importing the package. However, feel free to use some
of the utility functions for your own code -- it can be useful beyond Pandas!

# Compatibility Notes

At the moment the requirements are pretty strict on Pandas and the OpenEye Toolkits. This is simply because the
package hasn't been tested for backwards compatibility on previous versions.