# OEPandas

**Author:** Scott Arne Johnson ([scott.johnson@bms.com](mailto:scott.johnson@bms.com))

## Description

Deep integration of OpenEye objects into Pandas. This introduces the ```MoleculeArray``` type for holding molecules in
NumPy arrays, and ```MoleculeArrayDtype``` for typing Pandas series.

# Usage

This package makes use of [Pandas extensions](https://pandas.pydata.org/docs/development/extending.html) to make working
with OpenEye datatypes easier. Nothing needs to be done other than importing the package. However, feel free to use some
of the utility functions for your own code -- it can be useful beyond Pandas!

```python
import pandas as pd

# This is it!
import oepandas

# TODO!
```

# Development

Coming soon...

# Compatibility Notes

At the moment the requirements are pretty strict on Pandas and the OpenEye Toolkits. This is simply because the
package hasn't been tested for backwards compatibility on previous versions.