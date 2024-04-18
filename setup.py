import re
import sys
import ast
from pathlib import Path
from json import dumps
from setuptools import setup, find_packages


########################################################################################################################
# Requirements for oepandas
########################################################################################################################

requirements = [
    "pandas>=2.1.0",
    "numpy",
    "openeye-toolkits>=2023.1.0"
    "more-itertools"
]

########################################################################################################################


if sys.version_info < (3, 10):
    sys.exit("Sorry, Python < 3.10 is not supported")


if "--requires" in sys.argv:
    print(dumps(requirements))
    exit()

if "--requirements" in sys.argv:
    print("\n".join(requirements))
    exit()

# Get the package version number
init_file = Path("oepandas", "__init__.py")
with open(init_file, "rt") as f:
    m = re.search(r'__version__\s+=\s+(.*)', f.read())

    # __version__ could not be found
    if m is None:
        raise AttributeError(f'__version__ not assigned in {init_file}')

    # Get the version string literal
    version = ast.literal_eval(m.group(1))


setup(
    name="oepandas",
    version=version,
    packages=find_packages(where=".", exclude=["*tests*", "*test*"]),
    author="Scott Arne Johnson",
    author_email="scott.johnson@bms.com",
    description="OpenEye Pandas integration",
    license="Other/Proprietary License",
    install_requires=requirements,
    extras_require={
        "dev": [
            "invoke"
        ]
    }
)
