import itertools
import math
import typing
from collections import namedtuple
from collections.abc import Iterable, Sequence
from copy import copy as shallow_copy
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import scipy.special
from numpy.typing import DTypeLike
from openeye import oechem, oegraphsim

# noinspection PyProtectedMember
from pandas._typing import Dtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from tqdm.auto import tqdm

from .base import OEExtensionArray

########################################################################################################################
# Fingerprint conversion
########################################################################################################################

OEFingerprintInfo = namedtuple("OEFingerPrintInfo", ("hex_length", "size", "ngroups"))


def get_openeye_fp_info(fp):
    """
    Get information on an OpenEye fingerprint for conversion
    :param fp: OEFingerPrint or OEFingerPrint hex string
    :type fp: oegraphsim.OEFingerPrint or str
    :return: OEFingerPrintInfo
    """
    if isinstance(fp, str):
        fp_hex = fp
    elif isinstance(fp, oegraphsim.OEFingerPrint):
        fp_hex = fp.ToHexString()  # type: str
    else:
        raise TypeError(f'Fingerprint must be either hex string or OEFingerprint not {type(fp).__name__}')

    # Size of the fingerprint
    hex_length = len(fp_hex) - 1
    fp_size = 4 * hex_length

    # Number of bits to nibble off at the end
    if (nibble_char := fp_hex[-1]) == "G":
        fp_size -= 1
    elif nibble_char == "H":
        fp_size -= 2
    elif nibble_char == "I":
        fp_size -= 3

    # Padding at end that may result given that numpy uses uint8 but OpenEye is using uint4
    # We are grouping two uint4 FPs into a uint8 for decoding so "ngroups" is the number of byte groups.
    return OEFingerprintInfo(hex_length=hex_length, size=fp_size, ngroups=math.ceil(hex_length / 2))


def to_numpy_fp(fp, fp_info=None):
    """
    Convert an OpenEye fingerprint to a numpy array
    :param fp: OpenEye fingerprint or hex string
    :type fp: str or oegraphsim.OEFingerPrint
    :param fp_info: Optional Fingerprint information (for efficiency)
    :type fp_info: OEFingerPrintInfo or None
    :return: Numpy fingerprint
    :rtype: np.ndarray or None
    """
    # Missing or invalid fingerprints
    if fp is None:
        return None
    elif isinstance(fp, str):
        fp_hex = fp
    elif isinstance(fp, oegraphsim.OEFingerPrint):
        fp_hex = fp.ToHexString()  # type: str
    else:
        raise TypeError(f'Fingerprint must be either hex string or OEFingerprint not {type(fp).__name__}')

    # Get the fingerprint info (if necessary)
    if fp_info is None:
        fp_info = get_openeye_fp_info(fp_hex)

    # OpenEye BitVector.ToHexString() starts from the least significant bit, so we need to reverse the string
    # in order for numpy.unand we also need to truncate out the nibble character
    fp_hex = fp_hex[-2::-1]

    packed = np.array(
        [int(f'{fp_hex[2 * i]}{fp_hex[j] if (j := 2 * i + 1) < fp_info.hex_length else 0}', 16)
         for i in range(fp_info.ngroups)],
        dtype=np.uint8
    )

    return np.array(np.unpackbits(packed, count=fp_info.size), dtype=bool)


def iterable_to_numpy_fps(fps, quiet=True):
    """
    Fast conversion of an OpenEye fingerprint to a numpy bit array

    Note: This is about 10x faster than looping through each bit and checking if it is on

    :param fps: List of OpenEye fingerprints
    :type fps: Iterable
    :param quiet: Disable the progress bar
    :return: List of numpy bitvectors
    :rtype: list of np.ndarray
    """
    bitvecs = []
    # Assume every fingerprint is the same type, so we only need to calculate these things once
    fp_info = None

    # noinspection PyTypeChecker
    for fp in tqdm(fps, desc="Converting Fingerprints", unit="fp", disable=quiet):  # type: oegraphsim.OEFingerPrint
        # If we need to calculate the fingerprint processing items
        # This is done once for efficiency - we assume every fingerprint is the same type and length
        if fp_info is None:
            fp_info = get_openeye_fp_info(fp)

        # Convert and track
        bitvecs.append(to_numpy_fp(fp, fp_info))

    return bitvecs


########################################################################################################################
# Fingerprint generation
########################################################################################################################

# Dynamic creation of a typemap for OpenEye atom type fingerprints
atom_fp_typemap = dict(
    (x.replace("OEFPAtomType_", "").lower(), getattr(oegraphsim, x))
    for x in list(filter(lambda x: x.startswith("OEFPAtomType_"), dir(oegraphsim)))
)

# Dynamic creation of a typemap for OpenEye bond type fingerprints
bond_fp_typemap = dict(
    (x.replace("OEFPBondType_", "").lower(), getattr(oegraphsim, x))
    for x in list(filter(lambda x: x.startswith("OEFPBondType_"), dir(oegraphsim)))
)


def get_atom_mask(atom_type):
    """
    Get the OEFingerprint atom type masks from "|" delimited strings

    The atom_type string is composed of "|" delimted members from the OEFPAtomType_ namespace. These are
    case-insensitive and only optionally need to be prefixed by "OEFPAtomType_".

    :param atom_type: Delimited string of OEFPAtomTypes
    :return: Bitmask for OpenEye fingerprint atom types
    :rtype: int
    """
    atom_mask = oegraphsim.OEFPAtomType_None
    for m in atom_type.split("|"):
        mask = atom_fp_typemap.get(m.strip().lower().replace("oefpatomtype_", ""))
        if mask is None:
            raise KeyError(f'{m} is not a known OEAtomFPType')
        atom_mask |= mask
    # Check validity
    if atom_mask == oegraphsim.OEFPAtomType_None:
        raise ValueError("No atom fingerprint types configured")
    return atom_mask


def get_bond_mask(bond_type):
    """
    Get the OEFingerprint bond type masks from "|" delimited strings

    The bond_type string is composed of "|" delimted members from the OEFPBondType_ namespace. These are
    case-insensitive and only optionally need to be prefixed by "OEFPBondType_".

    :param bond_type: Delimited string of OEFPBondTypes
    :return: Bitmask for OpenEye fingerprint bond types
    :rtype: int
    """
    # Bond mask
    bond_mask = oegraphsim.OEFPBondType_None
    for m in bond_type.split("|"):
        mask = bond_fp_typemap.get(m.strip().lower().replace("oefpbondtype_", ""))
        if mask is None:
            raise KeyError(f'{m} is not a known OEBondFPType')
        bond_mask |= mask
    # Check validity
    if bond_mask == oegraphsim.OEFPBondType_None:
        raise ValueError("No bond fingerprint types configured")
    return bond_mask


def fingerprint_maker(
        fptype: str,
        num_bits: int,
        min_distance: int,
        max_distance: int,
        atom_type: str | int,
        bond_type: str | int
) -> typing.Callable[[oechem.OEMolBase], oegraphsim.OEFingerPrint]:
    """
    Create a function that generates a fingerprint from a molecule
    :param fptype: Fingerprint type
    :param num_bits: Number of bits in the fingerprint
    :param min_distance: Minimum distance/radius for path/circular/tree
    :param max_distance: Maximum distance/radius for path/circular/tree
    :param atom_type: Atom type string delimited by "|" OR int bitmask from the oegraphsim.OEFPAtomType_ namespace
    :param bond_type: Bond type string delimited by "|" OR int bitmask from the oegraphsim.OEFPBondType_ namespace
    :return: Function that generates a fingerprint from a molecule
    """
    # Be forgiving with case
    _fptype = fptype.lower()

    # Convert atom type and bond type strings to masks if necessary
    atom_mask = get_atom_mask(atom_type) if isinstance(atom_type, str) else atom_type
    bond_mask = get_bond_mask(bond_type) if isinstance(bond_type, str) else bond_type
    if _fptype == "path":
        def _make_path_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakePathFP(fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask)
            return fp
        return _make_path_fp
    elif _fptype == "circular":
        def _make_circular_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask)
            return fp
        return _make_circular_fp
    elif _fptype == "tree":
        def _make_tree_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeTreeFP(fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask)
            return fp
        return _make_tree_fp
    elif _fptype == "maccs":
        def _make_maccs(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeMACCS166FP(fp, mol)
            return fp
        return _make_maccs
    elif _fptype == "lingo":
        def _make_lingo(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeLingoFP(fp, mol)
            return fp
        return _make_lingo
    raise KeyError(f'Unknown fingerprint type {fptype} (valid: path / tree / circular / maccs / lingo)')


def make_fingerprints(
        data: Iterable[oechem.OEMolBase],
        fptype: str,
        num_bits: int,
        min_distance: int,
        max_distance: int,
        atom_type: str | int,
        bond_type: str | int,
        desalt: bool = True,
        quiet: bool = True
) -> Iterable[oegraphsim.OEFingerPrint | None]:
    """
    Generate fingerprints from a list of molecules and returns the subset of valid molecules and fingerprints
    :param data: Iterable of OpenEye molecules
    :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs)
    :param num_bits: Number of bits in the Fingerprint
    :param min_distance: Minimum distance (path, tree) or radius (circular) for generating the fingerprint
    :param max_distance: Maximum distance (path, tree) or radius (circular) for generating the fingerprint
    :param atom_type: List of "|" separated members of the oegraphsim.OEFPAtomType_ namespace or the bitmask itself
    :param bond_type: List of "|" separated members of the oegraphsim.OEFPBondType_ namespace or the bitmask itself
    :param desalt: Rudamentary desalting by deleting everything but the first largest component in the molecule
    :param quiet: Silence the progress bar
    :return: Iterator over OpenEye FingerPrint objects, or None where molecules or fingerprints were invalid
    """
    # Fingerprint maker
    make_fp = fingerprint_maker(fptype, num_bits, min_distance, max_distance, atom_type, bond_type)

    # noinspection PyTypeChecker
    for mol in tqdm(data, desc="Generate Fingerprints", unit="mol", disable=quiet):
        # If getting rid of salts and other nonsense
        if desalt:
            # noinspection PyTypeChecker
            oechem.OEDeleteEverythingExceptTheFirstLargestComponent(mol)

        # Sanity check
        if not isinstance(mol, oechem.OEMolBase):
            raise TypeError(f'Expected list of molecule objects but encountered a {type(mol).__name__}')

        # Only generate fingerprints for valid molecules and do not keep invalid fingerprints
        if mol.IsValid():
            fp = make_fp(mol)
            if fp.IsValid():
                yield fp
            else:
                yield None
        else:
            yield None


def make_numpy_fingerprints(
        data: Iterable[oechem.OEMolBase],
        fptype: str,
        num_bits: int,
        min_distance: int,
        max_distance: int,
        atom_type: str | int,
        bond_type: str | int,
        desalt: bool = True,
        quiet: bool = True
    ) -> list[np.ndarray | None]:
    """
    Convenience function to make a list of Numpy fingerprints from an interable of molecules
    :param data: Iterable of OpenEye molecules
    :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs)
    :param num_bits: Number of bits in the Fingerprint
    :param min_distance: Minimum distance (path, tree) or radius (circular) for generating the fingerprint
    :param max_distance: Maximum distance (path, tree) or radius (circular) for generating the fingerprint
    :param atom_type: List of "|" separated members of the oegraphsim.OEFPAtomType_ namespace or the bitmask itself
    :param bond_type: List of "|" separated members of the oegraphsim.OEFPBondType_ namespace or the bitmask itself
    :param desalt: Rudamentary desalting by deleting everything but the first largest component in the molecule
    :param quiet: Silence the progress bar
    :return: List of Numpy fingerprints as ndarrays, or None where molecules or fingerprints were invalid
    """
    return iterable_to_numpy_fps(
        list(make_fingerprints(
            data,
            fptype,
            num_bits,
            min_distance,
            max_distance,
            atom_type,
            bond_type,
            desalt=desalt,
            quiet=quiet
        )),
        quiet=quiet
    )


########################################################################################################################
# OpenEye fingerprint implementation of a pdist-like function
########################################################################################################################

# noinspection PyPep8Naming
def OEJaccard(fpA, fpB):
    """
    Jaccard distance
    :param fpA: Fingerprint A
    :type fpA: oegraphsim.OEFingerPrint
    :param fpB: Fingerprint B
    :type fpB: oegraphsim.OEFingerPrint
    :return: Jaccard distance
    :rtype: float
    """
    onlyA, onlyB, bothAB, _ = oechem.OEGetBitCounts(fpA, fpB)
    union = onlyA + onlyB + bothAB
    return float((union - bothAB) / union)


# Fingerprint similarity metric stored as (function, is_similarity_metric)
openeye_comparison_metric = {
    "tanimoto": (oegraphsim.OETanimoto, True),
    "euclidean": (oegraphsim.OEEuclid, True),
    "dice": (oegraphsim.OEDice, True),
    "cosine": (oegraphsim.OECosine, True),
    "manhattan": (oegraphsim.OEManhattan, False),
    "jaccard": (OEJaccard, False)
}


def get_openeye_comparison_metric(name):
    """
    Get a comparison function
    :param name: Name of the comparison function
    :return: The comparison function
    """
    cf = openeye_comparison_metric.get(name.lower())
    if cf is None:
        raise KeyError(f'Unknown comparison function: {name}')
    return cf


def fpdist(fps, metric: str | typing.Callable):
    """
    Create a compressed pairwise distance matrix for a set of fingerprints
    :param fps: Iterable of fingerprints
    :type fps: list of oechem.OEFingerPrint
    :param metric: OpenEye comparison metric or function that compares to fingerprints
    :return: Compressed pairwise distance matrix
    """
    if isinstance(metric, str):
        cfunc, _ = get_openeye_comparison_metric(metric)
    else:
        cfunc = metric

    # noinspection PyUnresolvedReferences
    compressed = np.zeros(shape=scipy.special.comb(len(fps), 2, exact=True))
    for idx, (fp1, fp2) in enumerate(itertools.combinations(fps, 2)):
        compressed[idx] = cfunc(fp1, fp2)
    return compressed


########################################################################################################################
# FingerprintArray
########################################################################################################################


class FingerprintArray(OEExtensionArray[oegraphsim.OEFingerPrint]):
    """
    Custom extension array for OpenEye fingerprints.

    Stores OEFingerPrint objects internally and provides methods for similarity
    calculations, numpy conversion, and integration with pandas DataFrames.
    """

    _base_openeye_type = oegraphsim.OEFingerPrint

    def __init__(
            self,
            fps: None | oegraphsim.OEFingerPrint | Iterable[oegraphsim.OEFingerPrint | None],
            copy: bool = False,
            metadata: dict | None = None
    ):
        """
        Initialize a FingerprintArray.

        :param fps: Fingerprint or an iterable of fingerprints.
        :param copy: Create copy of the fingerprints if True.
        :param metadata: Optional metadata dictionary (e.g., fingerprint parameters).
        """
        # Handle singleton fingerprints
        fps_iter: Iterable[oegraphsim.OEFingerPrint | None]
        if fps is None:
            fps_iter = []
        elif isinstance(fps, oegraphsim.OEFingerPrint):
            fps_iter = (fps,)
        else:
            fps_iter = fps

        # Process fingerprints
        processed = []
        for fp in fps_iter:
            if isinstance(fp, oegraphsim.OEFingerPrint):
                processed.append(oegraphsim.OEFingerPrint(fp) if copy else fp)
            elif pd.isna(fp) or fp is None:
                processed.append(None)
            else:
                processed.append(None)

        # Superclass initialization
        super().__init__(processed, copy=False, metadata=metadata)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def dtype(self) -> PandasExtensionDtype:
        return FingerprintDtype()

    @property
    def num_bits(self) -> int | None:
        """
        Get the number of bits in the fingerprints.

        :returns: Number of bits, or None if array is empty or has no valid fingerprints.
        """
        for fp in self._objs:
            if fp is not None and fp.IsValid():
                return fp.GetSize()
        return None

    # --------------------------------------------------------
    # Factory Methods
    # --------------------------------------------------------

    @classmethod
    def from_molecules(
            cls,
            molecules: Iterable[oechem.OEMolBase],
            fptype: str,
            num_bits: int,
            min_distance: int,
            max_distance: int,
            atom_type: str | int,
            bond_type: str | int,
            desalt: bool = True,
            quiet: bool = True
    ) -> Self:
        """
        Create a FingerprintArray from an iterable of molecules.

        :param molecules: Iterable of OpenEye molecules.
        :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
        :param num_bits: Number of bits in the fingerprint.
        :param min_distance: Minimum distance (path, tree) or radius (circular).
        :param max_distance: Maximum distance (path, tree) or radius (circular).
        :param atom_type: "|" separated OEFPAtomType members or bitmask.
        :param bond_type: "|" separated OEFPBondType members or bitmask.
        :param desalt: Remove salts by keeping only the largest component.
        :param quiet: Silence the progress bar.
        :returns: FingerprintArray populated with generated fingerprints.
        """
        # Generate fingerprints
        fps = list(make_fingerprints(
            molecules,
            fptype=fptype,
            num_bits=num_bits,
            min_distance=min_distance,
            max_distance=max_distance,
            atom_type=atom_type,
            bond_type=bond_type,
            desalt=desalt,
            quiet=quiet
        ))

        # Store parameters in metadata for reproducibility
        metadata = {
            "fptype": fptype,
            "num_bits": num_bits,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "atom_type": atom_type,
            "bond_type": bond_type,
            "desalt": desalt
        }

        return cls(fps, copy=False, metadata=metadata)

    @classmethod
    def _from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,  # noqa
            copy: bool = False
    ) -> Self:
        """
        Initialize from a sequence of scalar values.

        Supports OEFingerPrint objects and hex strings.

        :param scalars: Scalars (fingerprints or hex strings).
        :param dtype: Not used (for pandas API compatibility).
        :param copy: Copy the fingerprints.
        :returns: New FingerprintArray instance.
        """
        fps = []

        for obj in scalars:
            if obj is None or pd.isna(obj):
                fps.append(None)
            elif isinstance(obj, oegraphsim.OEFingerPrint):
                fps.append(obj)
            elif isinstance(obj, str):
                # Assume hex string
                fp = oegraphsim.OEFingerPrint()
                if fp.FromHexString(obj):
                    fps.append(fp)
                else:
                    fps.append(None)
            else:
                fps.append(None)

        return cls(fps, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,  # noqa
            copy: bool = False  # noqa
    ) -> Self:
        """
        Create FingerprintArray from hex strings.

        :param strings: Sequence of hex strings.
        :param dtype: Not used (for pandas API compatibility).
        :param copy: Not used (for pandas API compatibility).
        :returns: Array of fingerprints.
        """
        fps = []

        for s in strings:
            if s is None or (isinstance(s, str) and s == ""):
                fps.append(None)
            elif isinstance(s, str):
                fp = oegraphsim.OEFingerPrint()
                if fp.FromHexString(s):
                    fps.append(fp)
                else:
                    fps.append(None)
            else:
                fps.append(None)

        return cls(fps, copy=False)

    @classmethod
    def from_sequence(
            cls,
            scalars: Iterable[Any],
            *,
            dtype: Dtype | None = None,
            copy: bool = False
    ) -> Self:
        """
        Public alias of _from_sequence.

        :param scalars: Sequence of fingerprints or hex strings.
        :param dtype: Not used (for pandas API compatibility).
        :param copy: Copy the fingerprints.
        :returns: Array of fingerprints.
        """
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    @classmethod
    def from_sequence_of_strings(
            cls,
            strings: Sequence[str],
            *,
            dtype: Dtype | None = None,
            copy: bool = False
    ) -> Self:
        """
        Public alias of _from_sequence_of_strings.

        :param strings: Sequence of hex strings.
        :param dtype: Not used (for pandas API compatibility).
        :param copy: Not used (for pandas API compatibility).
        :returns: Array of fingerprints.
        """
        return cls._from_sequence_of_strings(strings, dtype=dtype, copy=copy)

    def deepcopy(self, metadata: bool | dict | None = True) -> Self:
        """
        Make a deep copy of this FingerprintArray.

        :param metadata: Metadata for copied object, whether to copy metadata, or None.
        :returns: Deep copy of this array.
        """
        resolved_metadata: dict | None
        if isinstance(metadata, bool):
            if metadata:
                resolved_metadata = shallow_copy(self.metadata)
            else:
                resolved_metadata = None
        else:
            resolved_metadata = metadata

        copied_fps = []
        for fp in self._objs:
            if fp is not None:
                copied_fps.append(oegraphsim.OEFingerPrint(fp))
            else:
                copied_fps.append(None)

        return cast(Self, FingerprintArray(copied_fps, copy=False, metadata=resolved_metadata))

    def fillna(
        self,
        value: object = None,
        method=None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill N/A values and invalid fingerprints.

        :param value: Fill value (should be an OEFingerPrint).
        :param method: Not used.
        :param limit: Maximum number of entries to fill.
        :param copy: Whether to copy the data.
        :returns: Filled FingerprintArray.
        """
        if limit is not None:
            limit = max(0, limit)

        filled = []
        fills_done = 0

        for fp in self._objs:
            if limit is not None and fills_done >= limit:
                if copy and fp is not None:
                    filled.append(oegraphsim.OEFingerPrint(fp))
                else:
                    filled.append(fp)
                continue

            if pd.isna(fp) or fp is None:
                filled.append(value)
                fills_done += 1
            elif isinstance(fp, oegraphsim.OEFingerPrint):
                if not fp.IsValid():
                    filled.append(value)
                    fills_done += 1
                else:
                    filled.append(oegraphsim.OEFingerPrint(fp) if copy else fp)
            else:
                filled.append(value)
                fills_done += 1

        return cast(Self, FingerprintArray(filled, copy=False, metadata=shallow_copy(self.metadata)))

    def dropna(self) -> Self:
        """
        Drop all NA and invalid fingerprints.

        :returns: FingerprintArray with no missing or invalid fingerprints.
        """
        non_missing = []

        for fp in self._objs:
            if fp is not None and isinstance(fp, oegraphsim.OEFingerPrint) and fp.IsValid():
                non_missing.append(fp)

        return cast(Self, FingerprintArray(non_missing, copy=False, metadata=shallow_copy(self.metadata)))

    def tolist(self, copy: bool = False) -> list[oegraphsim.OEFingerPrint | None]:
        """
        Convert to a list.

        :param copy: Whether to copy the fingerprints or return pointers.
        :returns: List of fingerprints.
        """
        if copy:
            return [oegraphsim.OEFingerPrint(fp) if fp is not None else None for fp in self._objs]
        return shallow_copy(self._objs)

    # --------------------------------------------------------
    # Conversion Methods
    # --------------------------------------------------------

    # noinspection PyTypeHints
    def to_numpy(
            self, dtype: DTypeLike | None = None,
            copy: bool = False,
            na_value: object = pd.api.extensions.no_default
    ) -> np.ndarray:
        """
        Convert to a 2D numpy boolean array.

        Each row is a fingerprint, each column is a bit position.
        Invalid/missing fingerprints are represented as rows of False.

        :param dtype: The desired NumPy dtype for the result. If None, bool is used.
        :param copy: Whether to ensure the returned array is a copy.
        :param na_value: The value to use for missing values (unused; missing FPs become rows of False).
        :returns: 2D numpy array of shape (n_fingerprints, n_bits).
        """
        # Get fingerprint size from first valid fingerprint
        n_bits = self.num_bits
        if n_bits is None:
            return np.array([], dtype=bool).reshape(0, 0)

        # Convert each fingerprint
        result = []
        fp_info = None

        for fp in self._objs:
            if fp is None or not fp.IsValid():
                result.append(np.zeros(n_bits, dtype=bool))
            else:
                if fp_info is None:
                    fp_info = get_openeye_fp_info(fp)
                result.append(to_numpy_fp(fp, fp_info))

        arr = np.vstack(result) if result else np.array([], dtype=bool).reshape(0, n_bits)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr

    def to_hex_strings(self) -> np.ndarray:
        """
        Convert fingerprints to hex strings.

        Invalid/missing fingerprints are represented as empty strings.

        :returns: Array of hex strings.
        """
        hex_strings = []

        for fp in self._objs:
            if fp is not None and fp.IsValid():
                hex_strings.append(fp.ToHexString())
            else:
                hex_strings.append("")

        return np.array(hex_strings, dtype=str)

    # --------------------------------------------------------
    # Similarity Methods
    # --------------------------------------------------------

    def tanimoto(
            self,
            other: 'FingerprintArray | oegraphsim.OEFingerPrint'
    ) -> np.ndarray:
        """
        Calculate Tanimoto similarity against another fingerprint or array.

        :param other: Single fingerprint or FingerprintArray.
        :returns: Array of Tanimoto similarities.
        :raises ValueError: If comparing arrays of different lengths.
        """
        if isinstance(other, oegraphsim.OEFingerPrint):
            # Compare all fingerprints to a single reference
            similarities = []
            for fp in self._objs:
                if fp is None or not fp.IsValid() or not other.IsValid():
                    similarities.append(np.nan)
                else:
                    similarities.append(oegraphsim.OETanimoto(fp, other))
            return np.array(similarities, dtype=float)

        elif isinstance(other, FingerprintArray):
            if len(self) != len(other):
                raise ValueError(
                    f"Cannot compare arrays of different lengths: {len(self)} vs {len(other)}"
                )
            similarities = []
            for fp1, fp2 in zip(self._objs, other._objs, strict=True):
                if fp1 is None or fp2 is None or not fp1.IsValid() or not fp2.IsValid():
                    similarities.append(np.nan)
                else:
                    similarities.append(oegraphsim.OETanimoto(fp1, fp2))
            return np.array(similarities, dtype=float)

        else:
            raise TypeError(
                f"Cannot compare FingerprintArray with {type(other).__name__}"
            )

    def jaccard(
            self,
            other: 'FingerprintArray | oegraphsim.OEFingerPrint'
    ) -> np.ndarray:
        """
        Calculate Jaccard distance against another fingerprint or array.

        :param other: Single fingerprint or FingerprintArray.
        :returns: Array of Jaccard distances.
        :raises ValueError: If comparing arrays of different lengths.
        """
        if isinstance(other, oegraphsim.OEFingerPrint):
            # Compare all fingerprints to a single reference
            distances = []
            for fp in self._objs:
                if fp is None or not fp.IsValid() or not other.IsValid():
                    distances.append(np.nan)
                else:
                    distances.append(OEJaccard(fp, other))
            return np.array(distances, dtype=float)

        elif isinstance(other, FingerprintArray):
            if len(self) != len(other):
                raise ValueError(
                    f"Cannot compare arrays of different lengths: {len(self)} vs {len(other)}"
                )
            distances = []
            for fp1, fp2 in zip(self._objs, other._objs, strict=True):
                if fp1 is None or fp2 is None or not fp1.IsValid() or not fp2.IsValid():
                    distances.append(np.nan)
                else:
                    distances.append(OEJaccard(fp1, fp2))
            return np.array(distances, dtype=float)

        else:
            raise TypeError(
                f"Cannot compare FingerprintArray with {type(other).__name__}"
            )

    def pdist(self, metric: str | typing.Callable = "tanimoto") -> np.ndarray:
        """
        Calculate pairwise distances/similarities between all fingerprints.

        :param metric: Comparison metric name or callable.
        :returns: Compressed pairwise distance matrix (scipy pdist format).
        """
        # Filter to valid fingerprints for pdist calculation
        valid_fps = [fp for fp in self._objs if fp is not None and fp.IsValid()]
        return fpdist(valid_fps, metric)


@register_extension_dtype
class FingerprintDtype(PandasExtensionDtype):
    """
    OpenEye fingerprint datatype for Pandas.
    """

    type: type = oegraphsim.OEFingerPrint  # noqa
    name: str = "fingerprint"  # noqa
    kind: str = "O"
    base = np.dtype("O")

    @classmethod
    def construct_array_type(cls):
        """
        Return array type associated with this dtype.
        """
        return FingerprintArray

    def __hash__(self) -> int:
        return hash(self.name)

    # noinspection PyTypeHints
    def __eq__(self, other: str | type) -> bool:
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, type(self))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
