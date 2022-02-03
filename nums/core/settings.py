import os
from pathlib import Path

pj = lambda *paths: os.path.abspath(os.path.expanduser(os.path.join(*paths)))
core_root = os.path.abspath(os.path.dirname(__file__))
package_root = pj(core_root, "../")
project_root = pj(package_root, "../")
data_dir = pj(project_root, "data")
Path(data_dir).mkdir(parents=True, exist_ok=True)


# System settings.
system_name = os.environ.get("NUMS_SYSTEM", "ray")
# TODO (hme):
#  - Make cluster shape an environment variable.
#  - use_head should be an environment variable.
use_head = True
head_ip = os.environ.get("NUMS_HEAD_IP", None)

# An address to which the system client should connect.
address = None


# Compute settings.
compute_name = os.environ.get("NUMS_COMPUTE", "numpy")


# Device grid settings.
num_cpus = None
cluster_shape = (1, 1)
device_grid_name = os.environ.get("NUMS_DEVICE_GRID", "packed")


# NumPy operator map.
np_ufunc_map = {
    "truediv": "true_divide",
    "sub": "subtract",
    "pow": "power",
    "mult": "multiply",
    "mul": "multiply",
    "tensordot": "multiply",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "eq": "equal",
    "ne": "not_equal",
}

np_bop_reduction_set = {"min", "amin", "max", "amax", "nanmax", "nanmin", "nansum"}

# Fallback on NumPy for these operations.
# This is achieved by converting the block array to a single block, performing the operation,
# and converting back to the original block shape.
doctest_fallbacks = {
    "argwhere",
    "asscalar",
    "clip",
    "compress",
    "convolve",
    "corrcoef",
    "cumprod",
    "cumproduct",
    "cumsum",
    "diag_indices_from",
    "diagflat",
    "fix",
    "fromiter",
    "full",
    "msort",
    "nancumprod",
    "nancumsum",
    "nanprod",
    "partition",
    "polysub",
    "product",
    "ravel_multi_index",
    "repeat",
    "resize",
    "roll",
    "roots",
    "rot90",
    "round",
    "round_",
    "searchsorted",
    "setdiff1d",
    "setxor1d",
    "sometrue",
    "sort_complex",
    "swapaxes",
    "tile",
    "trapz",
    "tri",
    "tril",
    "tril_indices_from",
    "triu",
    "triu_indices_from",
    "union1d",
}

tested_fallbacks = {
    "angle",
    "append",
    "argsort",
    "around",
    "apply_along_axis",
    "apply_over_axes",
    "bartlett",
    "cov",
    "kaiser",
}

untested_fallbacks = {
    "array_split",
    "argpartition",
    "asarray",
    "asarray_chkfinite",
    "average",
    "bincount",
    "blackman",
    "choose",
    "common_type",
    "correlate",
    "count_nonzero",
    "cross",
    "delete",
    "diag_indices",
    "diagonal",
    "diff",
    "digitize",
    "divmod",
    "dot",
    "dsplit",
    "ediff1d",
    "einsum",
    "einsum_path",
    "extract",
    "fill_diagonal",
    "flatnonzero",
    "flip",
    "fliplr",
    "flipud",
    "frexp",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "frompyfunc",
    "full_like",
    "geomspace",
    "gradient",
    "hamming",
    "hanning",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "hsplit",
    "i0",
    "imag",
    "in1d",
    "indices",
    "insert",
    "interp",
    "intersect1d",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
    "ix_",
    "kron",
    "lexsort",
    "maximum_sctype",
    "median",
    "meshgrid",
    "min_scalar_type",
    "mintypecode",
    "modf",
    "moveaxis",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "nonzero",
    "obj2sctype",
    "packbits",
    "pad",
    "percentile",
    "piecewise",
    "place",
    "poly",
    "polyadd",
    "polyder",
    "polydiv",
    "polyfit",
    "polyint",
    "polymul",
    "polyval",
    "prod",
    "promote_types",
    "ptp",
    "put",
    "put_along_axis",
    "putmask",
    "quantile",
    "ravel",
    "real",
    "real_if_close",
    "require",
    "result_type",
    "rollaxis",
    "sctype2char",
    "select",
    "sinc",
    "sort",
    "stack",
    "take",
    "take_along_axis",
    "trace",
    "tril_indices",
    "trim_zeros",
    "triu_indices",
    "unique",
    "unpackbits",
    "unravel_index",
    "unwrap",
    "vander",
    "vdot",
    "vsplit",
    "who",
}

excluded_fallbacks = set()
assert (
    len(doctest_fallbacks & tested_fallbacks & untested_fallbacks & excluded_fallbacks)
    == 0
)
fallback = doctest_fallbacks | tested_fallbacks | untested_fallbacks
