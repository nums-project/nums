# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse


import numpy as np

from nums.core.systems import utils as systems_utils
from nums.numpy import api as numpy_api


def execute(print_missing, count_fallback):

    for name, func in systems_utils.get_module_functions(numpy_api).items():
        if name in ("_not_implemented", "_instance"):
            continue
        if getattr(np, name, None) is None:
            raise Exception("Implemented method that does not exist! %s" % name)

    coverage = 0
    total = 0
    # Functions ignored for various reasons.
    ignore = {
        '_add_newdoc_ufunc', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc',
        # Deprecated
        'alen',
        # Order is generally not applicable w/ GPU backend.
        'asanyarray', 'ascontiguousarray', 'asfarray', 'asfortranarray',
        # There are no plans to provide a matrix type.
        'asmatrix', 'array2string',
        'base_repr', 'binary_repr', 'block', 'bmat', 'broadcast_arrays',
        'broadcast_to', 'busday_count', 'busday_offset', 'byte_bounds',
        'compare_chararrays',
        'datetime_as_string', 'datetime_data', 'deprecate', 'deprecate_with_doc', 'disp',
        'fastCopyAndTranspose', 'format_float_positional', 'format_float_scientific',
        'get_array_wrap', 'get_include', 'get_printoptions',
        'getbufsize', 'geterr', 'geterrcall', 'geterrobj',
        'info', 'is_busday', 'isfortran', 'isnat',
        'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable',
        'loads', 'lookfor',
        'mat', 'may_share_memory',
        'ndfromtxt', 'nested_iters',
        'printoptions', 'recfromcsv', 'recfromtxt',
        'safe_eval', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize',
        'seterr', 'seterrcall', 'seterrobj', 'show_config', 'source', 'typename'
    }

    not_implemented = {
        'array_repr', 'array_str',
        'can_cast', 'copy', 'copyto', 'find_common_type',
        'ndim', 'savetxt', 'shares_memory', 'shape', 'size'
    }

    # Fallback on NumPy for these operations.
    # This is achieved by converting the block array to a single block, performing the operation,
    # and converting back to the original block shape.
    fallback = {
        # TODO (hme): Organize API according to below.
        # Basic Unsupported
        # Manipulation
        # I/O
        # Unusual
        'all', 'alltrue', 'angle', 'any', 'append', 'apply_along_axis', 'apply_over_axes',
        'argpartition', 'argsort', 'argwhere', 'around', 'array_equal', 'array_equiv',
        'array_split', 'asarray', 'asarray_chkfinite', 'asscalar', 'atleast_1d',
        'atleast_2d', 'atleast_3d', 'average',
        'bartlett', 'bincount', 'blackman',
        'choose', 'clip', 'column_stack', 'common_type', 'compress', 'convolve', 'corrcoef',
        'correlate', 'count_nonzero', 'cov', 'cross', 'cumprod', 'cumproduct', 'cumsum',
        'delete', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize',
        'divmod', 'dot', 'dsplit', 'dstack',
        'ediff1d', 'einsum', 'einsum_path', 'expand_dims', 'extract',
        'fill_diagonal', 'fix', 'flatnonzero', 'flip', 'fliplr', 'flipud', 'frexp', 'frombuffer',
        'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full',
        'full_like', 'fv',
        'genfromtxt', 'geomspace', 'gradient',
        'hamming', 'hanning',
        'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'i0',
        'imag', 'in1d', 'indices', 'inner', 'insert', 'interp', 'intersect1d', 'ipmt',
        'irr', 'isclose', 'iscomplex', 'iscomplexobj', 'isin',
        'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar',
        'ix_', 'kaiser', 'kron', 'lexsort', 'load',
        'mafromtxt', 'mask_indices', 'matmul', 'maximum_sctype',
        'median', 'meshgrid', 'min_scalar_type', 'mintypecode', 'mirr', 'modf', 'moveaxis', 'msort',
        'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum',
        'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile',
        'nanstd', 'nansum', 'nanvar',
        'nonzero', 'nper', 'npv',
        'obj2sctype', 'outer',
        'packbits', 'pad', 'partition', 'percentile', 'piecewise', 'place',
        'pmt', 'poly', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polysub',
        'polyval', 'ppmt', 'prod', 'product', 'promote_types', 'ptp', 'put',
        'put_along_axis', 'putmask', 'pv',
        'quantile', 'rate', 'ravel', 'ravel_multi_index', 'real',
        'real_if_close', 'repeat', 'require', 'reshape', 'resize',
        'result_type', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack',
        'save', 'savez', 'savez_compressed', 'sctype2char', 'searchsorted',
        'select', 'setdiff1d', 'setxor1d',
        'sinc', 'sometrue', 'sort', 'sort_complex', 'squeeze',
        'stack', 'swapaxes', 'take', 'take_along_axis', 'tensordot', 'tile', 'trace', 'transpose',
        'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu',
        'triu_indices', 'triu_indices_from', 'union1d', 'unique', 'unpackbits',
        'unravel_index', 'unwrap', 'vander', 'vdot', 'vsplit', 'vstack', 'where', 'who'
    }

    missing = []
    for name, func in systems_utils.get_module_functions(np).items():
        if name in ignore or (count_fallback and name in fallback):
            continue
        total += 1
        print_tuple = None
        try:
            doc_lines = func.__doc__.split("\n")
            for line in doc_lines:
                descr = line.strip()
                if len(descr) != 0:
                    break
                print_tuple = name, descr
        except Exception as e:
            print_tuple = name, func.__code__.co_varnames
        if print_tuple is None:
            continue
        if name in fallback:
            print("Fallback %s: %s" % print_tuple)
        elif getattr(numpy_api, name, None) is None:
            print("Missing %s: %s" % print_tuple)
            missing.append("%s" % name)
        else:
            coverage += 1

    print("coverage", coverage)
    print("total", total)
    print("percent covered", "%.1f" % (coverage / total * 100))
    if print_missing:
        print(str(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fallback', action="store_true", help='Include fallback functions '
                                                                'in converage.')
    parser.add_argument('--print-missing', action="store_true", help='Output array of missing '
                                                                     'values.')
    args = parser.parse_args()
    args_dict = dict(vars(args).items())
    execute(**args_dict)


if __name__ == "__main__":
    main()
