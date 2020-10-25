# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from nums.core.systems import utils as systems_utils
from nums.numpy import api as numpy_api
from nums.numpy import numpy_utils


def _api_coverage():
    import numpy as np

    for name, func in systems_utils.get_module_functions(numpy_api).items():
        if name in ("_not_implemented",):
            continue
        if getattr(np, name, None) is None:
            raise Exception("Implemented method that does not exist! %s" % name)

    coverage = 0
    total = 0
    # Functions ignored for various reasons.
    ignore = {
        '_add_newdoc_ufunc', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc',
        'alen',  # Deprecated
        # Order is generally not applicable w/ GPU backend.
        'asanyarray', 'ascontiguousarray', 'asfarray', 'asfortranarray',
        # There are no plans to provide a matrix type.
        'asmatrix',
        # TODO (hme): Go through below functions.
        # 'base_repr', 'binary_repr', 'bincount', 'blackman', 'block', 'bmat', 'broadcast_arrays', 'broadcast_to', 'busday_count', 'busday_offset', 'byte_bounds',
        # 'can_cast', 'choose', 'clip', 'column_stack', 'common_type', 'compare_chararrays', 'compress', 'convolve', 'copy', 'copyto', 'corrcoef', 'correlate', 'count_nonzero', 'cov', 'cross', 'cumprod', 'cumproduct', 'cumsum',
        # 'datetime_as_string', 'datetime_data', 'delete', 'deprecate', 'deprecate_with_doc', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divmod', 'dot', 'dsplit', 'dstack',
        # 'ediff1d', 'einsum', 'einsum_path', 'expand_dims', 'extract',
        # 'fastCopyAndTranspose', 'fill_diagonal', 'find_common_type', 'fix', 'flatnonzero', 'flip', 'fliplr', 'flipud', 'format_float_positional', 'format_float_scientific', 'frexp', 'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'fv',
        # 'genfromtxt', 'geomspace', 'get_array_wrap', 'get_include', 'get_printoptions', 'getbufsize', 'geterr', 'geterrcall', 'geterrobj', 'gradient',
        # 'hamming', 'hanning', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack',
        # 'i0', 'identity', 'imag', 'in1d', 'indices', 'info', 'inner', 'insert', 'interp', 'intersect1d', 'ipmt', 'irr', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isfortran', 'isin', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable', 'ix_',
        # 'kaiser', 'kron',
        # 'lexsort', 'linspace', 'load', 'loads', 'logspace', 'lookfor',
        # 'mafromtxt', 'mask_indices', 'mat', 'matmul', 'maximum_sctype', 'may_share_memory', 'median', 'meshgrid', 'min_scalar_type', 'mintypecode', 'mirr', 'modf', 'moveaxis', 'msort',
        # 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'ndfromtxt', 'ndim', 'nested_iters', 'nonzero', 'nper', 'npv',
        # 'obj2sctype', 'outer',
        # 'packbits', 'pad', 'partition', 'percentile', 'piecewise', 'place', 'pmt', 'poly', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polysub', 'polyval', 'ppmt', 'printoptions', 'prod', 'product', 'promote_types', 'ptp', 'put', 'put_along_axis', 'putmask', 'pv',
        # 'quantile',
        # 'rate', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'recfromcsv', 'recfromtxt', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack',
        # 'safe_eval', 'save', 'savetxt', 'savez', 'savez_compressed', 'sctype2char', 'searchsorted', 'select', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize', 'setdiff1d', 'seterr', 'seterrcall', 'seterrobj', 'setxor1d', 'shape', 'shares_memory', 'show_config', 'sinc', 'size', 'sometrue', 'sort', 'sort_complex', 'source', 'squeeze', 'stack', 'swapaxes',
        # 'take', 'take_along_axis', 'tensordot', 'tile', 'trace', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'typename',
        # 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unwrap', 'vander', 'var', 'vdot', 'vsplit', 'vstack', 'where', 'who'
    }

    for name, func in systems_utils.get_module_functions(np).items():
        if name in ignore:
            continue
        total += 1
        if getattr(numpy_api, name, None) is None:
            try:
                print("Missing ", name, func.__code__.co_varnames)
            except Exception as e:
                sig: str = func.__doc__.split("\n")[0].strip()
                print("Missing ", sig)
        else:
            coverage += 1

    print("coverage", coverage)
    print("total", total)
    print("percent covered", "%.1f" % (coverage / total * 100))


def _stub():
    import numpy as np

    def _uop_template(op_name):
        s = """

def {op_name}(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="{op_name}",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))"""
        return s.format(op_name=op_name)

    def _bop_template(op_name):
        s = """

def {op_name}(x1: BlockArray, x2: BlockArray, 
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="{op_name}",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))"""
        return s.format(op_name=op_name)

    uops, bops, rest = numpy_utils.ufunc_op_signatures()
    for name, func in sorted(systems_utils.get_module_functions(np).items()):
        if name in ("deprecate_with_doc", "loads"):
            continue
        try:
            sig: str = func.__doc__.split("\n")[0].strip()
            op_name, args = sig.split("(")
            args = args.split(")")[0]
            has_subok = "subok" in args
            if has_subok:
                # We don't support subok.
                args = args.split("subok")[0].strip()[:-1]
            args = list(map(lambda x: x.strip(), args.split(",")))
            if args == ['x', '/', 'out=None', '*', 'where=True',
                        "casting='same_kind'", "order='K'", 'dtype=None']:
                # This is a uop.
                uops.append(_uop_template(name))
            elif args == ['x1', "x2", '/', 'out=None', '*', 'where=True',
                          "casting='same_kind'", "order='K'", 'dtype=None']:
                # This is a bop.
                bops.append(_bop_template(name))
            else:
                print(name, op_name, args)
        except Exception as e:
            print("FAILED", name)
    for sig in uops + bops:
        print(sig)
    print("num ufuncs", len(uops + bops))
