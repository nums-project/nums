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

import doctest
import io
import warnings
from contextlib import redirect_stdout

import numpy as np

from nums.core import settings
from nums.numpy.numpy_utils import update_doc_string

warnings.filterwarnings("ignore", category=RuntimeWarning)


# pylint: disable=import-outside-toplevel, possibly-unused-variable, eval-used, reimported


def test_doctest_fallback(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    passed = []
    failed = []
    excepted = []

    plot_funcs = {
        "kaiser",
        "bartlett",
        "hanning",
        "blackman",
        "histogram2d",
        "interp",
        "sinc",
    }

    for func in settings.doctest_fallbacks:
        nps_func = nps.__getattribute__(func)
        nps_func.__doc__ = update_doc_string(np.__getattribute__(func).__doc__)

        f = io.StringIO()
        with redirect_stdout(f):
            nps_func = nps.__getattribute__(func)

            if func in plot_funcs:
                # Skip plot functions and add them to the failed list
                print("Failure")
            else:
                optionflags = doctest.NORMALIZE_WHITESPACE | doctest.FAIL_FAST
                doctest.run_docstring_examples(
                    nps_func, locals(), optionflags=optionflags
                )

        if f.getvalue() == "":
            passed.append(func)
        else:
            failed.append(func)

    print("***DOCTESTS***")
    print("PASSED:")
    print(sorted(passed))
    print("FAILED:")
    print(sorted(failed))
    print("EXCEPTED:")
    print(sorted(excepted))


def test_manual_cov(nps_app_inst):
    assert nps_app_inst is not None

    x = np.array([[0, 2], [1, 1], [2, 0]]).T
    assert np.allclose(np.cov(x), np.array([[1.0, -1.0], [-1.0, 1.0]]))
    x = [-2.1, -1, 4.3]
    y = [3, 1.1, 0.12]
    X = np.stack((x, y), axis=0)
    assert np.allclose(np.cov(X), np.array([[11.71, -4.286], [-4.286, 2.144133]]))
    assert np.allclose(np.cov(x, y), np.array([[11.71, -4.286], [-4.286, 2.144133]]))
    assert np.allclose(np.cov(x), np.array(11.71))


def test_manual(nps_app_inst):
    assert nps_app_inst is not None
    import nums.numpy as nps

    test_set = {}
    test_set.update(get_test_set_a())
    test_set.update(get_test_set_b2f())
    test_set.update(get_test_set_g2n())
    test_set.update(get_test_set_o2z())

    for func_name in test_set:
        nps_func = nps.__getattribute__(func_name)
        np_func = np.__getattribute__(func_name)
        for params in test_set[func_name]:
            assert np.allclose(nps_func(*params).get(), np_func(*params))


def get_test_set_a():
    return {
        "angle": [([1.0, 1.0j, 1 + 1j],), (1 + 1j, True)],
        "append": [
            ([1, 2, 3], [[4, 5, 6], [7, 8, 9]]),
            ([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], 0),
        ],
        "argsort": [
            (np.array([3, 1, 2]),),
            (np.array([[0, 3], [2, 2]]), 0),
            (np.array([[0, 3], [2, 2]]), 1),
        ],
        "around": [(56294995342131.5, 3)],
        "apply_along_axis": [
            (
                lambda a: (a[0] + a[-1]) * 0.5,
                0,
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            )
        ],
        "apply_over_axes": [(np.sum, np.arange(24).reshape(2, 3, 4), [0, 2])],
        "array_split": [],
        "argpartition": [],
        "asarray": [],
        "asarray_chkfinite": [],
        "average": [],
    }


def get_test_set_b2f():
    return {
        "bartlett": [(12,)],
        "bincount": [],
        "blackman": [],
        "choose": [],
        "common_type": [],
        "correlate": [],
        "count_nonzero": [],
        "cov": [],
        "cross": [],
        "delete": [],
        "diag_indices": [],
        "diagonal": [],
        "diff": [],
        "digitize": [],
        "divmod": [],
        "dot": [],
        "dsplit": [],
        "ediff1d": [],
        "einsum": [],
        "einsum_path": [],
        "extract": [],
        "fill_diagonal": [],
        "flatnonzero": [],
        "flip": [],
        "fliplr": [],
        "flipud": [],
        "frexp": [],
        "frombuffer": [],
        "fromfile": [],
        "fromfunction": [],
        "frompyfunc": [],
        "full_like": [],
    }


def get_test_set_g2n():
    return {
        "geomspace": [],
        "gradient": [],
        "hamming": [],
        "hanning": [],
        "histogram": [],
        "histogram2d": [],
        "histogram_bin_edges": [],
        "histogramdd": [],
        "hsplit": [],
        "i0": [],
        "imag": [],
        "in1d": [],
        "indices": [],
        "insert": [],
        "interp": [],
        "intersect1d": [],
        "isclose": [],
        "iscomplex": [],
        "iscomplexobj": [],
        "isin": [],
        "isneginf": [],
        "isposinf": [],
        "isreal": [],
        "isrealobj": [],
        "isscalar": [],
        "ix_": [],
        "kaiser": [(12, 14)],
        "kron": [],
        "lexsort": [],
        "maximum_sctype": [],
        "meshgrid": [],
        "min_scalar_type": [],
        "mintypecode": [],
        "modf": [],
        "moveaxis": [],
        "nan_to_num": [],
        "nanargmax": [],
        "nanargmin": [],
        "nanmedian": [],
        "nanpercentile": [],
        "nanquantile": [],
        "nonzero": [],
    }


def get_test_set_o2z():
    return {
        "obj2sctype": [],
        "packbits": [],
        "pad": [],
        "piecewise": [],
        "place": [],
        "poly": [],
        "polyadd": [],
        "polyder": [],
        "polydiv": [],
        "polyfit": [],
        "polyint": [],
        "polymul": [],
        "polyval": [],
        "prod": [],
        "promote_types": [],
        "ptp": [],
        "put": [],
        "put_along_axis": [],
        "putmask": [],
        "ravel": [],
        "real": [],
        "real_if_close": [],
        "require": [],
        "result_type": [],
        "rollaxis": [],
        "sctype2char": [],
        "select": [],
        "sinc": [],
        "sort": [],
        "stack": [],
        "take": [],
        "take_along_axis": [],
        "trace": [],
        "tril_indices": [],
        "trim_zeros": [],
        "triu_indices": [],
        "unique": [],
        "unpackbits": [],
        "unravel_index": [],
        "unwrap": [],
        "vander": [],
        "vdot": [],
        "vsplit": [],
        "who": [],
    }


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    # test_doctest_fallback(nps_app_inst)
    # test_manual_cov(nps_app_inst)
    test_manual(nps_app_inst)
