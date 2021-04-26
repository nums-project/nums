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
    import numpy as np
    assert nps_app_inst is not None

    passed = []
    failed = []
    excepted = []

    plot_funcs = {"kaiser", "bartlett", "hanning", "blackman", "histogram2d", "interp", "sinc"}

    for func in settings.doctest_fallback:
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
                doctest.run_docstring_examples(nps_func, locals(), optionflags=optionflags)

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
    assert np.allclose(np.cov(x), np.array([[1., -1.], [-1., 1.]]))
    x = [-2.1, -1, 4.3]
    y = [3, 1.1, 0.12]
    X = np.stack((x, y), axis=0)
    assert np.allclose(np.cov(X), np.array([[11.71, -4.286], [-4.286, 2.144133]]))
    assert np.allclose(np.cov(x, y), np.array([[11.71, -4.286], [-4.286, 2.144133]]))
    assert np.allclose(np.cov(x), np.array(11.71))


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_doctest_fallback(nps_app_inst)
    test_manual_cov(nps_app_inst)
