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


import numpy as np

from nums.core.array.application import ArrayApplication


def test_stats(app_inst: ArrayApplication):
    np_x = np.arange(100)
    ba_x = app_inst.array(np_x, block_shape=np_x.shape)
    assert np.allclose(np.mean(np_x), app_inst.mean(ba_x).get())
    assert np.allclose(np.std(np_x), app_inst.std(ba_x).get())


def test_uops(app_inst: ArrayApplication):
    np_x = np.arange(100)
    ba_x = app_inst.array(np_x, block_shape=np_x.shape)
    assert np.allclose(np.abs(np_x), app_inst.abs(ba_x).get())
    assert np.allclose(np.linalg.norm(np_x), app_inst.norm(ba_x).get())


def test_bops(app_inst: ArrayApplication):
    pairs = [(1, 2),
             (2.0, 3.0),
             (2, 3.0),
             (2.0, 3)]
    for a, b in pairs:
        np_a, np_b = np.array(a), np.array(b)
        ba_a, ba_b = app_inst.scalar(a), app_inst.scalar(b)
        assert np.allclose(np_a + np_b, (ba_a + ba_b).get())
        assert np.allclose(np_a - np_b, (ba_a - ba_b).get())
        assert np.allclose(np_a * np_b, (ba_a * ba_b).get())
        assert np.allclose(np_a / np_b, (ba_a / ba_b).get())
        assert np.allclose(np_a ** np_b, (ba_a ** ba_b).get())


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_stats(app_inst)
    test_uops(app_inst)
    test_bops(app_inst)
