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

from nums.core.systems.systems import RaySystem
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block


# pylint: disable=protected-access


def test_options(app_inst):
    result = app_inst._system.get_options(cluster_entry=(0, 0), cluster_shape=(1, 1))
    assert len(result) > 0


def test_warmup(app_inst):
    sys = app_inst._system
    if isinstance(sys, RaySystem):
        sys.warmup(10)
    assert True


def test_block_grid_entry(app_inst: ArrayApplication):
    ba: BlockArray = app_inst.array(np.array([[1, 2, 3], [4, 5, 6]]), block_shape=(1, 3))
    block1: Block = ba.T.blocks[0, 1]
    assert block1.size() == 3
    assert block1.transposed
    assert block1.grid_entry == (0, 1)
    assert block1.grid_shape == (1, 2)
    assert block1.true_grid_entry() == (1, 0)
    assert block1.true_grid_shape() == (2, 1)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_options(app_inst)
    test_warmup(app_inst)
    test_block_grid_entry(app_inst)
