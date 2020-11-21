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


import numpy as np

from nums.core.systems.systems import RaySystem
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block


# pylint: disable=protected-access


def test_options(app_inst):
    result = app_inst.system.get_options(cluster_entry=(0, 0), cluster_shape=(1, 1))
    assert len(result) > 0


def test_warmup(app_inst):
    sys = app_inst.system
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
