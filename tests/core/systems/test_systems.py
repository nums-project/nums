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

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block
from nums.core.grid.grid import NoDeviceGrid, CyclicDeviceGrid
from nums.core.systems.systems import RaySystem


# pylint: disable=protected-access


def test_options(app_inst_all: ArrayApplication):
    device_id = app_inst_all.cm.device_grid.get_device_id(agrid_entry=(0, 0),
                                                          agrid_shape=(1, 1))
    if isinstance(app_inst_all.cm.device_grid, NoDeviceGrid):
        assert device_id is None
    if isinstance(app_inst_all.cm.device_grid, CyclicDeviceGrid):
        assert device_id is not None


def test_warmup(app_inst_all: ArrayApplication):
    sys = app_inst_all.cm.system
    if isinstance(sys, RaySystem):
        sys.warmup(10)
    assert True


def test_block_grid_entry(app_inst_all: ArrayApplication):
    ba: BlockArray = app_inst_all.array(np.array([[1, 2, 3], [4, 5, 6]]), block_shape=(1, 3))
    block1: Block = ba.T.blocks[0, 1]
    assert block1.size() == 3
    assert block1.transposed
    assert block1.grid_entry == (0, 1)
    assert block1.grid_shape == (1, 2)
    assert block1.true_grid_entry() == (1, 0)
    assert block1.true_grid_shape() == (2, 1)


if __name__ == "__main__":
    # pylint: disable=import-error
    import conftest

    app_inst = conftest.get_app("ray-none")
    test_options(app_inst)
    test_warmup(app_inst)
    test_block_grid_entry(app_inst)
