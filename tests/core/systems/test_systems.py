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
from nums.core.backends.backends import RayBackend


# pylint: disable=protected-access


def test_warmup(app_inst_all: ArrayApplication):
    sys = app_inst_all.cm.backend
    if isinstance(sys, RayBackend):
        sys.warmup(10)
    assert True


def test_transposed_block(app_inst_all: ArrayApplication):
    ba: BlockArray = app_inst_all.array(
        np.array([[1, 2, 3], [4, 5, 6]]), block_shape=(1, 3)
    )
    block1: Block = ba.T.blocks[0, 1]
    assert block1.size() == 3
    assert not block1.transposed
    assert block1.grid_entry == block1.true_grid_entry()
    assert block1.grid_shape == block1.true_grid_shape()


def test_deferred_transposed_block(app_inst_all: ArrayApplication):
    ba: BlockArray = app_inst_all.array(
        np.array([[1, 2, 3], [4, 5, 6]]), block_shape=(1, 3)
    )
    block1: Block = ba.transpose(defer=True).blocks[0, 1]
    assert block1.size() == 3
    assert block1.transposed
    assert block1.grid_entry == (0, 1)
    assert block1.grid_shape == (1, 2)
    assert block1.true_grid_entry() == (1, 0)
    assert block1.true_grid_shape() == (2, 1)


if __name__ == "__main__":
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "mpi"
    nps_app_inst = application_manager.instance()
    # test_warmup(nps_app_inst)
    test_transposed_block(nps_app_inst)
    test_deferred_transposed_block(nps_app_inst)
