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


import time

import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array import utils as array_utils


def test_reshape_basic(app_inst):
    def _reshape_by_entry(app_inst, arr: BlockArray, shape, block_shape) -> BlockArray:
        dst_arr = app_inst.empty(shape=shape, block_shape=block_shape, dtype=arr.dtype)
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_slice_selection = dst_arr.grid.get_slice(dst_grid_entry)
            dst_index_list = array_utils.slice_sel_to_index_list(dst_slice_selection)
            src_index_list = array_utils.translate_index_list(dst_index_list, shape, arr.shape)
            for i in range(len(dst_index_list)):
                dst_index = dst_index_list[i]
                src_index = src_index_list[i]
                dst_arr[dst_index] = arr[src_index]
        return dst_arr

    src_shape, dst_shape = (1, 20, 3), (10, 3, 1, 2, 1)
    dst_block_shape = (5, 1, 1, 1, 3)
    np_arr = np.arange(np.product(src_shape)).reshape(src_shape)
    true_arr = np_arr.reshape(dst_shape)
    src_arr: BlockArray = app_inst.array(np_arr, block_shape=(1, 10, 3))
    t = time.time()
    dst_arr_blockwise: BlockArray = src_arr.reshape(dst_shape, block_shape=dst_block_shape)
    dst_arr_blockwise.touch()
    print("blockwise time", time.time() - t)
    assert np.allclose(dst_arr_blockwise.get(), true_arr)
    t = time.time()
    dst_arr_entrywise: BlockArray = _reshape_by_entry(app_inst, src_arr,
                                                      dst_shape,
                                                      dst_block_shape)
    dst_arr_entrywise.touch()
    print("entrywise time", time.time() - t)
    assert np.allclose(dst_arr_entrywise.get(), true_arr)


def test_reshape_ones(app_inst: ArrayApplication):

    def _strip_ones(shape, block_shape):
        indexes = np.where(np.array(shape) != 1)
        return tuple(np.array(shape)[indexes]), tuple(np.array(block_shape)[indexes])

    # inject many different variants of ones, and ensure the block shapes match at every level.
    shapes = [
        [(10, 2, 20, 5, 3), (5, 1, 4, 3, 3)],
        [(10, 1, 5, 1, 3), (5, 1, 4, 1, 3)],
        [(1, 2, 3), (1, 1, 1)],
        [(10, 1), (2, 1)],
        [(1, 100, 10), (1, 10, 10)],
        [(), ()],
        [(1,), (1,)],
        [(1, 1), (1, 1)],
        [(1, 1, 1), (1, 1, 1)],
    ]
    num_ones = [1, 2, 3]

    for shape, block_shape in shapes:
        arr = app_inst.random_state(1337).random(shape, block_shape)
        arr_np = arr.get()

        # Try removing ones.
        new_shape, new_block_shape = _strip_ones(shape, block_shape)
        new_arr = arr.reshape(new_shape, block_shape=new_block_shape)
        for grid_entry in new_arr.grid.get_entry_iterator():
            new_block: Block = new_arr.blocks[grid_entry]
            new_block_np = new_block.get()
            assert new_block.shape == new_block_np.shape
        assert np.allclose(arr_np, new_arr.get().reshape(shape))

        # Try adding ones.
        for nones in num_ones:
            for pos in range(len(shape)+1):
                ones = [1]*nones
                new_shape = list(shape)
                new_shape = new_shape[:pos] + ones + new_shape[pos:]
                new_block_shape = list(block_shape)
                new_block_shape = new_block_shape[:pos] + ones + new_block_shape[pos:]
                new_arr = arr.reshape(new_shape, block_shape=new_block_shape)
                for grid_entry in new_arr.grid.get_entry_iterator():
                    new_block: Block = new_arr.blocks[grid_entry]
                    new_block_np = new_block.get()
                    assert new_block.shape == new_block_np.shape
                assert np.allclose(arr_np, new_arr.get().reshape(shape))


def test_reshape_blocks_only(app_inst):
    shape, block_shape = (3, 5, 10), (3, 2, 5)
    arr = app_inst.random_state(1337).random(shape, block_shape)
    arr_np = arr.get()
    assert np.allclose(arr_np, arr.reshape(shape, block_shape=(2, 2, 5)).get())
    assert np.allclose(arr_np, arr.reshape(shape, block_shape=(2, 3, 5)).get())
    assert np.allclose(arr_np, arr.reshape(shape, block_shape=(2, 3, 7)).get())


if __name__ == "__main__":
    # pylint: disable=import-error, no-member
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_reshape_basic(app_inst)
    test_reshape_ones(app_inst)
    test_reshape_blocks_only(app_inst)
