import time
import nums.numpy as nps

import numpy as np
import pytest

from nums.core.array import utils as array_utils
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray, Block
from nums.core import application_manager

app_inst = application_manager.instance()

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
            for pos in range(len(shape) + 1):
                ones = [1] * nones
                new_shape = list(shape)
                new_shape = new_shape[:pos] + ones + new_shape[pos:]
                new_block_shape = list(block_shape)
                new_block_shape = new_block_shape[:pos] + ones + new_block_shape[pos:]
                print("Shapes: {} {}".format(shape, new_shape))
                print("Block Shapes: {} {}".format(block_shape, new_block_shape))
                print("__________________________________________________________________________________")
                new_arr = arr.reshape(new_shape, block_shape=new_block_shape)
                for grid_entry in new_arr.grid.get_entry_iterator():
                    new_block: Block = new_arr.blocks[grid_entry]
                    new_block_np = new_block.get()
                    assert new_block.shape == new_block_np.shape
                assert np.allclose(arr_np, new_arr.get().reshape(shape))

test_reshape_ones(app_inst)