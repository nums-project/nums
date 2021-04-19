import numpy as np
import scipy as sp

from nums.core.array.blockarray import BlockArray
from nums.core.array.base import Block
from nums.core.array import utils as array_utils
from nums.core.storage.storage import ArrayGrid

class SparseBlockArray(BlockArray):        

    def get(self) -> np.ndarray:
        result: np.ndarray = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        block_shape: np.ndarray = np.array(self.grid.block_shape, dtype=np.int)
        arrays: list = self.system.get([self.blocks[grid_entry].oid
                                        for grid_entry in self.grid.get_entry_iterator()])
        for block_index, grid_entry in enumerate(self.grid.get_entry_iterator()):
            start = block_shape * grid_entry
            entry_shape = np.array(self.grid.get_block_shape(grid_entry), dtype=np.int)
            end = start + entry_shape
            slices = tuple(map(lambda item: slice(*item), zip(*(start, end))))
            block: Block = self.blocks[grid_entry]
            arr: np.ndarray = arrays[block_index].toarray()
            if block.transposed:
                arr = arr.T
            result[slices] = arr.reshape(block.shape)
        return result

    @classmethod
    def from_np(cls, arr, block_shape, copy, system):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = SparseBlockArray(grid, system)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = sp.sparse.csr_matrix(arr[grid_slice])

            rarr.blocks[grid_entry].oid = system.put(block)
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, system):
        sample_idx = tuple(0 for dim in arr.shape)
        if isinstance(arr, Block):
            sample_block = arr
            result_shape = ()
        else:
            sample_block = arr[sample_idx]
            if result_shape is None:
                result_shape = array_utils.shape_from_block_array(arr)
        result_block_shape = sample_block.shape
        result_dtype_str = sample_block.dtype.__name__
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=result_dtype_str)
        assert arr.shape == result_grid.grid_shape
        result = SparseBlockArray(result_grid, system)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, Block):
                block: Block = arr
            else:
                block: Block = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        other = self.check_or_convert_other(other)
        return self.from_blocks(
            self.blocks * other.blocks,
            result_shape=None,
            system=self.system,
        )
