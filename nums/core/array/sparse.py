from typing import Optional
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.blockarray import BlockArray
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
import numpy as np


class SparseBlock(Block):

    def __init__(
        self,
        grid_entry,
        grid_shape,
        shape,
        dtype,
        transposed,
        fill_value,
        cm: ComputeManager,
        id=None,
    ):
        super().__init__(grid_entry, grid_shape, shape, dtype, transposed, cm, id)
        self.fill_value = fill_value
        # Keep as ObjectRefs to avoid blocking on creation?
        self._nbytes: object = None
        self._nnz: object = None

    def __repr__(self):
        return f"SparseBlock({self.oid})"

    def size(self):
        return np.product(self.shape)
    
    def copy(self, shallow=True):
        assert shallow, "Only shallow copies are currently supported."
        block = SparseBlock(
            self.grid_entry,
            self.grid_shape,
            self.shape,
            self.dtype,
            self.transposed,
            self.fill_value,
            self._cm,
        )
        block.oid = self.oid
        return block

    def transpose(self, defer=False, redistribute=False):
        grid_entryT = tuple(reversed(self.grid_entry))
        grid_shapeT = tuple(reversed(self.grid_shape))
        blockT = SparseBlock(
            grid_entry = grid_entryT,
            grid_shape = grid_shapeT,
            shape = tuple(reversed(self.shape)),
            dtype = self.dtype,
            transposed = not self.transposed,
            cm = self._cm,
        )
        if not defer:
            blockT.transposed = False
            if redistribute:
                syskwargs = {"grid_entry": grid_entryT, "grid_shape": grid_shapeT}
            else:
                syskwargs = {
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                }
            blockT.oid = self._cm.transpose(self.oid, syskwargs=syskwargs)
        return blockT

    def ufunc(self, op_name, device_id=None):
        return self.uop_map(op_name, device_id=device_id)

    def uop_map(self, op_name, args=None, kwargs=None, device_id=None):
        block = self.copy()
        block.dtype = array_utils.get_uop_output_type(op_name, self.dtype)
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if device_id is None:
            syskwargs = {"grid_entry": block.grid_entry, "grid_shape": block.grid_shape}
        else:
            syskwargs = {"device_id": device_id}
        block.device_id = device_id
        block.oid = self._cm.sparse_uop_map(
            op_name, self.oid, args, kwargs, syskwargs=syskwargs
        )
        return block

    @staticmethod
    def init_block(op, block1, block2, args, device_id=None):
        result_grid_entry, result_grid_shape, result_shape, dtype = SparseBlock.block_meta(
            op, block1, block2, args
        )
        block = SparseBlock(
            grid_entry=result_grid_entry,
            grid_shape=result_grid_shape,
            shape=result_shape,
            dtype=dtype,
            transposed=False,
            cm=block1._cm,
        )
        block.device_id = device_id
        return block

    def bop(self, op, other, args: dict, device_id=None):
        if isinstance(other, SparseBlock):
            block = SparseBlock.init_block(op, self, other, args, device_id)
            if device_id is None:
                syskwargs = {
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                }
            else:
                syskwargs = {"device_id": device_id}
            block.oid = self._cm.sparse_bop(
                op,
                self.oid,
                other.oid,
                self.transposed,
                other.transposed,
                axes=args.get("axes"),
                syskwargs=syskwargs,
            )
            return block

    def tensordot(self, other, axes):
        return self.bop("sparse_tensordot", other, args={"axes": axes})

    def __add__(self, other):
        if isinstance(other, SparseBlock):
            return self.bop("sparse_add", other, args={})


class SparseBlockArray(BlockArray):

    def __init__(
        self,
        grid: ArrayGrid,
        cm: ComputeManager,
        fill_value = 0,
        blocks: np.ndarray = None,
    ):
        self.grid = grid
        self.cm = cm
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.grid_shape = self.grid.grid_shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        self.dtype = self.grid.dtype
        self.fill_value = fill_value
        self.blocks = blocks
        if self.blocks is None:
            self.blocks = np.empty(shape=self.grid_shape, dtype=SparseBlock)
            for grid_entry in self.grid.get_entry_iterator():
                self.blocks[grid_entry] = SparseBlock(
                    grid_entry,
                    self.grid_shape,
                    self.grid.get_block_shape(grid_entry),
                    self.dtype,
                    False,
                    fill_value,
                    self.cm,
                )
        self._nbytes: object = None
        self._nnz: object = None

    def __getattr__(self, item):
        if item == "nbytes":
            return self._nbytes #.get()
        elif item == "nnz":
            return self._nnz #.get()
        else:
            super().__getattr__(item)

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, fill_value=0):
        grid = ArrayGrid(ba.shape, ba.block_shape, ba.dtype.__name__)
        sba = SparseBlockArray(grid, ba.cm, fill_value)
        nbytes_oids = []
        nnz_oids = []
        for grid_entry in grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            sba.blocks[grid_entry].oid, nb, nz = sba.cm.dense_to_sparse(
                block.oid,
                fill_value,
                syskwargs = {
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                    "num_returns": 3,
                },
            )
            sba.blocks[grid_entry]._nbytes = nb
            sba.blocks[grid_entry]._nnz = nz
            nbytes_oids.append(nb)
            nnz_oids.append(nz)
        sba._nbytes = sba.cm.sum_reduce(
            *nbytes_oids,
            syskwargs = {"device_id": 0}
        )
        sba._nnz = sba.cm.sum_reduce(
            *nnz_oids,
            syskwargs = {"device_id": 0}
        )
        return sba

    def to_ba(self):
        grid = ArrayGrid(self.shape, self.block_shape,self.dtype.__name__)
        ba = BlockArray(grid, self.cm)
        for grid_entry in grid.get_entry_iterator():
            block: SparseBlock = self.blocks[grid_entry]
            ba.blocks[grid_entry].oid = ba.cm.sparse_to_dense(
                block.oid,
                syskwargs = {
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
            )
        return ba

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = SparseBlockArray(grid_copy, self.cm, self.fill_value)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def ufunc(self, op_name):
        result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        return result

    def __elementwise__(self, op_name, other):
        pass

    def __add__(self, other):
        if isinstance(other, SparseBlockArray):
            return self.__elementwise__("add", self)
        elif isinstance(other, BlockArray):
            return self.__elementwise__("sd_add", self)

    def __radd__(self, other):
        pass
