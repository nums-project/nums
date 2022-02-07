from nums.core.array.base import Block, BlockArrayBase
from nums.core.array.blockarray import BlockArray
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
from typing import Iterator, List, Tuple
import numpy as np


class SparseBlock(Block):
    """
    Probably use sparse.GCXS at the block level
    """

    def transpose(self):
        pass

    @staticmethod
    def init_block(op, block1, block2, args, device_id=None):
        result_grid_entry, result_grid_shape, result_shape, dtype = Block.block_meta(
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
            block: Block = self.init_block(op, self, other, args, device_id)
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

    def __add__(self, other):
        return self.bop("sparse_add", other, args={})

    def tensordot(self, other, axes):
        return self.bop("sparse_tensordot", other, args={"axes": axes})


class ArrayIndex(ArrayGrid):
    """
    For now, use COO
    """

    def __init__(self, shape: Tuple, block_shape: Tuple, dtype: str):
        super().__init__(shape, block_shape, dtype)
        self.coordinates = []  # Basically grid_entry
        self.blocks = []

    def get_sparse_entry_iterator(self) -> Iterator[Tuple]:
        return self.coordinates

    def _compare_coord(self, coord1, coord2):
        assert len(coord1) == len(coord2)
        for i in range(len(coord1)):
            if coord1[i] > coord2[i]:
                return 1
            elif coord1[i] < coord2[i]:
                return -1
        return 0

    def insert_block(self, coord, block):
        """
        Keep sorted by coordinate using binary insertion
        """
        top, bot = 0, len(self.coordinates) - 1
        while top < bot:
            mid = top + (bot - top) // 2
            comp = self._compare_coord(coord, self.coordinates[mid])
            if comp == 1:
                top = mid + 1
            elif comp == -1:
                bot = mid
            else:
                raise Exception(f"Block already exists at {coord}")
        self.coordinates.insert(top, coord)
        self.blocks.insert(top, block)

    def block_at(self, coord):
        # Inefficient
        for i, c in enumerate(self.coordinates):
            if c == coord:
                return self.blocks[i]
        return None


class SparseBlockArray(object):
    @classmethod
    def from_ba(cls, ba: BlockArrayBase, fill_value=0):
        index = ArrayIndex(ba.shape, ba.block_shape, ba.dtype.__name__)
        sp_ba = SparseBlockArray(index, ba.cm, fill_value)
        mask = ba != fill_value
        oids = []
        for grid_entry in mask.grid.get_entry_iterator():
            block: Block = mask.blocks[grid_entry]
            oid = mask.cm.any(
                block.oid,
                syskwargs={
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
            )
            oids.append((grid_entry, oid))
        for grid_entry, oid in oids:
            if sp_ba.cm.get(oid):
                block: Block = ba.blocks[grid_entry]
                oid = sp_ba.cm.dense_to_sparse(
                    block.oid,
                    fill_value,
                    syskwargs={
                        "grid_entry": block.grid_entry,
                        "grid_shape": block.grid_shape,
                    },
                )
                sp_block = SparseBlock(
                    grid_entry,
                    index.grid_shape,
                    index.shape,
                    dtype=ba.dtype,
                    transposed=False,
                    cm=sp_ba.cm,
                )
                sp_block.oid = oid
                sp_ba.index.insert_block(grid_entry, sp_block)
        return sp_ba

    def __init__(self, index: ArrayIndex, cm: ComputeManager, fill_value):
        self.index = index
        self.cm = cm
        self.shape = index.shape
        self.block_shape = index.block_shape
        self.grid_shape = index.grid_shape
        self.size = len(self.shape)
        self.ndim = np.product(self.shape)
        self.dtype = self.index.dtype
        self.fill_value = fill_value
        init_ops = {
            0: "zeros",
            1: "ones",
        }
        self.init_op = init_ops.get(fill_value, "empty")

    def __repr__(self):
        return f"SparseBlockArray({self.index.blocks})"

    def get(self):
        pass

    def to_ba(self):
        grid = ArrayGrid(self.shape, self.block_shape, self.dtype.__name__)
        grid_meta = grid.to_meta()
        ba = BlockArray(grid, self.cm)
        for grid_entry in self.index.get_entry_iterator():
            # FIXME: inefficient linear search
            if grid_entry in self.index.coordinates:
                sp_block = self.index.block_at(grid_entry)
                ba.blocks[grid_entry].oid = self.cm.sparse_to_dense(
                    sp_block.oid,
                    syskwargs={
                        "grid_entry": grid_entry,
                        "grid_shape": self.grid_shape,
                    },
                )
            else:
                ba.blocks[grid_entry].oid = self.cm.new_block(
                    self.init_op,
                    grid_entry,
                    grid_meta,
                    syskwargs={
                        "grid_entry": grid_entry,
                        "grid_shape": self.grid_shape,
                    },
                )
        return ba

    def __elementwise__(self, op_name, other):
        assert self.shape == other.shape and self.block_shape

    def __add__(self, other):
        return self.__elementwise__("add", other)
