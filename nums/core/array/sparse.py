from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.blockarray import BlockArray
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
from typing import Iterator, List, Tuple
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
        self.oid = None
        self.nbytes = None

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
            self.cm,
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
        block.oid = self._cm.sparse_map_uop(
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

    def __add__(self, other):
        if isinstance(other, SparseBlock):
            return self.bop("sparse_add", other, args={})

    def tensordot(self, other, axes):
        return self.bop("sparse_tensordot", other, args={"axes": axes})


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
        self.nbytes = 0 #
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

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, fill_value=0):
        grid = ArrayGrid(ba.shape, ba.block_shape, ba.dtype.__name__)
        sba = SparseBlockArray(grid, ba.cm, fill_value)
        for grid_entry in grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            sba.blocks[grid_entry].oid = sba.cm.dense_to_sparse(
                block.oid,
                fill_value,
                syskwargs = {
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
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


# class ArrayIndex(ArrayGrid):
#     """
#     For now, use COO
#     """

#     def __init__(self, shape: Tuple, block_shape: Tuple, dtype: str):
#         super().__init__(shape, block_shape, dtype)
#         self.coordinates = []  # Basically grid_entry
#         self.blocks = []

#     def get_sparse_entry_iterator(self) -> Iterator[Tuple]:
#         return self.coordinates

#     def _compare_coord(self, coord1, coord2):
#         assert len(coord1) == len(coord2)
#         for i in range(len(coord1)):
#             if coord1[i] > coord2[i]:
#                 return 1
#             elif coord1[i] < coord2[i]:
#                 return -1
#         return 0

#     def insert_block(self, coord, block):
#         """
#         Keep sorted by coordinate using binary insertion
#         """
#         top, bot = 0, len(self.coordinates) - 1
#         while top < bot:
#             mid = top + (bot - top) // 2
#             comp = self._compare_coord(coord, self.coordinates[mid])
#             if comp == 1:
#                 top = mid + 1
#             elif comp == -1:
#                 bot = mid
#             else:
#                 raise Exception(f"Block already exists at {coord}")
#         self.coordinates.insert(top, coord)
#         self.blocks.insert(top, block)

#     def block_at(self, coord):
#         # Inefficient
#         for i, c in enumerate(self.coordinates):
#             if c == coord:
#                 return self.blocks[i]
#         return None
    
#     def copy(self):
#         index = self.from_meta(self.to_meta())
#         index.coordinates = []
#         index.blocks = []
#         for i, grid_entry in enumerate(self.get_sparse_entry_iterator()):
#             index.coordinates.append(tuple(grid_entry))
#             index.blocks.append(self.blocks[i].copy())
#         return index


# class SparseBlockArrayI(object):
#     @classmethod
#     def from_ba(cls, ba: BlockArrayBase, fill_value=0):
#         index = ArrayIndex(ba.shape, ba.block_shape, ba.dtype.__name__)
#         sp_ba = SparseBlockArrayI(index, ba.cm, fill_value)
#         mask = ba != fill_value
#         oids = []
#         for grid_entry in mask.grid.get_entry_iterator():
#             block: Block = mask.blocks[grid_entry]
#             oid = mask.cm.any(
#                 block.oid,
#                 syskwargs={
#                     "grid_entry": block.grid_entry,
#                     "grid_shape": block.grid_shape,
#                 },
#             )
#             oids.append((grid_entry, oid))
#         for grid_entry, oid in oids:
#             if sp_ba.cm.get(oid):
#                 block: Block = ba.blocks[grid_entry]
#                 oid = sp_ba.cm.dense_to_sparse(
#                     block.oid,
#                     fill_value,
#                     syskwargs={
#                         "grid_entry": block.grid_entry,
#                         "grid_shape": block.grid_shape,
#                     },
#                 )
#                 sp_block = SparseBlock(
#                     grid_entry,
#                     index.grid_shape,
#                     index.shape,
#                     dtype=ba.dtype,
#                     transposed=False,
#                     cm=sp_ba.cm,
#                 )
#                 sp_block.oid = oid
#                 sp_ba.index.insert_block(grid_entry, sp_block)
#         return sp_ba

#     @classmethod
#     def random(
#         cls,
#         cm: ComputeManager, 
#         shape,
#         block_shape,
#         density,
#         random_state,
#         data_rvs,
#         dtype,
#         fill_value=0,
#     ):
#         index = ArrayIndex(shape, block_shape, dtype.__name__)
#         sp_ba = SparseBlockArrayI(index, cm, fill_value)
#         for grid_entry in sp_ba.index.get_entry_iterator():
#             sp_block = SparseBlock(
#                 grid_entry,
#                 index.grid_shape,
#                 index.shape,
#                 dtype=dtype,
#                 transposed=False,
#                 cm=cm
#             )
#             sp_block.oid = cm.sparse_random_block(
#                 block_shape,
#                 density,
#                 random_state,
#                 data_rvs,
#                 fill_value,
#                 syskwargs={
#                     "grid_entry": grid_entry,
#                     "grid_shape": index.grid_shape,
#                 },
#             )
#             sp_ba.index.insert_block(grid_entry, sp_block)
#         return sp_ba

#     def __init__(self, index: ArrayIndex, cm: ComputeManager, fill_value):
#         self.index = index
#         self.cm = cm
#         self.shape = index.shape
#         self.block_shape = index.block_shape
#         self.grid_shape = index.grid_shape
#         self.size = len(self.shape)
#         self.ndim = np.product(self.shape)
#         self.dtype = self.index.dtype
#         self.fill_value = fill_value
#         init_ops = {
#             0: "zeros",
#             1: "ones",
#         }
#         self.init_op = init_ops.get(fill_value, "empty")

#     def __repr__(self):
#         return f"SparseBlockArrayI({self.index.blocks})"

#     def get(self):
#         return self.to_ba(self).get()

#     def copy(self):
#         index_copy = self.index.copy()
#         rarr_copy = SparseBlockArrayI(index_copy, self.cm, self.fill_value)
#         return rarr_copy

#     def to_ba(self):
#         grid = ArrayGrid(self.shape, self.block_shape, self.dtype.__name__)
#         grid_meta = grid.to_meta()
#         ba = BlockArray(grid, self.cm)
#         for grid_entry in self.index.get_entry_iterator():
#             # FIXME: inefficient linear search
#             if grid_entry in self.index.coordinates:
#                 sp_block = self.index.block_at(grid_entry)
#                 ba.blocks[grid_entry].oid = self.cm.sparse_to_dense(
#                     sp_block.oid,
#                     syskwargs={
#                         "grid_entry": grid_entry,
#                         "grid_shape": self.grid_shape,
#                     },
#                 )
#             else:
#                 ba.blocks[grid_entry].oid = self.cm.new_block(
#                     self.init_op,
#                     grid_entry,
#                     grid_meta,
#                     syskwargs={
#                         "grid_entry": grid_entry,
#                         "grid_shape": self.grid_shape,
#                     },
#                 )
#         return ba

#     def __elementwise__(self, op_name, other):
#         assert self.shape == other.shape and self.block_shape

#     def __add__(self, other):
#         return self.__elementwise__("add", other)
