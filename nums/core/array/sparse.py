from typing import List
import itertools
import warnings
import numpy as np
import sparse

from nums.core.array import utils as array_utils
from nums.core.array.base import BlockBase, Block, BlockArrayBase
from nums.core.array.blockarray import BlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import ArrayGrid


# pylint: disable=protected-access, redefined-builtin


class SparseBlock(BlockBase):
    def __init__(
        self,
        grid_entry,
        grid_shape,
        shape,
        dtype,
        transposed,
        km: KernelManager,
        id=None,
        index_dtype=np.int64,
    ):
        super().__init__(grid_entry, grid_shape, shape, dtype, transposed, km, id)
        self.index_dtype = index_dtype
        self.oid = None
        self._nnz: object = None
        self._nbytes: object = None

    @property
    def is_dense(self):
        return False

    @property
    def nnz(self):
        if self._nnz is None:
            self._nnz = self.km.sparse_nnz(
                self.oid,
                syskwargs={
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                },
            )
        if not array_utils.is_int(self._nnz):
            self._nnz = self.km.get(self._nnz)
        return self._nnz

    @property
    def nbytes(self):
        if self._nbytes is None:
            self._nbytes = self.km.sparse_nbytes(
                self.oid,
                syskwargs={
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                },
            )
        if not array_utils.is_int(self._nbytes):
            self._nbytes = self.km.get(self._nbytes)
        return self._nbytes

    def __repr__(self):
        return f"SparseBlock({self.oid})"

    def copy(self, shallow=True):
        assert shallow, "Only shallow copies are currently supported."
        block = SparseBlock(
            self.grid_entry,
            self.grid_shape,
            self.shape,
            self.dtype,
            self.transposed,
            self.km,
        )
        block.oid = self.oid
        return block

    def map_uop(self, op_name, args=None, kwargs=None, device=None):
        densify = array_utils.get_sparse_uop_densify(op_name)
        if densify:
            block = Block(
                self.grid_entry,
                self.grid_shape,
                self.shape,
                self.dtype,
                self.transposed,
                self.km,
            )
        else:
            block = self.copy()
        block.dtype = array_utils.get_uop_output_type(op_name, self.dtype)
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if device is None:
            syskwargs = {"grid_entry": block.grid_entry, "grid_shape": block.grid_shape}
        else:
            syskwargs = {"device": device}
        block._device = device
        block.oid = self.km.sparse_map_uop(
            op_name, self.oid, args, kwargs, densify, syskwargs=syskwargs
        )
        block._nnz = self.km.sparse_nnz(block.oid, syskwargs=syskwargs)
        block._nbytes = self.km.sparse_nbytes(block.oid, syskwargs=syskwargs)
        return block

    def block_from_scalar(self, other):
        assert array_utils.is_scalar(other)
        # Construct sparse block only if value is 0.
        if np.isclose(other, 0):
            block = SparseBlock(
                self.grid_entry,
                self.grid_shape,
                (1,),
                self.dtype,
                False,
                self.km,
            )
            block.oid = self.km.sparse_block_from_scalar(
                other,
                syskwargs={
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                },
            )
        else:
            block = Block(
                self.grid_entry,
                self.grid_shape,
                (1,),
                self.dtype,
                False,
                self.km,
            )
            block.oid = self.km.block_from_scalar(
                other,
                syskwargs={
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                },
            )
        return block

    @staticmethod
    def init_block(op_name, block1, block2, args, device=None):
        (
            result_grid_entry,
            result_grid_shape,
            result_shape,
            dtype,
        ) = BlockBase.block_meta(op_name, block1, block2, args)
        # TODO: what happens when different index_dtype?
        block = SparseBlock(
            grid_entry=result_grid_entry,
            grid_shape=result_grid_shape,
            shape=result_shape,
            dtype=dtype,
            transposed=False,
            km=block1.km,
        )
        block._device = device
        return block

    def _check_bop_implemented(self, other):
        if isinstance(other, BlockBase) or array_utils.is_scalar(other):
            return True
        return False

    @staticmethod
    def binary_op(op_name, a: BlockBase, b: BlockBase, args: dict, device=None):
        if isinstance(a, SparseBlock) and array_utils.is_scalar(b):
            b = a.block_from_scalar(b)
        elif isinstance(b, SparseBlock) and array_utils.is_scalar(a):
            a = b.block_from_scalar(a)
        if not isinstance(a, BlockBase) or not isinstance(b, BlockBase):
            raise NotImplementedError()

        densify = array_utils.get_sparse_bop_densify(op_name, a.is_dense, b.is_dense)
        if densify:
            block = Block.init_block(op_name, a, b, args, device)
        else:
            block = SparseBlock.init_block(op_name, a, b, args, device)
        if device is None:
            syskwargs = {
                "grid_entry": block.grid_entry,
                "grid_shape": block.grid_shape,
            }
        else:
            syskwargs = {"device": device}
        block.oid = a.km.sparse_bop(
            op_name,
            a.oid,
            b.oid,
            a.transposed,
            b.transposed,
            axes=args.get("axes"),
            densify=densify,
            syskwargs=syskwargs,
        )
        if not densify:
            block._nnz = a.km.sparse_nnz(block.oid, syskwargs=syskwargs)
            block._nbytes = a.km.sparse_nbytes(block.oid, syskwargs=syskwargs)
        return block

    def bop(self, op_name, other, args: dict, device=None):
        return self.binary_op(op_name, self, other, args, device)

    def tensordot(self, other, axes):
        return self.binary_op("tensordot", self, other, args={"axes": axes})


class SparseBlockArray(BlockArrayBase):
    def __init__(
        self,
        grid: ArrayGrid,
        km: KernelManager,
        blocks: np.ndarray = None,
    ):
        if blocks is not None:
            assert (
                blocks.dtype == SparseBlock
            ), "SparseBlockArray must be initialized with SparseBlocks"
        super().__init__(grid, km, blocks)
        if self.blocks is None:
            self.blocks = np.empty(shape=self.grid_shape, dtype=SparseBlock)
            for grid_entry in self.grid.get_entry_iterator():
                self.blocks[grid_entry] = SparseBlock(
                    grid_entry,
                    self.grid_shape,
                    self.grid.get_block_shape(grid_entry),
                    self.dtype,
                    False,
                    self.km,
                )
        self._nnz = -1
        self._nbytes = -1

    @property
    def is_dense(self):
        return False

    @property
    def nnz(self):
        return self._get_nnz()

    def _get_nnz(self):
        if self._nnz == -1:
            self._nnz = 0
            for grid_entry in self.grid.get_entry_iterator():
                self._nnz += self.blocks[grid_entry].nnz
        return self._nnz

    @property
    def nbytes(self):
        return self._get_nbytes()

    def _get_nbytes(self):
        if self._nbytes == -1:
            self._nbytes = 0
            for grid_entry in self.grid.get_entry_iterator():
                self._nbytes += self.blocks[grid_entry].nbytes
        return self._nbytes

    def __repr__(self):
        return f"SparseBlockArray({self.blocks})"

    @classmethod
    def empty(cls, shape, block_shape, dtype, km: KernelManager):
        return SparseBlockArray.create("empty", shape, block_shape, dtype, km)

    @classmethod
    def create(cls, create_op_name, shape, block_shape, dtype, km: KernelManager):
        grid = ArrayGrid(shape=shape, block_shape=block_shape, dtype=dtype.__name__)
        grid_meta = grid.to_meta()
        arr = SparseBlockArray(grid, km)
        for grid_entry in grid.get_entry_iterator():
            arr.blocks[grid_entry].oid = km.new_sparse_block(
                create_op_name,
                grid_entry,
                grid_meta,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return arr

    @classmethod
    def from_np(cls, arr, block_shape, copy, km):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = SparseBlockArray(grid, km)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = np.copy(block)
            # TODO: generalize for different kernels
            block = sparse.COO.from_numpy(block, fill_value=0)
            rarr.blocks[grid_entry].oid = km.put(
                block,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_sparse(cls, arr, block_shape, copy, km, fill_value=0):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = SparseBlockArray(grid, km, fill_value)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = sparse.COO.copy(block)
            # TODO: generalize for different kernels
            rarr.blocks[grid_entry].oid = km.put(
                block,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_scalar(cls, val, km):
        # Only create SparseBlockArray with 0s. Other scalars should use dense BlockArray.
        if not np.isclose(val, 0):
            warnings.warn(
                "%s cannot fill SparseBlockArray. Converting to BlockArray." % val
            )
            return BlockArray.from_np(np.array(val), (), copy=False, km=km)
        return SparseBlockArray.from_np(np.array(val), (), copy=False, km=km)

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, km):
        sample_idx = tuple(0 for _ in arr.shape)
        if isinstance(arr, SparseBlock):
            sample_block = arr
            result_shape = ()
        else:
            sample_block = arr[sample_idx]
            if result_shape is None:
                result_shape = array_utils.shape_from_block_array(arr)
        result_block_shape = sample_block.shape
        result_dtype_str = sample_block.dtype.__name__
        result_grid = ArrayGrid(
            shape=result_shape, block_shape=result_block_shape, dtype=result_dtype_str
        )
        assert arr.shape == result_grid.grid_shape
        result = SparseBlockArray(result_grid, km)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, SparseBlock):
                block: SparseBlock = arr
            else:
                block: SparseBlock = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    @classmethod
    def from_ba(cls, ba: BlockArrayBase):
        assert (
            ba.shape != ()
        ), "from_ba does not support scalar BlockArray. Use from_scalar."
        grid = ArrayGrid(ba.shape, ba.block_shape, ba.dtype.__name__)
        sba = SparseBlockArray(grid, ba.km)
        for grid_entry in grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            sblock: SparseBlock = sba.blocks[grid_entry]
            syskwargs = {
                "grid_entry": grid_entry,
                "grid_shape": block.grid_shape,
            }
            sblock.oid = sba.km.dense_to_sparse(
                block.oid,
                syskwargs=syskwargs,
            )
            sblock._nnz = sba.km.sparse_nnz(sblock.oid, syskwargs=syskwargs)
            sblock._nbytes = sba.km.sparse_nbytes(sblock.oid, syskwargs=syskwargs)
        return sba

    def to_ba(self):
        ba = BlockArray(self.grid.copy(), self.km)
        for grid_entry in ba.grid.get_entry_iterator():
            block: SparseBlock = self.blocks[grid_entry]
            ba.blocks[grid_entry].oid = ba.km.sparse_to_dense(
                block.oid,
                syskwargs={
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
            )
        return ba

    def todense(self) -> BlockArray:
        return self.to_ba()

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = SparseBlockArray(grid_copy, self.km)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def astype(self, dtype):
        grid = ArrayGrid(self.shape, self.block_shape, dtype.__name__)
        result = BlockArray(grid, self.km)
        for grid_entry in result.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].astype(dtype)
        return result

    @staticmethod
    def to_block_array(obj, km: KernelManager, block_shape=None):
        if isinstance(obj, (BlockArray, SparseBlockArray)):
            return obj
        if isinstance(obj, np.ndarray):
            np_array = obj
        elif isinstance(obj, list):
            np_array = np.array(obj)
        elif array_utils.is_scalar(obj):
            return SparseBlockArray.from_scalar(obj, km)
        else:
            raise Exception("Unsupported type %s" % type(obj))
        if block_shape is None:
            block_shape = km.get_block_shape(np_array.shape, np_array.dtype)
        # Assume object is dense.
        return BlockArray.from_np(np_array, block_shape, False, km)

    def check_or_convert_other(self, other, compute_block_shape=False):
        block_shape = None if compute_block_shape else self.block_shape
        return SparseBlockArray.to_block_array(other, self.km, block_shape=block_shape)

    def _check_bop_implemented(self, other):
        if isinstance(
            other, (BlockArrayBase, np.ndarray, list)
        ) or array_utils.is_scalar(other):
            return True
        return False

    def ufunc(self, op_name):
        densify = array_utils.get_sparse_uop_densify(op_name)
        if densify:
            result = BlockArray(self.grid, self.km)
        else:
            result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        result._nnz = -1
        result._nbytes = -1
        return result

    def reduce_axis(self, op_name, axis, keepdims=False):
        if not (axis is None or isinstance(axis, (int, np.int32, np.int64))):
            raise NotImplementedError("Only integer axis is currently supported.")
        if 0 in self.shape:
            return SparseBlockArray.create("zeros", (), (), float, self.km)
        block_reduced_oids = np.empty_like(self.blocks, dtype=tuple)
        for grid_entry in self.grid.get_entry_iterator():
            block = self.blocks[grid_entry]
            block_oid = self.km.sparse_reduce_axis(
                op_name=op_name,
                arr=block.oid,
                axis=axis,
                keepdims=keepdims,
                transposed=block.transposed,
                syskwargs={
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                },
            )
            block_reduced_oids[grid_entry] = (
                block_oid,
                block.grid_entry,
                block.grid_shape,
                False,
            )
        result_shape = []
        result_block_shape = []
        for curr_axis in range(len(self.shape)):
            axis_size, axis_block_size = (
                self.shape[curr_axis],
                self.block_shape[curr_axis],
            )
            if curr_axis == axis or axis is None:
                if keepdims:
                    axis_size, axis_block_size = 1, 1
                else:
                    continue
            result_shape.append(axis_size)
            result_block_shape.append(axis_block_size)
        result_shape = tuple(result_shape)
        result_block_shape = tuple(result_block_shape)
        result_dtype = array_utils.get_reduce_output_type(op_name, self.dtype)
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=result_dtype.__name__,
        )
        result = BlockArray(result_grid, self.km)

        if axis is None:
            if result.shape == ():
                result_block: Block = result.blocks[()]
            else:
                result_block: Block = result.blocks[:].item()
            result_block.oid = self.tree_reduce(
                op_name,
                block_reduced_oids.flatten().tolist(),
                result_block.grid_entry,
                result_block.grid_shape,
                False,
            )
        else:
            for result_grid_entry in result_grid.get_entry_iterator():
                block_reduced_oids_axis = []
                for sum_dim in range(self.grid.grid_shape[axis]):
                    grid_entry = list(result_grid_entry)
                    if keepdims:
                        grid_entry[axis] = sum_dim
                    else:
                        grid_entry = grid_entry[:axis] + [sum_dim] + grid_entry[axis:]
                    grid_entry = tuple(grid_entry)
                    block_reduced_oids_axis.append(block_reduced_oids[grid_entry])
                result_block: Block = result.blocks[result_grid_entry]
                result_block.oid = self.tree_reduce(
                    op_name,
                    block_reduced_oids_axis,
                    result_block.grid_entry,
                    result_block.grid_shape,
                    False,
                )
        return result

    # pylint: disable=arguments-differ
    def tree_reduce(
        self,
        op_name,
        blocks_or_oids,
        result_grid_entry,
        result_grid_shape,
        densify,
        *args,
    ):
        """
        Basic tree reduce imp.
        Schedules op on same node as left operand.
        :param op_name: The reduction op.
        :param blocks_or_oids: A list of type Block or a list of tuples.
        Tuples must be of the form
        (oid, grid_entry, grid_shape, transposed)
        :param result_grid_entry: The grid entry of the result block. This will be used
        to compute the final reduction step.
        :param result_grid_shape: The grid entry of the result block. This will be used
        to compute the final reduction step.
        :return: The oid of the result.
        """
        oid_list = blocks_or_oids
        if isinstance(blocks_or_oids[0], Block):
            oid_list = [
                (b.oid, b.grid_entry, b.grid_shape, b.transposed)
                for b in blocks_or_oids
            ]
        if len(oid_list) == 1:
            return oid_list[0][0]
        q = oid_list
        if densify:
            while len(q) > 1:
                a_oid, a_ge, a_gs, a_T = q.pop(0)
                b_oid, _, _, b_T = q.pop(0)
                ge, gs = (
                    (result_grid_entry, result_grid_shape)
                    if len(q) == 0
                    else (a_ge, a_gs)
                )
                c_oid = self.km.bop_reduce(
                    op_name,
                    a_oid,
                    b_oid,
                    a_T,
                    b_T,
                    syskwargs={
                        "grid_entry": ge,
                        "grid_shape": gs,
                    },
                )
                q.append((c_oid, ge, gs, False))
        else:
            while len(q) > 1:
                a_oid, a_ge, a_gs, a_T = q.pop(0)
                b_oid, _, _, b_T = q.pop(0)
                ge, gs = (
                    (result_grid_entry, result_grid_shape)
                    if len(q) == 0
                    else (a_ge, a_gs)
                )
                c_oid = self.km.sparse_bop_reduce(
                    op_name,
                    a_oid,
                    b_oid,
                    a_T,
                    b_T,
                    syskwargs={
                        "grid_entry": ge,
                        "grid_shape": gs,
                    },
                )
                q.append((c_oid, ge, gs, False))
        r_oid, r_ge, r_gs, _ = q.pop(0)
        assert r_ge == result_grid_entry
        assert r_gs == result_grid_shape
        return r_oid

    #################
    # Arithmetic
    #################

    @staticmethod
    def elementwise(op_name, a, b):
        if isinstance(a, SparseBlockArray):
            b = a.check_or_convert_other(b)
        elif isinstance(b, SparseBlockArray):
            a = b.check_or_convert_other(a)
        else:
            raise NotImplementedError()

        densify = array_utils.get_sparse_bop_densify(op_name, a.is_dense, b.is_dense)
        if a.shape == b.shape and a.block_shape == b.block_shape:
            return SparseBlockArray._fast_elementwise(op_name, a, b, densify)
        else:
            blocks_op = a.blocks.__getattribute__(f"__{op_name}__")
            if densify:
                result = BlockArray.from_blocks(
                    blocks_op(b.blocks),
                    result_shape=None,
                    km=a.km,
                )
            else:
                result = SparseBlockArray.from_blocks(
                    blocks_op(b.blocks),
                    result_shape=None,
                    km=a.km,
                )
            return result

    @staticmethod
    def _fast_elementwise(op_name, a, b, densify):
        # a, b have the same grid_shape and block_shape
        dtype = array_utils.get_bop_output_type(op_name, a.dtype, b.dtype)
        if densify:
            block_type = Block
        else:
            block_type = SparseBlock
        blocks = np.empty(shape=a.grid_shape, dtype=block_type)
        for grid_entry in a.grid.get_entry_iterator():
            a_block: BlockBase = a.blocks[grid_entry]
            b_block: BlockBase = b.blocks[grid_entry]
            blocks[grid_entry] = block_type(
                grid_entry,
                a_block.grid_shape,
                a_block.shape,
                dtype,
                transposed=False,
                km=a.km,
            )
            blocks[grid_entry].oid = a.km.sparse_bop(
                op_name,
                a_block.oid,
                b_block.oid,
                a_block.transposed,
                b_block.transposed,
                axes={},
                densify=densify,
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": a.grid.grid_shape,
                },
            )
        grid = ArrayGrid(a.shape, a.block_shape, dtype.__name__)
        if densify:
            return BlockArray(grid, a.km, blocks=blocks)
        else:
            return SparseBlockArray(grid, a.km, blocks=blocks)

    #################
    # Linear Algebra
    #################

    @staticmethod
    def tensordot(a, b, axes=2):
        if isinstance(axes, int):
            pass
        elif array_utils.is_array_like(axes):
            raise NotImplementedError("Non-integer axes is currently not supported.")
        else:
            raise TypeError(f"Unexpected axes type '{type(axes).__name__}'")

        if isinstance(a, SparseBlockArray):
            b = a.check_or_convert_other(b, compute_block_shape=True)
        elif isinstance(b, SparseBlockArray):
            a = b.check_or_convert_other(a, compute_block_shape=True)
        else:
            raise NotImplementedError()

        if array_utils.np_tensordot_param_test(a.shape, a.ndim, b.shape, b.ndim, axes):
            raise ValueError("shape-mismatch for sum")

        densify = array_utils.get_sparse_bop_densify(
            "tensordot", a.is_dense, b.is_dense
        )

        if axes > 0:
            a_axes = a.grid.grid_shape[:-axes]
            a_sum_axes = a.grid.grid_shape[-axes:]
            b_axes = b.grid.grid_shape[axes:]
            b_sum_axes = b.grid.grid_shape[:axes]
            assert a_sum_axes == b_sum_axes
            result_shape = tuple(a.shape[:-axes] + b.shape[axes:])
            result_block_shape = tuple(a.block_shape[:-axes] + b.block_shape[axes:])
        else:
            a_axes = a.grid.grid_shape
            b_axes = b.grid.grid_shape
            a_sum_axes = ()
            result_shape = tuple(a.shape + b.shape)
            result_block_shape = tuple(a.block_shape + b.block_shape)

        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=array_utils.get_bop_output_type(
                "tensordot", a.dtype, b.dtype
            ).__name__,
        )
        assert result_grid.grid_shape == tuple(a_axes + b_axes)
        if densify:
            result = BlockArray(result_grid, a.km)
        else:
            result = SparseBlockArray(result_grid, a.km)
        a_dims = list(itertools.product(*map(range, a_axes)))
        b_dims = list(itertools.product(*map(range, b_axes)))
        sum_dims = list(itertools.product(*map(range, a_sum_axes)))
        for i in a_dims:
            for j in b_dims:
                grid_entry = tuple(i + j)
                result_block: Block = result.blocks[grid_entry]
                sum_oids = []
                for k in sum_dims:
                    a_block: Block = a.blocks[tuple(i + k)]
                    b_block: Block = b.blocks[tuple(k + j)]
                    dot_grid_args = a._compute_tensordot_syskwargs(a_block, b_block)
                    dotted_oid = a.km.sparse_bop(
                        "tensordot",
                        a_block.oid,
                        b_block.oid,
                        a_block.transposed,
                        b_block.transposed,
                        axes=axes,
                        densify=densify,
                        syskwargs={
                            "grid_entry": dot_grid_args[0],
                            "grid_shape": dot_grid_args[1],
                        },
                    )
                    sum_oids.append(
                        (dotted_oid, dot_grid_args[0], dot_grid_args[1], False)
                    )
                result_block.oid = a.tree_reduce(
                    "sum",
                    sum_oids,
                    result_block.grid_entry,
                    result_block.grid_shape,
                    densify,
                )
                if not densify:
                    syskwargs = {
                        "grid_entry": result_block.grid_entry,
                        "grid_shape": result_block.grid_shape,
                    }
                    result_block._nnz = a.km.sparse_nnz(
                        result_block.oid, syskwargs=syskwargs
                    )
                    result_block._nbytes = a.km.sparse_nbytes(
                        result_block.oid, syskwargs=syskwargs
                    )
        return result

    def sdtp(self, *block_arrays: List[BlockArray]):
        """
        Perform a sampled tensor product among an arbitrary number of block arrays.
        """
        for i, ba in enumerate(block_arrays):
            assert len(ba.shape) == 1
            assert ba.shape[0] == self.shape[i]
            assert ba.block_shape[0] == self.block_shape[i]
        # Sparsity of result is same as self.
        result: SparseBlockArray = SparseBlockArray(self.grid, self.km)
        for grid_entry in self.grid.get_entry_iterator():
            dense_oids = [
                ba.blocks[grid_entry[i]].oid for i, ba in enumerate(block_arrays)
            ]
            result.blocks[grid_entry].oid = self.km.sdtp(
                self.blocks[grid_entry].oid,
                *dense_oids,
                syskwargs={"grid_entry": grid_entry, "grid_shape": result.grid_shape},
            )
        return result

    def sdtd(self, x: BlockArray, y: BlockArray, axes: int):
        """
        Perform a sampled dense-dense tensor dot between two tensors x and y.
        In addition to several compatibility-related constraints to ensure
        the operation is actually valid, we also require that the fill value of
        the sparse array is 0, and that the axes over which the tensor dot
        is being performed are not partitioned.
        The last constraint can be eliminated if we add a sampled element-wise kernel.
        """
        if array_utils.np_tensordot_param_test(x.shape, x.ndim, y.shape, y.ndim, axes):
            raise ValueError("shape-mismatch for sum")

        x_axes = x.grid.grid_shape[:-axes]
        x_sum_axes = x.grid.grid_shape[-axes:]
        y_axes = y.grid.grid_shape[axes:]
        y_sum_axes = y.grid.grid_shape[:axes]
        assert x_sum_axes == y_sum_axes
        result_shape = tuple(x.shape[:-axes] + y.shape[axes:])
        result_block_shape = tuple(x.block_shape[:-axes] + y.block_shape[axes:])
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=array_utils.get_bop_output_type(
                "tensordot", x.dtype, y.dtype
            ).__name__,
        )
        # Ensure dims over which we sum are not partitioned.
        # This is currently required for scalable sampled dense.
        assert np.sum(x_sum_axes) == np.sum(y_sum_axes) == axes

        assert result_grid.grid_shape == tuple(x_axes + y_axes)
        assert result_grid.grid_shape == self.grid_shape
        assert result_grid.block_shape == self.block_shape
        result: SparseBlockArray = SparseBlockArray(self.grid, self.km)
        x_dims = list(itertools.product(*map(range, x_axes)))
        y_dims = list(itertools.product(*map(range, y_axes)))
        sum_dims = tuple([0] * axes)
        for i in x_dims:
            for j in y_dims:
                grid_entry = tuple(i + j)
                x_block: Block = x.blocks[tuple(i + sum_dims)]
                y_block: Block = y.blocks[tuple(sum_dims + j)]
                result.blocks[grid_entry].oid = self.km.sdtd(
                    self.blocks[grid_entry].oid,
                    x_block.oid,
                    y_block.oid,
                    axes=axes,
                    syskwargs={
                        "grid_entry": grid_entry,
                        "grid_shape": result.grid_shape,
                    },
                )
        return result

    #################
    # Inequalities
    #################

    def __inequality__(self, op_name, other):
        other = self.check_or_convert_other(other)
        assert (
            other.shape == () or other.shape == self.shape
        ), "Currently supports comparison with scalars only."
        shape = array_utils.broadcast(self.shape, other.shape).shape
        block_shape = array_utils.broadcast_block_shape(
            self.shape, other.shape, self.block_shape
        )
        dtype = bool.__name__
        grid = ArrayGrid(shape, block_shape, dtype)
        result = SparseBlockArray(grid, self.km)
        for grid_entry in result.grid.get_entry_iterator():
            other_block: Block = other.blocks.item()
            result.blocks[grid_entry] = self.blocks[grid_entry].bop(
                op_name, other_block, args={}
            )
        return result
