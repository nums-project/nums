from typing import List
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockBase, Block, BlockArrayBase
from nums.core.array.blockarray import BlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import ArrayGrid
import numpy as np
import itertools
import warnings
import sparse


class SparseBlock(BlockBase):
    def __init__(
        self,
        grid_entry,
        grid_shape,
        shape,
        dtype,
        transposed,
        km: KernelManager,
        fill_value=0,
        id=None,
        index_dtype=np.int64,
    ):
        super().__init__(grid_entry, grid_shape, shape, dtype, transposed, km, id)
        self.fill_value = fill_value
        self.index_dtype = index_dtype
        self.oid = None
        self._nnz: object = None
        self._nbytes: object = None

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

    def transpose(self, defer=False, redistribute=False):
        grid_entryT = tuple(reversed(self.grid_entry))
        grid_shapeT = tuple(reversed(self.grid_shape))
        blockT = SparseBlock(
            grid_entry=grid_entryT,
            grid_shape=grid_shapeT,
            shape=tuple(reversed(self.shape)),
            dtype=self.dtype,
            transposed=not self.transposed,
            km=self.km,
        )
        blockT.oid = self.oid
        if not defer:
            blockT.transposed = False
            if redistribute:
                syskwargs = {"grid_entry": grid_entryT, "grid_shape": grid_shapeT}
            else:
                syskwargs = {
                    "grid_entry": self.grid_entry,
                    "grid_shape": self.grid_shape,
                }
            blockT.oid = self.km.transpose(self.oid, syskwargs=syskwargs)
        return blockT

    def map_uop(self, op_name, args=None, kwargs=None, device=None):
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
            op_name, self.oid, args, kwargs, syskwargs=syskwargs
        )
        block._nnz = self.km.sparse_nnz(block.oid, syskwargs=syskwargs)
        block._nbytes = self.km.sparse_nbytes(block.oid, syskwargs=syskwargs)
        block.fill_value = np.__getattribute__(op_name)(self.fill_value)
        return block

    def block_from_scalar(self, other):
        assert array_utils.is_scalar(other)
        block = SparseBlock(
            self.grid_entry,
            self.grid_shape,
            (1,),
            self.dtype,
            False,
            self.km,
            fill_value=other,
        )
        block.oid = self.km.sparse_block_from_scalar(
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
        fill_value = array_utils.get_bop_fill_value(
            op_name, block1.fill_value, block2.fill_value
        )
        # TODO: what happens when different index_dtype?
        block = SparseBlock(
            grid_entry=result_grid_entry,
            grid_shape=result_grid_shape,
            shape=result_shape,
            dtype=dtype,
            transposed=False,
            fill_value=fill_value,
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

        densify = array_utils.get_sparse_bop_return_type(
            op_name, a.fill_value, b.fill_value
        )
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

    # TODO: densify when fill_value != 0
    def tensordot(self, other, axes):
        assert self.fill_value == 0
        if not other.is_dense:
            assert other.fill_value == 0
        return self.binary_op("tensordot", self, other, args={"axes": axes})


class SparseBlockArray(BlockArrayBase):
    def __init__(
        self,
        grid: ArrayGrid,
        km: KernelManager,
        fill_value=0,
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
                    fill_value,
                )
        self.fill_value = fill_value
        self._nnz = -1
        self._nbytes = -1

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
        return SparseBlockArray.create(
            "empty", shape, block_shape, dtype, km, fill_value=0
        )

    @classmethod
    def create(
        cls, create_op_name, shape, block_shape, dtype, km: KernelManager, fill_value=0
    ):
        grid = ArrayGrid(shape=shape, block_shape=block_shape, dtype=dtype.__name__)
        grid_meta = grid.to_meta()
        arr = SparseBlockArray(grid, km, fill_value)
        for grid_entry in grid.get_entry_iterator():
            arr.blocks[grid_entry].oid = km.new_sparse_block(
                create_op_name,
                grid_entry,
                grid_meta,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return arr

    @classmethod
    def from_np(cls, arr, block_shape, copy, km, fill_value=0):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = SparseBlockArray(grid, km, fill_value)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = arr[grid_slice]
            if copy:
                block = np.copy(block)
            # TODO: generalize for different kernels
            block = sparse.COO.from_numpy(block, fill_value)
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
        if not array_utils.is_scalar(val):
            raise ValueError("%s is not a scalar." % val)
        return SparseBlockArray.from_np(
            np.array(val), (), copy=False, km=km, fill_value=val
        )

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, km, fill_value):
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
        result = SparseBlockArray(result_grid, km, fill_value)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, SparseBlock):
                block: SparseBlock = arr
            else:
                block: SparseBlock = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, fill_value=0):
        assert (
            ba.shape != ()
        ), "from_ba does not support scalar BlockArray. Use from_scalar."
        grid = ArrayGrid(ba.shape, ba.block_shape, ba.dtype.__name__)
        sba = SparseBlockArray(grid, ba.km, fill_value)
        for grid_entry in grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            sblock: SparseBlock = sba.blocks[grid_entry]
            syskwargs = {
                "grid_entry": grid_entry,
                "grid_shape": block.grid_shape,
            }
            sblock.oid = sba.km.dense_to_sparse(
                block.oid,
                fill_value,
                syskwargs=syskwargs,
            )
            sblock._nnz = sba.km.sparse_nnz(sblock.oid, syskwargs=syskwargs)
            sblock._nbytes = sba.km.sparse_nbytes(sblock.oid, syskwargs=syskwargs)
        sba.fill_value = fill_value
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
        rarr_copy = SparseBlockArray(grid_copy, self.km, self.fill_value)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def astype(self, dtype):
        grid = ArrayGrid(self.shape, self.block_shape, dtype.__name__)
        result = BlockArray(grid, self.km)
        for grid_entry in result.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].astype(dtype)
        return result

    def transpose(self, defer=False, redistribute=False):
        if defer and redistribute:
            warnings.warn("defer is True, redistribute=True will be ignored.")
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        gridT = ArrayGrid.from_meta(metaT)
        rarrT = SparseBlockArray(gridT, self.km, self.fill_value)
        rarrT.blocks = np.copy(self.blocks.T)
        for grid_entry in rarrT.grid.get_entry_iterator():
            rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose(
                defer, redistribute
            )
        return rarrT

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
        result = self.copy()
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = self.blocks[grid_entry].ufunc(op_name)
        func = np.__getattribute__(op_name)
        result.fill_value = func(self.fill_value)
        result._nnz = -1
        result._nbytes = -1
        return result

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

        densify = array_utils.get_sparse_bop_return_type(
            op_name,
            a.fill_value,
            b.fill_value,
        )
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
                fill_value = array_utils.get_bop_fill_value(
                    op_name, a.fill_value, b.fill_value
                )
                result = SparseBlockArray.from_blocks(
                    blocks_op(b.blocks),
                    result_shape=None,
                    fill_value=fill_value,
                    km=a.km,
                )
            return result

    @staticmethod
    def _fast_elementwise(op_name, a, b, densify):
        # a, b have the same grid_shape and block_shape
        dtype = array_utils.get_bop_output_type(op_name, a.dtype, b.dtype)
        if densify:
            block_type = Block
            fill_value = None
        else:
            block_type = SparseBlock
            fill_value = array_utils.get_bop_fill_value(
                op_name, a.fill_value, b.fill_value
            )
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
            return SparseBlockArray(grid, a.km, fill_value, blocks=blocks)

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

        # PyData/Sparse only works with fill_value == 0
        # TODO: densify when fill_value != 0
        if not (a.is_dense or b.is_dense):
            assert (
                a.fill_value == 0 and b.fill_value == 0
            ), "Sparse-sparse tensordot with non-zero fill value is not supported."

        if array_utils.np_tensordot_param_test(a.shape, a.ndim, b.shape, b.ndim, axes):
            raise ValueError("shape-mismatch for sum")

        densify = array_utils.get_sparse_bop_return_type(
            "tensordot",
            a.fill_value,
            b.fill_value,
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
            result = SparseBlockArray(result_grid, a.km, a.fill_value)
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
                    "sum", sum_oids, result_block.grid_entry, result_block.grid_shape
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
        assert np.allclose(self.fill_value, 0)
        for i, ba in enumerate(block_arrays):
            assert len(ba.shape) == 1
            assert ba.shape[0] == self.shape[i]
            assert ba.block_shape[0] == self.block_shape[i]
        # Sparsity of result is same as self.
        result: SparseBlockArray = SparseBlockArray(self.grid, self.km, self.fill_value)
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
        assert np.allclose(self.fill_value, 0)

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
        result: SparseBlockArray = SparseBlockArray(self.grid, self.km, self.fill_value)
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
        fill_value = array_utils.get_bop_fill_value(
            op_name, self.fill_value, other.fill_value
        )
        result = SparseBlockArray(grid, self.km, fill_value)
        for grid_entry in result.grid.get_entry_iterator():
            other_block: Block = other.blocks.item()
            result.blocks[grid_entry] = self.blocks[grid_entry].bop(
                op_name, other_block, args={}
            )
        return result
