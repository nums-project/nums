from typing import Optional, List
from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.blockarray import BlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import ArrayGrid
from nums.core.settings import np_ufunc_map
import numpy as np
import itertools


class SparseBlock(Block):
    def __init__(
        self,
        grid_entry,
        grid_shape,
        shape,
        dtype,
        transposed,
        km: KernelManager,
        id=None,
    ):
        super().__init__(grid_entry, grid_shape, shape, dtype, transposed, km, id)
        # Keep as ObjectRefs to avoid blocking on creation?
        self._nbytes: object = None
        self._nnz: object = None

    def __getattr__(self, item):
        if item == "nbytes":
            return self._km.get(self._nbytes)
        elif item == "nnz":
            return self._km.get(self._nnz)
        else:
            super().__getattr__(item)

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
            self._km,
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
            km=self._km,
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
            blockT.oid = self._km.transpose(self.oid, syskwargs=syskwargs)
        return blockT

    def ufunc(self, op_name, device=None):
        return self.uop_map(op_name, device=device)

    def uop_map(self, op_name, args=None, kwargs=None, device=None):
        block = self.copy()
        block.dtype = array_utils.get_uop_output_type(op_name, self.dtype)
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if device is None:
            syskwargs = {"grid_entry": block.grid_entry, "grid_shape": block.grid_shape}
        else:
            syskwargs = {"device": device}
        syskwargs["options"] = {"num_returns": 3}
        block._device = device
        block.oid, block._nbytes, block._nnz = self._km.sparse_map_uop(
            op_name, self.oid, args, kwargs, syskwargs=syskwargs
        )
        return block

    @staticmethod
    def init_block(op, block1, block2, args, device=None):
        (
            result_grid_entry,
            result_grid_shape,
            result_shape,
            dtype,
        ) = SparseBlock.block_meta(op, block1, block2, args)
        # TODO: figure out fill_value semantics
        assert block1.fill_value == block2.fill_value
        block = SparseBlock(
            grid_entry=result_grid_entry,
            grid_shape=result_grid_shape,
            shape=result_shape,
            dtype=dtype,
            transposed=False,
            km=block1._km,
        )
        block._device = device
        return block

    def bop(self, op, other, args: dict, densify, device=None):
        # TODO: figure out sparse result semantics
        if isinstance(other, SparseBlock):
            block = SparseBlock.init_block(op, self, other, args, device)
        else:
            block = Block.init_block(op, self, other, args, device)
        if device is None:
            syskwargs = {
                "grid_entry": block.grid_entry,
                "grid_shape": block.grid_shape,
            }
        else:
            syskwargs = {"device": device}
        syskwargs["options"] = {"num_returns": 3}
        block.oid = self._km.sparse_bop(
            op,
            self.oid,
            other.oid,
            self.transposed,
            other.transposed,
            axes=args.get("axes"),
            densify=densify,
            syskwargs=syskwargs,
        )
        return block

    def tensordot(self, other, axes):
        return self.bop("tensordot", other, args={"axes": axes})

    def __add__(self, other):
        return self.bop("add", other, args={})

    def __mul__(self, other):
        return self.bop("mul", other, args={})


class SparseBlockArray(BlockArray):
    def __init__(
        self,
        grid: ArrayGrid,
        km: KernelManager,
        fill_value=0,
        blocks: np.ndarray = None,
    ):
        self.grid = grid
        self.km = km
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.grid_shape = self.grid.grid_shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        self.dtype = self.grid.dtype
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
                    self.km,
                )
        self.fill_value = fill_value
        self._nbytes: object = None
        self._nnz: object = None

    def __getattr__(self, item):
        if item == "nbytes":
            return self.km.get(self._nbytes)
        elif item == "nnz":
            return self.km.get(self._nnz)
        else:
            super().__getattr__(item)

    @classmethod
    def from_scalar(cls, val, km, fill_value=None):
        if fill_value is None:
            fill_value = val
        ba = BlockArray.from_np(np.array(val), block_shape=(), copy=False, km=km)
        return SparseBlockArray.from_ba(ba, fill_value)

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, km, fill_value):
        sample_idx = tuple(0 for dim in arr.shape)
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
        grid = ArrayGrid(ba.shape, ba.block_shape, ba.dtype.__name__)
        sba = SparseBlockArray(grid, ba.km, fill_value)
        nbytes_oids = []
        nnz_oids = []
        for grid_entry in grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            sba.blocks[grid_entry].oid, nb, nz = sba.km.dense_to_sparse(
                block.oid,
                fill_value,
                syskwargs={
                    "grid_entry": block.grid_entry,
                    "grid_shape": block.grid_shape,
                    "options": {"num_returns": 3},
                },
            )
            sba.blocks[grid_entry]._nbytes = nb
            sba.blocks[grid_entry]._nnz = nz
            nbytes_oids.append(nb)
            nnz_oids.append(nz)
        sba.fill_value = fill_value
        device_0 = sba.km.devices()[0]
        sba._nbytes = sba.km.sum_reduce(*nbytes_oids, syskwargs={"device": device_0})
        sba._nnz = sba.km.sum_reduce(*nnz_oids, syskwargs={"device": device_0})
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

    def copy(self):
        grid_copy = self.grid.from_meta(self.grid.to_meta())
        rarr_copy = SparseBlockArray(grid_copy, self.km, self.fill_value)
        for grid_entry in grid_copy.get_entry_iterator():
            rarr_copy.blocks[grid_entry] = self.blocks[grid_entry].copy()
        return rarr_copy

    def ufunc(self, op_name):
        result = self.copy()
        nbytes_oids = []
        nnz_oids = []
        for grid_entry in self.grid.get_entry_iterator():
            result.blocks[grid_entry] = block = self.blocks[grid_entry].ufunc(op_name)
            nbytes_oids.append(block._nbytes)
            nnz_oids.append(block._nnz)
        op_name = np_ufunc_map.get(op_name, op_name)
        func = np.__getattribute__(op_name)
        result.fill_value = func(self.fill_value)
        device_0 = result.km.devices()[0]
        result._nbytes = result.km.sum_reduce(
            *nbytes_oids, syskwargs={"device": device_0}
        )
        result._nnz = result.km.sum_reduce(*nnz_oids, syskwargs={"device": device_0})
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
        return BlockArray.from_np(np_array, block_shape, False, km)

    def check_or_convert_other(self, other, compute_block_shape=False):
        block_shape = None if compute_block_shape else self.block_shape
        return SparseBlockArray.to_block_array(other, self.km, block_shape=block_shape)

    def _fast_elementwise(self, op_name, other, densify):
        dtype = array_utils.get_bop_output_type(op_name, self.dtype, other.dtype)
        # Schedule the op first.
        if densify:
            blocks = np.empty(shape=self.grid.grid_shape, dtype=Block)
        else:
            blocks = np.empty(shape=self.grid.grid_shape, dtype=SparseBlock)
        nbytes_oids = []
        nnz_oids = []
        for grid_entry in self.grid.get_entry_iterator():
            self_block: SparseBlock = self.blocks[grid_entry]
            other_block = other.blocks[grid_entry]
            blocks[grid_entry] = block = Block(
                grid_entry=grid_entry,
                grid_shape=self_block.grid_shape,
                shape=self_block.shape,
                dtype=dtype,
                transposed=False,
                km=self.km,
            )
            block.oid, nb, nz = self.km.sparse_bop(
                op_name,
                self_block.oid,
                other_block.oid,
                self_block.transposed,
                other_block.transposed,
                axes={},
                densify=densify,
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": self.grid.grid_shape,
                    "options": {"num_returns": 3},
                },
            )
            if not densify:
                block._nbytes = nb
                block._nnz = nz
                nbytes_oids.append(nb)
                nnz_oids.append(nz)
        grid = ArrayGrid(self.shape, self.block_shape, dtype.__name__)
        if densify:
            return BlockArray(grid, self.km, blocks=blocks)
        else:
            op_name = np_ufunc_map.get(op_name, op_name)
            ufunc = np.__getattribute__(op_name)
            # FIXME: other may be dense
            fill_value = self.fill_value
            if isinstance(other, SparseBlockArray):
                fill_value = ufunc(self.fill_value, other.fill_value)
            result = SparseBlockArray(grid, self.km, fill_value, blocks=blocks)
            device_0 = result.km.devices()[0]
            result._nbytes = result.km.sum_reduce(
                *nbytes_oids, syskwargs={"device": device_0}
            )
            result._nnz = result.km.sum_reduce(
                *nnz_oids, syskwargs={"device": device_0}
            )
            return result

    @staticmethod
    def ba_meta(a):
        meta = {}
        if isinstance(a, SparseBlockArray):
            meta["type"] = "sparse"
            meta["fill_value"] = a.fill_value
        elif isinstance(a, BlockArray):
            meta["type"] = "dense"
        return meta

    def todense(self) -> BlockArray:
        return self.to_ba()

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
            dense_oids = [ba.blocks[grid_entry[i]].oid for i, ba in enumerate(block_arrays)]
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
        this_dims = list(itertools.product(*map(range, x_axes)))
        other_dims = list(itertools.product(*map(range, y_axes)))
        sum_dims = tuple([0] * axes)
        for i in this_dims:
            for j in other_dims:
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

    def __elementwise__(self, op_name, other):
        other = self.check_or_convert_other(other)  # other is dense BlockArray
        densify = self.km.get(
            self.km.sparse_bop_densify(
                op_name,
                SparseBlockArray.ba_meta(self),
                SparseBlockArray.ba_meta(other),
                syskwargs={"device": self.km.devices()[0]},
            )
        )
        if self.shape == other.shape and self.block_shape == other.block_shape:
            return self._fast_elementwise(op_name, other, densify)
        # Add densify dunder method signature?
        blocks_op = self.blocks.__getattribute__("__%s__" % op_name)
        if densify:
            result = BlockArray.from_blocks(
                blocks_op(other.blocks),
                result_shape=None,
                km=self.km,
            )
        else:
            op_name = np_ufunc_map.get(op_name, op_name)
            ufunc = np.__getattribute__(op_name)
            # FIXME: other may be dense
            fill_value = ufunc(self.fill_value, other.fill_value)
            result = SparseBlockArray.from_blocks(
                blocks_op(other.blocks),
                result_shape=None,
                fill_value=fill_value,
                km=self.km,
            )
        return result

    def tensordot(self, other, axes=2):
        if isinstance(axes, int):
            pass
        elif array_utils.is_array_like(axes):
            raise NotImplementedError("Non-integer axes is currently not supported.")
        else:
            raise TypeError(f"Unexpected axes type '{type(axes).__name__}'")

        other = self.check_or_convert_other(other, compute_block_shape=True)

        if array_utils.np_tensordot_param_test(
            self.shape, self.ndim, other.shape, other.ndim, axes
        ):
            raise ValueError("shape-mismatch for sum")
        densify = self.km.get(
            self.km.sparse_bop_densify(
                "tensordot",
                SparseBlockArray.ba_meta(self),
                SparseBlockArray.ba_meta(other),
                syskwargs={"device": self.km.devices()[0]},
            )
        )
        this_axes = self.grid.grid_shape[:-axes]
        this_sum_axes = self.grid.grid_shape[-axes:]
        other_axes = other.grid.grid_shape[axes:]
        other_sum_axes = other.grid.grid_shape[:axes]
        assert this_sum_axes == other_sum_axes
        result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
        result_block_shape = tuple(self.block_shape[:-axes] + other.block_shape[axes:])
        result_grid = ArrayGrid(
            shape=result_shape,
            block_shape=result_block_shape,
            dtype=array_utils.get_bop_output_type(
                "tensordot", self.dtype, other.dtype
            ).__name__,
        )
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result = BlockArray(result_grid, self.km)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                grid_entry = tuple(i + j)
                result_block: Block = result.blocks[grid_entry]
                sum_oids = []
                for k in sum_dims:
                    self_block: Block = self.blocks[tuple(i + k)]
                    other_block: Block = other.blocks[tuple(k + j)]
                    dot_grid_args = self._compute_tensordot_syskwargs(
                        self_block, other_block
                    )
                    dotted_oid = self.km.sparse_bop(
                        "tensordot",
                        self_block.oid,
                        other_block.oid,
                        self_block.transposed,
                        other_block.transposed,
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
                result_block.oid = self._tree_reduce(
                    "sum", sum_oids, result_block.grid_entry, result_block.grid_shape
                )
        return result

    def __add__(self, other):
        return self.__elementwise__("add", other)

    def __radd__(self, other):
        pass

    def __sub__(self, other):
        return self.__elementwise__("sub", other)

    def __rsub__(self, other):
        pass

    def __mul__(self, other):
        return self.__elementwise__("mul", other)

    def __rmul__(self, other):
        pass
