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


# pylint: disable = protected-access
# pylint: disable=too-many-lines

import warnings

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.grid.grid import ArrayGrid


class BlockBase:
    block_id_counter = -1

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
        self.km = km
        self.grid_entry: tuple = grid_entry
        self.grid_shape: tuple = grid_shape
        self.oid: np.object = None
        self.shape: tuple = shape
        self.dtype = dtype
        self.ndim = len(self.shape)
        self.transposed = transposed
        self.id = id
        if self.id is None:
            Block.block_id_counter += 1
            self.id = Block.block_id_counter
        # Set if a device id was used to compute this block.
        self._device = None
        self.fill_value = None

    @property
    def is_dense(self):
        return self.fill_value is None

    @property
    def nbytes(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def size(self):
        return np.product(self.shape)

    def copy(self):
        raise NotImplementedError()

    def get(self):
        return self.km.get(self.oid)

    def astype(self, dtype):
        block = self.copy()
        block.dtype = dtype
        block.oid = self.km.astype(
            self.oid,
            dtype.__name__,
            syskwargs={"grid_entry": block.grid_entry, "grid_shape": block.grid_shape},
        )
        return block

    def true_grid_entry(self):
        if self.transposed:
            return tuple(reversed(self.grid_entry))
        return self.grid_entry

    def true_grid_shape(self):
        if self.transposed:
            return tuple(reversed(self.grid_shape))
        return self.grid_shape

    def transpose(self, defer=False, redistribute=False):
        raise NotImplementedError()

    def device(self):
        if self._device is not None:
            return self._device
        return self.km.device_grid.get_device(
            self.true_grid_entry(), self.true_grid_shape()
        )

    def swapaxes(self, axis1, axis2):
        block = self.copy()
        grid_entry = list(block.grid_entry)
        grid_shape = list(block.grid_shape)
        shape = list(block.shape)

        grid_entry[axis1], grid_entry[axis2] = grid_entry[axis2], grid_entry[axis1]
        grid_shape[axis1], grid_shape[axis2] = grid_shape[axis2], grid_shape[axis1]
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]

        block.grid_entry = tuple(grid_entry)
        block.grid_shape = tuple(grid_shape)
        block.shape = tuple(shape)

        block.oid = self.km.swapaxes(
            block.oid,
            axis1,
            axis2,
            syskwargs={"grid_entry": block.grid_entry, "grid_shape": block.grid_shape},
        )
        return block

    def map_uop(self, op_name, args=None, kwargs=None, device=None):
        raise NotImplementedError()

    def ufunc(self, op_name, device=None):
        return self.map_uop(op_name, device=device)

    def block_from_scalar(self, other):
        raise NotImplementedError()

    def conjugate(self):
        return self.ufunc("conjugate")

    def sqrt(self):
        return self.ufunc("sqrt")

    @staticmethod
    def block_meta(op, block1, block2, args):
        if op == "tensordot":
            (
                result_shape,
                result_grid_entry,
                result_grid_shape,
            ) = array_utils.get_tensordot_block_params(
                block1.shape,
                block1.grid_entry,
                block1.grid_shape,
                block2.shape,
                block2.grid_entry,
                block2.grid_shape,
                args["axes"],
            )
        else:
            (
                result_shape,
                result_grid_entry,
                result_grid_shape,
            ) = array_utils.get_elementwise_bop_block_params(
                block1.shape,
                block1.grid_entry,
                block1.grid_shape,
                block2.shape,
                block2.grid_entry,
                block2.grid_shape,
            )

        dtype = array_utils.get_bop_output_type(op, block1.dtype, block2.dtype)
        return result_grid_entry, result_grid_shape, result_shape, dtype

    @staticmethod
    def init_block(op_name, block1, block2, args, device=None):
        raise NotImplementedError()

    def _check_bop_implemented(self, other):
        raise NotImplementedError()

    @staticmethod
    def binary_op(op_name, a, b, args: dict, device=None):
        raise NotImplementedError()

    def bop(self, op_name, other, args: dict, device=None, **kwargs):
        raise NotImplementedError()

    def tensordot(self, other, axes):
        raise NotImplementedError()

    def __add__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("add", self, other, args={})

    def __radd__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("add", other, self, args={})

    def __sub__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("sub", self, other, args={})

    def __rsub__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("sub", other, self, args={})

    def __mul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("mul", self, other, args={})

    def __rmul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("mul", other, self, args={})

    def __truediv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("truediv", self, other, args={})

    def __rtruediv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("trudiv", other, self, args={})

    def __pow__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("pow", self, other, args={})

    def __rpow__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("pow", other, self, args={})

    def __matmul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.tensordot(other, axes=1)

    def __ge__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("ge", self, other, args={})

    def __gt__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("gt", self, other, args={})

    def __le__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("le", self, other, args={})

    def __lt__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("lt", self, other, args={})

    def __eq__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("eq", self, other, args={})

    def __ne__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.binary_op("ne", self, other, args={})

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __imatmul__ = __matmul__
    __itruediv__ = __truediv__
    __ipow__ = __pow__


class Block(BlockBase):
    # pylint: disable=redefined-builtin, global-statement

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

    @property
    def nbytes(self):
        return np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def __repr__(self):
        return "Block(" + str(self.oid) + ")"

    def copy(self, shallow=True):
        assert shallow, "Only shallow copies are currently supported."
        block = Block(
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
        # If defer is True, this operation does not modify the remote object.
        # If defer is True and redistribute is False,
        # this operation does not move the remote object.
        grid_entryT = tuple(reversed(self.grid_entry))
        grid_shapeT = tuple(reversed(self.grid_shape))
        blockT = Block(
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
        # This retains transpose.
        block = self.copy()
        block.dtype = array_utils.get_uop_output_type(op_name, self.dtype)
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if device is None:
            syskwargs = {"grid_entry": block.grid_entry, "grid_shape": block.grid_shape}
        else:
            syskwargs = {"device": device}
        block._device = device
        block.oid = self.km.map_uop(
            op_name, self.oid, args, kwargs, syskwargs=syskwargs
        )
        return block

    def block_from_scalar(self, other):
        # Assume other is numeric.
        # This only occurs during some numpy operations (e.g. np.mean),
        # where a literal is used in the operation.
        assert array_utils.is_scalar(other)
        block = Block(
            self.grid_entry,
            self.grid_shape,
            (1,),
            self.dtype,
            False,
            self.km,
        )
        # We pass syskwargs here for correct node placement for `other`,
        # which should be local to self.
        block.oid = self.km.put(
            np.array(other, dtype=self.dtype),
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
        block = Block(
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
        if isinstance(other, Block) or array_utils.is_scalar(other):
            return True
        return False

    @staticmethod
    def binary_op(op_name, a, b, args: dict, device=None):
        if isinstance(a, Block) and array_utils.is_scalar(b):
            b = a.block_from_scalar(b)
        elif isinstance(b, Block) and array_utils.is_scalar(a):
            a = b.block_from_scalar(a)
        if not isinstance(a, Block) or not isinstance(b, Block):
            raise NotImplementedError()

        block: Block = a.init_block(op_name, a, b, args, device)
        if device is None:
            syskwargs = {"grid_entry": block.grid_entry, "grid_shape": block.grid_shape}
        else:
            syskwargs = {"device": device}
        block.oid = a.km.bop(
            op_name,
            a.oid,
            b.oid,
            a.transposed,
            b.transposed,
            axes=args.get("axes"),
            syskwargs=syskwargs,
        )
        return block

    def bop(self, op_name, other, args: dict, device=None):
        try:
            return self.binary_op(op_name, self, other, args, device)
        except NotImplementedError:
            return other.binary_op(op_name, self, other, args, device)

    def tensordot(self, other, axes):
        try:
            return self.binary_op("tensordot", self, other, args={"axes": axes})
        except NotImplementedError:
            return other.binary_op("tensordot", self, other, args={"axes": axes})


class BlockArrayBase:
    def __init__(self, grid: ArrayGrid, km: KernelManager, blocks: np.ndarray = None):
        self.grid = grid
        self.km = km
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.grid_shape = self.grid.grid_shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        self.dtype = self.grid.dtype
        self.blocks = blocks
        self.fill_value = None

    @property
    def is_dense(self):
        return self.fill_value is None

    def __repr__(self):
        return "BlockArray(" + str(self.blocks) + ")"

    def __getattr__(self, item):
        if item == "__array_priority__" or item == "__array_struct__":
            # This is triggered by a numpy array on the LHS.
            raise TypeError(
                "Unexpected conversion attempt from BlockArrayBase to ndarray."
            )
        elif item == "ndim":
            return len(self.shape)
        elif item == "T":
            return self.transpose()
        else:
            raise NotImplementedError(item)

    def get(self) -> np.ndarray:
        result: np.ndarray = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        block_shape: np.ndarray = np.array(self.grid.block_shape, dtype=np.int64)
        arrays: list = self.km.get(
            [
                self.blocks[grid_entry].oid
                for grid_entry in self.grid.get_entry_iterator()
            ]
        )
        for block_index, grid_entry in enumerate(self.grid.get_entry_iterator()):
            start = block_shape * grid_entry
            entry_shape = np.array(
                self.grid.get_block_shape(grid_entry), dtype=np.int64
            )
            end = start + entry_shape
            slices = tuple(map(lambda item: slice(*item), zip(*(start, end))))
            block: BlockBase = self.blocks[grid_entry]
            arr: np.ndarray = arrays[block_index]
            if block.transposed:
                arr = arr.T
            result[slices] = arr.reshape(block.shape)
        return result

    def touch(self):
        """
        "Touch" an array. This is an efficient distributed "wait" operation.
        """
        oids = []
        for grid_entry in self.grid.get_entry_iterator():
            block: BlockBase = self.blocks[grid_entry]
            oids.append(
                self.km.touch(
                    block.oid,
                    syskwargs={
                        "grid_entry": block.grid_entry,
                        "grid_shape": block.grid_shape,
                    },
                )
            )
        self.km.get(oids)
        return self

    def copy(self):
        raise NotImplementedError()

    def astype(self, dtype):
        raise NotImplementedError()

    def flattened_oids(self):
        oids = []
        for grid_entry in self.grid.get_entry_iterator():
            oid = self.blocks[grid_entry].oid
            oids.append(oid)
        return oids

    @classmethod
    def empty(cls, shape, block_shape, dtype, km: KernelManager):
        raise NotImplementedError()

    @staticmethod
    def to_block_array(obj, km: KernelManager, block_shape=None):
        raise NotImplementedError()

    def transpose(self, defer=False, redistribute=False):
        raise NotImplementedError()

    def broadcast_to(self, shape):
        b = array_utils.broadcast(self.shape, shape)
        result_block_shape = array_utils.broadcast_block_shape(
            self.shape, shape, self.block_shape
        )
        result: BlockArrayBase = BlockArrayBase(
            ArrayGrid(b.shape, result_block_shape, self.grid.dtype.__name__), self.km
        )
        extras = []
        # Below taken directly from _broadcast_to in numpy's stride_tricks.py.
        it = np.nditer(
            (self.blocks,),
            flags=["multi_index", "refs_ok", "zerosize_ok"] + extras,
            op_flags=["readonly"],
            itershape=result.grid.grid_shape,
            order="C",
        )
        with it:
            # never really has writebackifcopy semantics
            broadcast = it.itviews[0]
        result.blocks = broadcast
        return result

    def reshape(self, *shape, **kwargs):
        block_shape = kwargs.get("block_shape", None)
        if array_utils.is_int(shape):
            shape = (shape,)
        elif len(shape) == 0:
            shape = self.shape
        elif isinstance(shape[0], (tuple, list)):
            assert len(shape) == 1
            shape = shape[0]
        else:
            assert all(np.issubdtype(type(n), int) for n in shape)
        shape = Reshape.compute_shape(self.shape, shape)
        if block_shape is None:
            if shape == self.shape:
                # This is a noop.
                block_shape = self.block_shape
            else:
                block_shape = self.km.get_block_shape(shape, self.dtype)
        return Reshape()(self, shape, block_shape)

    def _preprocess_subscript(self, item):
        if not isinstance(item, tuple):
            ss = (item,)
        else:
            ss = item
        # We need to fetch any block arrays.
        tmp = []
        for entry in ss:
            if isinstance(entry, BlockArrayBase):
                val = entry.get()
            else:
                val = entry
            if isinstance(val, list):
                val = np.array(val)
            if isinstance(val, np.ndarray):
                # If this is a Boolean mask, convert it to integers.
                if array_utils.is_bool(val.dtype, type_test=True):
                    val = np.arange(len(val))[val]
                if val.shape == ():
                    val = val.item()
            tmp.append(val)
        ss = tuple(tmp)
        is_handled_advanced = False
        array_encountered = False
        axis = None

        # Check if this is a supported advanced indexing operation.
        for i, entry in enumerate(ss):
            if isinstance(entry, slice) and entry.start is None and entry.stop is None:
                continue
            elif array_utils.is_int(entry) or array_utils.is_uint(entry):
                continue
            elif array_utils.is_array_like(entry):
                if array_encountered:
                    raise NotImplementedError(
                        "Advanced indexing is only supported along a single axis."
                    )
                is_handled_advanced = True
                array_encountered = True
                axis = i
                if not (np.all(0 <= entry) and np.all(entry < self.shape[axis])):
                    raise IndexError(
                        "Advanced indexing array along axis %s is out of bounds." % axis
                    )
            else:
                if array_encountered:
                    raise NotImplementedError(
                        "Advanced indexing is only supported "
                        "with full slices and integers along other axes."
                    )
                is_handled_advanced = False
                break

        return ss, is_handled_advanced, axis

    def __getitem__(self, item):
        ss, is_handled_advanced, axis = self._preprocess_subscript(item)

        if is_handled_advanced:
            # Treat this as a shuffle.
            return self._advanced_single_array_select(ss, axis=axis)

        # This is to deal with circular imports. Little overhead since this happens once per call.
        # However, would be better to rearrange modules in the future.
        from nums.core.array.view import ArrayView

        av: ArrayView = ArrayView.from_block_array(self)
        # TODO (hme): We don't have to create, but do so for now until we need to optimize.
        return av[ss].create()

    def _advanced_single_array_select(self, ss: tuple, axis: int = 0):
        # Create output array along the axis of the selection operation.
        # We don't allocate zeros for output array. Instead, we let the update kernel
        # create the initial set of zeros to save some memory.
        array = ss[axis]
        assert len(array.shape) == 1

        # TODO: We may encounter block shape incompatability due to this.
        block_size = self.block_shape[axis]
        self.km.update_block_shape_map(array.shape[0], block_size)

        dst_axis = None
        shape = []
        block_shape = []
        for i in range(len(self.shape)):
            if i == axis:
                dst_axis = len(shape)
                shape.append(array.shape[0])
                block_shape.append(block_size)
            elif i < len(ss):
                if isinstance(ss[i], slice):
                    shape.append(self.shape[i])
                    block_shape.append(self.block_shape[i])
                else:
                    # It's an index. We drop the indices.
                    continue
            else:
                shape.append(self.shape[i])
                block_shape.append(self.block_shape[i])

        dst_arr = type(self)(
            ArrayGrid(
                shape=tuple(shape),
                block_shape=tuple(block_shape),
                dtype=self.dtype.__name__,
            ),
            km=self.km,
        )

        src_arr = self
        np_ss = ss
        ss = self.km.put(
            ss,
            syskwargs={
                "grid_entry": (0,),
                "grid_shape": (1,),
            },
        )
        for src_grid_entry in src_arr.grid.get_entry_iterator():
            src_coord: tuple = src_arr.grid.get_entry_coordinates(src_grid_entry)
            src_block: Block = src_arr.blocks[src_grid_entry]

            # Make sure index values in subscript are within bounds of src_arr.
            # We also prepare dst_grid_entry here.
            dst_grid_entry_list = []
            skip = False
            for curr_axis in range(len(np_ss)):
                if curr_axis == axis:
                    dst_grid_entry_list.append(None)
                elif isinstance(np_ss[curr_axis], slice):
                    dst_grid_entry_list.append(src_grid_entry[curr_axis])
                elif not (
                    src_coord[curr_axis]
                    <= np_ss[curr_axis]
                    < src_coord[curr_axis] + src_block.shape[curr_axis]
                ):
                    skip = True
                    break
            if skip:
                continue
            for curr_axis in range(len(np_ss), len(src_grid_entry)):
                dst_grid_entry_list.append(src_grid_entry[curr_axis])

            for j in range(dst_arr.grid.grid_shape[dst_axis]):
                dst_grid_entry_list[dst_axis] = j
                dst_grid_entry = tuple(dst_grid_entry_list)
                dst_block: Block = dst_arr.blocks[dst_grid_entry]
                dst_coord: tuple = dst_arr.grid.get_entry_coordinates(dst_grid_entry)

                if dst_block.oid is None:
                    dst_arg = (dst_block.shape, dst_block.dtype)
                else:
                    dst_arg = dst_block.oid
                dst_block.oid = self.km.advanced_select_block_along_axis(
                    dst_arg,
                    src_block.oid,
                    ss,
                    dst_axis,
                    axis,
                    dst_coord,
                    src_coord,
                    syskwargs={
                        "grid_entry": dst_grid_entry,
                        "grid_shape": dst_arr.grid.grid_shape,
                    },
                )
        return dst_arr

    def __setitem__(self, key, value):
        value: BlockArrayBase = self.to_block_array(value, self.km)
        ss, is_handled_advanced, axis = self._preprocess_subscript(key)
        if is_handled_advanced:
            return self._advanced_single_array_assign(ss, value, axis)

        # This is to deal with circular imports. Little overhead since this happens once per call.
        # However, would be better to rearrange modules in the future.
        from nums.core.array.view import ArrayView

        av: ArrayView = ArrayView.from_block_array(self)
        av[key] = value

    def _advanced_single_array_assign(
        self,
        ss: tuple,
        value,
        axis: int,
    ):
        array = ss[axis]
        assert len(array.shape) == 1

        # The subscript contains a single array. We therefore know one of two things is true:
        # 1. value is the same shape as self along axes != axis.
        # 2. value is scalar or 1-dimensional.
        # We currently don't support the case where value may broadcasted if it has more dims.
        # This should be a straight-forward future task.
        value: BlockArrayBase = value
        mode = None
        if len(value.shape) == 0:
            # subscripted value per block will broadcast to other dimensions.
            mode = "scalar"
        elif len(value.shape) == 1:
            # assert len(value.shape) == len(ss)
            mode = "single-dim"
            # Can broadcast if trailing dim matches.
            assert len(ss[axis]) == value.shape[0]

            for i in range(len(self.shape)):
                if i == axis:
                    assert len(ss[i]) == value.shape[0]
                elif i < axis:
                    # Nothing to check here.
                    # These entries are : or integer.
                    pass
                else:
                    if i < len(ss):
                        if not isinstance(ss[i], slice):
                            # ss[i] is an integer.
                            continue
                    # If we're here, then the rest of the subscript operator
                    # will resolve to :, which is not broadcastable.
                    raise ValueError(
                        "Cannot broadcast input array "
                        "from shape %s into shape %s"
                        % (value.shape, tuple([value.shape[0]] + list(self.shape[i:])))
                    )
        elif len(value.shape) == len(self.shape):
            mode = "multi-dim"
            new_block_shape = []
            for i in range(len(self.shape)):
                if i == axis:
                    new_block_shape.append(value.block_shape[i])
                elif i < len(ss) and (
                    array_utils.is_int(ss[i]) or array_utils.is_uint(ss[i])
                ):
                    # These entries are : or integer.
                    # assert array_utils.is_int(ss[i]) or array_utils.is_uint(ss[i])
                    assert value.shape[i] == 1
                    new_block_shape.append(1)
                else:
                    assert value.shape[i] == self.shape[i], "Shape mismatch."
                    new_block_shape.append(self.block_shape[i])
            new_block_shape = tuple(new_block_shape)
            if new_block_shape != value.block_shape:
                # TODO: This message occurs on X[idx[:n]] = X[idx[n:]] + 0.5,
                #  even when n is a multiple of block_shape[0].
                warnings.warn(
                    ("Assigned value block shape %s " % str(value.block_shape))
                    + (
                        "does not match block shape %s of assignee. "
                        % str(new_block_shape)
                    )
                    + "Applying reshape to assigned value."
                )
                value = value.reshape(block_shape=new_block_shape)

        # Like select, iterate over destination blocks along the axis being updated.
        # e.g. if self is 2-dim and axis=0, then fix the row and iterate over the columns.
        # If value has the same shape as self, then for each destination block,
        # iterate over the blocks in value along axis.
        # e.g. if self is 2-dim and axis=0, then for the given column, iterate over the rows
        # of value.
        # If value is scalar, then attempt to assign it to every destination block.
        # If value is 1-dim, the just iterate over the dim and assign accordingly.

        dst_arr = self
        src_arr = value
        src_grid_shape = src_arr.grid.grid_shape
        np_ss = ss
        ss = self.km.put(
            ss,
            syskwargs={
                "grid_entry": (0,),
                "grid_shape": (1,),
            },
        )
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: BlockBase = dst_arr.blocks[dst_grid_entry]
            dst_coord: tuple = dst_arr.grid.get_entry_coordinates(dst_grid_entry)

            # Make sure index values in subscript are within bounds of dst_arr.
            # We don't need to check src_arr:
            # 1) The block shapes of dst_arr and src_arr are the same except along axis
            #    and indices in ss. We are not concerned with axes the indices in ss correspond to,
            #    because they are of size 1 in src_arr => we only need to check that indices
            #    fall within bounds of dst_arr.
            # 2) For each dst_arr, we test the values
            #    to assign to dst_arr by traverse the src_arr along axis.
            #    Thus, size along all other axes are equal or broadcasted.
            skip = False
            for curr_axis in range(len(np_ss)):
                if curr_axis == axis or isinstance(np_ss[curr_axis], slice):
                    continue
                if not (
                    dst_coord[curr_axis]
                    <= np_ss[curr_axis]
                    < dst_coord[curr_axis] + dst_block.shape[curr_axis]
                ):
                    skip = True
                    break
            if skip:
                continue

            if mode == "scalar":
                src_block: BlockBase = src_arr.blocks.item()
                src_coord: tuple = src_arr.grid.get_entry_coordinates(
                    src_block.grid_entry
                )
                dst_block.oid = self.km.advanced_assign_block_along_axis(
                    dst_block.oid,
                    src_block.oid,
                    ss,
                    axis,
                    dst_coord,
                    src_coord,
                    syskwargs={
                        "grid_entry": dst_grid_entry,
                        "grid_shape": dst_arr.grid.grid_shape,
                    },
                )
            elif mode == "single-dim":
                for src_grid_entry in src_arr.grid.get_entry_iterator():
                    src_block: BlockBase = src_arr.blocks[src_grid_entry]
                    src_coord: tuple = src_arr.grid.get_entry_coordinates(
                        src_grid_entry
                    )
                    dst_block.oid = self.km.advanced_assign_block_along_axis(
                        dst_block.oid,
                        src_block.oid,
                        ss,
                        axis,
                        dst_coord,
                        src_coord,
                        syskwargs={
                            "grid_entry": dst_grid_entry,
                            "grid_shape": dst_arr.grid.grid_shape,
                        },
                    )
            elif mode == "multi-dim":
                for j in range(src_grid_shape[axis]):
                    # Apply sel from each block along axis of src_arr.
                    # e.g. for 2 dim array, we fix the column blocks
                    # given by dst_grid_entry, and iterate over the rows.
                    src_grid_entry = tuple(
                        list(dst_grid_entry[:axis])
                        + [j]
                        + list(dst_grid_entry[axis + 1 :])
                    )
                    src_block: BlockBase = src_arr.blocks[src_grid_entry]
                    src_coord: tuple = src_arr.grid.get_entry_coordinates(
                        src_grid_entry
                    )
                    dst_block.oid = self.km.advanced_assign_block_along_axis(
                        dst_block.oid,
                        src_block.oid,
                        ss,
                        axis,
                        dst_coord,
                        src_coord,
                        syskwargs={
                            "grid_entry": dst_grid_entry,
                            "grid_shape": dst_arr.grid.grid_shape,
                        },
                    )
        return dst_arr

    def tree_reduce(
        self, op_name, blocks_or_oids, result_grid_entry, result_grid_shape
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
        while len(q) > 1:
            a_oid, a_ge, a_gs, a_T = q.pop(0)
            b_oid, _, _, b_T = q.pop(0)
            ge, gs = (
                (result_grid_entry, result_grid_shape) if len(q) == 0 else (a_ge, a_gs)
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
        r_oid, r_ge, r_gs, _ = q.pop(0)
        assert r_ge == result_grid_entry
        assert r_gs == result_grid_shape
        return r_oid

    def check_or_convert_other(self, other, compute_block_shape=False):
        raise NotImplementedError()

    def _check_bop_implemented(self, other):
        raise NotImplementedError()

    # All operators: https://docs.python.org/3/library/operator.html

    #################
    # Unary functions
    #################

    def ufunc(self, op_name):
        raise NotImplementedError()

    def __neg__(self):
        return self.ufunc("negative")

    def __pos__(self):
        return self

    def __abs__(self):
        return self.ufunc("abs")

    def __invert__(self):
        return self.ufunc("invert")

    #################
    # Arithmetic
    #################

    @staticmethod
    def elementwise(op_name, a, b):
        raise NotImplementedError()

    def __mod__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("mod", self, other)

    def __rmod__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("mod", other, self)

    __imod__ = __mod__

    def __add__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("add", self, other)

    def __radd__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("add", other, self)

    __iadd__ = __add__

    def __sub__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("sub", self, other)

    def __rsub__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("sub", other, self)

    __isub__ = __sub__

    def __mul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("mul", self, other)

    def __rmul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("mul", other, self)

    __imul__ = __mul__

    def __truediv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("truediv", self, other)

    def __rtruediv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("truediv", other, self)

    __itruediv__ = __truediv__

    def __floordiv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("floor_divide", self, other)

    def __rfloordiv__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("floor_divide", other, self)

    __ifloordiv__ = __floordiv__

    def __pow__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("pow", self, other)

    def __rpow__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("pow", other, self)

    __ipow__ = __pow__

    ##################
    # Boolean
    ##################

    # TODO (hme): Type check bool ops.
    def __bool__(self):
        # pylint: disable=no-member
        if np.sum(self.shape) == len(self.shape):
            # If all ones or scalar, then this is defined.
            return self.get().__bool__()
        return True

    def __or__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_or", self, other)

    def __ror__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_or", other, self)

    __ior__ = __or__

    def __and__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_and", self, other)

    def __rand__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_and", other, self)

    __iand__ = __and__

    def __xor__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_xor", self, other)

    def __rxor__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("bitwise_xor", other, self)

    __ixor__ = __xor__

    def __lshift__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("left_shift", self, other)

    def __rlshift__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("left_shift", other, self)

    __ilshift__ = __lshift__

    def __rshift__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("right_shift", self, other)

    def __rrshift__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.elementwise("right_shift", other, self)

    __irshift__ = __rshift__

    #################
    # Linear Algebra
    #################

    def _compute_tensordot_syskwargs(
        self, self_block: BlockBase, other_block: BlockBase
    ):
        # Schedule on larger block.
        if np.product(self_block.shape) >= np.product(other_block.shape):
            return self_block.true_grid_entry(), self_block.true_grid_shape()
        else:
            return other_block.true_grid_entry(), other_block.true_grid_shape()

    @staticmethod
    def tensordot(a, b, axes=2):
        raise NotImplementedError()

    def __matmul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        if len(self.shape) > 2:
            # TODO (bcp): NumPy's implementation does a stacked matmul, which is not supported yet.
            raise NotImplementedError(
                "Matrix multiply for tensors of rank > 2 not supported yet."
            )
        else:
            return self.tensordot(self, other, 1)

    def __rmatmul__(self, other):
        if not self._check_bop_implemented(other):
            return NotImplemented
        return self.tensordot(other, self, 1)

    __imatmul__ = __matmul__

    #################
    # Inequalities
    #################

    def __inequality__(self, op, other):
        raise NotImplementedError()

    def __ge__(self, other):
        return self.__inequality__("ge", other)

    def __rge__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("ge", self)

    def __gt__(self, other):
        return self.__inequality__("gt", other)

    def __rgt__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("gt", self)

    def __le__(self, other):
        return self.__inequality__("le", other)

    def __rle__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("le", self)

    def __lt__(self, other):
        return self.__inequality__("lt", other)

    def __rlt__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("lt", self)

    def __eq__(self, other):
        return self.__inequality__("eq", other)

    def __req__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("eq", self)

    def __ne__(self, other):
        return self.__inequality__("ne", other)

    def __rne__(self, other):
        other = self.check_or_convert_other(other)
        return other.__inequality__("ne", self)


class Reshape:
    @staticmethod
    def compute_shape(shape, input_shape):
        size = np.product(shape)
        if -1 in input_shape:
            new_shape = []
            other_dim_prod = 1
            negative_one_seen = False
            for dim in input_shape:
                if dim == -1:
                    if negative_one_seen:
                        raise Exception("Only one -1 permitted in reshape.")
                    negative_one_seen = True
                    continue
                other_dim_prod *= dim
            if size % other_dim_prod != 0:
                raise Exception("Invalid shape.")
            for dim in input_shape:
                if dim == -1:
                    new_shape.append(size // other_dim_prod)
                else:
                    new_shape.append(dim)
        else:
            new_shape = input_shape
        assert size == np.product(new_shape)
        return new_shape

    def _group_index_lists_by_block(
        self, dst_slice_tuples, src_grid: ArrayGrid, dst_index_list, src_index_list
    ):
        # TODO(hme): Keep this function here until it's needed for greater support of
        #  selection/assignment operations.
        # Block grid entries needed to write to given dst_slice_selection.
        src_blocks = {}
        dst_slice_np = np.array(dst_slice_tuples).T
        dst_index_arr = np.array(dst_index_list)
        src_index_arr = np.array(src_index_list)
        # Pick the smallest type to represent indices.
        # A set of these indices may be transmitted over the network,
        # so we want to pick the smallest encoding possible.
        index_types = [
            (2**8, np.uint8),
            (2**16, np.uint16),
            (2**32, np.uint32),
            (2**64, np.uint64),
        ]
        index_type = None
        for bound, curr_index_type in index_types:
            if np.all(np.array(src_grid.block_shape) < bound) and np.all(
                dst_slice_np[1] < bound
            ):
                index_type = curr_index_type
                break
        if index_type is None:
            raise Exception("Unable to encode block indices, blocks are too large.")
        for grid_entry in src_grid.get_entry_iterator():
            src_slice_np = np.array(src_grid.get_slice_tuples(grid_entry)).T
            index_pairs = []
            for i in range(src_index_arr.shape[0]):
                src_index = src_index_arr[i]
                dst_index = dst_index_arr[i]
                if np.all(
                    (src_slice_np[0] <= src_index) & (src_index < src_slice_np[1])
                ):
                    index_pair = (
                        (dst_index - dst_slice_np[0]).astype(index_type),
                        (src_index - src_slice_np[0]).astype(index_type),
                    )
                    index_pairs.append(index_pair)
            if len(index_pairs) > 0:
                src_blocks[grid_entry] = index_pairs
        return src_blocks

    def _arbitrary_reshape(
        self, arr: BlockArrayBase, shape, block_shape
    ) -> BlockArrayBase:
        # This is the worst-case scenario.
        # Generate index mappings per block, and group source indices to minimize
        # RPCs and generation of new objects.
        km = arr.km
        dst_arr = type(arr).empty(
            shape=shape, block_shape=block_shape, dtype=arr.dtype, km=km
        )
        for dst_grid_entry in dst_arr.grid.get_entry_iterator():
            dst_block: BlockBase = dst_arr.blocks[dst_grid_entry]
            dst_slice_selection = dst_arr.grid.get_slice(dst_grid_entry)
            dst_index_list = array_utils.slice_sel_to_index_list(dst_slice_selection)
            src_index_list = array_utils.translate_index_list(
                dst_index_list, shape, arr.shape
            )
            src_blocks = self._group_index_lists_by_block(
                dst_arr.grid.get_slice_tuples(dst_grid_entry),
                arr.grid,
                dst_index_list,
                src_index_list,
            )
            for src_grid_entry in src_blocks:
                src_block: BlockBase = arr.blocks[src_grid_entry]
                index_pairs = src_blocks[src_grid_entry]
                syskwargs = {
                    "grid_entry": dst_grid_entry,
                    "grid_shape": dst_arr.grid.grid_shape,
                }
                dst_block.oid = km.update_block_by_index(
                    dst_block.oid, src_block.oid, index_pairs, syskwargs=syskwargs
                )
        return dst_arr

    def _block_shape_reshape(self, arr, block_shape):
        rarr: BlockArrayBase = type(arr).empty(
            arr.shape, block_shape, arr.dtype, arr.km
        )
        for grid_entry in rarr.grid.get_entry_iterator():
            grid_entry_slice = rarr.grid.get_slice(grid_entry)
            # TODO (hme): This could be less costly.
            rarr[grid_entry_slice] = arr[grid_entry_slice]
        return rarr

    def _strip_ones(self, shape):
        return tuple(filter(lambda x: x != 1, shape))

    def _check_positions_ones(self, shape, block_shape):
        # If a position in the shape is 1, then the corresponding
        # position in block_shape should also be 1.
        for i in range(len(shape)):
            if shape[i] == 1:
                if shape[i] != block_shape[i]:
                    return False
        return True

    def _is_simple_reshape(self, arr: BlockArrayBase, shape, block_shape):
        # Is the reshape a difference of factors of 1?
        # Strip out 1s and compare.
        # If a position in the shape is 1, then the corresponding
        # position in block_shape should also be 1.

        # If source shape and dest shape are the same or source block_shape and dest block_shape
        # are same, this is not a simple reshape.
        if shape == arr.shape or block_shape == arr.block_shape:
            return False

        # Checks if source shape and dest shape are same & source block_shape and dest
        # block_shape are same after stripping ones.
        if not (
            self._strip_ones(shape) == self._strip_ones(arr.shape)
            and self._strip_ones(block_shape) == self._strip_ones(arr.block_shape)
        ):
            return False
        if not self._check_positions_ones(shape, block_shape):
            return False
        return True

    def _simple_reshape(self, arr, shape, block_shape):
        # Reshape the array of blocks only.
        # This is only used when the difference in shape are factors of 1s,
        # and the ordering of other factors are maintained.

        # Check assumptions.
        assert len(self._strip_ones(arr.shape)) == len(self._strip_ones(shape))

        # Create new grid, and perform reshape on blocks
        # to simplify access to source blocks.
        grid = ArrayGrid(shape, block_shape, dtype=arr.dtype.__name__)
        src_blocks = arr.blocks.reshape(grid.grid_shape)
        if arr.is_dense:
            rarr = type(arr)(grid, arr.km)
        else:
            rarr = type(arr)(grid, arr.km, arr.fill_value)
        for grid_entry in grid.get_entry_iterator():
            src_block: BlockBase = src_blocks[grid_entry]
            dst_block: BlockBase = rarr.blocks[grid_entry]
            syskwargs = {"grid_entry": grid_entry, "grid_shape": grid.grid_shape}
            dst_block.oid = arr.km.reshape(
                src_block.oid, dst_block.shape, syskwargs=syskwargs
            )
        return rarr

    def _validate(self, arr, shape, block_shape):
        assert -1 not in shape
        assert -1 not in block_shape
        assert len(shape) == len(block_shape)
        assert np.product(arr.shape) == np.product(shape)

    def __call__(self, arr: BlockArrayBase, shape, block_shape):
        self._validate(arr, shape, block_shape)
        if arr.shape == shape and arr.block_shape == block_shape:
            return arr
        elif self._is_simple_reshape(arr, shape, block_shape):
            return self._simple_reshape(arr, shape, block_shape)
        elif arr.shape == shape and arr.block_shape != block_shape:
            return self._block_shape_reshape(arr, block_shape)
        elif arr.shape != shape and arr.block_shape == block_shape:
            # Just do full reshape for this case as well.
            # Though there may be a better solution, we generally expect
            # the block shape to change with array shape.
            return self._arbitrary_reshape(arr, shape, block_shape)
        else:
            assert arr.shape != shape and arr.block_shape != block_shape
            return self._arbitrary_reshape(arr, shape, block_shape)
