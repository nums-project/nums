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
from numba.core.errors import NumbaNotImplementedError

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

    @property
    def is_dense(self):
        raise NotImplementedError()

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
        # If defer is True, this operation does not modify the remote object.
        # If defer is True and redistribute is False,
        # this operation does not move the remote object.
        grid_entryT = tuple(reversed(self.grid_entry))
        grid_shapeT = tuple(reversed(self.grid_shape))
        blockT = type(self)(
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
    def is_dense(self):
        return True

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

    @property
    def is_dense(self):
        raise NotImplementedError()

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
        """
        Transpose this matrix. Only use defer with arithmetic operations.
        Setting redistribute to True may significantly impact performance.
        :param defer: When true, the transpose operation will be applied
        with the next arithmetic operation.
        :param redistribute: If defer is false, setting this to true will
        redistribute the data according to the device grid (data placement policy).
        This parameter has no effect when defer is true.
        :return: The transposed matrix.
        """
        if defer and redistribute:
            warnings.warn("defer is True, redistribute=True will be ignored.")
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        gridT = ArrayGrid.from_meta(metaT)
        rarrT = type(self)(gridT, self.km)
        rarrT.blocks = np.copy(self.blocks.T)
        for grid_entry in rarrT.grid.get_entry_iterator():
            rarrT.blocks[grid_entry] = rarrT.blocks[grid_entry].transpose(
                defer, redistribute
            )
        return rarrT

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
