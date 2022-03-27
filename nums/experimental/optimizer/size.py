import numpy as np

from nums.core.array import utils as array_utils


# TODO: integrate this class with FuseGraph.
class TreeNodeSize:
    """
    Encapsulated by each TreeNode to keep track of estimated or observed sizes of blocks.
    Sparse binary operations use B(n, p) to estimate nnz as needed.
    """

    def __init__(self, shape, nnz, dtype, fill_value=None, index_dtype=np.int64):
        self.shape = shape
        self.nnz = nnz
        self.dtype = dtype
        self.fill_value = fill_value
        self.index_dtype = index_dtype
        if self.is_dense:
            assert nnz == np.prod(shape)
        else:
            assert index_dtype is not None

        self.bop_estimation_map = {
            "add": self.add,
            "sum": self.add,
            "sub": self.add,
            "mul": self.mul,
            "prod": self.mul,
            "truediv": self.truediv,
            "pow": self.pow,
            "matmul": lambda other: self.tensordot(other, axes=1),
            "tensordot": self.tensordot,
            "lt": lambda other: self.inequality("lt", other),
            "le": lambda other: self.inequality("le", other),
            "gt": lambda other: self.inequality("gt", other),
            "ge": lambda other: self.inequality("ge", other),
            "eq": lambda other: self.inequality("eq", other),
            "ne": lambda other: self.inequality("ne", other),
        }

    def copy(self):
        return TreeNodeSize(
            self.shape,
            self.nnz,
            self.dtype,
            self.fill_value,
            self.index_dtype,
        )

    @property
    def is_dense(self):
        return self.fill_value is None

    @property
    def nbytes(self):
        if self.is_dense:
            return self.nnz * np.dtype(self.dtype).itemsize
        else:
            # Assuming COO format.
            return (
                self.nnz * np.dtype(self.dtype).itemsize
                + self.nnz * len(self.shape) * np.dtype(self.index_dtype).itemsize
            )

    def uop(self, op_name):
        if self.fill_value is None:
            fill_value = None
        else:
            fill_value = np.__getattribute__(op_name)(self.fill_value)
        return TreeNodeSize(
            shape=self.shape,
            nnz=self.nnz,
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            fill_value=fill_value,
        )

    def reduce_axis(self, op_name, axis, keepdims, transposed):
        # Assume dense for now
        shape = list(self.shape)
        if axis is None:
            shape = []
        elif keepdims:
            shape[axis] = 1
        else:
            shape.pop(axis)
        return TreeNodeSize(
            shape=tuple(shape),
            nnz=np.prod(shape),
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            fill_value=None,
        )

    def _nnz_disjunction(self, other, shape):
        # If either element is nonzero, result is nonzero.
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        return int((1 - (1 - p1) * (1 - p2)) * np.prod(shape))

    def _nnz_conjunction(self, other, shape):
        # If both element is nonzero, result is nonzero.
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        return int(p1 * p2 * np.prod(shape))

    def _nnz_selection(self, other, shape):
        # If self element is nonzero, result is nonzero.
        assert self.fill_value == 0
        n1 = np.prod(self.shape)
        p1 = self.nnz / n1
        return int(p1 * np.prod(shape))

    def add(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("add", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "add",
            self.fill_value,
            other.fill_value,
        )
        if is_dense:
            return TreeNodeSize(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                fill_value=None,
            )
        if self.is_dense or other.is_dense:
            raise ValueError(
                "TreeNodeSize.__add__ is inconsistent with sparse bop rules."
            )
        return TreeNodeSize(
            shape=shape,
            nnz=self._nnz_disjunction(other, shape),
            dtype=dtype,
            fill_value=array_utils.get_bop_fill_value(
                "add", self.fill_value, other.fill_value
            ),
        )

    __add__ = add

    def mul(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("mul", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "mul",
            self.fill_value,
            other.fill_value,
        )
        if is_dense:
            return TreeNodeSize(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                fill_value=None,
            )
        if not self.is_dense and not other.is_dense:
            if self.fill_value == 0 and other.fill_value == 0:
                nnz = self._nnz_conjunction(other, shape)
            elif self.fill_value == 0:
                nnz = self._nnz_selection(other, shape)
            elif other.fill_value == 0:
                nnz = other._nnz_selection(self, shape)
            else:
                nnz = self._nnz_disjunction(other, shape)
        elif self.fill_value == 0 and other.is_dense:
            nnz = self._nnz_selection(other, shape)
        elif self.is_dense and other.fill_value == 0:
            nnz = other._nnz_selection(self, shape)
        else:
            raise ValueError(
                "TreeNodeSize.__mul__ is inconsistent with sparse bop rules."
            )
        return TreeNodeSize(
            shape=shape,
            nnz=nnz,
            dtype=dtype,
            fill_value=array_utils.get_bop_fill_value(
                "mul", self.fill_value, other.fill_value
            ),
        )

    __mul__ = mul

    def truediv(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("truediv", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "truediv",
            self.fill_value,
            other.fill_value,
        )
        if is_dense:
            return TreeNodeSize(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                fill_value=None,
            )
        if self.is_dense or other.is_dense:
            raise ValueError(
                "TreeNodeSize.__add__ is inconsistent with sparse bop rules."
            )
        if other.fill_value == 0:  # nan
            nnz = other._nnz_selection(self, shape)
        elif self.fill_value == 0:
            nnz = self._nnz_selection(other, shape)
        else:
            nnz = self._nnz_disjunction(other, shape)
        return TreeNodeSize(
            shape=shape,
            nnz=nnz,
            dtype=dtype,
            fill_value=array_utils.get_bop_fill_value(
                "truediv", self.fill_value, other.fill_value
            ),
        )

    __truediv__ = truediv

    def floordiv(self, other):
        raise NotImplementedError()

    def pow(self, other):
        raise NotImplementedError()

    __pow__ = pow

    def tensordot(self, other, axes):
        if axes > 0:
            shape = tuple(self.shape[:-axes] + other.shape[axes:])
            sum_shape = tuple(self.shape[-axes:])
        else:
            shape = tuple(self.shape + other.shape)
            sum_shape = (1,)
        dtype = array_utils.get_bop_output_type("tensordot", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "tensordot", self.fill_value, other.fill_value
        )
        if is_dense:
            return TreeNodeSize(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                fill_value=None,
            )
        if (
            self.is_dense
            or other.is_dense
            or self.fill_value != 0
            or other.fill_value != 0
        ):
            raise ValueError(
                "TreeNodeSize.tensordot is inconsistent with sparse bop rules."
            )
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        m = np.prod(shape)
        k = np.prod(sum_shape)
        return TreeNodeSize(
            shape=shape,
            nnz=int((1 - (1 - p1 * p2) ** k) * m),
            dtype=dtype,
            fill_value=0,
        )

    def inequality(self, op_name, other):
        assert other.shape == ()
        dtype = array_utils.get_bop_output_type(op_name, self.dtype, other.dtype)
        fill_value = array_utils.get_bop_fill_value(
            op_name, self.fill_value, other.fill_value
        )
        return TreeNode(
            shape=self.shape,
            nnz=self.nnz,
            dtype=dtype,
            fill_value=fill_value,
        )

    def bop_dense(self, other):
        # NOTE: as catch-all fallback, this may be wrong for unknown non-elementwise binary ops.
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("add", self.dtype, other.dtype)
        return TreeNodeSize(
            shape=shape,
            nnz=np.prod(shape),
            dtype=dtype,
            fill_value=None,
        )

    def bop(self, op_name, other, **kwargs):
        return self.bop_estimation_map.get(op_name, self.bop_dense)(other, **kwargs)
