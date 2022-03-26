import numpy as np

from nums.core.array import utils as array_utils


# TODO: integrate this class with TreeNode and FuseGraph.
class TreeNodeSize:
    """
    Encapsulated by each TreeNode to keep track of estimated or observed sizes of blocks.
    Sparse binary operations use B(n, p) to estimate nnz as needed.
    """

    def __init__(
        self, is_dense, shape, nnz, dtype, fill_value=None, index_dtype=np.int64
    ):
        if is_dense:
            assert nnz == np.prod(shape)
        else:
            assert fill_value is not None and index_dtype is not None
        self.is_dense = is_dense
        self.shape = shape
        self.nnz = nnz
        self.dtype = dtype
        self.fill_value = fill_value
        self.index_dtype = index_dtype

    def copy(self):
        return TreeNodeSize(
            self.is_dense,
            self.shape,
            self.nnz,
            self.dtype,
            self.fill_value,
            self.index_dtype,
        )

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
        return TreeNodeSize(
            is_dense=self.is_dense,
            shape=self.shape,
            nnz=self.nnz,
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            fill_value=np.__getattribute__(op_name)(self.fill_value),
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

    def __add__(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("add", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "add", self.fill_value, other.fill_value
        )
        if is_dense:
            return TreeNodeSize(
                is_dense=True,
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
            is_dense=False,
            shape=shape,
            nnz=self._nnz_disjunction(other, shape),
            dtype=dtype,
            fill_value=array_utils.get_bop_fill_value(
                "add", self.fill_value, other.fill_value
            ),
        )

    def __mul__(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("mul", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_return_type(
            "mul", self.fill_value, other.fill_value
        )
        if is_dense:
            return TreeNodeSize(
                is_dense=True,
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
            is_dense=False,
            shape=shape,
            nnz=nnz,
            dtype=dtype,
            fill_value=array_utils.get_bop_fill_value(
                "mul", self.fill_value, other.fill_value
            ),
        )

    def tensordot(self, other, axes=2):
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
                is_dense=True,
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
            is_dense=False,
            shape=shape,
            nnz=int((1 - (1 - p1 * p2) ** k) * m),
            dtype=dtype,
            fill_value=0,
        )
