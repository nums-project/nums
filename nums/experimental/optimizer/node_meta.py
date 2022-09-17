import numpy as np

from nums.core.array import utils as array_utils


class TreeNodeMeta:
    """
    Encapsulated by each TreeNode to keep track of node metadata like shape and output density.
    """

    def __init__(self, dtype, is_dense):
        self.dtype = dtype
        self.is_dense = is_dense

    def copy(self):
        return TreeNodeMeta(
            self.dtype,
            self.is_dense,
        )

    def uop_partial(self, op_name):
        """
        Does not update nnz, as it can't be estimated for non-leaf nodes.
        """
        is_dense = self.is_dense or array_utils.get_sparse_uop_densify(op_name)
        return TreeNodeMeta(
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            is_dense=is_dense,
        )

    def reduce_axis_partial(self, op_name, axis, keepdims):
        return TreeNodeMeta(
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            is_dense=True,
        )

    def bop_partial(self, op_name, other, **kwargs):
        dtype = array_utils.get_bop_output_type(op_name, self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_densify(
            op_name,
            self.is_dense,
            other.is_dense,
        )
        return TreeNodeMeta(dtype, is_dense)


class LeafMeta(TreeNodeMeta):
    """
    Encapsulated by each Leaf to keep track of sizes of blocks in addition to TreeNodeMeta.
    Sparse binary operations use B(n, p) to estimate nnz as needed.
    """

    def __init__(self, shape, nnz, dtype, is_dense, index_dtype=np.int64):
        super().__init__(dtype, is_dense)
        self.shape = shape
        self.nnz = nnz
        self.index_dtype = index_dtype
        if self.is_dense:
            assert nnz == np.prod(shape)
        else:
            assert index_dtype is not None

    def copy(self):
        return LeafMeta(
            self.shape,
            self.nnz,
            self.dtype,
            self.is_dense,
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
        if op_name == "transpose":
            return LeafMeta(
                shape=tuple(reversed(self.shape)),
                nnz=self.nnz,
                dtype=self.dtype,
                is_dense=self.is_dense,
            )
        nnz = self.nnz
        is_dense = self.is_dense
        if not is_dense and array_utils.get_sparse_uop_densify(op_name):
            nnz = np.prod(self.shape)
            is_dense = True
        return LeafMeta(
            shape=self.shape,
            nnz=nnz,
            dtype=array_utils.get_uop_output_type(op_name, self.dtype),
            is_dense=is_dense,
        )

    def reduce_axis(self, op_name, axis, keepdims):
        shape = list(self.shape)
        if axis is None:
            shape = []
        elif keepdims:
            shape[axis] = 1
        else:
            shape.pop(axis)
        if self.is_dense:
            return LeafMeta(
                shape=tuple(shape),
                nnz=np.prod(shape),
                dtype=array_utils.get_uop_output_type(op_name, self.dtype),
                is_dense=True,
            )
        else:
            # If any element in reduced axis is nonzero, result is nonzero.
            p1 = self.nnz / np.prod(self.shape)
            nnz = int((1 - (1 - p1) ** self.shape[axis]) * np.prod(shape))
            return LeafMeta(
                shape=tuple(shape),
                nnz=nnz,
                dtype=array_utils.get_uop_output_type(op_name, self.dtype),
                is_dense=False,
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
        n1 = np.prod(self.shape)
        p1 = self.nnz / n1
        return int(p1 * np.prod(shape))

    def add(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("add", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_densify(
            "add",
            self.is_dense,
            other.is_dense,
        )
        if is_dense:
            return LeafMeta(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                is_dense=True,
            )
        if self.is_dense or other.is_dense:
            raise ValueError(
                "TreeNodeSize.__add__ is inconsistent with sparse bop rules."
            )
        return LeafMeta(
            shape=shape,
            nnz=self._nnz_disjunction(other, shape),
            dtype=dtype,
            is_dense=False,
        )

    __add__ = add

    def mul(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("mul", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_densify(
            "mul",
            self.is_dense,
            other.is_dense,
        )
        if is_dense:
            return LeafMeta(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                is_dense=True,
            )
        if not self.is_dense and not other.is_dense:
            nnz = self._nnz_conjunction(other, shape)
        elif not self.is_dense and other.is_dense:
            nnz = self._nnz_selection(other, shape)
        elif self.is_dense and not other.is_dense:
            nnz = other._nnz_selection(self, shape)
        else:
            raise ValueError(
                "TreeNodeSize.__mul__ is inconsistent with sparse bop rules."
            )
        return LeafMeta(
            shape=shape,
            nnz=nnz,
            dtype=dtype,
            is_dense=False,
        )

    __mul__ = mul

    def truediv(self, other):
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("truediv", self.dtype, other.dtype)
        is_dense = array_utils.get_sparse_bop_densify(
            "truediv",
            self.is_dense,
            other.is_dense,
        )
        if is_dense:
            return LeafMeta(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                is_dense=True,
            )
        if self.is_dense or other.is_dense:
            raise ValueError(
                "TreeNodeSize.__add__ is inconsistent with sparse bop rules."
            )
        if not other.is_dense:  # nan
            nnz = other._nnz_selection(self, shape)
        elif not self.is_dense:
            nnz = self._nnz_selection(other, shape)
        else:
            nnz = self._nnz_disjunction(other, shape)
        return LeafMeta(
            shape=shape,
            nnz=nnz,
            dtype=dtype,
            is_dense=False,
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
        is_dense = array_utils.get_sparse_bop_densify(
            "tensordot", self.is_dense, other.is_dense
        )
        if is_dense:
            return LeafMeta(
                shape=shape,
                nnz=np.prod(shape),
                dtype=dtype,
                is_dense=True,
            )
        if self.is_dense or other.is_dense:
            raise ValueError(
                "TreeNodeSize.tensordot is inconsistent with sparse bop rules."
            )
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        m = np.prod(shape)
        k = np.prod(sum_shape)
        return LeafMeta(
            shape=shape,
            nnz=int((1 - (1 - p1 * p2) ** k) * m),
            dtype=dtype,
            is_dense=False,
        )

    def inequality(self, op_name, other):
        assert other.shape == ()
        dtype = array_utils.get_bop_output_type(op_name, self.dtype, other.dtype)
        return LeafMeta(
            shape=self.shape,
            nnz=self.nnz,
            dtype=dtype,
            is_dense=self.is_dense,
        )

    def bop_dense(self, other, **kwargs):
        # NOTE: as catch-all fallback, this may be wrong for unknown non-elementwise binary ops.
        shape = array_utils.broadcast_shape(self.shape, other.shape)
        dtype = array_utils.get_bop_output_type("add", self.dtype, other.dtype)
        return LeafMeta(
            shape=shape,
            nnz=np.prod(shape),
            dtype=dtype,
            is_dense=False,
        )

    def bop(self, op_name, other, **kwargs):
        bop_estimation_map = {
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
        return bop_estimation_map.get(op_name, self.bop_dense)(other, **kwargs)
