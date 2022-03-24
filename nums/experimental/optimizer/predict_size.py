import numpy as np

# TODO: integrate this class with TreeNode and FuseGraph.
class PredictedSize:
    """
    Object encapsulated by each TreeNode to keep track of predicted sizes of blocks.
    Sparse operations use B(n, p) to estimate nnz as needed.
    """

    def __init__(self, is_dense, shape, nnz, dtype, fill_value=None, index_dtype=None):
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
        return PredictedSize(
            self.is_dense,
            self.shape,
            self.nnz,
            self.dtype,
            self.fill_value,
            self.index_dtype,
        )

    def nbytes(self):
        if self.fill_value is not None:
            return self.nnz * np.dtype(self.dtype).itemsize
        else:
            # Assuming COO format.
            return (
                self.nnz * np.dtype(self.dtype).itemsize
                + self.nnz * len(self.shape) * np.dtype(self.index_dtype).itemsize
            )

    def _disjunction(self, other):
        # If either elements are nonzero, result is nonzero.
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        result = self.copy()
        result.nnz = int((1 - (1 - p1) * (1 - p2)) * n1)
        return result

    def _conjunction(self, other):
        # If both elements are nonzero, result is nonzero.
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        p2 = other.nnz / n2
        result = self.copy()
        result.nnz = int(p1 * p2 * n1)
        return result

    def _selection(self, other):
        # If self element is nonzero, result is nonzero.
        assert self.fill_value == 0
        n1 = np.prod(self.shape)
        n2 = np.prod(other.shape)
        p1 = self.nnz / n1
        result = self.copy()
        result.nnz = int(p1 * n2)
        return result

    def __add__(self, other):
        if not self.is_dense and not other.is_dense:
            return self._disjunction(other)
        if self.is_dense:
            return self.copy()
        return other.copy()

    def __mul__(self, other):
        if not self.is_dense and not other.is_dense:
            if self.fill_value == 0 and other.fill_value == 0:
                return self._conjunction(other)
            if self.fill_value == 0:
                return self._selection(other)
            if other.fill_value == 0:
                return other._selection(self)
            return self._disjunction(other)
        if self.fill_value == 0 and other.is_dense:
            return self._selection(other)
        if self.is_dense and other.fill_value == 0:
            return other._selection(self)
        if self.is_dense:
            return self.copy()
        return other.copy()

    def tensordot(self, other, axes=2):
        if axes > 0:
            result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
            sum_shape = tuple(self.shape[-axes:])
        else:
            result_shape = tuple(self.shape + other.shape)
            sum_shape = (1,)
        if not self.is_dense and not other.is_dense:
            assert self.fill_value == 0 and other.fill_value == 0
            n1 = np.prod(self.shape)
            n2 = np.prod(other.shape)
            p1 = self.nnz / n1
            p2 = other.nnz / n2
            m = np.prod(result_shape)
            k = np.prod(sum_shape)
            return PredictedSize(
                is_dense=False,
                shape=result_shape,
                nnz=int((1 - (1 - p1 * p2) ** k) * m),
                dtype=self.dtype,  # FIXME: what happens for different dtypes? Use larger one?
                fill_value=0,
                index_dtype=self.dtype,  # FIXME
            )
        return PredictedSize(
            is_dense=True,
            shape=result_shape,
            nnz=np.prod(result_shape),
            dtype=self.dtype,  # FIXME
        )
