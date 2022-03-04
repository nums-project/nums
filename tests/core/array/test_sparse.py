import numpy as np
import sparse
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparse import SparseBlockArray
from nums.core.array.random import NumsRandomState


def test_sparse_init(app_inst: ArrayApplication):
    X = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X_ba = app_inst.array(X, block_shape=(2, 2))
    X_sp = SparseBlockArray.from_ba(X_ba)
    assert X_sp.nnz == 8
    print(X_sp.nbytes)
    Y_ba = X_sp.to_ba()
    assert np.array_equal(X, Y_ba.get())


def test_sparse_random(app_inst: ArrayApplication):
    rs: NumsRandomState = app_inst.random_state(1337)
    sba = rs.sparse_randint(1, high=5, dtype=int, shape=(15, 10), block_shape=(5, 5), p=0.1, fill_value=0)
    print(sba.nbytes, sba.nnz)
    ba = sba.to_ba()
    print(ba.get())


def test_sparse_uop(app_inst: ArrayApplication):
    X = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X_sp = sparse.GCXS.from_numpy(X, fill_value=1)
    Y_sp = sparse.elemwise(np.negative, X_sp)
    X_ba = app_inst.array(X, block_shape=(2, 2))
    X_sba = SparseBlockArray.from_ba(X_ba, fill_value=1)
    Y_sba = X_sba.ufunc("negative")
    assert Y_sba.fill_value == Y_sp.fill_value  # -1
    assert Y_sba.nnz == Y_sp.nnz  # 12
    Y_ba = Y_sba.to_ba()
    assert np.array_equal(np.negative(X), Y_ba.get())


def test_sparse_add(app_inst: ArrayApplication):
    X1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X2 = X1 * 2
    X1_sp = sparse.GCXS.from_numpy(X1, fill_value=2)
    X2_sp = sparse.GCXS.from_numpy(X2, fill_value=2)
    X1_ba = app_inst.array(X1, block_shape=(2, 2))
    X2_ba = app_inst.array(X2, block_shape=(2, 2))
    X1_sba = SparseBlockArray.from_ba(X1_ba, fill_value=2)
    X2_sba = SparseBlockArray.from_ba(X2_ba, fill_value=2)
    Y_sp = X1_sp + X2_sp
    Y_sba = X1_sba + X2_sba
    assert Y_sba.fill_value == Y_sp.fill_value  # 4
    print(Y_sp.fill_value)
    assert Y_sba.nnz == Y_sp.nnz  # 16
    print(Y_sp.nnz)
    Y_ba = Y_sba.to_ba()
    assert np.array_equal(X1 + X2, Y_ba.get())

    Y_ba = X1_sba + X2_ba
    assert np.array_equal(X1_sp + X2, Y_ba.get())


def test_sparse_mul(app_inst: ArrayApplication):
    X1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    X1_sp = sparse.GCXS.from_numpy(X1, fill_value=2)
    X2_sp = sparse.GCXS.from_numpy(X2, fill_value=2)
    X1_ba = app_inst.array(X1, block_shape=(2, 2))
    X2_ba = app_inst.array(X2, block_shape=(2, 2))
    X1_sba = SparseBlockArray.from_ba(X1_ba, fill_value=2)
    X2_sba = SparseBlockArray.from_ba(X2_ba, fill_value=2)
    Y_sp = X1_sp * X2_sp
    Y_sba = X1_sba * X2_sba
    assert Y_sba.fill_value == Y_sp.fill_value  # 4
    print(Y_sp.fill_value)
    assert Y_sba.nnz == Y_sp.nnz  # 16
    print(Y_sp.nnz)
    Y_ba = Y_sba.to_ba()
    assert np.array_equal(X1 * X2, Y_ba.get())

    # Sparse-dense
    X1_sp = sparse.GCXS.from_numpy(X1, fill_value=0)
    X1_sba = SparseBlockArray.from_ba(X1_ba, fill_value=0)
    Y_sp = X1_sp * X1
    Y_sba = X1_sba * X2_ba
    assert Y_sba.fill_value == Y_sp.fill_value
    print(Y_sp.fill_value)
    assert Y_sba.nnz == Y_sp.nnz
    print(Y_sp.nnz)
    Y_ba = Y_sba.to_ba()
    assert np.array_equal(X1 * X2, Y_ba.get())


if __name__ == "__main__":
    # pylint disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_sparse_init(app_inst)
    test_sparse_random(app_inst)
    test_sparse_uop(app_inst)
    test_sparse_add(app_inst)
