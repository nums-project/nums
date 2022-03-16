import numpy as np
import sparse
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparse import SparseBlockArray
from nums.core.array.random import NumsRandomState


def test_sparse_init(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x_ba = app_inst.array(x1, block_shape=(2, 2))
    x_sp = SparseBlockArray.from_ba(x_ba)
    assert x_sp.nnz == 8
    print(x_sp.nbytes)
    y_ba = x_sp.to_ba()
    assert np.array_equal(x1, y_ba.get())


def test_sparse_random(app_inst: ArrayApplication):
    rs: NumsRandomState = app_inst.random_state(1337)
    sba = rs.sparse_randint(1, high=5, dtype=int, shape=(15, 10), block_shape=(5, 5), p=0.1, fill_value=0)
    print(sba.nbytes, sba.nnz)
    ba = sba.to_ba()
    print(ba.get())


def test_sparse_uop(app_inst: ArrayApplication):
    x = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x_sp = sparse.GCXS.from_numpy(x, fill_value=1)
    x_sp = sparse.elemwise(np.negative, x_sp)
    x_ba = app_inst.array(x, block_shape=(2, 2))
    x_sba = SparseBlockArray.from_ba(x_ba, fill_value=1)
    y_sba = x_sba.ufunc("negative")
    assert y_sba.fill_value == x_sp.fill_value  # -1
    assert y_sba.nnz == x_sp.nnz  # 12
    y_ba = y_sba.to_ba()
    assert np.array_equal(np.negative(x), y_ba.get())


def test_sparse_add(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x2 = x1 * 2
    x1_sp = sparse.GCXS.from_numpy(x1, fill_value=2)
    x2_sp = sparse.GCXS.from_numpy(x2, fill_value=2)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x2_ba = app_inst.array(x2, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=2)
    x2_sba = SparseBlockArray.from_ba(x2_ba, fill_value=2)
    y_sp = x1_sp + x2_sp
    y_sba = x1_sba + x2_sba
    assert y_sba.fill_value == y_sp.fill_value  # 4
    print(y_sp.fill_value)
    assert y_sba.nnz == y_sp.nnz  # 16
    print(y_sp.nnz)
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 + x2, y_ba.get())

    y_ba = x1_sba + x2_ba
    assert np.array_equal(x1_sp + x2, y_ba.get())


def test_sparse_mul(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    x1_sp = sparse.GCXS.from_numpy(x1, fill_value=2)
    x2_sp = sparse.GCXS.from_numpy(x2, fill_value=2)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x2_ba = app_inst.array(x2, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=2)
    x2_sba = SparseBlockArray.from_ba(x2_ba, fill_value=2)
    y_sp = x1_sp * x2_sp
    y_sba = x1_sba * x2_sba
    assert y_sba.fill_value == y_sp.fill_value  # 4
    print(y_sp.fill_value)
    assert y_sba.nnz == y_sp.nnz  # 16
    print(y_sp.nnz)
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 * x2, y_ba.get())

    # Sparse-dense
    rs: NumsRandomState = app_inst.random_state(1337)
    x1_sba = rs.sparse_randint(1, high=5, dtype=int, shape=(100, 50), block_shape=(5, 5), p=0.1, fill_value=0)
    x2_sba = rs.sparse_randint(1, high=5, dtype=int, shape=(100, 50), block_shape=(5, 5), p=0.1, fill_value=0)
    x1_ba = x1_sba.to_ba()
    x2_ba = x2_sba.to_ba()
    x1 = x1_ba.get()
    x2 = x2_ba.get()
    x1_sp = sparse.GCXS.from_numpy(x1, fill_value=0)
    x2_sp = sparse.GCXS.from_numpy(x2, fill_value=0)
    y_sp = x1_sp * x2
    y_sba = x1_sba * x2_ba
    assert y_sba.fill_value == y_sp.fill_value
    assert y_sba.nnz == y_sp.nnz
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 * x2, y_ba.get())


if __name__ == "__main__":
    # pylint disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_sparse_init(app_inst)
    test_sparse_random(app_inst)
    test_sparse_uop(app_inst)
    test_sparse_add(app_inst)
