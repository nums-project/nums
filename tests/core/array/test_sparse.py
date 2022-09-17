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
    x_sba = SparseBlockArray.from_ba(x_ba, fill_value=0)
    assert x_sba.nnz == 8
    assert x_sba.nbytes == 8 * 4 + 2 * 8 * 8
    y_ba = x_sba.to_ba()
    assert np.array_equal(x1, y_ba.get())


def test_from_coo(app_inst: ArrayApplication):
    row_coords = [0, 1, 0, 1, 2, 2, 3, 3]
    col_coords = [3, 2, 2, 3, 0, 1, 0, 1]
    values     = [1, 1, 1, 1, 2, 2, 2, 2]
    x_sp = sparse.COO([row_coords, col_coords], values)
    x_de = x_sp.todense()
    x_sba = app_inst.array(x_sp, block_shape=(2, 2))
    assert x_sba.nnz == 8
    assert x_sba.nbytes == 8 * 4 + 2 * 8 * 8
    y_ba = x_sba.to_ba()
    assert np.array_equal(x_de, y_ba.get())


def test_sparse_random(app_inst: ArrayApplication):
    rs: NumsRandomState = app_inst.random_state(1337)
    x_sba = rs.sparse_randint(
        1, high=5, dtype=int, shape=(15, 10), block_shape=(5, 5), p=0.1, fill_value=0
    )
    x_ba = x_sba.to_ba()
    x_np = x_ba.get()
    x_sp = sparse.COO.from_numpy(x_np, fill_value=0)
    assert x_sba.nnz == x_sp.nnz
    assert x_sba.nbytes == x_sp.nbytes
    print(x_np)


def test_sparse_uop(app_inst: ArrayApplication):
    x = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x_sp = sparse.COO.from_numpy(x, fill_value=1)
    x_sp = sparse.elemwise(np.negative, x_sp)
    x_ba = app_inst.array(x, block_shape=(2, 2))
    x_sba = SparseBlockArray.from_ba(x_ba, fill_value=1)
    y_sba = -x_sba
    assert y_sba.fill_value == x_sp.fill_value  # -1
    assert y_sba.nnz == x_sp.nnz  # 12
    y_ba = y_sba.to_ba()
    assert np.array_equal(np.negative(x), y_ba.get())


def test_sparse_add(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x2 = x1 * 2
    x1_sp = sparse.COO.from_numpy(x1, fill_value=2)
    x2_sp = sparse.COO.from_numpy(x2, fill_value=2)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x2_ba = app_inst.array(x2, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=2)
    x2_sba = SparseBlockArray.from_ba(x2_ba, fill_value=2)
    y_sp = x1_sp + x2_sp
    y_sba = x1_sba + x2_sba
    assert y_sba.fill_value == y_sp.fill_value  # 4
    assert y_sba.nnz == y_sp.nnz  # 16
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 + x2, y_ba.get())

    # Test sparse-dense.
    y_ba = x2_ba + x1_sba  # __radd__
    assert np.array_equal(x2 + x1_sp, y_ba.get())

    # Test sparse-scalar.
    y_sp = x1_sp - 1
    y_sba = x1_sba - 1
    assert y_sba.fill_value == y_sp.fill_value  # 4
    assert y_sba.nnz == y_sp.nnz  # 16
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 - 1, y_ba.get())


def test_sparse_mul(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    x1_sp = sparse.COO.from_numpy(x1, fill_value=2)
    x2_sp = sparse.COO.from_numpy(x2, fill_value=2)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x2_ba = app_inst.array(x2, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=2)
    x2_sba = SparseBlockArray.from_ba(x2_ba, fill_value=2)
    y_sp = x1_sp * x2_sp
    y_sba = x1_sba * x2_sba
    assert y_sba.fill_value == y_sp.fill_value  # 4
    assert y_sba.nnz == y_sp.nnz  # 16
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 * x2, y_ba.get())

    # Test sparse-dense.
    rs: NumsRandomState = app_inst.random_state(1337)
    x1_sba = rs.sparse_randint(
        1, high=5, dtype=int, shape=(100, 50), block_shape=(5, 5), p=0.1, fill_value=0
    )
    x2_sba = rs.sparse_randint(
        1, high=5, dtype=int, shape=(1, 50), block_shape=(5, 5), p=0.1, fill_value=0
    )
    x1_ba = x1_sba.to_ba()
    x2_ba = x2_sba.to_ba()
    x1 = x1_ba.get()
    x2 = x2_ba.get()
    x1_sp = sparse.COO.from_numpy(x1, fill_value=0)
    x2_sp = sparse.COO.from_numpy(x2, fill_value=0)
    y_sp = x1_sp * x2
    y_sba = x2_ba * x1_sba  # __rmul__
    assert y_sba.fill_value == y_sp.fill_value
    assert y_sba.nnz == y_sp.nnz
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 * x2, y_ba.get())


def test_neq(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x1_sp = sparse.COO.from_numpy(x1, fill_value=2)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=2)
    y_sp = x1_sp > 0
    y_sba = x1_sba > 0
    assert y_sba.fill_value == y_sp.fill_value  # True
    assert y_sba.nnz == y_sp.nnz  # 8 (nnz changes!)
    y_ba = y_sba.to_ba()
    assert np.array_equal(x1 > 0, y_ba.get())


def test_tensordot(app_inst: ArrayApplication):
    x1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    x2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    x1_sp = sparse.COO.from_numpy(x1, fill_value=0)
    x2_sp = sparse.COO.from_numpy(x2, fill_value=0)
    x1_ba = app_inst.array(x1, block_shape=(2, 2))
    x2_ba = app_inst.array(x2, block_shape=(2, 2))
    x1_sba = SparseBlockArray.from_ba(x1_ba, fill_value=0)
    x2_sba = SparseBlockArray.from_ba(x2_ba, fill_value=0)
    y_sp = sparse.tensordot(x1_sp, x2_sp, axes=1)
    y_sba = x1_sba.tensordot(x2_sba, axes=1)
    assert y_sba.fill_value == y_sp.fill_value  # 0
    assert y_sba.nnz == y_sp.nnz  #
    y_ba = y_sba.to_ba()
    assert np.array_equal(np.tensordot(x1, x2, axes=1), y_ba.get())


def test_sdtp(app_inst: ArrayApplication):
    shape = 50, 50, 50
    block_shape = 10, 10, 10
    s: SparseBlockArray = app_inst.random.sparse_uniform(
        shape=shape, block_shape=block_shape, p=0.001
    )
    vecs = [
        app_inst.random.random(shape=(n,), block_shape=(bn,))
        for n, bn in zip(*(shape, block_shape))
    ]

    r_true = s * app_inst.tensordot(
        app_inst.tensordot(vecs[0], vecs[1], axes=0), vecs[2], axes=0
    )
    r_sdtp = s.sdtp(*vecs)
    assert app_inst.allclose(r_true.todense(), r_sdtp.todense())


def test_sdtd(app_inst):
    shape = 12, 23, 34, 45
    block_shape = 6, 7, 8, 9
    s: SparseBlockArray = app_inst.random.sparse_uniform(
        shape=shape, block_shape=block_shape, p=0.001
    )

    sum_shape = 67, 89
    sum_block_shape = 67, 89
    x = app_inst.random.random(
        shape=shape[:2] + sum_shape, block_shape=block_shape[:2] + sum_block_shape
    )
    y = app_inst.random.random(
        shape=sum_shape + shape[2:], block_shape=sum_block_shape + block_shape[2:]
    )
    axes = len(sum_shape)
    r_true: SparseBlockArray = s * app_inst.tensordot(x, y, axes=axes)
    r_sdtd: SparseBlockArray = s.sdtd(x, y, axes)
    assert app_inst.allclose(r_true.todense(), r_sdtd.todense())


if __name__ == "__main__":
    # pylint disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_from_coo(app_inst)
    test_sparse_init(app_inst)
    test_sdtp(app_inst)
    # test_sdtd(app_inst)
    # test_sparse_init(app_inst)
    # test_sparse_random(app_inst)
    # test_sparse_uop(app_inst)
    # test_sparse_add(app_inst)
