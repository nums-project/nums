import numpy as np
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparse import SparseBlockArray
from nums.core.array.random import NumsRandomState
from nums.core.grid.grid import ArrayGrid


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

def test_sparse_ufunc(app_inst: ArrayApplication):
    X = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X_ba = app_inst.array(X, block_shape=(2, 2))
    X_sp = SparseBlockArray.from_ba(X_ba)
    Y_sp = X_sp.ufunc("negative")
    print(type(X_sp), type(Y_sp))
    Y_ba = Y_sp.to_ba()
    assert np.array_equal(np.negative(X), Y_ba.get())


if __name__ == "__main__":
    # pylint disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_sparse_init(app_inst)
    test_sparse_random(app_inst)
    test_sparse_ufunc(app_inst)
