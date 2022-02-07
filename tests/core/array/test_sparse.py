import numpy as np
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparse import SparseBlockArray
from nums.core.grid.grid import ArrayGrid


def test_sparse_init(app_inst: ArrayApplication):
    X_np = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])
    X_ba = app_inst.array(X_np, block_shape=(2, 2))
    X_sp = SparseBlockArray.from_ba(X_ba)
    assert len(X_sp.index.blocks) == 2
    Y_ba = X_sp.to_ba()
    assert np.array_equal(X_np, Y_ba.get())

    X_sp = SparseBlockArray.from_ba(X_ba, fill_value=1)
    assert len(X_sp.index.blocks) == 3
    Y_ba = X_sp.to_ba()
    assert np.array_equal(X_np, Y_ba.get())


if __name__ == "__main__":
    # pylint disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_sparse_init(app_inst)
