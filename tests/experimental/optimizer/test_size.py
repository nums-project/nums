import numpy as np
import sparse

from nums.experimental.optimizer.size import TreeNodeSize


def test_nbytes():
    x1 = np.eye(10)
    x1_sp = sparse.COO.from_numpy(x1, fill_value=0)
    x1_ts = TreeNodeSize(
        False,
        (10, 10),
        10,
        np.int64,
        0,
    )
    assert x1_sp.nbytes == x1_ts.nbytes


def test_add():
    x1_ts = TreeNodeSize(
        False,
        (10, 10),
        10,
        np.int64,
        0,
    )
    x2_ts = TreeNodeSize(
        False,
        (10, 10),
        20,
        np.int64,
        0,
    )
    y_ts = x1_ts + x2_ts
    assert np.allclose(y_ts.nnz, int((1 - 0.9 * 0.8) * 100))


def test_mul():
    x1_ts = TreeNodeSize(
        False,
        (10, 10),
        10,
        np.int64,
        0,
    )
    x2_ts = TreeNodeSize(
        False,
        (10, 10),
        20,
        np.int64,
        0,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 0.2 * 100))

    x2_ts = TreeNodeSize(
        False,
        (10, 1),
        10,
        np.int64,
        1,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 100))

    x2_ts = TreeNodeSize(
        True,
        (10, 1),
        10,
        np.int64,
        None,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 100))


def test_tensordot():
    x1_ts = TreeNodeSize(
        False,
        (10, 10),
        10,
        np.int64,
        0,
    )
    x2_ts = TreeNodeSize(
        False,
        (10, 10),
        20,
        np.int64,
        0,
    )
    y_ts = x1_ts.tensordot(x2_ts, axes=1)
    assert np.allclose(y_ts.nnz, int((1 - (1 - 0.1 * 0.2) ** 10) * 100))
