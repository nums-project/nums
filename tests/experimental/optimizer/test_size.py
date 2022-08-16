import numpy as np
import sparse

from nums.experimental.optimizer.size import TreeNodeSize


def test_nbytes():
    x1 = np.eye(10)
    x1_sp = sparse.COO.from_numpy(x1, fill_value=0)
    x1_ts = TreeNodeSize(
        (10, 10),
        10,
        np.int64,
        False,
    )
    assert x1_sp.nbytes == x1_ts.nbytes


def test_uop():
    x1_ts = TreeNodeSize(
        (10, 10),
        10,
        np.int64,
        False,
    )
    y_ts = x1_ts.uop("negative")
    assert np.allclose(y_ts.nnz, x1_ts.nnz)
    assert not y_ts.is_dense


def test_add():
    x1_ts = TreeNodeSize(
        (10, 10),
        10,
        np.int64,
        False,
    )
    x2_ts = TreeNodeSize(
        (10, 10),
        20,
        np.int64,
        False,
    )
    y_ts = x1_ts + x2_ts
    assert np.allclose(y_ts.nnz, int((1 - 0.9 * 0.8) * 100))
    assert not y_ts.is_dense


def test_mul():
    x1_ts = TreeNodeSize(
        (10, 10),
        10,
        np.int64,
        False,
    )
    x2_ts = TreeNodeSize(
        (10, 10),
        20,
        np.int64,
        False,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 0.2 * 100))

    x2_ts = TreeNodeSize(
        (10, 1),
        10,
        np.int64,
        False,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 100))

    x2_ts = TreeNodeSize(
        (10, 1),
        10,
        np.int64,
        True,
    )
    y_ts = x1_ts * x2_ts
    assert np.allclose(y_ts.nnz, int(0.1 * 100))
    assert not y_ts.is_dense


def test_tensordot():
    x1_ts = TreeNodeSize(
        (10, 10),
        10,
        np.int64,
        False,
    )
    x2_ts = TreeNodeSize(
        (10, 10),
        20,
        np.int64,
        False,
    )
    y_ts = x1_ts.tensordot(x2_ts, axes=1)
    assert np.allclose(y_ts.nnz, int((1 - (1 - 0.1 * 0.2) ** 10) * 100))
