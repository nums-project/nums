# pylint: disable=import-outside-toplevel
def test_dask_system():
    import numpy as np
    from nums.core import linalg
    import nums.numpy as nps
    from nums.core import settings
    from nums.core.array.application import ArrayApplication
    from nums.core.array.blockarray import BlockArray
    from nums.core import application_manager
    from nums.experimental.nums_dask.dask_system import DaskSystem

    settings.system_name = "dask"
    assert not application_manager.is_initialized()
    app: ArrayApplication = application_manager.instance()

    assert isinstance(app.cm.system, DaskSystem)

    X: BlockArray = app.random.normal(shape=(3, 3), block_shape=(3, 3))
    Y: BlockArray = app.random.normal(shape=(3, 3), block_shape=(3, 3))
    Z: BlockArray = X @ Y
    assert np.allclose(Z.get(), X.get() @ Y.get())

    X: BlockArray = app.random.normal(shape=(10, 20), block_shape=(3, 5))
    Y: BlockArray = app.random.normal(shape=(20, 30), block_shape=(5, 6))
    Z: BlockArray = X @ Y
    assert np.allclose(Z.get(), X.get() @ Y.get())

    X: BlockArray = app.random.normal(shape=(100, 20), block_shape=(23, 5))
    Q, R = linalg.indirect_tsqr(app, X)
    assert nps.allclose(Q @ R, X)

    Q, R = linalg.direct_tsqr(app, X)
    assert nps.allclose(Q @ R, X)
    application_manager.destroy()


if __name__ == "__main__":
    test_dask_system()
