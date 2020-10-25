import numpy as np

from nums.numpy import BlockArray
from nums.core.storage.storage import ArrayGrid


def test_basic_creation(nps_app_inst):
    import nums.numpy as nps
    ops = "empty", "zeros", "ones"
    for op in ops:
        ba: BlockArray = nps.__getattribute__(op)(shape=(100, 200, 3000))
        grid: ArrayGrid = ba.grid
        assert grid.grid_shape[0] <= grid.grid_shape[1] < grid.grid_shape[2]


def test_eye(nps_app_inst):
    import nums.numpy as nps
    eyes = [
        (10, 10),
        (7, 10),
        (10, 13),
    ]
    for N, M in eyes:
        ba: BlockArray = nps.eye(N, M)
        np_arr = np.eye(N, M)
        assert np.allclose(ba.get(), np_arr)
        # Also test identity.
        ba: BlockArray = nps.identity(N)
        np_arr = np.identity(N)
        assert np.allclose(ba.get(), np_arr)


def test_diag(nps_app_inst):
    import nums.numpy as nps
    ba: BlockArray = nps.array([1.0, 2.0, 3.0])
    np_arr = ba.get()
    # Make a diag matrix.
    ba = nps.diag(ba)
    np_arr = np.diag(np_arr)
    assert np.allclose(ba.get(), np_arr)
    # Take diag of diag matrix.
    ba = nps.diag(ba)
    np_arr = np.diag(np_arr)
    assert np.allclose(ba.get(), np_arr)


def test_arange(nps_app_inst):
    import nums.numpy as nps
    ba: BlockArray = nps.arange(5)
    np_arr = np.arange(5)
    assert np.allclose(ba.get(), np_arr)


def test_concatenate(nps_app_inst):
    import nums.numpy as nps
    ba1: BlockArray = nps.arange(5)
    ba2: BlockArray = nps.arange(6)
    ba = nps.concatenate((ba1, ba2))
    np_arr = np.concatenate((np.arange(5), np.arange(6)))
    assert np.allclose(ba.get(), np_arr)


def test_split(nps_app_inst):
    import nums.numpy as nps
    ba: BlockArray = nps.arange(10)
    np_arr = np.arange(10)
    ba_list = nps.split(ba, 2)
    np_arr_list = np.split(np_arr, 2)
    for i in range(len(np_arr_list)):
        assert np.allclose(ba_list[i].get(), np_arr_list[i])


def test_func_space(nps_app_inst):
    import nums.numpy as nps
    ba: BlockArray = nps.linspace(12.3, 45.6, 23).reshape(block_shape=(10,))
    np_arr = np.linspace(12.3, 45.6, 23)
    assert np.allclose(ba.get(), np_arr)
    ba: BlockArray = nps.logspace(12.3, 45.6, 23).reshape(block_shape=(10,))
    np_arr = np.logspace(12.3, 45.6, 23)
    assert np.allclose(ba.get(), np_arr)


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_basic_creation(nps_app_inst)
    test_eye(nps_app_inst)
    test_diag(nps_app_inst)
    test_arange(nps_app_inst)
    test_concatenate(nps_app_inst)
    test_split(nps_app_inst)
    test_func_space(nps_app_inst)
