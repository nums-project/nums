import numpy as np

from nums.numpy import BlockArray


def test_basic(nps_app_inst):
    import nums.numpy as nps
    app = nps_app_inst

    x_api = nps.random.RandomState(1337).random_sample((500, 1))
    x_app = app.random_state(1337).random(shape=x_api.shape, block_shape=x_api.block_shape)
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).rand(500, 1)
    x_app = app.random_state(1337).random(shape=x_api.shape, block_shape=x_api.block_shape)
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).randn(500, 1) + 5.0
    x_app = app.random_state(1337).normal(loc=5.0, shape=x_api.shape, block_shape=x_api.block_shape)
    assert nps.allclose(x_api, x_app)

    x_api = nps.random.RandomState(1337).randint(0, 10, size=(100, 1))
    x_app = app.random_state(1337).integers(0, 10, shape=x_api.shape, block_shape=x_api.block_shape)
    assert nps.allclose(x_api, x_app)


def test_shuffle(nps_app_inst):
    import nums.numpy as nps
    shape = (12, 34, 56)
    block_shape = (2, 5, 7)
    arr: BlockArray = nps.arange(np.product(shape)).reshape(shape=shape, block_shape=block_shape)
    np_arr = arr.get()

    for axis in range(3):
        for axis_frac in (1.0, 0.5):
            rs = nps.random.RandomState(1337)
            idx: BlockArray = rs.permutation(int(shape[axis]*axis_frac))
            np_idx = idx.get()
            if axis == 0:
                arr_shuffle = arr[idx]
                np_arr_shuffle = np_arr[np_idx]
            else:
                arr_shuffle = arr._advanced_single_array_subscript((np_idx,), axis=axis)
                np_ss = [slice(None, None) for _ in range(3)]
                np_ss[axis] = np_idx
                np_ss = tuple(np_ss)
                np_arr_shuffle = np_arr[np_ss]
            assert np.all(np_arr_shuffle == arr_shuffle.get())


def test_shuffle_subscript_ops(nps_app_inst):
    import nums.numpy as nps
    shape = (123, 45)
    block_shape = (10, 20)
    arr: BlockArray = nps.arange(np.product(shape)).reshape(shape=shape, block_shape=block_shape)
    np_arr = arr.get()
    rs = nps.random.RandomState(1337)
    idx: BlockArray = rs.permutation(shape[1])
    np_idx = idx.get()
    arr_shuffle = arr[:, idx]
    np_arr_shuffle = np_arr[:, np_idx]
    assert np.all(np_arr_shuffle == arr_shuffle.get())


if __name__ == "__main__":
    import nums.core.settings
    nums.core.settings.system_name = "serial"
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_basic(nps_app_inst)
    test_shuffle(nps_app_inst)
    test_shuffle_subscript_ops(nps_app_inst)
