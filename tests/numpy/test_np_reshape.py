import numpy as np


# pylint: disable=import-outside-toplevel


def test_reshape_int(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    shape = (3, 5, 10)
    arr = nps.arange(np.product(shape))
    np_arr = arr.get()
    assert np.allclose(np_arr.reshape(shape), arr.reshape(shape).get())
    assert np.allclose(
        np_arr.reshape(shape).reshape(-1), arr.reshape(shape).reshape(-1).get()
    )
    assert np.allclose(
        np_arr.reshape(shape).reshape(np.product(shape)),
        arr.reshape(shape).reshape(np.product(shape)).get(),
    )


def test_reshape_noops(nps_app_inst):
    shape, block_shape = (3, 5, 10), (3, 2, 5)
    arr = nps_app_inst.random_state(1337).random(shape, block_shape)
    new_arr = arr.reshape()
    assert arr is new_arr
    new_arr = arr.reshape(shape)
    assert arr is new_arr
    new_arr = arr.reshape(block_shape=block_shape)
    assert arr is new_arr
    new_arr = arr.reshape(shape, block_shape=block_shape)
    assert arr is new_arr


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_reshape_int(nps_app_inst)
    test_reshape_noops(nps_app_inst)
