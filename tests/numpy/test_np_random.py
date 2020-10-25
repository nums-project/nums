from nums.numpy import BlockArray


def test_basic(nps_app_inst):
    import nums.numpy as nps
    app = nps.app

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


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_basic(nps_app_inst)
