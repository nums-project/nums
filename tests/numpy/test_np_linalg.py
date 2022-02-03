# pylint: disable=import-outside-toplevel


def test_basic(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    x = nps.random.RandomState(1337).random_sample((100, 10))
    q, r = nps.linalg.qr(x)
    assert nps.allclose(x, q @ r)


if __name__ == "__main__":
    from nums.core import application_manager

    nps_app_inst = application_manager.instance()
    test_basic(nps_app_inst)
