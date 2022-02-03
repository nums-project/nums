import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.storage.storage import BimodalGaussian


def test_log(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(10, 2))
    assert np.allclose(app_inst.log(X).get(), np.log(real_X))


def test_stats(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(3, 2)
    X = app_inst.array(real_X, block_shape=(2, 1))
    assert np.allclose(app_inst.mean(X, axis=0).get(), np.mean(real_X, axis=0))
    assert np.allclose(app_inst.std(X, axis=1).get(), np.std(real_X, axis=1))

    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(10, 2))
    assert np.allclose(app_inst.mean(X, axis=0).get(), np.mean(real_X, axis=0))
    assert np.allclose(app_inst.std(X, axis=1).get(), np.std(real_X, axis=1))


def test_sum(app_inst: ArrayApplication):
    shape = (5, 6, 7)
    real_X = np.random.random_sample(np.product(shape)).reshape(shape)
    X = app_inst.array(real_X, block_shape=(2, 1, 4))
    assert np.allclose(app_inst.sum(X, axis=1).get(), np.sum(real_X, axis=1))


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_log(app_inst)
    test_stats(app_inst)
    test_sum(app_inst)
