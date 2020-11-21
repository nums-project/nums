# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools

import tqdm
import numpy as np
import pytest

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication

# pylint: disable=wrong-import-order
import common


def test_matmul(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(100, 1))
    X_sqr = X.T @ X
    assert np.allclose(X_sqr.get(), real_X.T @ real_X)


def test_matvec(app_inst: ArrayApplication):
    X = app_inst.array(np.arange(200).reshape(2, 100), block_shape=(1, 10))
    y1 = app_inst.array(np.arange(100).reshape(100, 1), block_shape=(10, 1))
    assert np.allclose((X @ y1).get(), X.get() @ y1.get())
    y2 = app_inst.array(np.arange(100).reshape(100), block_shape=(10,))
    assert np.allclose((X @ y2).get(), X.get() @ y2.get())
    # This won't trigger optimized routine, but it is rare and strange so not worth addressing.
    # TODO (hme): Apparently this is invalid. Figure out why.
    # y3 = app_inst.array(np.arange(100).reshape((100, 1, 1)), block_shape=(10, 1, 1))
    # assert np.allclose((X @ y3).get(), X.get() @ y3.get())


def test_vecdot(app_inst: ArrayApplication):
    size = 9
    block_size = 3
    y1 = app_inst.array(np.arange(size).reshape(size, 1), block_shape=(block_size, 1))
    y2 = app_inst.array(np.arange(size).reshape(size, 1), block_shape=(block_size, 1))
    assert np.allclose((y1.T @ y2).get(), y1.T.get() @ y2.get())
    y1 = app_inst.array(np.arange(size).reshape(size), block_shape=(block_size,))
    y2 = app_inst.array(np.arange(size).reshape(size), block_shape=(block_size,))
    assert np.allclose((y1.T @ y2).get(), y1.T.get() @ y2.get())
    y1 = app_inst.array(np.arange(size).reshape(size), block_shape=(block_size,))
    y2 = app_inst.array(np.arange(size).reshape(size, 1), block_shape=(block_size, 1))
    assert np.allclose((y1.T @ y2).get(), y1.T.get() @ y2.get())
    assert np.allclose((y2.T @ y1).get(), y2.T.get() @ y1.get())
    y1 = app_inst.array(np.arange(size).reshape(1, size), block_shape=(1, block_size))
    y2 = app_inst.array(np.arange(size).reshape(size, 1), block_shape=(block_size, 1))
    assert np.allclose((y1 @ y2).get(), y1.get() @ y2.get())
    y1 = app_inst.array(np.arange(size).reshape(1, size), block_shape=(1, block_size))
    y2 = app_inst.array(np.arange(size).reshape(1, size), block_shape=(1, block_size))
    assert np.allclose((y1 @ y2.T).get(), y1.get() @ y2.T.get())


def test_tensordot_basic(app_inst: ArrayApplication):
    shape = 2, 4, 10, 15
    npX = np.arange(np.product(shape)).reshape(*shape)
    rX = app_inst.array(npX, block_shape=(1, 2, 10, 3))

    rResult = rX.T.tensordot(rX, axes=1)
    assert np.allclose(
        rResult.get(),
        (np.tensordot(npX.T, npX, axes=1))
    )
    common.check_block_integrity(rResult)


def test_tensordot_large_shape(app_inst: ArrayApplication):
    a = np.arange(4 * 6 * 10 * 90).reshape((90, 10, 6, 4))
    b = np.arange(4 * 6 * 10 * 75).reshape((4, 6, 10, 75))
    c = np.tensordot(a, b, axes=1)

    block_a = app_inst.array(a, block_shape=(30, 5, 3, 2))
    block_b = app_inst.array(b, block_shape=(2, 3, 5, 25))
    block_c = block_a.tensordot(block_b, axes=1)
    assert np.allclose(block_c.get(), c)
    common.check_block_integrity(block_c)


@pytest.mark.skip
def test_tensordot_all_shapes(app_inst: ArrayApplication):
    for axes in [0, 1, 2]:
        if axes == 2:
            a = np.arange(7 * 6 * 4).reshape((7, 6, 4))
            b = np.arange(6 * 4 * 9).reshape((6, 4, 9))
            c = np.tensordot(a, b, axes=axes)
        elif axes in (1, 0):
            a = np.arange(7 * 6 * 4).reshape((7, 6, 4))
            b = np.arange(6 * 4 * 9).reshape((4, 6, 9))
            c = np.tensordot(a, b, axes=axes)
        else:
            raise Exception()
        a_block_shapes = list(itertools.product(*list(map(lambda x: list(range(1, x + 1)),
                                                          a.shape))))
        b_block_shapes = list(itertools.product(*list(map(lambda x: list(range(1, x + 1)),
                                                          b.shape))))
        pbar = tqdm.tqdm(total=np.product([len(a_block_shapes), len(b_block_shapes)]))
        for a_block_shape in a_block_shapes:
            for b_block_shape in b_block_shapes:
                pbar.update(1)
                if a_block_shape[-axes:] != b_block_shape[:axes]:
                    continue
                pbar.set_description("axes=%s %s @ %s" % (str(axes),
                                                          str(a_block_shape),
                                                          str(b_block_shape)))
                block_a = app_inst.array(a, block_shape=a_block_shape)
                block_b = app_inst.array(b, block_shape=b_block_shape)
                block_c = block_a.tensordot(block_b, axes=axes)
                assert np.allclose(block_c.get(), c)
                common.check_block_integrity(block_c)


@pytest.fixture(scope="module", params=["same_dim", "broadcasting", "scalars"])
def bop_data(request, app_inst):

    def same_dim():
        X_shape = 6, 8
        Y_shape = 6, 8
        npX = np.random.random_sample(np.product(X_shape)).reshape(*X_shape)
        X = app_inst.array(npX, block_shape=(3, 2))
        npY = np.random.random_sample(np.product(Y_shape)).reshape(*Y_shape)
        Y = app_inst.array(npY, block_shape=(3, 2))
        return X, Y, npX, npY

    def broadcasting():
        X_shape = 6, 1
        Y_shape = 8,
        npX = np.random.random_sample(np.product(X_shape)).reshape(*X_shape)
        X = app_inst.array(npX, block_shape=(3, 1))
        npY = np.random.random_sample(np.product(Y_shape)).reshape(*Y_shape)
        Y = app_inst.array(npY, block_shape=(2,))
        return X, Y, npX, npY

    def scalars():
        X_shape = 6, 8
        npX = np.random.random_sample(np.product(X_shape)).reshape(*X_shape)
        X = app_inst.array(npX, block_shape=(3, 2))
        npY = .5
        Y = app_inst.scalar(npY)
        return X, Y, npX, npY

    return {
        "same_dim": same_dim,
        "broadcasting": broadcasting,
        "scalars": scalars,
    }[request.param]()


def test_bops(bop_data: tuple):
    X, Y, npX, npY = bop_data

    # Add
    Z = X + Y
    npZ = npX + npY
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)

    # Subtract
    Z = X - Y
    npZ = npX - npY
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)

    # Multiply
    Z = X * Y
    npZ = npX * npY
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)

    # Divide
    Z = X / Y
    npZ = npX / npY
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)

    # Power
    Z = X ** Y
    npZ = npX ** npY
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)


@pytest.fixture(scope="module", params=["scalar", "list", "ndarray"])
def conversions_data(request, app_inst):
    X_shape = 6, 6
    npX = np.random.random_sample(np.product(X_shape)).reshape(*X_shape)
    X = app_inst.array(npX, block_shape=(3, 3))

    if request.param == "scalar":
        Y = 1.23
    elif request.param == "list":
        Y = list(range(6))
    elif request.param == "ndarray":
        Y = np.arange(6)
    else:
        raise Exception("impossible.")

    return X, npX, Y


def test_conversions(conversions_data: tuple):
    X, npX, Y = conversions_data

    # Add
    Z = X + Y
    npZ = npX + Y
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)
    if isinstance(Y, np.ndarray):
        with pytest.raises(ValueError):
            Z = Y + X
    else:
        Z = Y + X
        npZ = Y + npX
        assert np.allclose(Z.get(), npZ)
        common.check_block_integrity(Z)

    # Subtract
    Z = X - Y
    npZ = npX - Y
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)
    if isinstance(Y, np.ndarray):
        with pytest.raises(ValueError):
            Z = Y - X
    else:
        Z = Y - X
        npZ = Y - npX
        assert np.allclose(Z.get(), npZ)
        common.check_block_integrity(Z)

    # Multiply
    Z = X * Y
    npZ = npX * Y
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)
    if isinstance(Y, np.ndarray):
        with pytest.raises(ValueError):
            Z = Y * X
    else:
        Z = Y * X
        npZ = Y * npX
        assert np.allclose(Z.get(), npZ)
        common.check_block_integrity(Z)

    # Divide
    Z = X / Y
    npZ = npX / Y
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)
    if isinstance(Y, np.ndarray):
        with pytest.raises(ValueError):
            Z = Y / X
    else:
        Z = Y / X
        npZ = Y / npX
        assert np.allclose(Z.get(), npZ)
        common.check_block_integrity(Z)

    # Power
    Z = X ** Y
    npZ = npX ** Y
    assert np.allclose(Z.get(), npZ)
    common.check_block_integrity(Z)
    if isinstance(Y, np.ndarray):
        with pytest.raises(ValueError):
            Z = Y ** X
    else:
        Z = Y ** X
        npZ = Y ** npX
        assert np.allclose(Z.get(), npZ)
        common.check_block_integrity(Z)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_tensordot_large_shape(app_inst)
    # test_matvec(app_inst)
    # test_vecdot(app_inst)
    # test_conversions(conversions_data(None, app_inst))
