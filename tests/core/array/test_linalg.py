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


import time

import numpy as np
import scipy.linalg
from scipy.linalg import lapack

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication


# pylint: disable=protected-access


def sample_sym_psd_mat(shape):
    np_X = np.random.random_sample(np.product(shape)).reshape(shape)
    # Check PSD assumptions.
    res = np_X.T @ np_X
    w, _ = np.linalg.eigh(res)
    assert np.all(w >= 0)
    return res


def sample_sym_pd_mat(shape):
    np_X = sample_sym_psd_mat(shape)
    w, v = np.linalg.eigh(np_X)
    # Check spectral decomposition assumptions.
    assert np.allclose(((v * w) @ np.linalg.inv(v)), np_X)
    w += 1.0
    return (v * w) @ np.linalg.inv(v)


def test_inv_assumptions(app_inst: ArrayApplication):
    # pylint: disable=no-member, unused-variable
    np_Z = sample_sym_pd_mat(shape=(10, 10))

    # Compute the inverse of np_Z using sym_psd routine.
    Z = app_inst.array(np_Z, np_Z.shape)
    Z_inv = app_inst.inv(Z).get()
    Z_true_inv = np.linalg.inv(np_Z)
    assert np.allclose(Z_true_inv, Z_inv)

    # Try Cholesky approach.
    np_L = np.linalg.cholesky(np_Z)
    np_L_inv = np.linalg.inv(np_L)
    Z_cho_inv = np_L_inv.T @ np_L_inv
    assert np.allclose(Z_cho_inv, Z_true_inv)

    # Test backsub.
    assert np_L.dtype == np.float64
    lp_L_inv, _ = lapack.dtrtri(np_L, lower=1, unitdiag=0, overwrite_c=0)
    assert np.allclose(np_L_inv, lp_L_inv)

    # Test overwrite.
    overwrite_L_inv = np_L.copy(order="F")
    overwrite_L_inv_res, info = lapack.dtrtri(overwrite_L_inv, lower=1, unitdiag=0, overwrite_c=1)
    assert np.allclose(overwrite_L_inv_res, overwrite_L_inv)
    assert np.allclose(np_L_inv, overwrite_L_inv)

    # This should copy.
    overwrite_L_inv = np_L.copy(order="C")
    overwrite_L_inv_res, info = lapack.dtrtri(overwrite_L_inv, lower=1, unitdiag=0, overwrite_c=1)
    assert not np.allclose(overwrite_L_inv_res, overwrite_L_inv)

    # scipy cholesky tests.
    scipy_L_inv, info = lapack.dtrtri(scipy.linalg.cholesky(np.asfortranarray(np_Z),
                                                            lower=True,
                                                            overwrite_a=True,
                                                            check_finite=False),
                                      lower=1,
                                      unitdiag=0,
                                      overwrite_c=1)
    assert np.allclose(scipy_L_inv, np_L_inv)

    # Benchmark test.
    np_Z = sample_sym_pd_mat((1500, 1500))
    scipy_runtime = time.time()
    scipy_L_inv, info = lapack.dtrtri(scipy.linalg.cholesky(np.asfortranarray(np_Z),
                                                            lower=True,
                                                            overwrite_a=True,
                                                            check_finite=False),
                                      lower=1,
                                      unitdiag=0,
                                      overwrite_c=1)
    scipy_Z_inv = scipy_L_inv.T @ scipy_L_inv
    scipy_runtime = time.time() - scipy_runtime

    np_runtime = time.time()
    np_Z_inv = np.linalg.inv(np_Z)
    np_runtime = time.time() - np_runtime
    assert scipy_runtime < np_runtime


def test_inv(app_inst: ArrayApplication):
    shape = (5, 5)
    for dtype in (np.float32, np.float64):
        mat = app_inst.array(sample_sym_pd_mat(shape=shape).astype(dtype), block_shape=shape)
        _, r = np.linalg.qr(mat.get())
        r_inv = app_inst.inv(app_inst.array(r, block_shape=shape)).get()
        assert np.allclose(np.linalg.inv(r), r_inv, rtol=1e-4, atol=1e-4)
        L = app_inst.cholesky(mat).get()
        assert np.allclose(np.linalg.cholesky(mat.get()), L, rtol=1e-4, atol=1e-4)


def test_qr(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(2345, 9)
    X = app_inst.array(real_X, block_shape=(123, 4))
    Q, R = app_inst.indirect_tsqr(X)
    assert np.allclose(Q.get() @ R.get(), real_X)
    Q, R = app_inst.direct_tsqr(X)
    assert np.allclose(Q.get() @ R.get(), real_X)


def test_svd(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(2345, 9)
    X = app_inst.array(real_X, block_shape=(123, 4))
    U, S, VT = app_inst.svd(X)
    assert np.allclose((U.get() * S.get()) @ VT.get(), real_X)


def test_lr(app_inst: ArrayApplication):
    num_features = 13
    rs = np.random.RandomState(1337)
    for dtype in (np.float32, np.float64):
        real_theta = rs.random_sample(num_features).astype(dtype)
        real_X, real_y = BimodalGaussian.get_dataset(233, num_features, theta=real_theta)
        real_X = real_X.astype(dtype)
        real_y = real_y.astype(dtype)
        X = app_inst.array(real_X, block_shape=(15, 5))
        y = app_inst.array(real_y, block_shape=(15,))

        # Direct TSQR LR
        theta = app_inst.linear_regression(X, y)
        error = app_inst.sum((((X @ theta) - y)**2)).get()
        if dtype == np.float64:
            assert np.allclose(0, error), error
        else:
            # Need to account for lower precision.
            assert np.allclose(0, error, rtol=1.e-4, atol=1.e-4), error

        # Fast LR
        theta = app_inst.fast_linear_regression(X, y)
        error = app_inst.sum((((X @ theta) - y)**2)).get()
        if dtype == np.float64:
            assert np.allclose(0, error), error
        else:
            # Need to account for lower precision.
            assert np.allclose(0, error, rtol=1.e-4, atol=1.e-4), error


def test_rr(app_inst: ArrayApplication):
    num_features = 13
    rs = np.random.RandomState(1337)
    real_theta = rs.random_sample(num_features)
    real_X, real_y = BimodalGaussian.get_dataset(100, num_features, p=0.5, theta=real_theta)
    extra_X, extra_y = BimodalGaussian.get_dataset(10, num_features, p=0.5, theta=real_theta)

    # Perturb some examples.
    extra_X = extra_X * rs.random_sample(np.product(extra_X.shape)).reshape(extra_X.shape)
    extra_y = extra_y * rs.random_sample(extra_y.shape).reshape(extra_y.shape)
    real_X = np.concatenate([real_X, extra_X], axis=0)
    real_y = np.concatenate([real_y, extra_y], axis=0)

    X = app_inst.array(real_X, block_shape=(15, 5))
    y = app_inst.array(real_y, block_shape=(15,))
    theta = app_inst.ridge_regression(X, y, lamb=0.0)
    robust_theta = app_inst.ridge_regression(X, y, lamb=10000.0)

    # Generate a test set to evaluate robustness to outliers.
    test_X, test_y = BimodalGaussian.get_dataset(100, num_features, p=0.5, theta=real_theta)
    test_X = app_inst.array(test_X, block_shape=(15, 5))
    test_y = app_inst.array(test_y, block_shape=(15,))
    theta_error = np.sum((((test_X @ theta) - test_y)**2).get())
    robust_theta_error = np.sum((((test_X @ robust_theta) - test_y)**2).get())
    assert robust_theta_error < theta_error


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    # test_inv_assumptions(app_inst)
    test_inv(app_inst)
    # test_qr(app_inst)
    # test_svd(app_inst)
    # test_lr(app_inst)
    # test_rr(app_inst)
