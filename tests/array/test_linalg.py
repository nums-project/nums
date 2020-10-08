# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time

import numpy as np
import scipy.linalg
from scipy.linalg import lapack

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication
from nums.core.models import LogisticRegression, LinearRegression, PoissonRegression


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
        mat_inv = app_inst.inv_sym_psd(mat).get()
        assert np.allclose(np.linalg.inv(mat.get()), mat_inv, rtol=1e-4, atol=1e-4)
        _, r = np.linalg.qr(mat.get())
        r_inv = app_inst.inverse_triangular(app_inst.array(r, block_shape=shape), lower=False).get()
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


def test_logistic(app_inst: ArrayApplication):
    num_samples, num_features = 1000, 10
    real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = app_inst.array(real_X, block_shape=(100, 3))
    y = app_inst.array(real_y, block_shape=(100,))
    opt_param_set = [
        ("gd", {"lr": 1e-6, "tol": 1e-8, "max_iter": 10}),
        ("block_sync_sgd", {"lr": 1e-6, "tol": 1e-8, "max_iter": 10}),
        ("block_async_sgd", {"lr": 1e-6, "tol": 1e-8, "max_iter": 10}),
        ("newton", {"tol": 1e-8, "max_iter": 10}),
        ("irls", {"tol": 1e-8, "max_iter": 10})
    ]
    for opt, opt_params in opt_param_set:
        runtime = time.time()
        lr_model: LogisticRegression = LogisticRegression(app_inst, opt, opt_params)
        lr_model.fit(X, y)
        runtime = time.time() - runtime
        y_pred = (lr_model.predict(X).get() > 0.5).astype(int)
        print("opt", opt)
        print("runtime", runtime)
        print("norm", lr_model.grad_norm_sq(X, y).get())
        print("objective", lr_model.objective(X, y).get())
        print("accuracy", np.sum(y.get() == y_pred)/num_samples)


def test_logistic_cv(app_inst: ArrayApplication):
    num_samples, num_features = 1000, 10
    num_bad = 100
    block_shape = (200, 10)
    folds = num_samples // block_shape[0]
    rs = np.random.RandomState(1337)

    real_X, real_y = BimodalGaussian.get_dataset(num_samples-num_bad, num_features, p=0.5)
    extra_X, extra_y = BimodalGaussian.get_dataset(num_bad, num_features, p=0.5)

    # Perturb some examples.
    extra_X = extra_X * rs.random_sample(np.product(extra_X.shape)).reshape(extra_X.shape)
    extra_y = rs.randint(0, 2, extra_y.shape).reshape(extra_y.shape)
    perm = rs.permutation(np.arange(num_samples))
    real_X = np.concatenate([real_X, extra_X], axis=0)[perm]
    real_y = np.concatenate([real_y, extra_y], axis=0)[perm]

    # real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = app_inst.array(real_X, block_shape=block_shape)
    y = app_inst.array(real_y, block_shape=(block_shape[0],))
    opt_param_set = [
        ("newton", {"l2": None, "tol": 1e-8, "max_iter": 10}),
        ("newton", {"l2": 0.1, "tol": 1e-8, "max_iter": 10}),
        ("newton", {"l2": 0.2, "tol": 1e-8, "max_iter": 10}),
        ("newton", {"l2": 0.4, "tol": 1e-8, "max_iter": 10}),
        ("newton", {"l2": 0.8, "tol": 1e-8, "max_iter": 10}),
    ]
    X_train = app_inst.empty((num_samples - X.block_shape[0], num_features), X.block_shape, X.dtype)
    y_train = app_inst.empty((num_samples - y.block_shape[0],), y.block_shape, y.dtype)
    num_hps = len(opt_param_set)
    mean_accuracies = app_inst.empty((num_hps,), (num_hps,))
    for i, (opt, opt_params) in enumerate(opt_param_set):
        accuracies = app_inst.empty((folds,), (folds,))
        for fold in range(folds):
            print(i, fold)
            pos = X.block_shape[0]*fold
            block_size, _ = X.grid.get_block_shape((fold, 0))
            start = pos
            stop = pos + block_size
            X_train[:start] = X[:start]
            X_train[start:] = X[stop:]
            y_train[:start] = y[:start]
            y_train[start:] = y[stop:]
            X_test, y_test = X[start:stop], y[start:stop]
            lr_model: LogisticRegression = LogisticRegression(app_inst, opt, opt_params)
            lr_model.fit(X_train, y_train)
            y_pred = lr_model.predict(X_test) > 0.5
            accuracies[fold] = app_inst.sum(y_test == y_pred) / (stop-start)
        mean_accuracies[i] = app_inst.mean(accuracies)
    print(mean_accuracies.get())


def test_glm_lr(app_inst: ArrayApplication):
    num_samples, num_features = 1000, 10
    num_features = 13
    rs = np.random.RandomState(1337)
    real_theta = rs.random_sample(num_features)
    real_X, real_y = BimodalGaussian.get_dataset(233, num_features, theta=real_theta)
    X = app_inst.array(real_X, block_shape=(100, 3))
    y = app_inst.array(real_y, block_shape=(100,))
    opt_param_set = [
        ("gd", {"lr": 1e-6, "tol": 1e-8, "max_iter": 100}),
        ("newton", {"tol": 1e-8, "max_iter": 10})
    ]
    for opt, opt_params in opt_param_set:
        runtime = time.time()
        model: LinearRegression = LinearRegression(app_inst, opt, opt_params)
        model.fit(X, y)
        assert model._beta.shape == real_theta.shape and model._beta0.shape == ()
        runtime = time.time() - runtime
        y_pred = model.predict(X).get()
        print("opt", opt)
        print("runtime", runtime)
        print("norm", model.grad_norm_sq(X, y).get())
        print("objective", model.objective(X, y).get())
        print("error", np.sum((y.get() - y_pred)**2)/num_samples)
        print("D^2", model.deviance_sqr(X, y))


def test_poisson_basic(app_inst):
    coef = np.array([0.2, -0.1])
    X_real = np.array([[0, 1, 2, 3, 4]]).T
    y_real = np.exp(np.dot(X_real, coef[0]) + coef[1]).reshape(-1)
    X = app_inst.array(X_real, block_shape=X_real.shape)
    y = app_inst.array(y_real, block_shape=y_real.shape)
    model: PoissonRegression = PoissonRegression(app_inst, "newton", {"tol": 1e-8, "max_iter": 10})
    model.fit(X, y)
    print("norm", model.grad_norm_sq(X, y).get())
    print("objective", model.objective(X, y).get())
    print("D^2", model.deviance_sqr(X, y).get())
    assert app_inst.allclose(model._beta,
                             app_inst.array(coef[:-1], block_shape=(1,)), rtol=1e-4).get()
    assert app_inst.allclose(model._beta0,
                             app_inst.scalar(coef[-1]), rtol=1e-4).get()


def test_poisson(app_inst: ArrayApplication):
    # TODO (hme): Is there a more appropriate distribution for testing Poisson?
    num_samples, num_features = 1000, 1
    rs = np.random.RandomState(1337)
    real_beta = rs.random_sample(num_features)
    real_model: PoissonRegression = PoissonRegression(app_inst, "newton", {})
    real_model._beta = app_inst.array(real_beta, block_shape=(3,))
    real_model._beta0 = app_inst.scalar(rs.random_sample())
    real_X = rs.random_sample(size=(num_samples, num_features))
    X = app_inst.array(real_X, block_shape=(100, 3))
    y = real_model.predict(X)
    opt_param_set = [
        ("newton", {"tol": 1e-8, "max_iter": 10})
    ]
    for opt, opt_params in opt_param_set:
        runtime = time.time()
        model: PoissonRegression = PoissonRegression(app_inst, opt, opt_params)
        model.fit(X, y)
        runtime = time.time() - runtime
        print("opt", opt)
        print("runtime", runtime)
        print("norm", model.grad_norm_sq(X, y).get())
        print("objective", model.objective(X, y).get())
        print("D^2", model.deviance_sqr(X, y))
        assert app_inst.allclose(real_model._beta, model._beta).get()
        assert app_inst.allclose(real_model._beta0, model._beta0).get()


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_inv_assumptions(app_inst)
    test_inv(app_inst)
    # test_qr(app_inst)
    # test_svd(app_inst)
    # test_lr(app_inst)
    # test_rr(app_inst)
    # test_logistic(app_inst)
    # test_logistic_cv(app_inst)
    # test_glm_lr(app_inst)
    # test_poisson_basic(app_inst)
    # test_poisson(app_inst)
