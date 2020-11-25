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

import pytest
import numpy as np

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication
from nums.models.glms import LogisticRegression, LinearRegression, PoissonRegression


# pylint: disable = protected-access, import-outside-toplevel, import-error


def test_logistic(nps_app_inst: ArrayApplication):
    num_samples, num_features = 1000, 10
    real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = nps_app_inst.array(real_y, block_shape=(100,))
    param_set = [
        {"solver": "gd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        {"solver": "sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        {"solver": "block_sgd", "lr": 1e-6, "tol": 1e-8, "max_iter": 10},
        {"solver": "newton", "tol": 1e-8, "max_iter": 10},
        {"solver": "irls", "tol": 1e-8, "max_iter": 10}
    ]
    for kwargs in param_set:
        runtime = time.time()
        lr_model: LogisticRegression = LogisticRegression(**kwargs)
        lr_model.fit(X, y)
        runtime = time.time() - runtime
        y_pred = lr_model.predict(X).get()
        y_pred_proba = lr_model.predict_proba(X).get()
        np.allclose(np.ones(shape=(y.shape[0],)), y_pred_proba[:, 0] + y_pred_proba[:, 1])
        print("opt", kwargs["solver"])
        print("runtime", runtime)
        print("norm", lr_model.grad_norm_sq(X, y).get())
        print("objective", lr_model.objective(X, y).get())
        print("accuracy", np.sum(y.get() == y_pred)/num_samples)


def test_logistic_cv(nps_app_inst: ArrayApplication):
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
    X = nps_app_inst.array(real_X, block_shape=block_shape)
    y = nps_app_inst.array(real_y, block_shape=(block_shape[0],))
    param_set = [
        {"solver": "newton", "tol": 1e-8, "max_iter": 10},
        {"solver": "newton", "penalty": "l2", "C": 1.0/0.1, "tol": 1e-8, "max_iter": 10},
        {"solver": "newton", "penalty": "l2", "C": 1.0/0.2, "tol": 1e-8, "max_iter": 10},
        {"solver": "newton", "penalty": "l2", "C": 1.0/0.4, "tol": 1e-8, "max_iter": 10},
        {"solver": "newton", "penalty": "l2", "C": 1.0/0.8, "tol": 1e-8, "max_iter": 10},
    ]
    X_train = nps_app_inst.empty((num_samples - X.block_shape[0], num_features), X.block_shape,
                                 X.dtype)
    y_train = nps_app_inst.empty((num_samples - y.block_shape[0],), y.block_shape, y.dtype)
    num_hps = len(param_set)
    mean_accuracies = nps_app_inst.empty((num_hps,), (num_hps,))
    for i, kwargs in enumerate(param_set):
        accuracies = nps_app_inst.empty((folds,), (folds,))
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
            lr_model: LogisticRegression = LogisticRegression(**kwargs)
            lr_model.fit(X_train, y_train)
            y_pred = lr_model.predict(X_test)
            accuracies[fold] = nps_app_inst.sum(y_test == y_pred) / (stop - start)
        mean_accuracies[i] = nps_app_inst.mean(accuracies)
    print(mean_accuracies.get())


def test_lr(nps_app_inst: ArrayApplication):
    num_samples, num_features = 1000, 10
    rs = np.random.RandomState(1337)
    real_theta = rs.random_sample(num_features)
    real_X, real_y = BimodalGaussian.get_dataset(233, num_features, theta=real_theta)
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = nps_app_inst.array(real_y, block_shape=(100,))
    param_set = [
        {"solver": "gd", "lr": 1e-6, "tol": 1e-8, "max_iter": 100},
        {"solver": "newton", "tol": 1e-8, "max_iter": 10}
    ]
    for kwargs in param_set:
        runtime = time.time()
        model: LinearRegression = LinearRegression(**kwargs)
        model.fit(X, y)
        assert model._beta.shape == real_theta.shape and model._beta0.shape == ()
        runtime = time.time() - runtime
        y_pred = model.predict(X).get()
        print("opt", kwargs["solver"])
        print("runtime", runtime)
        print("norm", model.grad_norm_sq(X, y).get())
        print("objective", model.objective(X, y).get())
        print("error", np.sum((y.get() - y_pred)**2)/num_samples)
        print("D^2", model.deviance_sqr(X, y).get())


def test_poisson_basic(nps_app_inst: ArrayApplication):
    coef = np.array([0.2, -0.1])
    X_real = np.array([[0, 1, 2, 3, 4]]).T
    y_real = np.exp(np.dot(X_real, coef[0]) + coef[1]).reshape(-1)
    X = nps_app_inst.array(X_real, block_shape=X_real.shape)
    y = nps_app_inst.array(y_real, block_shape=y_real.shape)
    model: PoissonRegression = PoissonRegression(**{"solver": "newton",
                                                    "tol": 1e-8,
                                                    "max_iter": 10})
    model.fit(X, y)
    print("norm", model.grad_norm_sq(X, y).get())
    print("objective", model.objective(X, y).get())
    print("D^2", model.deviance_sqr(X, y).get())
    assert nps_app_inst.allclose(model._beta,
                                 nps_app_inst.array(coef[:-1], block_shape=(1,)), rtol=1e-4).get()
    assert nps_app_inst.allclose(model._beta0,
                                 nps_app_inst.scalar(coef[-1]), rtol=1e-4).get()


def test_poisson(nps_app_inst: ArrayApplication):
    # TODO (hme): Is there a more appropriate distribution for testing Poisson?
    num_samples, num_features = 1000, 1
    rs = np.random.RandomState(1337)
    real_beta = rs.random_sample(num_features)
    real_model: PoissonRegression = PoissonRegression(solver="newton")
    real_model._beta = nps_app_inst.array(real_beta, block_shape=(3,))
    real_model._beta0 = nps_app_inst.scalar(rs.random_sample())
    real_X = rs.random_sample(size=(num_samples, num_features))
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = real_model.predict(X)
    param_set = [
        {"solver": "newton", "tol": 1e-8, "max_iter": 10}
    ]
    for kwargs in param_set:
        runtime = time.time()
        model: PoissonRegression = PoissonRegression(**kwargs)
        model.fit(X, y)
        runtime = time.time() - runtime
        print("opt", kwargs["solver"])
        print("runtime", runtime)
        print("norm", model.grad_norm_sq(X, y).get())
        print("objective", model.objective(X, y).get())
        print("D^2", model.deviance_sqr(X, y).get())
        assert nps_app_inst.allclose(real_model._beta, model._beta).get()
        assert nps_app_inst.allclose(real_model._beta0, model._beta0).get()


@pytest.mark.skip
def test_sklearn_linear_regression(nps_app_inst: ArrayApplication):
    from sklearn.linear_model import LinearRegression as SKLinearRegression

    _, num_features = 1000, 10
    rs = np.random.RandomState(1337)
    real_theta = rs.random_sample(num_features)
    real_X, real_y = BimodalGaussian.get_dataset(233, num_features, theta=real_theta)
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = nps_app_inst.array(real_y, block_shape=(100,))
    param_set = [
        {"solver": "newton-cg", "tol": 1e-8, "max_iter": 10},
    ]
    for kwargs in param_set:
        lr_model: LinearRegression = LinearRegression(**kwargs)
        lr_model.fit(X, y)
        y_pred = lr_model.predict(X).get()

        sk_lr_model = SKLinearRegression()
        sk_lr_model.fit(real_X, real_y)
        sk_y_pred = sk_lr_model.predict(real_X)
        np.allclose(sk_y_pred, y_pred)


@pytest.mark.skip
def test_sklearn_logistic_regression(nps_app_inst: ArrayApplication):
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
    num_samples, num_features = 1000, 10
    real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = nps_app_inst.array(real_X, block_shape=(100, 3))
    y = nps_app_inst.array(real_y, block_shape=(100,))
    param_set = [
        {"solver": "newton-cg", "tol": 1e-8, "max_iter": 10},
    ]
    for kwargs in param_set:
        runtime = time.time()
        lr_model: LogisticRegression = LogisticRegression(**kwargs)
        lr_model.fit(X, y)
        runtime = time.time() - runtime
        y_pred = lr_model.predict(X).get()
        y_pred_proba = lr_model.predict_proba(X).get()
        np.allclose(np.ones(shape=(y.shape[0],)), y_pred_proba[:, 0] + y_pred_proba[:, 1])

        sk_lr_model = SKLogisticRegression(**kwargs)
        sk_lr_model.fit(real_X, real_y)
        sk_y_pred = sk_lr_model.predict(real_X)
        sk_y_pred_proba = sk_lr_model.predict_proba(real_X)
        np.allclose(np.ones(shape=(y.shape[0],)), sk_y_pred_proba[:, 0] + sk_y_pred_proba[:, 1])
        np.allclose(sk_y_pred, y_pred)


@pytest.mark.skip
def test_sklearn_poisson_regression(nps_app_inst: ArrayApplication):
    def dsqr(dev_func, y, _y_pred):
        dev = dev_func(y, _y_pred)
        y_mean = nps_app_inst.mean(y)
        dev_null = dev_func(y, y_mean)
        return 1 - dev / dev_null

    from sklearn.linear_model import PoissonRegressor as SKPoissonRegressor
    coef = np.array([0.2, -0.1])
    real_X = np.array([[0, 1, 2, 3, 4]]).T
    real_y = np.exp(np.dot(real_X, coef[0]) + coef[1]).reshape(-1)
    X = nps_app_inst.array(real_X, block_shape=real_X.shape)
    y = nps_app_inst.array(real_y, block_shape=real_y.shape)
    param_set = [
        {"tol": 1e-4, "max_iter": 100},
    ]
    for kwargs in param_set:
        lr_model: PoissonRegression = PoissonRegression(**kwargs)
        lr_model.fit(X, y)
        y_pred = lr_model.predict(X).get()
        print("D^2", dsqr(lr_model.deviance, y, y_pred).get())

        sk_lr_model = SKPoissonRegressor(**kwargs)
        sk_lr_model.fit(real_X, real_y)
        sk_y_pred = sk_lr_model.predict(real_X)
        print("D^2", dsqr(lr_model.deviance, y, sk_y_pred).get())


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    # test_logistic(nps_app_inst)
    # test_logistic_cv(nps_app_inst)
    # test_lr(nps_app_inst)
    # test_poisson_basic(nps_app_inst)
    # test_poisson(nps_app_inst)
    test_sklearn_linear_regression(nps_app_inst)
    test_sklearn_logistic_regression(nps_app_inst)
    test_sklearn_poisson_regression(nps_app_inst)
