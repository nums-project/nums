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


import numpy as np

from nums.core.application_manager import instance as _instance
from nums.core.array import utils as array_utils
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.random import NumsRandomState
from nums.core import linalg

# The GLMs are expressed in the following notation.
# f(y) = exp((y.T @ theta - b(theta))/phi + c(y, phi))
# phi is the dispersion parameter.
# theta is the parameter of a model in canonical form.
# b is the cumulant generating function.
#
# The link function is expressed as follows.
# E(Y | X) = mu
# Define the linear predictor eta = X.T @ beta
# Define g as the link function, so that g(mu) = eta
# E(Y | X) = g^{-1}(eta)
#
# The canonical link is given by g(mu) = (b')^{-1}(mu) = theta
#
# Note, for GLMs, the mean mu is some function of the model's parameter.
# Normal: mu(mu) = mu
# Bernoulli: mu(p) = p
# Exponential: mu(lambda) = 1 / lambda
# Poisson: mu(lambda) = lambda
# Dirichlet: mu_i(a) = a_i / sum(a)
#
# Theta is generally a function of the model parameter:
# Normal: theta(mu) = mu
# Bernoulli: theta(p) = ln(p/(1-p))
# Exponential: theta(lambda) = -lambda
# Poisson: theta(lambda) = ln(lambda)
# ...
#
# The canonical link maps mu to theta
# Normal: mu(mu) = mu, theta(mu) = mu, b(theta) = theta^2/2, g(mu) = mu
# Bernoulli:
#   mu(p) = p, p(mu) = mu
#   theta(p) = ln(p/(1-p)) = theta(mu) = ln(mu/(1-mu))
#   b(theta) = log(1 + exp(theta))
#   g(mu) = (b')^{-1}(mu) = ln(mu/(1-mu)) = ln(p/(1-p)) = theta(p)


class GLM(object):
    def __init__(
        self,
        penalty="none",
        C=1.0,
        tol=0.0001,
        max_iter=100,
        solver="newton-cg",
        lr=0.01,
        random_state=None,
        fit_intercept=True,
        normalize=False,
    ):

        if fit_intercept is False:
            raise NotImplementedError("fit_incercept=False currently not supported.")
        if normalize is True:
            raise NotImplementedError("normalize=True currently not supported.")

        self._app = _instance()
        if random_state is None:
            self.rs: NumsRandomState = self._app.random
        elif array_utils.is_int(random_state):
            self.rs: NumsRandomState = NumsRandomState(
                cm=self._app.cm, seed=random_state
            )
        elif isinstance(random_state, NumsRandomState):
            self.rs: NumsRandomState = random_state
        else:
            raise Exception(
                "Unexpected type for random_state %s" % str(type(random_state))
            )
        self._penalty = None if penalty == "none" else penalty
        if not (self._penalty is None or self._penalty == "l2"):
            raise NotImplementedError("%s penalty not supported" % self._penalty)
        self._lambda = 1.0 / C
        self._lambda_vec = None
        self._tol = tol
        self._max_iter = max_iter
        self._opt = solver
        self._lr = lr
        self._beta = None
        self._beta0 = None

    def fit(self, X: BlockArray, y: BlockArray):
        # Note, it's critically important from a performance point-of-view
        # to maintain the original block shape of X below, along axis 1.
        # Otherwise, the concatenation operation will not construct the new X
        # by referencing X's existing blocks.
        # TODO: Option to do concat.
        # TODO: Provide support for batching.
        if np.issubdtype(X.dtype, np.integer):
            X = X.astype(float)
        X = self._app.concatenate(
            [
                X,
                self._app.ones(
                    shape=(X.shape[0], 1),
                    block_shape=(X.block_shape[0], 1),
                    dtype=X.dtype,
                ),
            ],
            axis=1,
            axis_block_size=X.block_shape[1],
        )
        assert len(X.shape) == 2 and len(y.shape) == 1
        beta: BlockArray = self._app.zeros(
            (X.shape[1],), (X.block_shape[1],), dtype=X.dtype
        )
        tol: BlockArray = self._app.scalar(self._tol)
        max_iter: int = self._max_iter
        if self._penalty == "l2":
            self._lambda_vec = (
                self._app.ones(beta.shape, beta.block_shape, beta.dtype) * self._lambda
            )
        if self._opt == "gd" or self._opt == "sgd" or self._opt == "block_sgd":
            lr: BlockArray = self._app.scalar(self._lr)
            if self._opt == "gd":
                beta = gd(self, beta, X, y, tol, max_iter, lr)
            elif self._opt == "sgd":
                beta = sgd(self, beta, X, y, tol, max_iter, lr)
            else:
                beta = block_sgd(self, beta, X, y, tol, max_iter, lr)
        elif self._opt == "newton" or self._opt == "newton-cg":
            beta = newton(self._app, self, beta, X, y, tol, max_iter)
        elif self._opt == "irls":
            # TODO (hme): Provide irls for all GLMs.
            assert isinstance(self, LogisticRegression)
            beta = irls(self._app, self, beta, X, y, tol, max_iter)
        elif self._opt == "lbfgs":
            beta = lbfgs(self._app, self, beta, X, y, tol, max_iter)
        else:
            raise Exception("Unsupported optimizer specified %s." % self._opt)
        self._beta0 = beta[-1]
        self._beta = beta[:-1]

    def forward(self, X, beta=None):
        if beta:
            return self.link_inv(X @ beta)
        return self.link_inv(self._beta0 + X @ self._beta)

    def grad_norm_sq(self, X: BlockArray, y: BlockArray, beta=None):
        g = self.gradient(X, y, self.forward(X, beta), beta=beta)
        return g.T @ g

    def predict(self, X):
        raise NotImplementedError()

    def link_inv(self, eta: BlockArray):
        raise NotImplementedError()

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        raise NotImplementedError()

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        # gradient w.r.t. beta.
        raise NotImplementedError()

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        # Hessian w.r.t. beta.
        raise NotImplementedError()

    def deviance(self, y, y_pred):
        raise NotImplementedError()

    def deviance_sqr(self, X, y):
        y_pred = self.predict(X)
        dev = self.deviance(y, y_pred)
        y_mean = self._app.mean(y)
        dev_null = self.deviance(y, y_mean)
        return 1 - dev / dev_null


class LinearRegression(GLM):

    # Assume Sigma = I
    # canonical parameter: theta = mu
    # b(theta) = theta^2/2
    # b'(theta) = theta
    # canonical link: g(mu) = mu = theta = eta = X @ beta
    # inverse canonical link: mu = b(theta) = b(eta) = eta

    def link_inv(self, eta: BlockArray):
        return eta

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        assert beta is not None or self._beta is not None
        mu = self.forward(X, beta) if mu is None else mu
        return self._app.sum((y - mu) ** self._app.two)

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        if mu is None:
            mu = self.forward(X)
        return X.T @ (mu - y)

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        return X.T @ X

    def deviance(self, y, y_pred):
        return self._app.sum((y - y_pred) ** self._app.two)

    def predict(self, X: BlockArray) -> BlockArray:
        return self.forward(X)


class LogisticRegression(GLM):
    def link_inv(self, eta: BlockArray):
        return self._app.one / (self._app.one + self._app.exp(-eta))

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        assert beta is not None or self._beta is not None
        log, one = self._app.log, self._app.one
        mu = self.forward(X, beta) if mu is None else mu
        return -self._app.sum(y * log(mu) + (one - y) * log(one - mu))

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        if mu is None:
            mu = self.forward(X)
        if self._penalty is None:
            return X.T @ (mu - y)
        else:
            assert beta is not None
            return X.T @ (mu - y) + self._lambda_vec * beta

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        dim, block_dim = mu.shape[0], mu.block_shape[0]
        s = (mu * (self._app.one - mu)).reshape((dim, 1), block_shape=(block_dim, 1))
        if self._penalty is None:
            return X.T @ (s * X)
        else:
            # TODO (hme): Construct diag of _lambda_vec once.
            return X.T @ (s * X) + self._app.diag(self._lambda_vec)

    def deviance(self, y, y_pred):
        raise NotImplementedError()

    def predict(self, X: BlockArray) -> BlockArray:
        return (self.forward(X) > 0.5).astype(np.int)

    def predict_proba(self, X: BlockArray) -> BlockArray:
        y_pos = self.forward(X).reshape(
            (X.shape[0], 1), block_shape=(X.block_shape[0], 1)
        )
        y_neg = 1 - y_pos
        return self._app.concatenate([y_pos, y_neg], axis=1, axis_block_size=2)


class PoissonRegression(GLM):
    def link_inv(self, eta: BlockArray):
        return self._app.exp(eta)

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        if beta is None:
            eta = X @ self._beta + self._beta0
        else:
            eta = X @ beta
        mu = self._app.exp(eta) if mu is None else mu
        return self._app.sum(mu - y * eta)

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        if mu is None:
            mu = self.forward(X)
        return X.T @ (mu - y)

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        # TODO (hme): This is sub-optimal as it forces the computation of X.T.
        return (X.T * mu) @ X

    def deviance(self, y: BlockArray, y_pred: BlockArray) -> BlockArray:
        return self._app.sum(
            self._app.two * self._app.xlogy(y, y / y_pred) - y + y_pred
        )

    def predict(self, X: BlockArray) -> BlockArray:
        return self.forward(X)


class ExponentialRegression(GLM):
    # canonical parameter: theta = - lambda
    # b(theta) = -log(-theta)
    # b'(theta) = -1/theta
    # canonical link: g(mu) = theta = eta = X @ beta
    # inverse canonical link: mu = b'(theta) = -1/theta = -1/eta

    def link_inv(self, eta: BlockArray):
        raise NotImplementedError()

    def objective(
        self,
        X: BlockArray,
        y: BlockArray,
        beta: BlockArray = None,
        mu: BlockArray = None,
    ):
        raise NotImplementedError()

    def gradient(
        self,
        X: BlockArray,
        y: BlockArray,
        mu: BlockArray = None,
        beta: BlockArray = None,
    ):
        raise NotImplementedError()

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        raise NotImplementedError()


# Scikit-Learn aliases.
PoissonRegressor = PoissonRegression


def sgd(
    model: GLM,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    # Classic SGD.
    app = _instance()
    for _ in range(max_iter):
        # Sample an entry uniformly at random.
        idx = model.rs.numpy().integers(X.shape[0])
        X_sample, y_sample = X[idx : idx + 1], y[idx : idx + 1]
        mu = model.forward(X_sample, beta)
        g = model.gradient(X_sample, y_sample, mu, beta=beta)
        beta += -lr * g
        if app.max(app.abs(g)) <= tol:
            # sklearn uses max instead of l2 norm.
            break
    return beta


def block_sgd(
    model: GLM,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    # SGD with batches equal to block shape along first axis.
    app = _instance()
    for _ in range(max_iter):
        for (start, stop) in X.grid.grid_slices[0]:
            X_batch, y_batch = X[start:stop], y[start:stop]
            mu = model.forward(X_batch, beta)
            g = model.gradient(X_batch, y_batch, mu, beta=beta)
            beta += -lr * g
            if app.max(app.abs(g)) <= tol:
                break
    return beta


def gd(
    model: GLM,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
    lr: BlockArray,
):
    app = _instance()
    for _ in range(max_iter):
        mu = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)
        beta += -lr * g
        if app.max(app.abs(g)) <= tol:
            break
    return beta


def newton(
    app: ArrayApplication,
    model: GLM,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
):
    for _ in range(max_iter):
        mu: BlockArray = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)
        # These are PSD, but inv is faster than psd inv.
        beta += -linalg.inv(app, model.hessian(X, y, mu)) @ g
        if app.max(app.abs(g)) <= tol:
            break
    return beta


def irls(
    app: ArrayApplication,
    model: LogisticRegression,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
):
    for _ in range(max_iter):
        eta: BlockArray = X @ beta
        mu: BlockArray = model.link_inv(eta)
        s = mu * (1 - mu) + 1e-16
        XT_s = X.T * s
        # These are PSD, but inv is faster than psd inv.
        XTsX_inv = linalg.inv(app, XT_s @ X)
        z = eta + (y - mu) / s
        beta = XTsX_inv @ XT_s @ z
        g = model.gradient(X, y, mu, beta)
        if app.max(app.abs(g)) <= tol:
            break
    return beta


# pylint: disable = unused-argument
def lbfgs(
    app: ArrayApplication,
    model: GLM,
    beta,
    X: BlockArray,
    y: BlockArray,
    tol: BlockArray,
    max_iter: int,
):
    # TODO (hme): Enable a way to provide memory length and line search parameters.
    from nums.models.lbfgs import LBFGS  # pylint: disable = import-outside-toplevel

    lbfgs_optimizer = LBFGS(model, m=10, max_iter=max_iter, dtype=X.dtype, thresh=tol)
    return lbfgs_optimizer.execute(X, y, beta)


def admm():
    raise NotImplementedError()
