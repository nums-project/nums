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


import numpy as np

from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication

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

    def __init__(self, app_inst: ArrayApplication, opt: str, opt_params: dict):
        self._app = app_inst
        self._opt = opt
        self._opt_params = opt_params
        self._l2 = self._opt_params.get("l2", None)
        self._l2_vec = None
        self._beta = None
        self._beta0 = None

    def fit(self, X: BlockArray, y: BlockArray):
        # Note, it's critically important from a performance point-of-view
        # to maintain the original block shape of X below, along axis 1.
        # Otherwise, the concatenation operation will not construct the new X
        # by referencing X's existing blocks.
        X = self._app.concatenate([X, self._app.ones(shape=(X.shape[0], 1),
                                                     block_shape=(X.block_shape[0], 1),
                                                     dtype=X.dtype)],
                                  axis=1,
                                  axis_block_size=X.block_shape[1])
        assert len(X.shape) == 2 and len(y.shape) == 1
        beta: BlockArray = self._app.zeros((X.shape[1],), (X.block_shape[1],), dtype=X.dtype)
        tol: BlockArray = self._app.scalar(self._opt_params.get("tol", 1e-8))
        max_iter: int = self._opt_params.get("max_iter", 10)
        if self._l2 is not None:
            self._l2_vec = self._app.ones(beta.shape, beta.block_shape, beta.dtype) * self._l2
        if self._opt == "gd" or self._opt == "block_sync_sgd" or self._opt == "block_async_sgd":
            assert "lr" in self._opt_params
            lr: BlockArray = self._app.scalar(self._opt_params["lr"])
            if self._opt == "gd":
                beta = gd(self, beta, X, y, tol, max_iter, lr)
            elif self._opt == "block_sync_sgd":
                beta = block_sync_sgd(self, beta, X, y, tol, max_iter, lr)
            else:
                beta = block_async_sgd(self, beta, X, y, tol, max_iter, lr)
        elif self._opt == "newton":
            beta = newton(self._app, self, beta, X, y, tol, max_iter)
        elif self._opt == "irls":
            # TODO: Provide irls for all GLMs.
            assert isinstance(self, LogisticRegression)
            beta = irls(self._app, self, beta, X, y, tol, max_iter)
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
        return self.forward(X)

    def link_inv(self, eta: BlockArray):
        raise NotImplementedError()

    def objective(self, X: BlockArray, y: BlockArray, beta=None):
        raise NotImplementedError()

    def gradient(self, X: BlockArray, y: BlockArray,
                 mu: BlockArray = None, beta: BlockArray = None):
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

    def objective(self, X: BlockArray, y: BlockArray, beta=None):
        assert beta is not None or self._beta is not None
        mu = self.forward(X, beta)
        return self._app.sum((y-mu)**self._app.two)

    def gradient(self, X: BlockArray, y: BlockArray,
                 mu: BlockArray = None, beta: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        return X.T @ (mu - y)

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        return X.T @ X

    def deviance(self, y, y_pred):
        return self._app.sum((y - y_pred) ** self._app.two)


class LogisticRegression(GLM):

    def link_inv(self, eta: BlockArray):
        return self._app.one / (self._app.one + self._app.exp(-eta))

    def objective(self, X: BlockArray, y: BlockArray, beta=None):
        assert beta is not None or self._beta is not None
        log, one = self._app.log, self._app.one
        mu = self.forward(X, beta)
        return - self._app.sum(y * log(mu) + (one - y) * log(one - mu))

    def gradient(self, X: BlockArray, y: BlockArray,
                 mu: BlockArray = None, beta: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        if self._l2 is None:
            return X.T @ (mu - y)
        else:
            assert beta is not None
            return X.T @ (mu - y) + self._l2 * beta

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        s = (mu * (self._app.one - mu)).reshape(shape=(-1, 1), block_shape=(-1, 1))
        if self._l2 is None:
            return X.T @ (s * X)
        else:
            return X.T @ (s * X) + self._l2_vec

    def deviance(self, y, y_pred):
        raise NotImplementedError()


class PoissonRegression(GLM):

    def link_inv(self, eta: BlockArray):
        return self._app.exp(eta)

    def objective(self, X: BlockArray, y: BlockArray, beta=None):
        if beta is None:
            eta = X @ self._beta + self._beta0
        else:
            eta = X @ beta
        mu = self._app.exp(eta)
        return self._app.sum(mu - y * eta)

    def gradient(self, X: BlockArray, y: BlockArray,
                 mu: BlockArray = None, beta: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        return X.T @ (mu - y)

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        if mu is None:
            mu = self.forward(X)
        # TODO: This is sub-optimal as it forces the computation of X.T.
        return (X.T * mu) @ X

    def deviance(self, y, y_pred):
        return self._app.sum(self._app.two * self._app.xlogy(y, y / y_pred) - y + y_pred)


class ExponentialRegression(GLM):
    # canonical parameter: theta = - lambda
    # b(theta) = -log(-theta)
    # b'(theta) = -1/theta
    # canonical link: g(mu) = theta = eta = X @ beta
    # inverse canonical link: mu = b'(theta) = -1/theta = -1/eta

    def link_inv(self, eta: BlockArray):
        raise NotImplementedError()

    def objective(self, X: BlockArray, y: BlockArray, beta=None):
        raise NotImplementedError()

    def gradient(self, X: BlockArray, y: BlockArray,
                 mu: BlockArray = None, beta: BlockArray = None):
        raise NotImplementedError()

    def hessian(self, X: BlockArray, y: BlockArray, mu: BlockArray = None):
        raise NotImplementedError()


def line_search():
    pass


def block_sync_sgd(model: GLM, beta,
                   X: BlockArray, y: BlockArray,
                   tol: BlockArray, max_iter: int, lr: BlockArray):
    for _ in range(max_iter):
        for (start, stop) in X.grid.grid_slices[0]:
            X_batch, y_batch = X[start:stop], y[start:stop]
            mu = model.forward(X_batch, beta)
            g = model.gradient(X_batch, y_batch, mu, beta=beta)
            beta += - lr * g
            if g.T @ g < tol:
                break
    return beta


def block_async_sgd(model: GLM, beta,
                    X: BlockArray, y: BlockArray,
                    tol: BlockArray, max_iter: int, lr: BlockArray):
    max_staleness = 5
    staleness = 0
    for _ in range(max_iter):
        beta0 = beta
        randomized_slices = np.random.permutation(X.grid.grid_slices[0])
        for (start, stop) in randomized_slices:
            X_batch, y_batch = X[start:stop], y[start:stop]
            mu = model.forward(X_batch, beta0)
            g = model.gradient(X_batch, y_batch, mu, beta=beta)
            # Computing on the same beta enables parallelization of this inner loop,
            # achieving a controlled "async" sgd.
            # With an array-based wait operation, we could update beta0 as
            # computations of betas complete.
            beta += - lr * g
            if g.T @ g < tol:
                break
            staleness += 1
            if staleness >= max_staleness:
                # Force use of current beta once max staleness is reached.
                staleness = 0
                beta0 = beta
    return beta


def gd(model: GLM, beta,
       X: BlockArray, y: BlockArray,
       tol: BlockArray, max_iter: int, lr: BlockArray):
    for _ in range(max_iter):
        mu = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)
        beta += - lr * g
        if g.T @ g < tol:
            break
    return beta


def newton(app: ArrayApplication, model: GLM, beta,
           X: BlockArray, y: BlockArray,
           tol: BlockArray, max_iter: int):
    for _ in range(max_iter):
        mu: BlockArray = model.forward(X, beta)
        g = model.gradient(X, y, mu, beta=beta)
        # These are PSD, but inv is faster than psd inv.
        beta += - app.inv(model.hessian(X, y, mu)) @ g
        if g.T @ g < tol:
            break
    return beta


def irls(app: ArrayApplication, model: LogisticRegression, beta,
         X: BlockArray, y: BlockArray,
         tol: BlockArray, max_iter: int):
    for _ in range(max_iter):
        eta: BlockArray = X @ beta
        mu: BlockArray = model.link_inv(eta)
        s = mu * (1 - mu) + 1e-16
        XT_s = (X.T * s)
        # These are PSD, but inv is faster than psd inv.
        XTsX_inv = app.inv(XT_s @ X)
        z = eta + (y-mu)/s
        beta = XTsX_inv @ XT_s @ z
        gnorm = model.grad_norm_sq(X, y, beta)
        if gnorm < tol:
            break
    return beta


def lbfgs():
    pass
