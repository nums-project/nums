# originally from https://github.com/elibol/nums-1/blob/plan_support/nums/experimental/lbfgs.py
# modified for numerical stability and multiple classes

import time
from typing import List, Union

import numpy as np
import ray

from nums.core import settings
from nums.core.array.application import ArrayApplication
from nums.core.application_manager import instance
from nums import numpy as nps


def forward(app, X, theta):
    if X.shape[1] < theta.shape[0]:
        assert X.shape[1] + 1 == theta.shape[0]
        eta = theta[-1] + X @ theta[:-1]
    else:
        eta = X @ theta
    eta = eta - app.max(eta, axis=1).expand_dims(-1)
    unnormalized_probs = app.exp(eta)
    mu = unnormalized_probs / app.sum(unnormalized_probs, axis=1).expand_dims(-1)
    # print('mu', mu.get()[0])
    return mu # probabilities for each class


def objective(app: ArrayApplication, y, mu):
    # neg log likelihood of correct class. y is an array of onehots
    return - app.sum(y * app.log(mu + 1e-10))
    # one = app.one
    # return - app.sum(y * app.log(mu) + (one - y) * app.log(one - mu + 1e-14))


def grad(X, y, mu):
    return X.T @ (mu - y) # this is still correct


def bt_linesearch(app,
                  X, y, theta,
                  grad, p,
                  rho=1.e-1, init_alpha=1.0, c=1e-4, min_alpha=1e-10):

    def f(theta_prime):
        return objective(app, y, forward(app, X, theta_prime))

    alpha = init_alpha
    f_val = f(theta)
    f_next = f(theta + alpha * p)
    # print(f_val.get(), f_next.get())
    while nps.isnan(f_next) or f_next > f_val + c * alpha * app.sum(grad * p):
        alpha *= rho
        if alpha < min_alpha:
            return min_alpha
        # print("btls step alpha=%s" % alpha)
        f_next = f(theta + alpha * p)
    return alpha


class LBFGSMemory(object):

    def __init__(self, app, k, s, y):
        self.k = k
        self.s = s
        self.y = y
        # print('s', self.s.get())
        # print('y', self.y.get())
        ys_inner = app.sum(s * y)
        self.rho = 1.0 / ys_inner
        # print('rho', self.rho.get())
        self.gamma = ys_inner / (app.sum(y * y))
        # print('gamma', self.gamma.get())


class LBFGS(object):

    def __init__(self, app: ArrayApplication,
                 m=3, max_iter=100,  thresh=1e-5, dtype=np.float32):
        self.app: ArrayApplication = app
        self.m = m
        self.max_iter = max_iter
        self.thresh = thresh
        self.dtype = dtype
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None]*m

    def get_H(self):
        if self.k == 0:
            return self.identity
        else:
            mem: LBFGSMemory = self.memory[-1]
            assert mem.k == self.k-1
            return mem.gamma * self.identity

    def get_p(self, H, g):
        q = g
        forward_vars = []
        for i in range(-1, -self.m-1, -1):
            mem_i: LBFGSMemory = self.memory[i]
            if mem_i is None:
                break
            alpha = mem_i.rho * self.app.sum(mem_i.s * q)
            q -= alpha * mem_i.y
            forward_vars.insert(0, (alpha, mem_i))
        r = H @ q
        for alpha, mem_i in forward_vars:
            beta = mem_i.rho * self.app.sum(mem_i.y * r)
            r += mem_i.s * (alpha - beta)
        return r

    def execute(self, X, y, theta):

        if self.k != 0:
            raise Exception("Unexpected state.")

        self.identity = self.app.eye((X.shape[1], X.shape[1]),
                                     (X.block_shape[1], X.block_shape[1]),
                                     dtype=self.dtype)

        # TODO: Try sampling a new block every iteration.
        # TODO: Try stochastic approach, given line search...
        # X_btls = X[:X.block_shape[0]]
        # y_btls = y[:y.block_shape[0]]
        X_btls = X
        y_btls = y

        g = grad(X, y, forward(self.app, X, theta))
        next_g = None
        next_theta = None
        while self.k < self.max_iter:
            # print('iter', self.k)
            H = self.get_H()
            p = - self.get_p(H, g)
            init_alpha = 1

            alpha = bt_linesearch(self.app, X_btls, y_btls,
                                  theta, g, p,
                                  rho=0.1,
                                  init_alpha=init_alpha,
                                  c=1e-4,
                                  min_alpha=1e-10)
            # print("alpha", alpha)

            next_theta = theta + alpha * p
            if self.k + 1 >= self.max_iter:
                # Terminate immediately if this is the last iteration.
                theta = next_theta
                break
            next_g = grad(X, y, forward(self.app, X, next_theta))
            theta_diff = next_theta - theta
            grad_diff = next_g - g
            mem: LBFGSMemory = LBFGSMemory(self.app, k=self.k, s=theta_diff, y=grad_diff)
            self.memory.append(mem)
            self.memory.pop(0)
            self.k += 1
            theta = next_theta
            g = next_g
            # print('theta', theta.get())
            if self.app.max(self.app.abs(g)) <= self.thresh:
            # if self.converged(next_g):
                break

        # Reset vars.
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None]*self.m

        return theta

    def converged(self, g):
        return self.app.sqrt(self.app.sum(g * g)) < self.thresh


def logistic(app, X, y, max_iter, m):
    Xc = app.concatenate([X, app.ones(shape=(X.shape[0], 1),
                                      block_shape=(X.block_shape[0], 1),
                                      dtype=X.dtype)],
                         axis=1,
                         axis_block_size=X.block_shape[1])
    theta = app.zeros((Xc.shape[1],), (Xc.block_shape[1],), dtype=Xc.dtype)
    lbfgs_optimizer = LBFGS(app, m=m, max_iter=max_iter, dtype=Xc.dtype)
    theta = lbfgs_optimizer.execute(Xc, y, theta)
    return forward(app, Xc, theta)


def sample_set(app: ArrayApplication):
    shape = (500, 10)
    block_shape = (100, 10)
    rs = app.random_state(1337)
    X1 = rs.normal(loc=5.0, shape=shape, block_shape=block_shape)
    y1 = app.zeros(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X2 = rs.normal(loc=10.0, shape=shape, block_shape=block_shape)
    y2 = app.ones(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X = app.concatenate([X1, X2], axis=0)
    y = app.concatenate([y1, y2], axis=0)
    return X, y


def load_set(app: ArrayApplication, read_func, dataset):
    X = read_func("%s_X" % dataset)
    y = read_func("%s_y" % dataset)
    return X, y


def execute(dataset, cluster_shape, address, use_s3):

    settings.cluster_shape = cluster_shape
    ray.init(address=address)
    app: ArrayApplication = instance()
    time.sleep(0.1)

    start_time = time.time()
    read_func = app.read_s3 if use_s3 else app.read_fs
    # X, y = load_set(app, read_func, dataset)
    X, y = sample_set(app)
    y_pred_proba = logistic(app, X, y, max_iter=10, m=3)
    print("scheduling submitted.")
    y_pred = (y_pred_proba > 0.5).astype(np.float32)
    print("prediction submitted.")
    error = (app.sum(app.abs(y - y_pred)) / X.shape[0]).astype(np.float32).get()
    total_time = time.time() - start_time
    print("opt", "lbfgs")
    print("total time", total_time)
    print("error (1-accuracy)", error)
    # print("norm", model.grad_norm_sq(X, y).get())
    # print("objective", model.objective(X, y).get())
    return total_time, float(error)


if __name__ == "__main__":
    execute(dataset=None, cluster_shape=(1, 1),
            address=None, use_s3=False)
