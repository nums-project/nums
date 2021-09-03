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


from typing import List, Union

import numpy as np
from nums.core.array.application import ArrayApplication
from nums.core.application_manager import instance as _instance
from nums.models.glms import GLM


# Based on Nocedal and Wright, chapters 2, 3, 6 and 7.


class BackTrackingLineSearch(object):
    def __init__(self, model: GLM):
        self.app = _instance()
        self.model = model

    def f(self, theta_prime, X, y):
        return self.model.objective(
            X, y, theta_prime, self.model.forward(X, theta_prime)
        )

    def execute(
        self, X, y, theta, grad, p, rho=1.0e-1, init_alpha=1.0, c=1e-4, min_alpha=1e-10
    ):

        alpha = init_alpha
        f_val = self.f(X, y, theta)
        f_next = self.f(X, y, theta + alpha * p)
        while self.app.isnan(f_next) or f_next > f_val + c * alpha * grad.T @ p:
            alpha *= rho
            if alpha < min_alpha:
                return min_alpha
            # print("btls step alpha=%s" % alpha)
            f_next = self.f(X, y, theta + alpha * p)
        return max(alpha, min_alpha)


class LBFGSMemory(object):
    def __init__(self, k, s, y):
        self.k = k
        self.s = s
        self.y = y
        ys_inner = s.T @ y
        self.rho = 1.0 / (ys_inner + 1e-30)
        self.gamma = ys_inner / (y.T @ y + 1e-30)


class LBFGS(object):
    def __init__(self, model: GLM, m=3, max_iter=100, thresh=1e-5, dtype=np.float64):
        self.app: ArrayApplication = _instance()
        self.model: GLM = model
        self.m = m
        self.max_iter = max_iter
        self.thresh = thresh
        self.dtype = dtype
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None] * m
        self.ls = BackTrackingLineSearch(model)

    def get_H(self):
        if self.k == 0:
            return self.identity
        else:
            mem: LBFGSMemory = self.memory[-1]
            assert mem.k == self.k - 1
            return mem.gamma * self.identity

    def get_p(self, H, g):
        q = g
        forward_vars = []
        for i in range(-1, -self.m - 1, -1):
            mem_i: LBFGSMemory = self.memory[i]
            if mem_i is None:
                break
            alpha = mem_i.rho * mem_i.s.T @ q
            q -= alpha * mem_i.y
            forward_vars.insert(0, (alpha, mem_i))
        r = H @ q
        for alpha, mem_i in forward_vars:
            beta = mem_i.rho * mem_i.y.T @ r
            r += mem_i.s * (alpha - beta)
        return r

    def execute(self, X, y, theta):

        if self.k != 0:
            raise Exception("Unexpected state.")

        self.identity = self.app.eye(
            (X.shape[1], X.shape[1]),
            (X.block_shape[1], X.block_shape[1]),
            dtype=self.dtype,
        )

        g = self.model.gradient(X, y, self.model.forward(X, theta))
        next_g = None
        next_theta = None
        while self.k < self.max_iter:
            H = self.get_H()
            p = -self.get_p(H, g)
            init_alpha = 1.0
            alpha = self.ls.execute(
                X,
                y,
                theta,
                g,
                p,
                rho=1e-2,
                init_alpha=init_alpha,
                c=1e-4,
                min_alpha=1e-30,
            )
            print("alpha", alpha)
            # print("alpha", alpha,
            #       "objective", f(theta).get(),
            #       "grad_norm", self.app.sqrt(g.T @ g).get())
            next_theta = theta + alpha * p
            if self.k + 1 >= self.max_iter:
                # Terminate immediately if this is the last iteration.
                theta = next_theta
                break
            next_g = self.model.gradient(X, y, self.model.forward(X, next_theta))
            theta_diff = next_theta - theta
            grad_diff = next_g - g
            mem: LBFGSMemory = LBFGSMemory(k=self.k, s=theta_diff, y=grad_diff)
            self.memory.append(mem)
            self.memory.pop(0)
            self.k += 1
            theta = next_theta
            g = next_g
            if self.converged(next_g):
                break

        # Reset vars.
        self.k = 0
        self.identity = None
        self.memory: Union[List[LBFGSMemory], List[None]] = [None] * self.m

        return theta

    def converged(self, g):
        return self.app.sqrt(g.T @ g) < self.thresh


if __name__ == "__main__":
    pass
