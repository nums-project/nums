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


import itertools
import time

import numpy as np

from nums.core.storage.storage import BimodalGaussian
from nums.core.array.application import ArrayApplication, BlockArray

from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
    TreeNode,
    BinaryOp,
    ReductionOp,
    Leaf,
)
import conftest


def test_logistic(app_inst_mock_small):
    app: ArrayApplication = app_inst_mock_small
    num_samples, num_features = 16 * 10, 3
    real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = app.array(real_X, block_shape=(10, 3))
    y = app.array(real_y, block_shape=(10,))
    Xc = app.concatenate(
        [
            X,
            app.ones(
                shape=(X.shape[0], 1), block_shape=(X.block_shape[0], 1), dtype=X.dtype
            ),
        ],
        axis=1,
        axis_block_size=X.block_shape[1],
    )
    theta: BlockArray = app.zeros((Xc.shape[1],), (Xc.block_shape[1],), dtype=Xc.dtype)
    # Sigmoid
    Z = Xc @ theta
    mu: BlockArray = app.one / (app.one + app.exp(-Z))
    # Gradient
    g = Xc.T @ (mu - y)
    # Hessian
    s = mu * (app.one - mu)
    # TODO: Transpose here may have unexpected scheduling implications.
    hess = (Xc.T * s) @ Xc
    # These are PSD, but inv is faster than psd inv.
    theta += -app.inv(hess) @ g
    Z = theta[-1] + X @ theta[:-1]
    y_pred_proba = app.one / (app.one + app.exp(-Z))
    y_pred = (y_pred_proba > 0.5).astype(np.float32)
    error = (app.sum(app.abs(y - y_pred)) / X.shape[0]).get()
    print("error", error)


def test_graph_array_logistic(app_inst_mock_small):
    app = app_inst_mock_small

    def derivatives(local_X, local_y, local_theta, local_one):
        # Sigmoid
        Z = local_X @ local_theta
        mu = local_one / (local_one + app.exp(-Z))
        # Gradient
        g = local_X.T @ (mu - local_y)
        # # Hessian
        s = mu * (local_one - mu)
        hess = (local_X.T * s) @ local_X
        return g, hess

    def update_theta(g, hess, local_theta):
        return local_theta - app.inv(hess) @ g

    num_samples, num_features = 64 * 10, 3
    real_X, real_y = BimodalGaussian.get_dataset(num_samples, num_features)
    X = app.array(real_X, block_shape=(10, 3))
    y = app.array(real_y, block_shape=(10,))
    Xc = app.concatenate(
        [
            X,
            app.ones(
                shape=(X.shape[0], 1), block_shape=(X.block_shape[0], 1), dtype=X.dtype
            ),
        ],
        axis=1,
        axis_block_size=X.block_shape[1],
    )
    theta: BlockArray = app.zeros((Xc.shape[1],), (Xc.block_shape[1],), dtype=Xc.dtype)

    cluster_state = ClusterState(app.cm.devices())
    X_ga = GraphArray.from_ba(Xc, cluster_state)
    y_ga = GraphArray.from_ba(y, cluster_state)
    theta_ga = GraphArray.from_ba(theta, cluster_state)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    grad_ga, hess_ga = derivatives(X_ga, y_ga, theta_ga, one_ga)
    t = time.time()
    grad_ga_ba: BlockArray = conftest.compute_graph_array(grad_ga)
    hess_ga_ba: BlockArray = conftest.compute_graph_array(hess_ga)
    print("opt time", time.time() - t)
    grad_ba, hess_ba = derivatives(Xc, y, theta, app.one)
    result_ga_ba = update_theta(grad_ga_ba, hess_ga_ba, theta)
    result_ba = update_theta(grad_ba, hess_ba, theta)
    print(app.allclose(result_ga_ba, result_ba).get())
    assert app.allclose(result_ga_ba, result_ba)
    print("mem", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])


if __name__ == "__main__":
    import conftest

    app = conftest.mock_cluster((4, 1))
    test_logistic(app)
    test_graph_array_logistic(app)
    conftest.destroy_mock_cluster(app)
