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

from nums.core.array.application import BlockArray
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
)
from nums.experimental.optimizer.tree_search import RandomTS


def test_neg(app_inst_mock_small):
    app = app_inst_mock_small
    cluster_state = ClusterState(app.cm.devices())

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = -A_ga
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.cm, result_ga.to_blocks())
    assert app.allclose(-A, result_ba)


def test_root_uop(app_inst_mock_small):
    app = app_inst_mock_small
    cluster_state = ClusterState(app.cm.devices())

    one_ba: BlockArray = app.one
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = one_ga / (one_ga + app.exp(-(A_ga + A_ga)))
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.cm, result_ga.to_blocks())
    print(app.allclose(result_ba, one_ba / (one_ba + app.exp(-(A + A)))).get())
    assert app.allclose(result_ba, one_ba / (one_ba + app.exp(-(A + A))))


def test_transpose(app_inst_mock_small):
    app = app_inst_mock_small
    cluster_state = ClusterState(app.cm.devices())

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = A_ga.T @ A_ga
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.cm, result_ga.to_blocks())
    assert app.allclose(A.transpose(defer=True) @ A, result_ba)


def test_reduce(app_inst_mock_small):
    import itertools

    app = app_inst_mock_small
    test_axis = (None, 0, 1, 2)
    test_keepdims = (True, False)
    test_op = ("sum", "prod")
    test_block_shape = ((2, 5, 7), (3, 16, 20), (1, 1, 1))
    test_inputs = list(
        itertools.product(test_axis, test_keepdims, test_op, test_block_shape)
    )
    for axis, keepdims, op, block_shape in test_inputs:
        cluster_state = ClusterState(app.cm.devices())
        X: BlockArray = app.random.random(shape=(3, 16, 20), block_shape=block_shape)
        X_np: np.ndarray = X.get()
        X_ga: GraphArray = GraphArray.from_ba(X, cluster_state)
        reduced_ga: GraphArray = X_ga.reduce_axis(op, axis=axis, keepdims=keepdims)
        result_ga: GraphArray = RandomTS(
            seed=1337,
            max_samples_per_step=1,
            max_reduction_pairs=1,
            force_final_action=True,
        ).solve(reduced_ga)
        result_ba = BlockArray(result_ga.grid, app.cm, result_ga.to_blocks())
        if op == "sum":
            reduced_np = X_np.sum(axis=axis, keepdims=keepdims)
        else:
            reduced_np = X_np.prod(axis=axis, keepdims=keepdims)
        assert np.allclose(result_ba.get(), reduced_np)


def test_bop(app_inst_mock_small):
    from nums.core.array.application import ArrayApplication
    from nums.experimental.optimizer.grapharray import GraphArray
    from nums.experimental.optimizer.tree_search import RandomTS

    random_seed = 1337

    def collapse_graph_array(app: ArrayApplication, ga):
        return RandomTS(
            seed=random_seed,
            max_samples_per_step=1,
            max_reduction_pairs=1,
            force_final_action=True,
        ).solve(ga)

    def compute_graph_array(app: ArrayApplication, ga) -> BlockArray:
        result_ga: GraphArray = RandomTS(
            seed=random_seed,
            max_samples_per_step=1,
            max_reduction_pairs=1,
            force_final_action=True,
        ).solve(ga)
        return BlockArray(result_ga.grid, app.cm, result_ga.to_blocks())

    app = app_inst_mock_small
    cluster_state = ClusterState(app.cm.devices())
    X = app.random.normal(shape=(10, 3), block_shape=(5, 3))
    # y = app.random.integers(0, 2, shape=(10,), block_shape=(5,))
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
    theta: GraphArray = app.zeros((Xc.shape[1],), (Xc.block_shape[1],), dtype=Xc.dtype)
    X_ga: GraphArray = GraphArray.from_ba(Xc, cluster_state)
    # y_ga: GraphArray = GraphArray.from_ba(y, cluster_state)
    theta_ga: GraphArray = GraphArray.from_ba(theta, cluster_state)
    Z_ga: GraphArray = X_ga @ theta_ga
    Z_ga: GraphArray = collapse_graph_array(app, Z_ga)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    mu_ga: GraphArray = collapse_graph_array(app, one_ga / (one_ga + app.exp(-Z_ga)))
    mu_ba: BlockArray = compute_graph_array(app, mu_ga)
    mu_ba.touch()


if __name__ == "__main__":
    import conftest

    app = conftest.mock_dask_cluster((8, 1))
    # test_neg(app)
    # test_root_uop(app)
    # test_transpose(app)
    # test_reduce(app)
    test_bop(app)
    conftest.destroy_mock_cluster(app)
