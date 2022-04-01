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

from nums.core.array.application import BlockArray
from nums.core.array.base import BlockArrayBase
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
)
from nums.experimental.optimizer.tree_search import RandomTS

import conftest


def optimized_tensordot(
    lhs: BlockArrayBase, rhs: BlockArrayBase, axes, copy_on_op=True
) -> BlockArray:
    cluster_state: ClusterState = ClusterState(lhs.km.devices())
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=copy_on_op)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=copy_on_op)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)
    global random_state
    print("*" * 50)
    print("op grid shape", tensordot_ga.grid.grid_shape)
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(tensordot_ga)

    # print("mem", resources[0] / np.sum(resources[0]))
    print("mem", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])
    print("*" * 50)
    return BlockArray(result_ga.grid, lhs.km, result_ga.to_blocks())

def test_matvec(app_inst_mock_none):
    app = app_inst_mock_none
    A_shape, A_block_shape = (5, 10), (5, 5)
    x_shape, x_block_shape = (10, 1), (5, 1)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    x: BlockArray = app.array(real_x, x_block_shape)
    result: BlockArray = A @ x
    opt_result: BlockArray = optimized_tensordot(A, x, axes=1)
    assert np.allclose(result.get(), real_A @ real_x)
    assert app.allclose(result, opt_result).get()


def test_matmat(app_inst_mock_none):
    app = app_inst_mock_none
    X_shape, X_block_shape = (5, 10), (5, 5)
    Y_shape, Y_block_shape = (10, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app.array(real_X, X_block_shape)
    Y: BlockArray = app.array(real_Y, Y_block_shape)
    Z: BlockArray = X @ Y
    opt_Z: BlockArray = optimized_tensordot(X, Y, axes=1)
    assert np.allclose(Z.get(), real_X @ real_Y)
    assert app.allclose(Z, opt_Z).get()


def test_big_matmat(app_inst_mock_none):
    app = app_inst_mock_none
    num_blocks = 10 ** 3
    X_shape, X_block_shape = (5, 5 * num_blocks), (5, 5)
    Y_shape, Y_block_shape = (5 * num_blocks, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app.array(real_X, X_block_shape)
    Y: BlockArray = app.array(real_Y, Y_block_shape)
    Z: BlockArray = X @ Y
    t = time.time()
    opt_Z: BlockArray = optimized_tensordot(X, Y, axes=1)
    print(time.time() - t)
    assert np.allclose(Z.get(), real_X @ real_Y)
    assert app.allclose(Z, opt_Z).get()


def test_load_sqr(app_inst_mock_big):
    app = app_inst_mock_big
    num_blocks = 100
    X_shape, X_block_shape = (5 * num_blocks, 5), (5, 5)
    Y_shape, Y_block_shape = (5 * num_blocks, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app.array(real_X, X_block_shape)
    Y: BlockArray = app.array(real_Y, Y_block_shape)

    lhs, rhs, axes = X.transpose(defer=True), Y, 1
    cluster_state: ClusterState = ClusterState(app.km.devices())
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)

    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    assert mem_diff == net_in_diff == net_out_diff == 0
    # Block-cyclic distribution of 100 blocks of size 25 over 10 nodes == 10*25 == 250
    # We have 2 such matrices, so expect initial memory to be 500.
    assert max(cluster_state.resources[0]) == 500
    assert max(cluster_state.resources[1]) == max(cluster_state.resources[2]) == 0
    result_ga: GraphArray = RandomTS(
        seed=np.random.RandomState(1337),
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(tensordot_ga)
    print("memory", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])

    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    # Can we predict the worst-case for a stochastic scheduler?
    # All blocks (input and output) are 25 in size.
    # For now, just go off known values.
    # assert mem_diff <= 200 and net_in_diff <= 100 and net_out_diff <= 100
    print(mem_diff, net_in_diff, net_out_diff)


def test_load_single_block_rhs(app_inst_mock_big):
    app = app_inst_mock_big
    num_blocks = 100
    X_shape, X_block_shape = (5 * num_blocks, 5), (5, 5)
    Y_shape, Y_block_shape = (5, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app.array(real_X, X_block_shape)
    Y: BlockArray = app.array(real_Y, Y_block_shape)

    lhs, rhs, axes = X, Y, 1
    cluster_state: ClusterState = ClusterState(app.km.devices())
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)

    print("memory", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])
    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    print(mem_diff, net_in_diff, net_out_diff)
    assert mem_diff == 25  # b/c single block array is placed in node 1.
    assert net_in_diff == net_out_diff == 0
    assert max(cluster_state.resources[1]) == max(cluster_state.resources[2]) == 0
    result_ga: GraphArray = RandomTS(
        seed=np.random.RandomState(1337),
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(tensordot_ga)
    print("memory", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])

    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    print(mem_diff, net_in_diff, net_out_diff)
    # assert mem_diff <= 25 and net_in_diff <= 50 and net_out_diff <= 250


if __name__ == "__main__":
    import conftest

    app = conftest.mock_cluster((1, 1))
    # test_matvec(app)
    test_matmat(app)
    # test_big_matmat(app)
    conftest.destroy_mock_cluster(app)

    # app = conftest.mock_cluster((10, 1))
    # test_load_sqr(app)
    # test_load_single_block_rhs(app)
    # conftest.destroy_mock_cluster(app)
