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
from nums.experimental.optimizer.fusion import FuseGraph
from nums.experimental.optimizer.graph import TreeNode, Leaf, FunctionNode

import conftest
def fusion1(app, x, y):
    # An element-wise expression that benefits from fusion.
    return 1.0 / (1.0 + app.exp(x - y))

def fusion2(app, x, y):
    return x @ y

def fusion3(app, s, q, p):
    return s * (q @ p.T)

def ga_op(app, func, x: BlockArray, y: BlockArray, copy_on_op=True, max_args=2) -> BlockArray:
    x_ga: GraphArray = GraphArray.from_ba(x, cluster_state, copy_on_op=copy_on_op)
    y_ga: GraphArray = GraphArray.from_ba(y, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, x_ga, y_ga)
    start_time = time.time()
    fused_ga: GraphArray = FuseGraph.fuse_ga(app, op_ga, max_args)
    end_time = time.time()
    print(end_time - start_time)
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)

    return BlockArray(result_ga.grid, x.cm, result_ga.to_blocks())

def ga_op_sparse(app, func, s: BlockArray, p: BlockArray, q: BlockArray, copy_on_op=True
    , max_args=3) -> BlockArray:
    cluster_state: ClusterState = ClusterState(s.cm.devices())
    s_ga: GraphArray = GraphArray.from_ba(s, cluster_state, copy_on_op=copy_on_op)
    p_ga: GraphArray = GraphArray.from_ba(p, cluster_state, copy_on_op=copy_on_op)
    q_ga: GraphArray = GraphArray.from_ba(q, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, s_ga, p_ga, q_ga)
    start_time = time.time()
    fused_ga: GraphArray = FuseGraph.fuse_ga(app, op_ga, max_args)
    end_time = time.time()
    print(end_time - start_time)
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)

    return BlockArray(result_ga.grid, s.cm, result_ga.to_blocks())


def test_fusion(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (100,), (2,)
    y_shape, y_block_shape = (100,), (2,)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    real_y = np.random.random(np.product(y_shape)).reshape(y_shape)
    x: BlockArray = app.array(real_x, x_block_shape)
    y: BlockArray = app.array(real_y, y_block_shape)
    z: BlockArray = fusion1(app, x, y)
    start_time = time.time()
    opt_z: BlockArray = ga_op(app, fusion1, x, y)
    end_time = time.time()
    print(end_time - start_time)
    assert np.allclose(z.get(), fusion1(np, real_x, real_y))
    assert app.allclose(z, opt_z).get()

# matrix multiply 
# tensordot operation
def test_tensordot(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (4,2), (2, 2)
    y_shape, y_block_shape = (2,4), (2, 2)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    real_y = np.random.random(np.product(y_shape)).reshape(y_shape)
    x: BlockArray = app.array(real_x, x_block_shape)
    y: BlockArray = app.array(real_y, y_block_shape)
    z: BlockArray = fusion2(app, x, y)
    start_time = time.time()
    opt_z: BlockArray = ga_op(app, fusion2, x, y)
    end_time = time.time()
    print(end_time - start_time)
    assert np.allclose(z.get(), fusion2(np, real_x, real_y))
    assert app.allclose(z, opt_z).get()

def test_sparse_array(app_inst_mock_none):
    app = app_inst_mock_none
    q_shape, q_block_shape = (100,2), (2, 2)
    p_shape, p_block_shape = (200,2), (2, 2)
    s_shape, s_block_shape = (100, 200), (2, 2)
    real_q = np.random.random(np.product(q_shape)).reshape(q_shape)
    real_p = np.random.random(np.product(p_shape)).reshape(p_shape)
    real_s = np.random.random(np.product(s_shape)).reshape(s_shape)
    q: BlockArray = app.array(real_q, q_block_shape)
    p: BlockArray = app.array(real_p, p_block_shape)
    s: BlockArray = app.array(real_s, s_block_shape)
    z: BlockArray = fusion3(app, s, q, p)
    start_time = time.time()
    opt_z: BlockArray = ga_op_sparse(app, fusion3, s, q, p)
    end_time = time.time()
    print(end_time - start_time)
    assert np.allclose(z.get(), fusion3(np, real_s, real_q, real_p))
    assert app.allclose(z, opt_z).get()


if __name__ == "__main__":
    import conftest

    app = conftest.mock_cluster((1, 1))
    test_sparse_array(app)
    #test_fusion(app)
    conftest.destroy_mock_cluster(app)

    # app = conftest.mock_cluster((10, 1))
    # test_load_sqr(app)
    # test_load_single_block_rhs(app)
    # conftest.destroy_mock_cluster(app)
