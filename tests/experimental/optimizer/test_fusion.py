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
from nums.core.array.sparse import SparseBlockArray
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
)
from nums.experimental.optimizer.tree_search import RandomTS
from nums.experimental.optimizer.fusion import FuseGraph
from nums.experimental.optimizer.fusion_utils import print_graph
from nums.experimental.optimizer.graph import TreeNode, Leaf, FunctionNode
import conftest


def fusion1(app, x, y):
    # An element-wise expression that benefits from fusion.
    return 1.0 / (1.0 + app.exp(x - y))


def fusion2(app, x, y):
    return x @ y


def ga_op(
    app, func, x: BlockArrayBase, y: BlockArrayBase, copy_on_op=True, max_args=2
) -> BlockArray:
    cluster_state: ClusterState = ClusterState(x.km.devices())
    x_ga: GraphArray = GraphArray.from_ba(x, cluster_state, copy_on_op=copy_on_op)
    y_ga: GraphArray = GraphArray.from_ba(y, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, x_ga, y_ga)
    start_time = time.time()
    fused_ga: GraphArray = op_ga.compile(max_args)
    end_time = time.time()
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)

    return BlockArray(result_ga.grid, x.km, result_ga.to_blocks())


def test_fusion(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (10,), (5,)
    y_shape, y_block_shape = (10,), (5,)
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
    for grid_entry in opt_z.grid.get_entry_iterator():
        block = opt_z.blocks[grid_entry]
        assert block.grid_entry == grid_entry
        assert block.grid_shape == opt_z.grid_shape
        assert block.shape == opt_z.grid.get_block_shape(grid_entry)
        assert block.dtype == opt_z.dtype


# matrix multiply
# tensordot operation
def test_tensordot(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (4, 2), (2, 2)
    y_shape, y_block_shape = (2, 4), (2, 2)
    t = time.time()
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
    print("schedule time", time.time() - t)
    for grid_entry in opt_z.grid.get_entry_iterator():
        block = opt_z.blocks[grid_entry]
        assert block.grid_entry == grid_entry
        assert block.grid_shape == opt_z.grid_shape
        assert block.shape == opt_z.grid.get_block_shape(grid_entry)
        assert block.dtype == opt_z.dtype


def ga_op_sparse_2(
    app, func, x: BlockArrayBase, y: BlockArrayBase, copy_on_op=True, max_args=2
) -> SparseBlockArray:
    cluster_state: ClusterState = ClusterState(x.km.devices())
    x_ga: GraphArray = GraphArray.from_ba(x, cluster_state, copy_on_op=copy_on_op)
    y_ga: GraphArray = GraphArray.from_ba(y, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, x_ga, y_ga)
    start_time = time.time()
    fused_ga: GraphArray = op_ga.compile(max_args)
    end_time = time.time()
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)

    return SparseBlockArray(result_ga.grid, x.km, result_ga.to_blocks())


def spmm(app, p, q):
    return p @ q


def test_spmm(app_inst_mock_none):
    app = app_inst_mock_none
    p_shape, p_block_shape = (10, 4), (2, 2)
    q_shape, q_block_shape = (4, 10), (2, 2)
    p: SparseBlockArray = app.random.sparse_normal(
        shape=p_shape, block_shape=p_block_shape, p=0.1
    )
    q: SparseBlockArray = app.random.sparse_normal(
        shape=q_shape, block_shape=q_block_shape, p=0.1
    )
    real_p = p.to_ba().get()
    real_q = q.to_ba().get()
    z: SparseBlockArray = spmm(app, p, q)
    start_time = time.time()
    opt_z: SparseBlockArray = ga_op_sparse_2(app, spmm, p, q)
    end_time = time.time()
    print(end_time - start_time)
    assert z.nnz == opt_z.nnz
    assert np.allclose(z.to_ba().get(), spmm(np, real_p, real_q))
    assert app.allclose(z.to_ba(), opt_z.to_ba()).get()


def ga_op_sparse_3(
    app,
    func,
    s: BlockArrayBase,
    p: BlockArrayBase,
    q: BlockArrayBase,
    copy_on_op=True,
    max_args=3,
) -> SparseBlockArray:
    cluster_state: ClusterState = ClusterState(s.km.devices())
    s_ga: GraphArray = GraphArray.from_ba(s, cluster_state, copy_on_op=copy_on_op)
    p_ga: GraphArray = GraphArray.from_ba(p, cluster_state, copy_on_op=copy_on_op)
    q_ga: GraphArray = GraphArray.from_ba(q, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, s_ga, p_ga, q_ga)
    start_time = time.time()
    fused_ga: GraphArray = op_ga.compile(max_args)
    end_time = time.time()
    print(fused_ga.graphs[0, 0])
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)

    return SparseBlockArray(result_ga.grid, s.km, result_ga.to_blocks())


def sparse_fusion(app, s, p, q):
    return (s + s) * p * q


def test_sparse_fusion(app_inst_mock_none):
    app = app_inst_mock_none
    s_shape, s_block_shape = (10, 4), (2, 2)
    p_shape, p_block_shape = (10, 4), (2, 2)
    q_shape, q_block_shape = (10, 4), (2, 2)
    real_p = np.random.random(np.product(p_shape)).reshape(p_shape)
    real_q = np.random.random(np.product(q_shape)).reshape(q_shape)
    s: SparseBlockArray = app.random.sparse_normal(
        shape=s_shape, block_shape=s_block_shape, p=0.1
    )
    p: BlockArray = app.array(real_p, p_block_shape)
    q: BlockArray = app.array(real_q, q_block_shape)
    real_s = s.to_ba().get()
    z: SparseBlockArray = sparse_fusion(app, s, p, q)
    start_time = time.time()
    opt_z: SparseBlockArray = ga_op_sparse_3(app, sparse_fusion, s, p, q)
    end_time = time.time()
    print(end_time - start_time)
    assert z.nnz == opt_z.nnz
    assert np.allclose(z.to_ba().get(), sparse_fusion(np, real_s, real_p, real_q))
    assert app.allclose(z.to_ba(), opt_z.to_ba()).get()


def sddmm(app, s, p, q):
    return s * (p @ q.T)


def test_sddmm(app_inst_mock_none):
    app = app_inst_mock_none
    s_shape, s_block_shape = (20, 10), (2, 2)
    p_shape, p_block_shape = (20, 4), (2, 2)
    q_shape, q_block_shape = (10, 4), (2, 2)
    real_p = np.random.random(np.product(p_shape)).reshape(p_shape)
    real_q = np.random.random(np.product(q_shape)).reshape(q_shape)
    s: SparseBlockArray = app.random.sparse_normal(
        shape=s_shape, block_shape=s_block_shape, p=0.1
    )
    p: BlockArray = app.array(real_p, p_block_shape)
    q: BlockArray = app.array(real_q, q_block_shape)
    real_s = s.to_ba().get()
    z: SparseBlockArray = sddmm(app, s, p, q)
    start_time = time.time()
    opt_z: SparseBlockArray = ga_op_sparse_3(app, sddmm, s, p, q)
    end_time = time.time()
    print(end_time - start_time)
    assert z.nnz == opt_z.nnz
    assert np.allclose(z.to_ba().get(), sddmm(np, real_s, real_p, real_q))
    assert app.allclose(z.to_ba(), opt_z.to_ba()).get()


if __name__ == "__main__":
    import conftest

    app = conftest.mock_ray_cluster((1, 1))
    # test_sparse_array(app)
    test_fusion(app)
    # test_tensordot_variant2(app)
    conftest.destroy_mock_cluster(app)

    # app = conftest.mock_cluster((10, 1))
    # test_load_sqr(app)
    # test_load_single_block_rhs(app)
    # conftest.destroy_mock_cluster(app)
