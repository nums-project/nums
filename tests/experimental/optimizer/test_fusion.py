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

import conftest


def fusion1(app, x, y):
    # An element-wise expression that benefits from fusion.
    return 1.0 / (1.0 + app.exp(x - y))


def fuse_ga(app, r: GraphArray) -> GraphArray:
    result_graphs = np.empty_like(r.graphs, dtype=r.graphs.dtype)
    t = time.time()
    for grid_entry in r.grid.get_entry_iterator():
        graph = r.graphs[grid_entry]
        result_graphs[grid_entry] = FuseGraph(graph, app.cm)()
    print("fusion time", time.time() - t)
    return GraphArray(r.grid.copy(), r.cluster_state, result_graphs, r.cm)


def ga_op(app, func, x: BlockArray, y: BlockArray, copy_on_op=True) -> BlockArray:
    cluster_state: ClusterState = ClusterState(x.cm.devices())
    x_ga: GraphArray = GraphArray.from_ba(x, cluster_state, copy_on_op=copy_on_op)
    y_ga: GraphArray = GraphArray.from_ba(y, cluster_state, copy_on_op=copy_on_op)
    op_ga: GraphArray = func(app, x_ga, y_ga)
    fused_ga: GraphArray = fuse_ga(app, op_ga)
    t = time.time()
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(fused_ga)
    print("tree search time", time.time() - t)
    return BlockArray(result_ga.grid, x.cm, result_ga.to_blocks())


def test_fusion(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (10,), (5,)
    y_shape, y_block_shape = (10,), (5,)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    real_y = np.random.random(np.product(y_shape)).reshape(y_shape)
    x: BlockArray = app.array(real_x, x_block_shape)
    y: BlockArray = app.array(real_y, y_block_shape)
    z: BlockArray = fusion1(app, x, y)
    opt_z: BlockArray = ga_op(app, fusion1, x, y)
    assert np.allclose(z.get(), fusion1(np, real_x, real_y))
    assert app.allclose(z, opt_z).get()
    for grid_entry in opt_z.grid.get_entry_iterator():
        block = opt_z.blocks[grid_entry]
        assert block.grid_entry == grid_entry
        assert block.grid_shape == opt_z.grid_shape
        assert block.shape == opt_z.grid.get_block_shape(grid_entry)
        assert block.dtype == opt_z.dtype


def test_fusion_time(app_inst_mock_none):
    app = app_inst_mock_none
    x_shape, x_block_shape = (1024,), (1,)
    y_shape, y_block_shape = (1024,), (1,)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    real_y = np.random.random(np.product(y_shape)).reshape(y_shape)
    x: BlockArray = app.array(real_x, x_block_shape)
    y: BlockArray = app.array(real_y, y_block_shape)
    t = time.time()
    opt_z: BlockArray = ga_op(app, fusion1, x, y)
    print("schedule time", time.time() - t)
    for grid_entry in opt_z.grid.get_entry_iterator():
        block = opt_z.blocks[grid_entry]
        assert block.grid_entry == grid_entry
        assert block.grid_shape == opt_z.grid_shape
        assert block.shape == opt_z.grid.get_block_shape(grid_entry)
        assert block.dtype == opt_z.dtype


def test_fused_placement():
    pass


if __name__ == "__main__":
    import conftest

    app = conftest.mock_ray_cluster((1, 1))
    # test_fusion(app)
    test_fusion_time(app)
    conftest.destroy_mock_cluster(app)

    # app = conftest.mock_cluster((10, 1))
    # test_load_sqr(app)
    # test_load_single_block_rhs(app)
    # conftest.destroy_mock_cluster(app)
