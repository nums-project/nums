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

from nums.core.systems.systems import RaySystem
from nums.core.array.application import ArrayApplication, BlockArray
from nums.core.array.base import BlockArrayBase

from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
    TreeNode,
    UnaryOp,
    BinaryOp,
    ReductionOp,
    Leaf,
)
from nums.experimental.optimizer.tree_search import RandomTS, RandomPlan, Plan
import conftest


def random_solve(ga, seed):
    planner: RandomPlan = RandomPlan(
        seed=seed,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    )
    planner.solve(ga)
    return planner


def graphs_equal(ga1: GraphArray, ga2: GraphArray):
    # Generate a list of nodes ordered top to bottom and left to right.
    assert np.allclose(ga1.cluster_state.resources, ga2.cluster_state.resources)
    assert ga1 is not ga2
    assert ga1.cluster_state is not ga2.cluster_state
    nodes1 = list(ga1.iterator())
    nodes2 = list(ga2.iterator())
    for i in range(len(nodes1)):
        n1: TreeNode = nodes1[i]
        n2: TreeNode = nodes2[i]
        assert n1 is not n2
        assert n1.tree_node_id == n2.tree_node_id
        if n1.parent is None:
            assert n2.parent is None
        else:
            assert n1.parent.tree_node_id == n2.parent.tree_node_id
        assert type(n1) == type(n2)
        if type(n1) is Leaf:
            assert n1.block.id == n2.block.id
        elif type(n1) is UnaryOp:
            assert n1.op_name == n2.op_name
            assert n1.child.tree_node_id == n2.child.tree_node_id
        elif type(n1) is BinaryOp:
            assert n1.op_name == n2.op_name
            assert n1.args == n2.args
            assert n1.left.tree_node_id == n2.left.tree_node_id
            assert n1.right.tree_node_id == n2.right.tree_node_id
        elif type(n1) is ReductionOp:
            assert n1.op_name == n2.op_name
            # TODO (hme): How to ensure proper replication of random state?
            assert set(n1.children_dict.keys()) == set(n2.children_dict.keys())
            assert set(n1.leafs_dict.keys()) == set(n2.leafs_dict.keys())


def tensordot(
    lhs: BlockArrayBase, rhs: BlockArrayBase, axes, copy_on_op=True
) -> GraphArray:
    cluster_state = ClusterState(lhs.cm.devices())
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=copy_on_op)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=copy_on_op)
    return lhs_ga.tensordot(rhs_ga, axes=axes)


def optimized_tensordot(
    lhs: BlockArrayBase, rhs: BlockArrayBase, axes, copy_on_op=True
) -> BlockArray:
    tensordot_ga = tensordot(lhs, rhs, axes, copy_on_op)
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(tensordot_ga)
    return BlockArray(result_ga.grid, lhs.cm, result_ga.to_blocks())


def test_matmat(app_inst: ArrayApplication):
    X_shape, X_block_shape = (5, 10), (5, 5)
    Y_shape, Y_block_shape = (10, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)
    Z_ga: GraphArray = tensordot(X, Y, axes=1)
    graphs_equal(Z_ga, Z_ga.copy())

    plan1 = random_solve(Z_ga, 1337).plan
    plan2 = random_solve(Z_ga, 1337).plan
    action_seq1 = [(item[1], item[2][0], item[2][1]) for item in plan1.plan]
    action_seq2 = [(item[1], item[2][0], item[2][1]) for item in plan2.plan]
    for i in range(len(plan1.plan)):
        s1, tn1, a1, r1, ns1, is_done1 = plan1.plan[i]
        s2, tn2, a2, r2, ns2, is_done2 = plan2.plan[i]

        s1: ClusterState = s1
        s2: ClusterState = s2
        assert s1 is not s2
        assert np.allclose(s1.resources, s2.resources)

        ns1: ClusterState = ns1
        ns2: ClusterState = ns2
        assert ns1 is not ns2
        assert np.allclose(ns1.resources, ns2.resources)

        assert tn1 is not tn2
        tnid1, kwargs1 = a1
        tnid2, kwargs2 = a2
        assert tnid1 == tn1.tree_node_id == tnid2 == tn2.tree_node_id
        assert list(kwargs1.keys()) == list(kwargs2.keys())
        for key in kwargs1:
            assert kwargs1[key] == kwargs2[key]

        assert np.allclose(r1, r2)
        assert is_done1 == is_done2


if __name__ == "__main__":
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_matmat(app_inst)
