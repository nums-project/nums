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

from nums.core.array.application import BlockArray, ArrayApplication
from nums.core.array.base import BlockArrayBase
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.grapharray import (
    GraphArray,
)
from nums.experimental.optimizer.tree_search import RandomTS

import conftest


def optimized_einsum(km, copy_on_op,
                     subscript, *operands
) -> BlockArray:
    cluster_state: ClusterState = ClusterState(km.devices())
    ga_operands = []
    for operand in operands:
        assert isinstance(operand, BlockArray)
        ga_operands.append(GraphArray.from_ba(operand, cluster_state, copy_on_op=copy_on_op))
    einsum_ga = GraphArray.einsum(cluster_state, km, copy_on_op, subscript, *ga_operands)
    global random_state
    print("*" * 50)
    print("op grid shape", einsum_ga.grid.grid_shape)
    result_ga: GraphArray = RandomTS(
        seed=conftest.rs,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True,
    ).solve(einsum_ga)

    # print("mem", resources[0] / np.sum(resources[0]))
    print("mem", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])
    print("*" * 50)
    return BlockArray(result_ga.grid, km, result_ga.to_blocks())


def test_einsum(app_inst: ArrayApplication):
    rs: np.random.RandomState = np.random.RandomState(1337)
    I, J, K = 10, 11, 12
    X_np = rs.rand(I, J, K)

    F = 3
    A_np = np.random.rand(I, F)
    B_np = np.random.rand(J, F)
    C_np = np.random.rand(K, F)

    X_ba = app_inst.array(X_np, (5, 4, 3))
    B_ba = app_inst.array(B_np, (4, 2))
    C_ba = app_inst.array(C_np, (3, 2))

    # MTTKRP
    ss = "ijk,jf,kf->if"
    R_np = np.einsum(ss, X_np, B_np, C_np)
    assert R_np.shape == A_np.shape
    R_ba = app_inst.einsum(ss, X_ba, B_ba, C_ba)
    assert np.allclose(R_np, R_ba.get())
    R_ga = optimized_einsum(app_inst.km, True, ss, X_ba, B_ba, C_ba)
    assert np.allclose(R_np, R_ga.get())


if __name__ == "__main__":
    from tests import conftest

    # pylint: disable=import-error
    app_inst = conftest.get_app("serial", "cyclic")
    test_einsum(app_inst)
