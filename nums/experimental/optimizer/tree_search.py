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


from typing import Union

import numpy as np

from nums.core.grid.grid import DeviceGrid
from nums.experimental.optimizer.grapharray import GraphArray

from nums.experimental.optimizer.graph import (
    TreeNode,
    BinaryOp,
    Leaf,
    UnaryOp,
)

from nums.experimental.optimizer.reduction_ops import TreeReductionOp

random_state = np.random.RandomState(1337)


class TreeNodeActionPair(object):
    """
    Node corresponds to the tree node on which the given actions can be performed.
    Actions are a list of arguments that can be invoked on either
    TreeNode.simulate_on or TreeNode.execute_on.
    """

    def __init__(self, node: TreeNode, actions: list):
        self.node: TreeNode = node
        self.actions: list = actions

    def __repr__(self):
        return "TNAP(%s, %s)" % (str(self.node), str(self.actions))


class ProgramState(object):
    def __init__(
        self,
        arr: GraphArray,
        device_grid: DeviceGrid,
        max_reduction_pairs=None,
        force_final_action=True,
        unique_reduction_pairs=False,
        use_all_devices=False,
    ):
        self.arr: GraphArray = arr
        self.device_grid: DeviceGrid = device_grid
        self.force_final_action = force_final_action
        self.get_action_kwargs = {
            "max_reduction_pairs": max_reduction_pairs,
            "unique_reduction_pairs": unique_reduction_pairs,
            "use_all_devices": use_all_devices,
        }
        # Keys are tree_node_id
        # Values are a 2-tuple corresponding to the actual tree node and the actions that can
        # be performed on that tree node.
        self.tnode_map: [str, TreeNodeActionPair] = {}
        self.init_frontier()

    def num_nodes(self):
        r = 0
        for grid_entry in self.arr.grid.get_entry_iterator():
            root: TreeNode = self.arr.graphs[grid_entry]
            r += root.num_nodes()
        return r

    def init_frontier(self):
        for grid_entry in self.arr.grid.get_entry_iterator():
            self.add_frontier_tree(self.arr.graphs[grid_entry])

    def add_frontier_tree(self, start_node: TreeNode):
        for tnode in start_node.get_frontier():
            self.add_frontier_node(tnode)

    def get_final_action(self, tnode: TreeNode):
        # This is hacky, but no good way to do it w/ current abstractions.
        # Gets the final action to perform on a graph array.
        # This is to ensure the output graph array satisfies device grid layout
        # assumptions.
        grid_entry = self.get_tnode_grid_entry(tnode)
        grid_shape = self.arr.grid.grid_shape
        device = self.device_grid.get_device(grid_entry, grid_shape)
        if isinstance(tnode, BinaryOp):
            actions = [(tnode.tree_node_id, {"device": device})]
        elif isinstance(tnode, TreeReductionOp):
            # TreeReductionOp maintains an ordered list of leaves,
            # which is initialized upon invoking get_actions().
            # It may never be invoked if the reduction op consists of two input vertices.
            # We therefore need to execute the following check to ensure the list of leaves
            # is either initialized or can safely be initialized to execute the final op.
            tnode.final_action_check()
            leaf_ids = tuple(tnode.leafs_dict.keys())[:2]
            actions = [(tnode.tree_node_id, {"device": device, "leaf_ids": leaf_ids})]
        elif isinstance(tnode, UnaryOp):
            actions = [(tnode.tree_node_id, {"device": device})]
        else:
            raise Exception()
        return actions

    def add_frontier_node(self, tnode: TreeNode):
        # This is a frontier node.
        actions = None
        if self.force_final_action and tnode.parent is None:
            if isinstance(tnode, (BinaryOp, UnaryOp)) or (
                isinstance(tnode, TreeReductionOp) and len(tnode.children_dict) == 2
            ):
                # This is a root frontier op.
                # The next action is the last action,
                # so intercept action to force computation on device
                # to satisfy device placement assumptions.
                actions = self.get_final_action(tnode)
        if actions is None:
            actions = tnode.get_actions(**self.get_action_kwargs)
        self.tnode_map[tnode.tree_node_id] = TreeNodeActionPair(tnode, actions)

    def copy(self):
        return ProgramState(
            self.arr.copy(),
            self.device_grid,
            **self.get_action_kwargs,
            force_final_action=self.force_final_action,
        )

    def commit_action(self, action):
        tnode_id, kwargs = action
        entry: TreeNodeActionPair = self.tnode_map[tnode_id]
        old_node: TreeNode = entry.node
        new_node: TreeNode = old_node.execute_on(**kwargs)
        # The frontier needs to be updated, so remove the current node from frontier.
        del self.tnode_map[tnode_id]
        if old_node.parent is None and old_node is not new_node:
            # We operated on a root node, so update the array.
            self.update_root(old_node, new_node)
        if isinstance(new_node, Leaf):
            # If it's a leaf node, its parent may now be a frontier node.
            new_node_parent: TreeNode = new_node.parent
            if new_node_parent is not None and new_node_parent.is_frontier():
                self.add_frontier_node(new_node_parent)
        else:
            # There's still work that needs to be done to compute this node.
            # Add the returned node to the frontier.
            # Either a BinaryOp or TreeReductionOp.
            if new_node.is_frontier():
                self.add_frontier_node(new_node)
        # That's it. This program state is now updated.
        return self.objective(self.arr.cluster_state.resources)

    def simulate_action(self, action):
        tnode_id, kwargs = action
        entry: TreeNodeActionPair = self.tnode_map[tnode_id]
        node: TreeNode = entry.node
        new_resources: np.ndarray = node.simulate_on(**kwargs)
        return self.objective(new_resources)

    def objective(self, resources):
        # Our simple objective.
        # Max over second axis, as this is the axis corresponding to resource load
        # over cluster nodes.
        return np.sum(np.max(resources, axis=1))

    def get_tnode_grid_entry(self, tnode: TreeNode):
        if tnode.parent is None:
            root: TreeNode = tnode
        else:
            root: TreeNode = tnode.get_root()
        tree_root_grid_entry = None
        for grid_entry in self.arr.grid.get_entry_iterator():
            tree_node: TreeNode = self.arr.graphs[grid_entry]
            if tree_node is root:
                tree_root_grid_entry = grid_entry
                break
        if tree_root_grid_entry is None:
            raise Exception("Bad tree.")
        return tree_root_grid_entry

    def update_root(self, old_root, new_root):
        tree_root_grid_entry = self.get_tnode_grid_entry(old_root)
        self.arr.graphs[tree_root_grid_entry] = new_root

    def get_all_actions(self):
        # This is not deterministic due to hashing of children for reduction nodes.
        actions = []
        for tnode_id in self.tnode_map:
            actions += self.tnode_map[tnode_id].actions
        return actions


class TreeSearch(object):
    """
    Tree search base class.
    - seed is supported for stochastic tree search algorithms.
    - max_samples_per_step is the number of vertices from the frontier sampled per step.
      If this is None, then the entire frontier is considered.
    - max_reduction_pairs is a parameter provided to the reduction vertex in the computation tree.
      The reduction vertex may have high fan-in, and scheduling is performed sequentially by
      considering random pairs of input vertices. This parameter reduces the number of random
      input pairs considered when generating the invocable actions on a reduction node.
    """

    def __init__(
        self,
        seed: Union[int, np.random.RandomState] = 1337,
        max_samples_per_step=None,
        max_reduction_pairs=None,
        force_final_action=True,
    ):
        if isinstance(seed, np.random.RandomState):
            self.rs = seed
        else:
            assert isinstance(seed, (int, np.int))
            self.rs = np.random.RandomState(seed)
        self.max_samples_per_step = max_samples_per_step
        self.max_reduction_pairs = max_reduction_pairs
        self.force_final_action = force_final_action

    def step(self, state: ProgramState):
        raise NotImplementedError()

    def solve(self, arr: GraphArray):
        state: ProgramState = ProgramState(
            arr,
            arr.km.device_grid,
            max_reduction_pairs=self.max_reduction_pairs,
            force_final_action=self.force_final_action,
        )
        num_steps = 0
        while True:
            num_steps += 1
            state, cost, is_done = self.step(state)
            if is_done:
                break
        # print("solve completed", num_steps, cost)
        return state.arr


class DeviceGridTS(TreeSearch):
    def __init__(
        self,
        seed: Union[int, np.random.RandomState] = 1337,
        max_samples_per_step=None,
        max_reduction_pairs=None,
        force_final_action=True,
    ):
        super().__init__(
            seed, max_samples_per_step, max_reduction_pairs, force_final_action
        )

    def step(self, state: ProgramState):
        if len(state.tnode_map) == 0:
            # We're done.
            return state, state.objective(state.arr.cluster_state.resources), True
        action = None
        for tnode_id in state.tnode_map:
            action = state.get_final_action(state.tnode_map[tnode_id].node)[0]
            break
        curr_cost = state.commit_action(action)
        return state, curr_cost, False


class RandomTS(TreeSearch):
    def __init__(
        self,
        seed: Union[int, np.random.RandomState] = 1337,
        max_samples_per_step=None,
        max_reduction_pairs=None,
        force_final_action=True,
    ):
        super().__init__(
            seed, max_samples_per_step, max_reduction_pairs, force_final_action
        )

    def sample_actions(self, state: ProgramState) -> list:
        if self.max_samples_per_step is None:
            return state.get_all_actions()
        # Subsample a set of frontier nodes to try next.
        tnode_ids = list(state.tnode_map.keys())
        num_tnodes = len(tnode_ids)
        if num_tnodes <= self.max_samples_per_step:
            tnode_id_sample = tnode_ids
        else:
            idx_set = set()
            tnode_id_sample = []
            while len(idx_set) < self.max_samples_per_step:
                i = self.rs.randint(0, num_tnodes)
                if i not in idx_set:
                    idx_set.add(i)
                    tnode_id_sample.append(tnode_ids[i])
        actions = []
        for tnode_id in tnode_id_sample:
            actions += state.tnode_map[tnode_id].actions
        return actions

    def step(self, state: ProgramState):
        # Sampling slows things down because for some reason,
        # the lowest cost computations are the sums, so
        # an algorithm that finds local optima keeps the number of leafs for reductions
        # small by computing them whenever they occur.
        actions = self.sample_actions(state)
        if len(actions) == 0:
            # We're done.
            return state, state.objective(state.arr.cluster_state.resources), True
        min_action = None
        min_cost = np.float64("inf")
        # actions_info = [(state.tnode_map[action[0]].node, action[1]) for action in actions]
        # prior_resources = state.arr.cluster_state.resources.sum(axis=1)
        for i in range(len(actions)):
            action = actions[i]
            action_cost = state.simulate_action(action)
            if action_cost < min_cost:
                min_action = action
                min_cost = action_cost
        curr_cost = state.commit_action(min_action)
        # print("")
        # print("actions", actions_info)
        # print("min_action", min_action)
        # curr_resources = state.arr.cluster_state.resources.sum(axis=1)
        # print("cost diff", curr_resources - prior_resources)
        return state, curr_cost, False
