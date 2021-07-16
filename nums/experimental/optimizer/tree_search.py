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
import dill
import multiprocessing
import time

from nums.experimental.optimizer.grapharray import (
    GraphArray,
    TreeNode,
    BinaryOp,
    ReductionOp,
    Leaf,
    UnaryOp,
)
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.core.compute.compute_manager import ComputeManager

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
        max_reduction_pairs=None,
        force_final_action=True,
        unique_reduction_pairs=False,
        plan_only=False,
        use_all_devices=False,
    ):
        self.arr: GraphArray = arr
        self.force_final_action = force_final_action
        self.get_action_kwargs = {
            "max_reduction_pairs": max_reduction_pairs,
            "unique_reduction_pairs": unique_reduction_pairs,
            "use_all_devices": use_all_devices,
        }
        self.plan_only = plan_only
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
        cm: ComputeManager = ComputeManager.instance
        grid_entry = self.get_tnode_grid_entry(tnode)
        grid_shape = self.arr.grid.grid_shape
        device_id = cm.device_grid.get_device_id(grid_entry, grid_shape)
        if isinstance(tnode, BinaryOp):
            actions = [(tnode.tree_node_id, {"device_id": device_id})]
        elif isinstance(tnode, ReductionOp):
            leaf_ids = tuple(tnode.leafs_dict.keys())[:2]
            actions = [
                (tnode.tree_node_id, {"device_id": device_id, "leaf_ids": leaf_ids})
            ]
        elif isinstance(tnode, UnaryOp):
            actions = [(tnode.tree_node_id, {"device_id": device_id})]
        else:
            raise Exception()
        return actions

    def add_frontier_node(self, tnode: TreeNode):
        # This is a frontier node.
        actions = None
        if self.force_final_action and tnode.parent is None:
            if isinstance(tnode, (BinaryOp, UnaryOp)) or (
                isinstance(tnode, ReductionOp) and len(tnode.children_dict) == 2
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
            **self.get_action_kwargs,
            force_final_action=self.force_final_action,
            plan_only=self.plan_only
        )

    def commit_action(self, action):
        tnode_id, kwargs = action
        entry: TreeNodeActionPair = self.tnode_map[tnode_id]
        old_node: TreeNode = entry.node
        new_node: TreeNode = old_node.execute_on(**kwargs, plan_only=self.plan_only)
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
            # Either a BinaryOp or ReductionOp.
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
            max_reduction_pairs=self.max_reduction_pairs,
            force_final_action=self.force_final_action,
        )
        num_steps = 0
        while True:
            num_steps += 1
            state, cost, is_done = self.step(state)
            # print(num_steps, state.num_nodes(), cost)
            if is_done:
                break
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
        for i in range(len(actions)):
            action = actions[i]
            action_cost = state.simulate_action(action)
            if action_cost < min_cost:
                min_action = action
                min_cost = action_cost
        curr_cost = state.commit_action(min_action)
        return state, curr_cost, False


class Plan(object):
    def __init__(self, max_reduction_pairs, force_final_action):
        self.max_reduction_pairs = max_reduction_pairs
        self.force_final_action = force_final_action
        self.plan = []
        self.cost = 0

    def copy(self):
        p = Plan(self.max_reduction_pairs, self.force_final_action)
        p.plan = self.plan.copy()  # Semi-deep copy; not sure if it copies ProgramState.
        return p

    def append(
        self, cluster_state, tree_node, action, cost, next_cluster_state, is_done
    ):
        self.plan.append(
            (cluster_state, tree_node, action, cost, next_cluster_state, is_done)
        )

    def get_plan_actions(self):
        actions = []
        for step in self.plan:
            actions.append(step[1])
        return actions

    # Return cluster state at final step in the plan.
    def get_cluster_state(self):
        return self.plan[-1][0]

    def get_next_cluster_state(self):
        return self.plan[-1][4]

    def get_cost(self):
        return self.cost

    def execute(self, arr_inp: GraphArray):
        arr = arr_inp.copy()
        state: ProgramState = ProgramState(
            arr,
            max_reduction_pairs=self.max_reduction_pairs,
            force_final_action=self.force_final_action,
        )
        for (
            cluster_state,
            tree_node,
            action,
            cost,
            next_cluster_state,
            is_done,
        ) in self.plan:
            # TODO (hme): Remove these inline tests once actual tests are added.
            actual_cost = state.commit_action(action)
            assert np.allclose(actual_cost, cost)
            actual_cluster_state: ClusterState = state.arr.cluster_state
            expected_cluster_state: ClusterState = next_cluster_state
            assert np.allclose(
                actual_cluster_state.resources, expected_cluster_state.resources
            )
            actual_is_done = len(state.tnode_map) == 0
            assert actual_is_done == is_done

        return state.arr


class SubtreeRoot:
    def __init__(self, id, state: ProgramState, cost, plan: Plan, depth):
        self.id = id
        self.state = state
        self.cost = cost
        self.plan = plan
        self.depth = depth


class ExhaustiveProcess(multiprocessing.Process):
    def __init__(self, thread_id, start_points, queue, nprocs, timestamp_queue):
        multiprocessing.Process.__init__(self)
        self.thread_id = thread_id
        self.queue = queue
        self.nprocs = nprocs
        self.start_points = start_points
        self.ts_queue = timestamp_queue

    def add_subtree(self, subtree):
        self.start_points.append(subtree)

    def run(self):
        print("Starting ", self.thread_id, "with", len(self.start_points), "subtrees")
        print("Subtrees: {}".format(",".join([str(x.id) for x in self.start_points])))
        search_time = 0
        for start_point in self.start_points:
            plans = []
            planner = ExhaustivePlanner(self.nprocs)
            t0 = time.time()
            planner.make_plan_helper(
                start_point.state,
                start_point.cost,
                start_point.depth,
                start_point.plan,
                plans,
                None,
            )  # or make plan?
            t1 = time.time()
            self.ts_queue.put((t1 - t0, len(plans)))
            search_time += t1 - t0
            self.queue.put(plans)
        # self.ts_queue.put(search_time)
        # self.all_plans[str(self.thread_id)] = plans


#        print("Exiting ", self.thread_id)


class ExhaustivePlanner(object):
    def __init__(self, nprocs, force_final_action=True):
        # To ensure a fully exhaustive search, want to allow reduction nodes
        # to consider all pairs of incoming vertices.
        self.max_reduction_pairs = None
        self.nprocs = nprocs
        self.force_final_action = force_final_action
        self.plan = Plan(self.max_reduction_pairs, force_final_action)
        self.pessimal_plan = Plan(self.max_reduction_pairs, force_final_action)

    def find_best_and_worst_plans(self, all_plans):
        min_cost = all_plans[0][1]  # cost is the second entry in the tuples
        max_cost = all_plans[0][1]
        print("Total plans: ", len(all_plans))
        print("Reviewing plans...")
        for p in all_plans:
            if p[1] <= min_cost:
                min_cost = p[1]
                self.plan = p[0]
                self.plan.cost = p[1]
            if p[1] >= max_cost:
                max_cost = p[1]
                self.pessimal_plan = p[0]
                self.pessimal_plan.cost = p[1]
        print("Chosen plan: ", self.plan, self.plan.cost)
        print("Worst plan: ", self.pessimal_plan, self.pessimal_plan.cost)

    def make_plan_parallel_unroll(self, state: ProgramState):
        # Make a thread for each exhaustive process
        q = multiprocessing.Queue()
        ts_queue = multiprocessing.Queue()
        actions = self.get_frontier_actions(state)

        unroll = True
        if self.nprocs <= len(actions):
            unroll = False

        # First layer: branch on first possible actions
        plans = []
        states = []
        next_actions = []
        costs = []
        subtrees = []
        for i, a in enumerate(actions):
            tree_node: TreeNode = state.tnode_map[a[0]].node
            next_plan = Plan(self.max_reduction_pairs, self.force_final_action)
            next_state = state.copy()  # copying the ProgramState invokes
            # init_frontier, which finds nodes
            # designated as frontier nodes.
            cluster_state = next_state.arr.cluster_state.copy()
            step_cost = next_state.commit_action(a)
            is_done = len(next_state.tnode_map) == 0
            next_plan.append(
                cluster_state,
                tree_node,
                a,
                step_cost,
                next_state.arr.cluster_state,
                is_done,
            )
            next_actions.append(self.get_frontier_actions(next_state))
            plans.append(next_plan)
            states.append(next_state)
            costs.append(step_cost)
            if not unroll:
                subtrees.append(SubtreeRoot(i, next_state, step_cost, next_plan, 1))

        # Second layer: create processes for sets of paths.
        processes = []
        #        print(len(next_actions))
        #        print(next_actions)
        subtree_id = 0
        # next_actions is a list of lists.
        if unroll:
            for i, action_set in enumerate(next_actions):
                for a in action_set:
                    tree_node: TreeNode = state.tnode_map[a[0]].node
                    next_plan = plans[i].copy()
                    next_state = states[i].copy()  # copying the ProgramState invokes
                    # init_frontier, which finds nodes
                    # designated as frontier nodes.
                    cluster_state = next_state.arr.cluster_state.copy()
                    step_cost = next_state.commit_action(a)
                    is_done = len(next_state.tnode_map) == 0
                    next_plan.append(
                        cluster_state,
                        tree_node,
                        a,
                        step_cost,
                        next_state.arr.cluster_state,
                        is_done,
                    )
                    subtrees.append(
                        SubtreeRoot(
                            subtree_id, next_state, step_cost + costs[i], next_plan, 2
                        )
                    )
                    subtree_id += 1

        # Round robin pickup of tasks
        for i in range(len(subtrees)):
            if i < self.nprocs:
                processes.append(
                    ExhaustiveProcess(i, [subtrees[i]], q, self.nprocs, ts_queue)
                )
            else:
                processes[i % self.nprocs].add_subtree(subtrees[i])

        # Run all processes.
        for p in processes:
            p.start()

        all_plans = []
        # print("proc plans length:", len(proc_plans))
        items = len(subtrees)
        while items > 0:
            if not q.empty():
                print(
                    "Picked up plans from queue. {}/{}".format(
                        (len(subtrees) - items) + 1, len(subtrees)
                    )
                )
                items -= 1
                all_plans += q.get()

        search_times = []
        times = len(subtrees)
        while times > 0:
            times -= 1
            search_times.append(ts_queue.get())

        print("Waiting for all processes to join")
        for p in processes:
            p.join()
        print("All joined")

        for ts, length in search_times:
            print("Time spent on search:", ts)
            print("Plans found:", length)

        # Find minimum cost plan
        self.find_best_and_worst_plans(all_plans)
        return all_plans

    def make_plan_parallel(self, state: ProgramState):
        # Make a thread for each exhaustive planner object
        q = multiprocessing.Queue()
        ts_queue = multiprocessing.Queue()
        actions = self.get_frontier_actions(state)
        # proc_plans = {}
        processes = []
        subtrees = []
        if self.nprocs > len(actions):
            self.nprocs = len(actions)

        for i, a in enumerate(actions):
            tree_node: TreeNode = state.tnode_map[a[0]].node
            next_plan = Plan(self.max_reduction_pairs, self.force_final_action)
            next_state = state.copy()  # copying the ProgramState invokes
            # init_frontier, which finds nodes
            # designated as frontier nodes.
            cluster_state = next_state.arr.cluster_state.copy()
            step_cost = next_state.commit_action(a)
            is_done = len(next_state.tnode_map) == 0
            next_plan.append(
                cluster_state,
                tree_node,
                a,
                step_cost,
                next_state.arr.cluster_state,
                is_done,
            )
            # self.make_plan_helper(next_state, step_cost, next_plan, all_plans)
            # processes.append(ExhaustiveProcess(i, next_state, step_cost, next_plan, q, self.nprocs))
            subtrees.append(SubtreeRoot(i, next_state, step_cost, next_plan))

        # Round robin pickup of tasks
        for i in range(len(subtrees)):
            if i < self.nprocs:
                processes.append(
                    ExhaustiveProcess(i, [subtrees[i]], q, self.nprocs, ts_queue)
                )
            else:
                processes[i % self.nprocs].add_subtree(subtrees[i])

        # Run all processes.
        for p in processes:
            p.start()

        all_plans = []
        # print("proc plans length:", len(proc_plans))
        items = len(subtrees)
        while items > 0:
            if not q.empty():
                print(
                    "Picked up plans from queue. {}/{}".format(
                        (len(subtrees) - items) + 1, len(subtrees)
                    )
                )
                items -= 1
                all_plans += q.get()

        search_times = []
        times = len(subtrees)
        while times > 0:
            times -= 1
            search_times.append(ts_queue.get())

        print("Waiting for all processes to join")
        for p in processes:
            p.join()
        print("All joined")

        for ts, length in search_times:
            print("Time spent on search:", ts)
            print("Plans found:", length)

        # Find minimum cost plan
        self.find_best_and_worst_plans(all_plans)
        return all_plans

    # Generate an optimal plan via exhaustive search
    def make_plan(self, state: ProgramState):
        # Generate + save all possible plans and their associated costs
        all_plans = []
        self.make_plan_helper(
            state,
            0,
            0,
            Plan(self.max_reduction_pairs, self.force_final_action),
            all_plans,
            None,
        )
        # Find minimum cost plan
        self.find_best_and_worst_plans(all_plans)
        return all_plans

    # Helper to make_plan: recursively generates all possible plans while tracking
    # cumulative cost.
    # all_plans: array of (Plan, int cost)
    def make_plan_helper(
        self, state: ProgramState, cost, depth, plan: Plan, all_plans, min_cost
    ):
        # hack for debugging
        #        if len(all_plans) > 0:
        #            return

        if min_cost is not None and cost > min_cost:
            return min_cost

        # Get all actions possible from current frontier.
        actions = self.get_frontier_actions(state)
        #        print("make_plan_helper actions", actions)
        #        print("cost", cost)
        #        print("current plan", plan.get_plan_actions())

        # Base case: if no actions, return plan and cost
        if len(actions) == 0:
            #            print("encountered base case, depth", depth)
            all_plans.append((plan, cost))
            if min_cost is None or cost < min_cost:
                min_cost = cost
            return min_cost

        # For each action, construct a new ProgramState that takes
        # that action at this step.
        # Keep a running sum of the total cost so far.
        for a in actions:
            #            print(a)
            tree_node: TreeNode = state.tnode_map[a[0]].node
            next_plan = plan.copy()
            next_state = state.copy()  # copying the ProgramState invokes init_frontier,
            # which finds nodes designated as frontier nodes.
            cluster_state = next_state.arr.cluster_state.copy()
            step_cost = next_state.commit_action(a)
            is_done = len(next_state.tnode_map) == 0
            next_plan.append(
                cluster_state,
                tree_node,
                a,
                step_cost,
                next_state.arr.cluster_state,
                is_done,
            )
            min_cost = self.make_plan_helper(
                next_state, cost + step_cost, depth + 1, next_plan, all_plans, min_cost
            )
        #        print("----")
        return min_cost

    def get_frontier_actions(self, state: ProgramState):
        # Get all frontier nodes.
        tnode_ids = list(state.tnode_map.keys())
        # print("frontier:", tnode_ids)
        # Collect all possible actions on each node.
        actions = []
        for tnode_id in tnode_ids:
            actions += state.tnode_map[tnode_id].actions
        # print("actions: (total)", len(actions), actions)
        return actions

    def solve(self, arr_in: GraphArray):
        arr = arr_in.copy()
        state: ProgramState = ProgramState(
            arr,
            max_reduction_pairs=self.max_reduction_pairs,
            force_final_action=self.force_final_action,
            plan_only=True,
        )
        t0 = time.time()
        if self.nprocs == 1:
            all_plans = self.make_plan(state)
        else:
            all_plans = self.make_plan_parallel_unroll(state)
        t1 = time.time()
        print("Time to make plan:", t1 - t0)
        return all_plans

    def serialize(self, pessimal=False, filename=None):
        p = self.plan
        if pessimal:
            p = self.pessimal_plan
        if filename:
            with open(filename, "wb") as f:
                dill.dump(p, f, recurse=True)
            return None
        else:
            return dill.dumps(p)


class RandomPlan(object):
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
        self.plan = Plan(max_reduction_pairs, force_final_action)

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
        actions = self.sample_actions(state)
        i = self.rs.randint(0, len(actions))
        action = actions[i]
        tree_node: TreeNode = state.tnode_map[action[0]].node
        cluster_state = state.arr.cluster_state.copy()
        cost = state.commit_action(action)
        next_cluster_state = state.arr.cluster_state.copy()
        is_done = len(state.tnode_map) == 0

        self.plan.append(
            cluster_state, tree_node, action, cost, next_cluster_state, is_done
        )
        return is_done

    def solve(self, arr_in: GraphArray):
        arr = arr_in.copy()
        state: ProgramState = ProgramState(
            arr,
            max_reduction_pairs=self.max_reduction_pairs,
            force_final_action=self.force_final_action,
            plan_only=True,
        )
        if len(state.tnode_map) == 0:
            return arr

        num_steps = 0
        while True:
            num_steps += 1
            is_done = self.step(state)
            if is_done:
                break
        return state.arr
