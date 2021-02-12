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


from types import FunctionType
from typing import Any, Union, List, Tuple, Dict

from nums.core.systems.utils import method_meta


class SystemInterface(object):

    def init(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()

    def put(self, value: Any):
        """
        Put object into system storage.
        """
        raise NotImplementedError()

    def get(self, object_ids: Union[Any, List]):
        """
        Get object from system storage.
        """
        raise NotImplementedError()

    def remote(self, function: FunctionType, remote_params: dict):
        """
        Return a callable remote function with remote_params.
        """
        raise NotImplementedError()

    def nodes(self):
        raise NotImplementedError()

    def get_block_addresses(self, grid):
        """
        Maps each grid entry in a grid array to a node address.
        """
        raise NotImplementedError()

    def register(self, name: str, func: callable, remote_params: dict = None):
        raise NotImplementedError("Implements a way to register new remote functions.")

    def call(self, name: str, *args, **kwargs):
        raise NotImplementedError("Implement RPC as e.g. "
                                  "self.remote_functions[name](*args, **new_kwargs)")

    def call_with_options(self, name: str, args, kwargs, options):
        raise NotImplementedError("Implement RPC with options support.")

    def get_options(self, cluster_entry, cluster_shape):
        # TODO (hme): API needs improvements in this area.
        raise NotImplementedError()


class ComputeInterface(object):

    def touch(self, arr, syskwargs: Dict):
        """
        "Touch" the given array. This returns nothing and can be used to wait for
        a computation without pulling data to the head node.
        """
        raise NotImplementedError()

    def empty(self, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def new_block(self, op_name: str, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def random_block(self, rng_params, rfunc_name, rfunc_args, shape, dtype, syskwargs: Dict):
        raise NotImplementedError()

    def permutation(self, rng_params, size, syskwargs: Dict):
        raise NotImplementedError()

    def diag(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def arange(self, start, stop, step, dtype, syskwargs: Dict):
        raise NotImplementedError()

    def sum_reduce(self, *arrs, syskwargs: Dict):
        raise NotImplementedError()

    def transpose(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def create_block(self, *src_arrs, src_params, dst_params, dst_shape, dst_shape_bc,
                     syskwargs: Dict):
        raise NotImplementedError()

    def update_block(self, dst_arr, *src_arrs, src_params, dst_params, syskwargs: Dict):
        raise NotImplementedError()

    def update_block_by_index(self, dst_arr, src_arr, index_pairs, syskwargs: Dict):
        raise NotImplementedError()

    def update_block_along_axis(self, dst_arr, src_arr, index_pairs, axis, syskwargs: Dict):
        raise NotImplementedError()

    def bop(self, op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes, syskwargs: Dict):
        raise NotImplementedError()

    def bop_reduce(self, op, a1, a2, a1_T, a2_T, syskwargs: Dict):
        raise NotImplementedError()

    def split(self, arr, indices_or_sections, axis, transposed, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_returns=2)
    def qr(self, *arrays, mode="reduced", axis=None, syskwargs: Dict):
        raise NotImplementedError()

    def cholesky(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_returns=3)
    def svd(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def inv(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def array_compare(self, func_name, a, b, args, syskwargs: Dict):
        raise NotImplementedError()

    def map_uop(self, op_name, arr, args, kwargs, syskwargs: Dict):
        raise NotImplementedError()

    def where(self, arr, x, y, block_slice_tuples, syskwargs: Dict):
        raise NotImplementedError()

    def reduce_axis(self, op_name, arr, axis, keepdims, transposed, syskwargs: Dict):
        raise NotImplementedError()

    # Scipy

    def xlogy(self, arr_x, arr_y, syskwargs: Dict):
        raise NotImplementedError()

    def logical_and(self, *bool_list, syskwargs: Dict):
        raise NotImplementedError()

    def astype(self, arr, dtype_str, syskwargs: Dict):
        raise NotImplementedError()

    def arg_op(self, op_name, arr, block_slice, other_argoptima, other_optima,
               syskwargs: Dict):
        raise NotImplementedError()

    def reshape(self, arr, shape, syskwargs: Dict):
        raise NotImplementedError()


class ComputeImp(object):
    pass


class RNGInterface(object):

    def new_block_rng_params(self):
        raise NotImplementedError()
