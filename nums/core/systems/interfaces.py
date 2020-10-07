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


import inspect
from functools import wraps

from types import FunctionType
from typing import Any, Union, List, Tuple, Dict


def method_meta(num_return_vals=1):
    def inner(func):
        func.remote_params = {
            "num_return_vals": num_return_vals
        }
        return func
    return inner


def extract_functions(imp, remove_self=True):

    def wrap_func(func):
        # This works for Ray, because ray.remote extracts signatures by following wrapped functions.
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    imp_functions = {}
    for name, obj in inspect.getmembers(imp()):
        if inspect.ismethod(obj):
            if remove_self:
                imp_functions[name] = wrap_func(obj)
            else:
                imp_functions[name] = obj
    return imp_functions


def check_implementation(interface_cls, imp):
    imp_functions = extract_functions(imp, remove_self=False)
    required_methods = inspect.getmembers(interface_cls(), predicate=inspect.ismethod)
    for name, func in required_methods:
        # Make sure the function exists.
        assert name in imp_functions, "%s not implemented." % name
        # Make sure all parameters are there.
        for varname in func.__code__.co_varnames:
            # Ignore matching on self and scheduling args.
            if varname == "self":
                continue
            if varname == "syskwargs":
                continue
            assert varname in imp_functions[name].__code__.co_varnames


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_interface_method_meta(interface_cls):
    required_methods = inspect.getmembers(interface_cls(), predicate=inspect.ismethod)
    result = {}
    for name, func in required_methods:
        try:
            remote_params = func.remote_params
        except Exception as _:
            remote_params = get_default_args(method_meta)
        result[name] = remote_params
    return result


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
        # TODO: API needs improvements in this area.
        raise NotImplementedError()


class ComputeInterface(object):

    def touch(self, arr, syskwargs: Dict):
        """
        "Touch" the given array. This returns nothing and can be used to wait for
        a computation without pulling data to the head node.
        """
        raise NotImplementedError()

    def zeros(self, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def empty(self, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def new_block(self, op_name: str, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def random_block(self, rng_params, rfunc_name, rfunc_args, shape, dtype, syskwargs: Dict):
        raise NotImplementedError()

    def diag(self, arr, syskwargs: Dict):
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

    def bop(self, op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes, syskwargs: Dict):
        raise NotImplementedError()

    def split(self, arr, indices_or_sections, axis, transposed, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_return_vals=2)
    def qr(self, *arrays, mode="reduced", axis=None, syskwargs: Dict):
        raise NotImplementedError()

    def cholesky(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def inv_sym_psd(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_return_vals=3)
    def svd(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def inv(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def allclose(self, a, b, rtol, atol, syskwargs: Dict):
        raise NotImplementedError()

    # This is essentially a map.
    def ufunc(self, op_name, arr, syskwargs: Dict):
        raise NotImplementedError()

    def xlogy(self, arr_x, arr_y, syskwargs: Dict):
        raise NotImplementedError()

    def reduce_axis(self, op_name, arr, axis, keepdims, transposed, syskwargs: Dict):
        raise NotImplementedError()

    # Lapack

    def lapack_dtrtri(self, arr, lower, unitdiag, overwrite_c, syskwargs: Dict):
        raise NotImplementedError()

    def lapack_strtri(self, arr, lower, unitdiag, overwrite_c, syskwargs: Dict):
        raise NotImplementedError()

    def logical_and(self, *bool_list, syskwargs: Dict):
        raise NotImplementedError()

    def astype(self, arr, dtype_str, syskwargs: Dict):
        raise NotImplementedError()


class ComputeImp(object):
    pass


class RNGInterface(object):

    def new_block_rng_params(self):
        raise NotImplementedError()
