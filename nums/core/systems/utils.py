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


import errno
import inspect
import multiprocessing
import socket
import types
from functools import wraps

import numpy as np


def get_num_cores():
    return multiprocessing.cpu_count()


def method_meta(num_returns=1):
    def inner(func):
        func.remote_params = {"num_returns": num_returns}
        return func

    return inner


def extract_functions(imp_cls, remove_self=True):
    def wrap_func(func):
        # This works for Ray, because ray.remote extracts signatures by following wrapped functions.
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    imp_functions = {}
    for name, obj in inspect.getmembers(imp_cls()):
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


def get_module_functions(module):
    # From project JAX: https://github.com/google/jax/blob/master/jax/util.py
    """
    Finds functions in module.
    :param module: module: A Python module.
    :return: A dict of names mapped to functions, builtins or ufuncs in `module`.
    """
    module_fns = {}
    for key in dir(module):
        # Omitting module level __getattr__, __dir__ which was added in Python 3.7
        # https://www.python.org/dev/peps/pep-0562/
        if key in ("__getattr__", "__dir__"):
            continue
        attr = getattr(module, key)
        # NOTE: I suspect that pylint is falsely labeling some of these types
        # as instances. See https://github.com/PyCQA/pylint/issues/3507.
        function_types = (types.BuiltinFunctionType, types.FunctionType, np.ufunc)
        if isinstance(attr, function_types):  # pylint: disable=W1116
            module_fns[key] = attr
    return module_fns


def get_instance_functions(class_inst):
    class_inst_funcs = {}
    for name, obj in inspect.getmembers(class_inst):
        if inspect.ismethod(obj):
            class_inst_funcs[name] = obj
    return class_inst_funcs


def get_private_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        private_ip = s.getsockname()[0]
    except OSError as e:
        private_ip = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                host_name = socket.getfqdn(socket.gethostname())
                private_ip = socket.gethostbyname(host_name)
            except Exception as _:
                pass
    finally:
        s.close()
    return private_ip
