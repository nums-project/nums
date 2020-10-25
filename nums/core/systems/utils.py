import types
import inspect
from functools import wraps
import multiprocessing

import numpy as np


def get_num_cores():
    return multiprocessing.cpu_count()


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
        if key in ('__getattr__', '__dir__'):
            continue
        attr = getattr(module, key)
        if isinstance(attr, (types.BuiltinFunctionType, types.FunctionType, np.ufunc)):
            module_fns[key] = attr
    return module_fns
