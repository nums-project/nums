import types

import numpy as np


def get_module_functions(module):
    # From project JAX: https://github.com/google/jax/blob/master/jax/util.py
    """Finds functions in module.
    Args:
      module: A Python module.
    Returns:
      module_fns: A dict of names mapped to functions, builtins or ufuncs in `module`.
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
