# pylint: disable = redefined-builtin

from nums.core import settings
from nums.numpy import linalg
from nums.numpy import random
from nums.numpy.api import *
from nums.numpy.api import _default_to_numpy
from nums.numpy.api import _not_implemented


# TODO(hme): Generate __all__, or control import hints some other way.


def _init():
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from nums.core.systems import utils as system_utils

    for name, func in system_utils.get_module_functions(np).items():
        if name not in globals():
            # TODO(mwe): Allow failed fallback functions to be used in default function doctests
            if hasattr(np, func.__name__) and func.__name__ in (
                settings.doctest_fallbacks
                | settings.tested_fallbacks
                | settings.untested_fallbacks
            ):
                globals()[name] = _default_to_numpy(func)
            else:
                globals()[name] = _not_implemented(func)
            if hasattr(np, func.__name__):
                globals()[name].__doc__ = np.__getattribute__(func.__name__).__doc__


_init()
del _init
