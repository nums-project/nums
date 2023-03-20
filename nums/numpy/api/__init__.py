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

# pylint: disable = redefined-builtin, too-many-lines, anomalous-backslash-in-string, unused-wildcard-import, wildcard-import

import warnings

from nums.numpy.api.algebra import *
from nums.numpy.api.constants import *
from nums.numpy.api.creation import *
from nums.numpy.api.generated import *
from nums.numpy.api.manipulation import *
from nums.numpy.api.nan import *
from nums.numpy.api.arithmetic import *
from nums.numpy.api.properties import *
from nums.numpy.api.sort import *
from nums.numpy.api.stats import *
from nums.numpy.api.logic import *


def _not_implemented(func):
    # From project JAX: https://github.com/google/jax/blob/master/jax/numpy/lax_numpy.py
    def wrapped(*args, **kwargs):
        # pylint: disable=unused-argument
        msg = "NumPy function {} not yet implemented."
        raise NotImplementedError(msg.format(func))

    return wrapped


# TODO (mwe): Convert this to invoke the NumPy op on a worker instead of the driver.
def _default_to_numpy(func):
    def wrapped(*args, **kwargs):
        warnings.warn(
            "Operation "
            + func.__name__
            + " not implemented, falling back to NumPy. "
            + "If this is too slow or failing, please open an issue on GitHub.",
            RuntimeWarning,
        )
        new_args = [arg.get() if isinstance(arg, BlockArray) else arg for arg in args]
        new_kwargs = {
            k: v.get() if isinstance(v, BlockArray) else v
            for k, v in zip(kwargs.keys(), kwargs.values())
        }
        res = np.__getattribute__(func.__name__)(*new_args, **new_kwargs)
        if isinstance(res, tuple):
            nps_res = tuple(array(x) for x in res)
        else:
            nps_res = array(res)
        return nps_res

    return wrapped
