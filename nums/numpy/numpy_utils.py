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


from nums.core.array.application import ArrayApplication
from nums.core.systems import utils as systems_utils
from nums.core.systems.systems import RaySystem
from nums.core.systems.schedulers import BlockCyclicScheduler
import numpy as np


def ufunc_kwargs(kwargs):
    # kwargs are currently not supported.
    if len(kwargs) > 0:
        raise NotImplementedError("ufunc kwargs currently not supported.")
    return kwargs


def ufunc_op_signatures():
    uops, bops = [], []
    for name, func in sorted(systems_utils.get_module_functions(np).items()):
        if name in ("deprecate_with_doc", "loads"):
            continue
        if name in ("isnat",):
            # These operate on data types which are currently not supported.
            # "isnat" operates on dates.
            continue
        try:
            sig: str = func.__doc__.split("\n")[0].strip()
            op_name, args = sig.split("(")
            args = args.split(")")[0]
            has_subok = "subok" in args
            if has_subok:
                # We don't support subok.
                args = args.split("subok")[0].strip()[:-1]
            args = list(map(lambda x: x.strip(), args.split(",")))
            if args == ['x', '/', 'out=None', '*', 'where=True',
                        "casting='same_kind'", "order='K'", 'dtype=None']:
                # This is a uop.
                uops.append((name, args))
            elif args == ['x1', "x2", '/', 'out=None', '*', 'where=True',
                          "casting='same_kind'", "order='K'", 'dtype=None']:
                # This is a bop.
                bops.append((name, args))
            else:
                pass
                # print(name, op_name, args)
        except Exception as e:
            pass
            # print("FAILED", name)
    return uops, bops


def get_num_cores(app: ArrayApplication):
    if isinstance(app._system, RaySystem):
        system: RaySystem = app._system
        nodes = system.nodes()
        return sum(map(lambda n: n["Resources"]["CPU"], nodes))
    else:
        raise NotImplementedError("NumPy API currently supports Ray only.")


