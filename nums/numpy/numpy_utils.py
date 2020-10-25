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


