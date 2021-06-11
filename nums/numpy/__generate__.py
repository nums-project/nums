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


from nums.core.systems import utils as systems_utils
from nums.numpy import numpy_utils


# pylint: disable = import-outside-toplevel


def api_stub():
    import numpy as np

    def _uop_template(op_name):
        s = """

def {op_name}(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_uop(op_name="{op_name}",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))"""
        return s.format(op_name=op_name)

    def _bop_template(op_name):
        s = """

def {op_name}(x1: BlockArray, x2: BlockArray, 
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return _instance().map_bop(op_name="{op_name}",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))"""
        return s.format(op_name=op_name)

    uops, bops = numpy_utils.ufunc_op_signatures()
    for name, func in sorted(systems_utils.get_module_functions(np).items()):
        if name in ("deprecate_with_doc", "loads"):
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
            if args == [
                "x",
                "/",
                "out=None",
                "*",
                "where=True",
                "casting='same_kind'",
                "order='K'",
                "dtype=None",
            ]:
                # This is a uop.
                uops.append(_uop_template(name))
            elif args == [
                "x1",
                "x2",
                "/",
                "out=None",
                "*",
                "where=True",
                "casting='same_kind'",
                "order='K'",
                "dtype=None",
            ]:
                # This is a bop.
                bops.append(_bop_template(name))
            else:
                print(name, op_name, args)
        except Exception as _:
            print("FAILED", name)
    for sig in uops + bops:
        print(sig)
    print("num ufuncs", len(uops + bops))


def random_stub():
    # pylint: disable = unused-variable
    import numpy.random as numpy_module
    from nums.core.array.random import NumsRandomState
    from nums.core.application_manager import instance

    app = instance()
    sys = app.cm
    rs_inst = NumsRandomState(cm=sys, seed=1337)
    numpy_items = sorted(systems_utils.get_module_functions(numpy_module).items())
    nums_items = sorted(systems_utils.get_instance_functions(rs_inst).items())
    raise NotImplementedError()


def api_fallback():
    # I/O
    pass
