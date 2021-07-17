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

import re

import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.systems import utils as systems_utils


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
            _, args = sig.split("(")
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
                uops.append((name, args))
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
                bops.append((name, args))
            else:
                pass
                # print(name, op_name, args)
        except Exception as _:
            pass
            # print("FAILED", name)
    return uops, bops


def get_num_cores(app: ArrayApplication):
    return app.cm.num_cores_total()


def update_doc_string(doc_string):
    # TODO(mwe): Continue to improve regex edge cases
    # np -> nps
    doc_string = re.sub("np(?![a-zA-Z_])", "nps", str(doc_string))
    # Remove comments
    doc_string = re.sub("#.*", "", doc_string)
    # Round all numbers
    nums_funcs = list(
        set(re.findall(r">>>\s[\.a-zA-Z0-9, ()].*\n(?!\s*>)", doc_string))
    )
    for nums_func in nums_funcs:
        doc_string = doc_string.replace(
            nums_func,
            '>>> eval("np.around(' + nums_func[4:].rstrip() + '.get(), 5)")\n',
        )
    numbers = list(set(re.findall(r"-?[0-9]\d{0,9}\.\d+", doc_string)))
    for num in numbers:
        doc_string = doc_string.replace(num, str(np.round(float(num), 5)))
    # Convert array(<num>) to num: e.g. array(11) -> 11
    ret_nums = list(
        set(re.findall(r"(?<=\n\s)*array\(-?[0-9]\d{0,9}\.\d+\)", doc_string))
    )
    for ret_num in ret_nums:
        doc_string = doc_string.replace(ret_num, ret_num[6:-1])
    return doc_string
