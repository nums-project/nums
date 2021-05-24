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


import argparse

from nums.core import settings
from nums.core.systems import utils as systems_utils


# pylint: disable = import-outside-toplevel


def execute(module_name, print_missing, count_fallback):
    if module_name == "api":
        return api_coverage(print_missing, count_fallback)
    elif module_name == "random":
        import numpy.random as numpy_module
        import nums.numpy.random as nums_module
    elif module_name == "linalg":
        import numpy.linalg as numpy_module
        import nums.numpy.linalg as nums_module
    elif module_name == "fft":
        import numpy.fft as numpy_module
        import nums.numpy.fft as nums_module
    else:
        raise Exception("Unknown module %s" % module_name)
    return module_coverage(module_name, print_missing, count_fallback,
                           numpy_module, nums_module)


def module_coverage(module_name, print_missing, count_fallback,
                    numpy_module, nums_module, ignore=None, fallback=None):

    print()
    print("-"*75)
    print(module_name)
    print("-"*75)

    for name, func in systems_utils.get_module_functions(nums_module).items():
        if name in ("_not_implemented", "_instance", "_default_to_numpy"):
            continue
        if getattr(numpy_module, name, None) is None:
            raise Exception("Implemented method that does not exist! %s" % name)

    ignore = [] if ignore is None else ignore
    fallback = [] if fallback is None else fallback
    coverage = 0.0
    total = 0.0
    missing = []
    for name, func in systems_utils.get_module_functions(numpy_module).items():
        if name in ignore:
            continue
        total += 1.0
        if count_fallback and name in fallback:
            coverage += 1.0
            continue
        try:
            doc_lines = func.__doc__.split("\n")
            descr = ""
            for line in doc_lines:
                descr = line.strip()
                if len(descr) != 0:
                    break
            print_tuple = name, descr
        except Exception as _:
            try:
                print_tuple = name, func.__code__.co_varnames
            except Exception as _:
                print_tuple = name, "Unavailable"
        if print_tuple is None:
            continue
        if name in fallback:
            print("Fallback %s: %s" % print_tuple)
        elif getattr(nums_module, name, None) is None:
            print("Missing %s: %s" % print_tuple)
            missing.append("%s" % name)
        else:
            coverage += 1.0

    print("-"*75)
    print(module_name)
    print("-"*75)
    print("coverage", coverage)
    print("total", total)
    print("percent covered", "%.1f" % (coverage / total * 100))
    if print_missing:
        print(str(missing))


def api_coverage(print_missing, count_fallback):
    # pylint: disable = unused-variable

    # Functions ignored for various reasons.
    ignore = {
        '_add_newdoc_ufunc', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc',
        # Deprecated
        'alen',
        # Order is generally not applicable w/ GPU backend.
        'asanyarray', 'ascontiguousarray', 'asfarray', 'asfortranarray',
        # There are no plans to provide a matrix type.
        'asmatrix', 'array2string',
        'base_repr', 'binary_repr', 'block', 'bmat', 'broadcast_arrays',
        'broadcast_to', 'busday_count', 'busday_offset', 'byte_bounds',
        'compare_chararrays',
        'datetime_as_string', 'datetime_data', 'deprecate', 'deprecate_with_doc', 'disp',
        'fastCopyAndTranspose', 'format_float_positional', 'format_float_scientific',
        'get_array_wrap', 'get_include', 'get_printoptions',
        'getbufsize', 'geterr', 'geterrcall', 'geterrobj',
        'info', 'is_busday', 'isfortran', 'isnat',
        'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable',
        'lookfor',
        'mat', 'may_share_memory',
        'ndfromtxt', 'nested_iters',
        'printoptions', 'recfromcsv', 'recfromtxt',
        'safe_eval', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize',
        'seterr', 'seterrcall', 'seterrobj', 'show_config', 'source', 'typename',
        'mafromtxt', 'mask_indices',
        # unclear whether we'll ever support these I/O operations.
        'loads', 'load', 'save',
        'savez', 'savez_compressed',
        'genfromtxt', 'fromregex', 'fromstring',
        # Memory ops that can't be supported.
        'copyto',
    }

    not_implemented = {
        'array_repr', 'array_str',
        'can_cast', 'find_common_type',
        'savetxt', 'shares_memory'
    }

    import numpy as numpy_module
    import nums.numpy.api as nums_module
    module_coverage("api", print_missing, count_fallback,
                    numpy_module, nums_module, ignore=ignore, fallback=settings.fallback)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-name', default="api", help='Which module to test.',
                        choices={"api", "random", "linalg", "fft"})

    parser.add_argument('--count-fallback', action="store_true", help='Include fallback functions '
                                                                      'in converage.')

    parser.add_argument('--print-missing', action="store_true", help='Output array of missing '
                                                                     'values.')

    args = parser.parse_args()
    args_dict = dict(vars(args).items())
    execute(**args_dict)


if __name__ == "__main__":
    main()
