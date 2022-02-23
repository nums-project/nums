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


import psutil

from nums.core.kernel import numpy_kernel
from nums.core.backends import utils as backend_utils


def test_utils():
    r = backend_utils.get_module_functions(backend_utils)
    assert len(r) > 0
    r = backend_utils.get_instance_functions(numpy_kernel.KernelCls())
    assert len(r) > 0


def test_num_cpus():
    all_cores = psutil.cpu_count(logical=False)
    returned_cores = backend_utils.get_num_cores(reserved_for_os=0)
    assert all_cores == returned_cores
    returned_cores = backend_utils.get_num_cores(reserved_for_os=2)
    if all_cores <= returned_cores:
        # CI machines may have few cores.
        assert all_cores == returned_cores
    else:
        assert all_cores - 2 == returned_cores


if __name__ == "__main__":
    # pylint: disable=import-error
    test_utils()
    test_num_cpus()
