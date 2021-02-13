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


import os
from pathlib import Path
import multiprocessing


pj = lambda *paths: os.path.abspath(os.path.expanduser(os.path.join(*paths)))
core_root = os.path.abspath(os.path.dirname(__file__))
package_root = pj(core_root, "../")
project_root = pj(package_root, "../")
data_dir = pj(project_root, "data")
Path(data_dir).mkdir(parents=True, exist_ok=True)


# System settings.
system_name = os.environ.get("NUMS_SYSTEM", "ray-cyclic")
# TODO (hme):
#  - Make cluster shape an environment variable. Default depends on available resources.
#  - use_head => use_driver, and should be an environment variable.
#  - Remove ray_init_default -- this should be handled in RaySystem.
use_head = True
cluster_shape = (1, 1)
ray_init_default = {
    "num_cpus": multiprocessing.cpu_count()
}


# Compute settings.
compute_name = os.environ.get("NUMS_COMPUTE", "numpy")


# NumPy operator map.
np_ufunc_map = {
    "truediv": "true_divide",
    "sub": "subtract",
    "pow": "power",
    "mult": "multiply",
    "mul": "multiply",
    "tensordot": "multiply",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "eq": "equal",
    "ne": "not_equal"
}

np_bop_reduction_set = {
    "min",
    "amin",
    "max",
    "amax",
    "nanmax",
    "nanmin",
    "nansum"
}
