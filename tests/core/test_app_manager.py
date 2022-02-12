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

import time

import numpy as np
import pytest

from nums.core import application_manager
from nums.core import settings
from nums.core.array.application import ArrayApplication
from nums.core.systems.utils import get_num_cores


@pytest.mark.parametrize("compute_name", ["numpy"])
@pytest.mark.parametrize("system_name", ["serial", "ray", "ray-scheduler"])
@pytest.mark.parametrize("device_grid_name", ["cyclic", "packed"])
@pytest.mark.parametrize("num_cpus", [2, 1, None])
def test_app_manager(compute_name, system_name, device_grid_name, num_cpus):
    settings.use_head = True
    settings.compute_name = compute_name
    settings.system_name = system_name
    settings.device_grid_name = device_grid_name
    settings.num_cpus = num_cpus

    app: ArrayApplication = application_manager.instance()
    print(settings.num_cpus, num_cpus, app.cm.num_cores_total())
    app_arange = app.arange(0, shape=(10,), block_shape=(10,))
    assert np.allclose(np.arange(10), app_arange.get())
    if num_cpus is None:
        assert app.cm.num_cores_total() == get_num_cores()
    else:
        assert app.cm.num_cores_total() == num_cpus
    application_manager.destroy()
    assert not application_manager.is_initialized()
    time.sleep(1)

    # Revert for other tests.
    settings.compute_name = "numpy"
    settings.system_name = "ray"
    settings.device_grid_name = "cyclic"
    settings.num_cpus = None


if __name__ == "__main__":
    test_app_manager("numpy", "mpi", "cyclic", 1)
