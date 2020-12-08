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

import time

import numpy as np

from nums.core import settings
from nums.core import application_manager
from nums.core.array.application import ArrayApplication


def test_app_manager():
    for compute_name in ["numpy"]:
        for system_name in ["serial", "ray-cyclic", "ray-task"]:
            settings.compute_name = compute_name
            settings.system_name = system_name
            app: ArrayApplication = application_manager.instance()
            assert np.allclose(np.arange(10), app.arange(shape=(10,), block_shape=(10,)).get())
            application_manager.destroy()
            assert not application_manager.is_initialized()
            time.sleep(1)


if __name__ == "__main__":
    test_app_manager()
