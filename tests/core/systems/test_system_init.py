# Copyright (C) NumS Development Team.
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


import ray

from nums.core.systems.systems import RaySystem
from nums.core.systems import utils as systems_utils
from nums.core import settings

# pylint: disable=protected-access


def test_head_detection():
    ray.init()

    assert settings.head_ip is None
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is not None
    sys.shutdown()

    settings.head_ip = "1.2.3.4"
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is None
    sys.shutdown()

    settings.head_ip = systems_utils.get_private_ip()
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is not None
    sys.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    test_head_detection()
