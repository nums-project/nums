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


import numpy as np

from nums.core.array.application import ArrayApplication


def test_quickselect(app_inst: ArrayApplication):
    np_x = np.array([1, 2, 3, 4, 5, 6, 7])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    ba_oids = ba_x.flattened_oids()
    k = app_inst.quickselect(ba_oids, -2)
    print(k)

if __name__ == "__main__":
    #pylint: disable=import-error
    from tests import conftest

    app_inst: ArrayApplication = conftest.get_app("serial")
    test_quickselect(app_inst)
