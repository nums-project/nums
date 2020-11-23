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

from nums.numpy import BlockArray


# pylint: disable=import-outside-toplevel


def test_where(nps_app_inst):
    import nums.numpy as nps

    assert nps_app_inst is not None

    shapes = [
        (),
        (10**6,),
        (10**6, 1),
        (10**5, 10)
    ]
    for shape in shapes:
        arr: BlockArray = nps.random.rand(*shape)
        if len(shape) == 1:
            arr = arr.reshape(block_shape=(arr.shape[0] // 12,))
        elif len(shape) == 2:
            arr = arr.reshape(block_shape=(arr.shape[0] // 12,
                                           arr.shape[1]))
        results: tuple = nps.where(arr < 0.5)
        np_results = np.where(arr.get() < 0.5)
        for i in range(len(np_results)):
            assert np.allclose(np_results[i], results[i].get())
        results: tuple = nps.where(arr >= 0.5)
        np_results = np.where(arr.get() >= 0.5)
        for i in range(len(np_results)):
            assert np.allclose(np_results[i], results[i].get())


if __name__ == "__main__":
    from nums.core import application_manager
    from nums.core import settings
    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_where(nps_app_inst)
