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

import itertools
import numpy as np

# pylint: disable=import-outside-toplevel

def try_multiple_nd(np_atleast_nd, nps_atleast_nd):
    import nums.numpy as nps

    python_types = [
        1,
        [1, 2],
        [[1, 2]],
        [[[1, 2]]]
    ]
    all_types = python_types + list(map(np.array, python_types))
    test_cases = list(itertools.product(all_types, repeat=3))
    nps_types = list(map(nps.array, python_types))
    nps_cases = list(itertools.product(nps_types, repeat=3))

    for arguments in test_cases:
        np_res = np_atleast_nd(*arguments)
        nps_res = nps_atleast_nd(*arguments)

        for i in range(len(np_res)):
            assert np.allclose(np.array(nps_res[i].get()), np.array(np_res[i]))

    for arguments in nps_cases:
        np_args = list(map(lambda obj: np.array(obj.get()), arguments))
        np_res = np_atleast_nd(*np_args)
        nps_res = nps_atleast_nd(*arguments)
        for i in range(len(np_res)):
            assert np.allclose(np.array(nps_res[i].get()), np.array(np_res[i]))


def test_atleast_1d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    x_np = np.arange(9).reshape(3, 3)
    x_nps = nps.arange(9).reshape(3, 3)
    assert np.allclose(nps.atleast_1d(x_np).get(), np.atleast_1d(x_np))
    assert np.allclose(nps.atleast_1d(x_nps).get(), np.atleast_1d(x_np))
    assert np.atleast_1d(x_np) is x_np
    assert nps.atleast_1d(x_nps) is x_nps

    try_multiple_nd(np.atleast_1d, nps.atleast_1d)


def test_atleast_2d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_2d(x).get(), np.atleast_2d(x))

    x = np.arange(3.0)
    assert np.allclose(nps.atleast_2d(x).get(), np.atleast_2d(x))

    try_multiple_nd(np.atleast_2d, nps.atleast_2d)


def test_atleast_3d(nps_app_inst):
    import nums.numpy as nps
    assert nps_app_inst is not None

    x = 1.0
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    x = np.arange(12.0).reshape(4, 3)
    assert np.allclose(nps.atleast_1d(x).get(), np.atleast_1d(x))

    try_multiple_nd(np.atleast_3d, nps.atleast_3d)

if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_atleast_1d(nps_app_inst)
    test_atleast_2d(nps_app_inst)
    test_atleast_3d(nps_app_inst)
