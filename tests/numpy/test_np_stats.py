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


# pylint: disable=import-outside-toplevel, no-member


def test_stats_1d(nps_app_inst):
    from nums import numpy as nps
    from nums.numpy import BlockArray

    assert nps_app_inst is not None

    ba: BlockArray = nps.array([5, -2, 4, 8, 3, 6, 1, 7])
    block_shapes = [(1,), (2,), (4,), (8,)]
    qs = [0, 50, 100]
    for block_shape in block_shapes:
        ba = ba.reshape(block_shape=block_shape)
        np_arr = ba.get()
        op_params = ["median", "percentile", "quantile"]
        axis_params = [None]
        keepdims_params = [False]

        for op, q, axis, keepdims in itertools.product(
            op_params, qs, axis_params, keepdims_params
        ):
            ns_op = nps.__getattribute__(op)
            np_op = np.__getattribute__(op)
            if op == "median":
                np_result = np_op(np_arr, axis=axis, keepdims=keepdims)
                ba_result: BlockArray = ns_op(ba, axis=axis, keepdims=keepdims)
            elif op == "quantile":
                q = q / 100
                np_result = np_op(np_arr, q, axis=axis, keepdims=keepdims)
                ba_result: BlockArray = ns_op(ba, q, axis=axis, keepdims=keepdims)
            assert ba_result.grid.grid_shape == ba_result.blocks.shape
            assert ba_result.size == np_result.size
            assert np.allclose(ba_result.get(), np_result)


if __name__ == "__main__":
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_stats_1d(nps_app_inst)
