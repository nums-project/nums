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
import warnings
import numpy as np

warnings.filterwarnings("ignore", "All-NaN (slice|axis) encountered")
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
warnings.filterwarnings("ignore", "Degrees of freedom <= 0 for slice.")


# pylint: disable=import-outside-toplevel, no-member


def test_nan_reductions(nps_app_inst):
    from nums import numpy as nps
    from nums.numpy import BlockArray

    assert nps_app_inst is not None

    ba: BlockArray = nps.array([[-1, 4, np.nan, 5],
                                [3, 2, nps.nan, 6]])
    block_shapes = [(1, 1),
                    (1, 2),
                    (1, 4),
                    (2, 1),
                    (2, 4)]
    for block_shape in block_shapes:
        ba = ba.reshape(block_shape=block_shape)
        np_arr = ba.get()
        op_params = ["nanmax", "nanmin", "nansum", "nanmean", "nanvar", "nanstd"]
        axis_params = [None, 0, 1]
        keepdims_params = [True, False]

        for op, axis, keepdims in itertools.product(op_params, axis_params, keepdims_params):
            ns_op = nps.__getattribute__(op)
            np_op = np.__getattribute__(op)
            np_result = np_op(np_arr, axis=axis, keepdims=keepdims)
            ba_result: BlockArray = ns_op(ba, axis=axis, keepdims=keepdims)
            assert ba_result.grid.grid_shape == ba_result.blocks.shape
            assert ba_result.shape == np_result.shape
            assert np.allclose(ba_result.get(), np_result, equal_nan=True)


if __name__ == "__main__":
    from nums.core import application_manager
    import nums.core.settings
    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_nan_reductions(nps_app_inst)
