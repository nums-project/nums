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


import itertools

import numpy as np


# pylint: disable=import-outside-toplevel, no-member


def test_reductions(nps_app_inst):
    from nums import numpy as nps
    from nums.numpy import BlockArray

    assert nps_app_inst is not None

    ba: BlockArray = nps.array([[5, -2, 4, 8], [3, 6, 1, 7]])
    block_shapes = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 4)]
    for block_shape in block_shapes:
        ba = ba.reshape(block_shape=block_shape)
        np_arr = ba.get()
        op_params = ["amin", "min", "amax", "max", "sum", "mean", "var", "std"]
        axis_params = [None, 0, 1]
        keepdims_params = [True, False]

        for op, axis, keepdims in itertools.product(
            op_params, axis_params, keepdims_params
        ):
            ns_op = nps.__getattribute__(op)
            np_op = np.__getattribute__(op)
            np_result = np_op(np_arr, axis=axis, keepdims=keepdims)
            ba_result: BlockArray = ns_op(ba, axis=axis, keepdims=keepdims)
            assert ba_result.grid.grid_shape == ba_result.blocks.shape
            assert ba_result.shape == np_result.shape
            assert np.allclose(ba_result.get(), np_result)

        # Also test ddof for std.
        # ba_result = nps.std(ba, axis=0, ddof=1)
        # np_result = np.std(np_arr, axis=0, ddof=1)
        # assert np.allclose(ba_result.get(), np_result)


def test_argops(nps_app_inst):
    from nums import numpy as nps
    from nums.numpy import BlockArray

    assert nps_app_inst is not None

    bas = [
        nps.array([5, -2, 4, 8]),
        nps.array([1, 2, 3, 4]),
        nps.array([3, 2, 1, 0]),
        nps.array([-1, -2, -3, -0]),
    ]
    block_shapes = [(1,), (2,), (3,), (4,)]
    for ba in bas:
        for block_shape in block_shapes:
            ba = ba.reshape(block_shape=block_shape)
            np_arr = ba.get()
            op_params = ["argmin", "argmax"]
            axis_params = [None, 0]

            for op, axis in itertools.product(op_params, axis_params):
                ns_op = nps.__getattribute__(op)
                np_op = np.__getattribute__(op)
                np_result = np_op(np_arr, axis=axis)
                ba_result: BlockArray = ns_op(ba, axis=axis)
                assert ba_result.grid.grid_shape == ba_result.blocks.shape
                assert ba_result.shape == np_result.shape
                assert np.allclose(ba_result.get(), np_result)


def test_average(nps_app_inst):
    from nums import numpy as nps
    from nums.numpy import BlockArray

    assert nps_app_inst is not None

    bas = [
        nps.array(
            [
                [[5, -2, 4, 8], [1, 2, 3, 4], [3, 2, 1, 0], [-1, -2, -3, 0]],
                [[6, -4, 2, 7], [4, 3, 2, 1], [1, 2, 0, 3], [0, -2, -3, -1]],
            ]
        ),
        nps.array(
            [
                [[5, -2, 4, 8], [1, 2, 3, 4], [3, 2, 1, 0], [-1, -2, -3, 0]],
                [[6, -4, 2, 7], [4, 3, 2, 1], [1, 2, 0, 3], [0, -2, -3, -1]],
            ]
        ),
    ]
    ba_wts = [
        None,
        nps.array(
            [
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                [
                    [-2, -3, -4, -5],
                    [-2, -3, -4, -5],
                    [-2, -3, -4, -5],
                    [-2, -3, -4, -5],
                ],
            ]
        ),
    ]
    ba_shapes = [(1, 1, 1), (1, 2, 2), (2, 1, 4), (2, 4, 2), (2, 4, 4)]
    for ba, ba_wt in zip(bas, ba_wts):
        for ba_shape in ba_shapes:
            ba = ba.reshape(block_shape=ba_shape)
            if ba_wt:
                ba_wt = ba_wt.reshape(block_shape=ba_shape)
            np_arr = ba.get()
            np_wt = ba_wt.get() if ba_wt else None
            op_params = ["average"]
            axis_params = [None, 0, 1, 2]

            for op, axis in itertools.product(op_params, axis_params):
                ns_op = nps.__getattribute__(op)
                np_op = np.__getattribute__(op)
                np_result = np_op(np_arr, axis=axis, weights=np_wt, returned=True)
                ba_result: BlockArray = ns_op(
                    ba, axis=axis, weights=ba_wt, returned=True
                )

                np_avg, np_ws = np_result[0], np_result[1]
                ba_avg, ba_ws = ba_result[0], ba_result[1]
                assert np.allclose(ba_ws.get(), np_ws)
                assert ba_avg.grid.grid_shape == ba_avg.blocks.shape
                assert ba_avg.shape == np_avg.shape
                assert np.allclose(ba_avg.get(), np_avg)

    # Test zero division error
    ba = nps.array(
        [
            [[5, -2, 4, 8], [1, 2, 3, 4], [3, 2, 1, 0], [-1, -2, -3, 0]],
            [[6, -4, 2, 7], [4, 3, 2, 1], [1, 2, 0, 3], [0, -2, -3, -1]],
        ]
    )
    ba_wt = nps.array(
        [
            [[1, 2, 3, 4], [1, 2, 3, -12], [1, 2, 3, 4], [1, 2, 3, 4]],  # axis 1
            [
                [-2, -3, 10, -5],  # axis 2
                [-2, -3, -4, -5],
                [-2, -2, -4, -5],  # axis 0
                [-2, -3, -4, -5],
            ],
        ]
    )
    for ax in range(3):
        err_match = True
        try:
            np.average(ba.get(), axis=ax, weights=ba_wt.get(), returned=False)
            err_match = False
        except ZeroDivisionError:
            pass
        try:
            nps.average(ba, axis=ax, weights=ba_wt, returned=False)
            err_match = False
        except ZeroDivisionError:
            pass
        assert err_match


if __name__ == "__main__":
    from nums.core import application_manager
    import nums.core.settings

    nums.core.settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_reductions(nps_app_inst)
    test_argops(nps_app_inst)
    test_average(nps_app_inst)
