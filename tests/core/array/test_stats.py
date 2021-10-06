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

from nums.core.array.application import ArrayApplication


def test_quantile_percentile(app_inst: ArrayApplication):
    # see https://github.com/dask/dask/blob/main/dask/array/tests/test_percentiles.py
    qs = [0, 50, 100]
    methods = ["tdigest"]
    interpolations = ["linear"]

    np_x = np.ones((10,))
    ba_x = app_inst.ones(shape=(10,), block_shape=(2,))
    for q, method, interpolation in itertools.product(qs, methods, interpolations):
        assert app_inst.quantile(
            ba_x, q / 100, method=method, interpolation=interpolation
        ).get() == np.quantile(np_x, q / 100)
        assert app_inst.percentile(
            ba_x, q, method=method, interpolation=interpolation
        ).get() == np.percentile(np_x, q)

    np_x = np.array([0, 0, 5, 5, 5, 5, 20, 20])
    ba_x = app_inst.array(np_x, block_shape=(3,))
    for q, method, interpolation in itertools.product(qs, methods, interpolations):
        assert app_inst.quantile(
            ba_x, q / 100, method=method, interpolation=interpolation
        ).get() == np.quantile(np_x, q / 100)
        assert app_inst.percentile(
            ba_x, q, method=method, interpolation=interpolation
        ).get() == np.percentile(np_x, q)


if __name__ == "__main__":
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    app_inst = application_manager.instance()

    test_quantile_percentile(app_inst)
