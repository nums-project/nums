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


import pytest


# pylint: disable=import-outside-toplevel
@pytest.mark.skip
def test_dask_system():
    import numpy as np
    from nums.core import linalg
    import nums.numpy as nps
    from nums.core import settings
    from nums.core.array.application import ArrayApplication
    from nums.core.array.blockarray import BlockArray
    from nums.core import application_manager
    from nums.experimental.nums_dask.dask_system import DaskSystem
    from nums.core.systems.utils import get_num_cores

    prev_settings = (
        settings.system_name,
        settings.device_grid_name,
        settings.cluster_shape,
    )
    settings.device_grid_name = "cyclic"
    settings.system_name = "dask"
    settings.cluster_shape = (get_num_cores(), 1)
    assert not application_manager.is_initialized()
    app: ArrayApplication = application_manager.instance()

    assert isinstance(app.cm.system, DaskSystem)

    X: BlockArray = app.random.normal(shape=(3, 3), block_shape=(3, 3))
    Y: BlockArray = app.random.normal(shape=(3, 3), block_shape=(3, 3))
    Z: BlockArray = X @ Y
    assert np.allclose(Z.get(), X.get() @ Y.get())

    X: BlockArray = app.random.normal(shape=(10, 20), block_shape=(3, 5))
    Y: BlockArray = app.random.normal(shape=(20, 30), block_shape=(5, 6))
    Z: BlockArray = X @ Y
    assert np.allclose(Z.get(), X.get() @ Y.get())

    X: BlockArray = app.random.normal(shape=(100, 20), block_shape=(23, 5))
    Q, R = linalg.indirect_tsqr(app, X)
    assert nps.allclose(Q @ R, X)

    Q, R = linalg.direct_tsqr(app, X)
    assert nps.allclose(Q @ R, X)
    application_manager.destroy()
    (
        settings.system_name,
        settings.device_grid_name,
        settings.cluster_shape,
    ) = prev_settings


if __name__ == "__main__":
    test_dask_system()
