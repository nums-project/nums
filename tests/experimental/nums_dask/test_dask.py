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


# pylint: disable=import-outside-toplevel
def test_dask_backend():
    import numpy as np
    from nums.core import linalg
    import nums.numpy as nps
    from nums.core import settings
    from nums.core.array.application import ArrayApplication
    from nums.core.array.blockarray import BlockArray
    from nums.core import application_manager
    from nums.experimental.nums_dask.dask_backend import DaskBackend

    settings.backend_name = "dask"
    assert not application_manager.is_initialized()
    app: ArrayApplication = application_manager.instance()

    assert isinstance(app.km.backend, DaskBackend)

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


if __name__ == "__main__":
    test_dask_backend()
