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


import numpy as np
from nums.models.glms import LogisticRegression
from nums.core.array.application import ArrayApplication


def sample_set(app: ArrayApplication):
    shape = (500, 10)
    block_shape = (100, 10)
    rs = app.random_state(1337)
    X1 = rs.normal(loc=5.0, shape=shape, block_shape=block_shape)
    y1 = app.zeros(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X2 = rs.normal(loc=10.0, shape=shape, block_shape=block_shape)
    y2 = app.ones(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X = app.concatenate([X1, X2], axis=0)
    y = app.concatenate([y1, y2], axis=0)
    return X, y


def test_lbfgs(nps_app_inst):
    assert nps_app_inst is not None
    X, y = sample_set(nps_app_inst)
    model = LogisticRegression(solver="lbfgs", max_iter=30)
    model.fit(X, y)
    y_pred = model.predict(X)
    error = (
        (nps_app_inst.sum(nps_app_inst.abs(y - y_pred)) / X.shape[0])
        .astype(np.float64)
        .get()
    )
    print("error", error)
    assert error < 0.25


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    test_lbfgs(nps_app_inst)
