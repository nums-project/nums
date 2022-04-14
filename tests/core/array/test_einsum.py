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


def test_einsum(app_inst: ArrayApplication):
    rs: np.random.RandomState = np.random.RandomState(1337)
    I, J, K = 10, 11, 12
    X_np = rs.rand(I, J, K)

    F = 3
    A_np = np.random.rand(I, F)
    B_np = np.random.rand(J, F)
    C_np = np.random.rand(K, F)

    X_ba = app_inst.array(X_np, (5, 4, 3))
    B_ba = app_inst.array(B_np, (4, 2))
    C_ba = app_inst.array(C_np, (3, 2))

    # MTTKRP
    R_np = np.einsum("ijk,jf,kf->if", X_np, B_np, C_np)
    assert R_np.shape == A_np.shape
    R_ba = app_inst.einsum("ijk,jf,kf->if", X_ba, B_ba, C_ba)
    assert np.allclose(R_np, R_ba.get())


if __name__ == "__main__":
    # pylint: disable=import-error
    import conftest

    app_inst = conftest.get_app("serial", "cyclic")
    test_einsum(app_inst)
