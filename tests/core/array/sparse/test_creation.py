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

import random

from nums.core.storage.storage import BimodalGaussian, ArrayGrid
from nums.core.array.application import ArrayApplication
from nums.core.array.sparseblockarray import SparseBlockArray

def test_from_np(app_inst: ArrayApplication):
    w = h = 400
    sparsity = int(w * h / 3)

    arr = np.zeros((w, h))
    ind = random.sample(range(w * h), sparsity)
    ind = [(i % w, i // w) for i in ind]
    for i in ind:
        arr[i] = np.random.randint(0, 100)

    block_shape = (1, 1)
    
    x1 = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app_inst.system
    )

    x2 = SparseBlockArray.from_np(
        arr,
        block_shape=(4, 4),
        copy=False,
        system=app_inst.system
    )

    assert np.all(arr == x1.get())
    assert np.all(arr == x2.get())

if __name__ == "__main__":
    # pylint: disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_from_np(app_inst)
