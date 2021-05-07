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

from nums.core.storage.storage import BimodalGaussian, ArrayGrid
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.sparseblockarray import SparseBlockArray

# pylint: disable=wrong-import-order

def test_from_np(app_inst: ArrayApplication):

    w = h = 400

    arr = np.zeros((w, h))
    
    dtype = np.__getattribute__(str(arr.dtype))
    shape = arr.shape
    block_shape = app_inst.compute_block_shape(shape, dtype)

    sparse_result = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app_inst.system
        ) 
    1+1

def test_from_np_blocks(app_inst: ArrayApplication):
    w = h = 400

    arr = np.zeros((w, h))
    
    dtype = np.__getattribute__(str(arr.dtype))
    shape = arr.shape
    block_shape = (100, 100)

    sparse_result = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app_inst.system
        )

def test_ops(app_inst: ArrayApplication):
    w = h = 400

    arr = np.zeros((w, h))
    
    dtype = np.__getattribute__(str(arr.dtype))
    shape = arr.shape
    block_shape = app_inst.compute_block_shape(shape, dtype)

    sparse_result = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app_inst.system
    )

    dense_result = BlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app_inst.system
    )

    funcs = [
        lambda x: x @ x,
        lambda x: x + x,
        lambda x: x - x,
        # lambda x: x ** x,
    ]
    for f in funcs:
        assert (f(sparse_result).get() == f(dense_result).get()).all()



if __name__ == "__main__":
    # pylint: disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_from_np_blocks(app_inst)



