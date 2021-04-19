import nums
import numpy as np
import scipy as sp

import random

from nums.core.application_manager import instance as _instance
from nums.core.application_manager import destroy

from nums.core.array.sparseblockarray import SparseBlockArray
from nums.core.array.blockarray import BlockArray

w = h = 400

sparsity = int(w * h / 3)

arr = np.zeros((w, h))
ind = random.sample(range(w * h), sparsity)
ind = [(i % w, i // w) for i in ind]

for i in ind:
    arr[i] = np.random.randint(0, 100)

dtype = np.__getattribute__(str(arr.dtype))
shape = arr.shape
app = _instance()
block_shape = app.compute_block_shape(shape, dtype)

sparse_result = SparseBlockArray.from_np(
    arr,
    block_shape=block_shape,
    copy=False,
    system=app.system
) 
dense_result = BlockArray.from_np(
    arr,
    block_shape=block_shape,
    copy=False,
    system=app.system
) 

funcs = [
    lambda x: x @ x,
    lambda x: x + x,
    lambda x: x - x,
    # lambda x: x ** x,
]
for f in funcs:
    assert (f(sparse_result).get() == f(dense_result).get()).all()

destroy()