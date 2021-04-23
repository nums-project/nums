import nums
import numpy as np
import nums.numpy as nps
import scipy as sp

import random
import time

from nums.core.application_manager import instance as _instance
from nums.core.application_manager import destroy

from nums.core.array.sparseblockarray import SparseBlockArray
from nums.core.array.blockarray import BlockArray

import argparse

import ray
import nums.numpy as nps
import nums
from nums.core import settings


def random_sparse(sparsity):

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

    x1 = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app.system
    )


def main(address, work_dir, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address, flush=True)
    ray.init(**{
        "address": address
    })

    print("running nums operation")
        
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

    x1 = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app.system
    )

    x2 = SparseBlockArray.from_np(
        arr,
        block_shape=block_shape,
        copy=False,
        system=app.system
    )

    # size = 10**2
    # # Memory used is 8 * (10**4)**2
    # # So this will be 800MB object.
    # x1 = nps.random.randn(size, size)
    # x2 = nps.random.randn(size, size)

    start = time.time()
    
    result = x1 @ x2
    print(result.touch())
    
    end = time.time()

    print("--- %s seconds ---" % (end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/m/matiashe/nums/test_ray")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")

    print("Starting", flush=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
