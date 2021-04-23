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

    size = 10**2
    # Memory used is 8 * (10**4)**2
    # So this will be 800MB object.
    x1 = nps.random.randn(size, size)
    x2 = nps.random.randn(size, size)

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
