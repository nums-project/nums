import time

import ray
import argparse

import nums.numpy as nps
from nums.core import settings

def execute():
    # Memory used is 8 * (10**4)**2
    # So this will be 800MB object.
    size = 10 ** 4

    x1 = nps.random.randn(size, size)
    x2 = nps.random.randn(size, size)

    start = time.time()
    
    r = x1 @ x2
    r.touch()
    
    stop = time.time()

    print("--- %s seconds ---" % (stop - start))

def main(address, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address, flush=True)
    ray.init(**{
        "address": address
    })

    print("running nums operation")
    execute()




if __name__ == "__main__":
    execute()
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
   # parser.add_argument('--work-dir', default="/global/homes/m/matiashe/nums/test_ray")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
