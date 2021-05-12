import argparse
import time

import ray
import nums.numpy as nps
from nums.core import settings

def routine(x1, x2):
    result = x1 @ x2
    print(result.touch())

def run():
    print("running nums operation")
    size = 5000
    density = 0.25

    # Memory used is 8 * (10**4)**2
    # So this will be 800MB object.
    x1 = nps.random.randn_sparse(size, size, density=density)
    x2 = nps.random.randn_sparse(size, size, density=density)

    start = time.time()
    routine(x1, x2)
    end = time.time()

    print("--- %s seconds ---" % (end - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', default=False, type=bool)
    parser.add_argument('--address', default="")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()

    if args.local:
        ray.init()
    else:
        settings.use_head = args.use_head
        settings.cluster_shape = tuple(map(lambda x: int(x), args.cluster_shape.split(",")))
        print("use_head", args.use_head)
        print("cluster_shape", args.cluster_shape)
        print("connecting to head node", args.address)
        ray.init(**{
            "address": args.address
        })

    run()


