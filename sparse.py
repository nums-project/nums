import time
import argparse
import ray

import nums
import numpy as np
import nums.numpy as nps
import scipy as sp

from nums.core.application_manager import instance as _instance
from nums.core.array.sparseblockarray import SparseBlockArray
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

def execute(size):
    shape = (size, size)
    dtype = np.float64
    app = _instance()
    block_shape = app.compute_block_shape(shape, dtype)

    x1 = sample(shape, block_shape, app.system)
    x2 = sample(shape, block_shape, app.system)

    start = time.time()

    r = x1 @ x2
    r.touch()

    stop = time.time()
    
    print("--- %s seconds ---" % (stop - start))


def sample(shape, block_shape, system):
    grid = ArrayGrid(shape, block_shape, 'float64')
    rarr = SparseBlockArray(grid, system)
    grid_entry_iterator = grid.get_entry_iterator()
    for grid_entry in grid_entry_iterator:
        grid_slice = grid.get_slice(grid_entry)
        
        fst = grid_slice[0]
        snd = grid_slice[1]
        
        block = sp.sparse.random(fst.stop - fst.start, snd.stop - snd.start, 0.25)

        rarr.blocks[grid_entry].oid = system.put(block)
        rarr.blocks[grid_entry].dtype = getattr(np, 'float64')

    return rarr


def main(address, use_head, cluster_shape, size):
    #settings.use_head = use_head
    #settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address, flush=True)
    # ray.init(**{
    #     "address": address
    # })
    ray.init()
    print("connected")
    
    execute(size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--size', type=int, default="100")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)

    print("Starting")
    main(**kwargs)
    print("Completed")

    #main(**kwargs)
