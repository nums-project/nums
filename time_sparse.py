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

from nums.core.storage.storage import ArrayGrid

import argparse

import ray
import nums.numpy as nps
import nums
from nums.core import settings
from scipy import sparse

def random_sparse_block(shape, density):
    m, n = shape
    return sparse.random(m, n, density)

def sample(shape, dtype, block_shape, system):
    dtype_str = str(dtype)
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



    if dtype is None:
        dtype = np.float64
    assert isinstance(dtype, type)

    dtype = np.float64
    app = _instance()
    block_shape = app.compute_block_shape(shape, dtype)

    grid = ArrayGrid(shape, block_shape, dtype=dtype.__name__)
    ba = BlockArray(grid, app.system)
    for grid_entry in ba.grid.get_entry_iterator():
        block = ba.blocks[grid_entry]
        block.oid = app.system.put(random_sparse_block(shape, 0.25))
        
    return ba

def main(address, work_dir, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address, flush=True)
    # ray.init(**{
    #     "address": address
    # })
    ray.init()

    print("running nums operation")
        
    start = time.time()
    
    print(x1.touch())
    
    end = time.time()

    print("--- %s seconds ---" % (end - start))


if __name__ == "__main__":

    shape = (4, 4)
    dtype = np.float64
    app = _instance()
    block_shape = app.compute_block_shape(shape, dtype)

    x1 = sample(shape, dtype, block_shape, app.system)
    x2 = sample(shape, dtype, block_shape, app.system)

    print(x1.get())

    r = x1 @ x2
    
    print(r.get())

    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/m/matiashe/nums/test_ray")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
