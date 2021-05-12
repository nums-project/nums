import time

import ray
import argparse

import nums.numpy as nps
from nums.core import settings

from scipy.sparse import random
from scipy import stats


def routine(x1, x2):
    result = x1 @ x2
    print(result.get_shape())

def run():
    print("running nums operation")
    size = 5000

    # Memory used is 8 * (10**4)**2
    # So this will be 800MB object.
    x1 = random(size, size, density=0.25)
    x2 = random(size, size, density=0.25)

    start = time.time()
    routine(x1, x2)
    end = time.time()

    print("--- %s seconds ---" % (end - start))

run()