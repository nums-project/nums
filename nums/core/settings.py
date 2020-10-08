# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
from pathlib import Path
import multiprocessing


pj = lambda *paths: os.path.abspath(os.path.expanduser(os.path.join(*paths)))


# System settings.
system_name = os.environ.get("NUMS_SYSTEM", "ray-cyclic")
use_head = True
cluster_shape = (1, 1)
ray_init_default = {
    "num_cpus": multiprocessing.cpu_count()
}


# Compute settings.
compute_name = os.environ.get("NUMS_COMPUTE", "numpy")


# Default block shapes for arrays with up to 2 axes.
# Block shapes can grow to approximately 1 gigabytes in size.
# Beyond 2 axes, block shape is a required parameter.
default_block_shape = (2**18, 2**9)
