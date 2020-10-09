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


from typing import Tuple, Iterator

import itertools
import numpy as np


def get_ufunc_output_type(op_name, dtype):
    a = np.array(1, dtype=dtype)
    result_dtype = np.__getattribute__(op_name)(a).dtype
    return np.__getattribute__(str(result_dtype))


def get_bop_output_type(op_name, dtype_a, dtype_b):
    a = np.array(1, dtype=dtype_a)
    b = np.array(2, dtype=dtype_b)
    op_name = {"truediv": "true_divide",
               "sub": "subtract",
               "pow": "power",
               "mult": "multiply",
               "mul": "multiply",
               "tensordot": "multiply",
               }.get(op_name, op_name)
    dtype = np.__getattribute__(op_name)(a, b).dtype
    return np.__getattribute__(str(dtype))


def shape_from_block_array(arr: np.ndarray):
    grid_shape = arr.shape
    num_axes = len(arr.shape)
    shape = np.zeros(num_axes, dtype=np.int)
    for j in range(num_axes):
        pos = [[0]] * num_axes
        pos[j] = range(grid_shape[j])
        j_iter = list(itertools.product(*pos))
        for j_access in j_iter:
            shape[j] += arr[j_access].shape[j]
    return tuple(shape)


def broadcast(a_shape, b_shape):
    a_view = np.lib.stride_tricks.broadcast_to(0, a_shape)
    b_view = np.lib.stride_tricks.broadcast_to(0, b_shape)
    return np.broadcast(a_view, b_view)


def broadcast_block_shape(a_shape, b_shape, a_block_shape):
    # Starting from last block shape dim and
    # map each shape dim to block shape dim as already defined,
    # and for the rest of dims, set block shape to 1.
    result_shape = broadcast(a_shape, b_shape).shape
    result_block_shape = []
    a_block_shape_r = list(reversed(a_block_shape))
    for i, _ in enumerate(reversed(result_shape)):
        if i < len(a_block_shape_r):
            result_block_shape.append(a_block_shape_r[i])
        else:
            result_block_shape.append(1)
    return tuple(reversed(result_block_shape))


def broadcast_shape(a_shape, b_shape):
    return broadcast(a_shape, b_shape).shape


def can_broadcast_shapes(a_shape, b_shape):
    try:
        assert broadcast_shape(a_shape, b_shape) is not None
        return True
    except ValueError as _:
        return False


def broadcast_shape_to(from_shape, to_shape):
    # Enforce broadcasting rules from an
    # array of references to 0 with shape from_shape.
    from_view = np.lib.stride_tricks.broadcast_to(0, from_shape)
    return np.lib.stride_tricks.broadcast_to(from_view, to_shape)


def can_broadcast_shape_to(from_shape, to_shape):
    # See: https://numpy.org/devdocs/user/theory.broadcasting.html
    try:
        broadcast_shape_to(from_shape, to_shape)
        return True
    except ValueError as _:
        return False


def broadcast_shape_to_alt(from_shape, to_shape):
    # This is heavily tested with shapes up to length 5.
    from_num_axes = len(from_shape)
    to_num_axes = len(to_shape)
    result_shape = []
    if to_num_axes < from_num_axes:
        raise ValueError("Input shape has more dimensions than allowed by the axis remapping.")
    if to_num_axes == 0 and from_shape != 0:
        raise ValueError("Cannot broadcast non-scalar shape to scalar shape ().")
    from_shape_r = list(reversed(from_shape))
    to_shape_r = list(reversed(to_shape))
    for i, from_dim in enumerate(from_shape_r):
        to_dim = to_shape_r[i]
        if from_dim == 1:
            result_shape.append(to_dim)
        elif to_dim == from_dim:
            result_shape.append(to_dim)
        else:
            raise ValueError("Cannot broadcast %s to %s." % (str(from_shape), str(to_shape)))
    return tuple(reversed(result_shape + to_shape_r[from_num_axes:]))


def is_array_like(obj):
    return isinstance(obj, (tuple, list, np.ndarray))


def block_shape_from_subscript(subscript: tuple, block_shape: tuple):
    new_block_shape = []
    for i, obj in enumerate(subscript):
        if isinstance(obj, slice):
            new_block_shape.append(block_shape[i])
        elif isinstance(obj, (int, np.intp)):
            continue
        else:
            raise NotImplementedError("No support for advanced indexing.")
    return tuple(new_block_shape)


def get_slices(total_size, batch_size, order, reverse_blocks=False):
    assert order in (-1, 1)
    if order > 0:
        if reverse_blocks:
            result = list(reversed(list(range(total_size, 0, -batch_size)) + [0]))
        else:
            result = list(range(0, total_size, batch_size)) + [total_size]
        return list(map(lambda s: slice(*s, order), zip(*(result[:-1], result[1:]))))
    else:
        if reverse_blocks:
            # If reverse order blocks are not multiples of axis dimension,
            # then the last block is smaller than block size and should be
            # the first block.
            result = list(reversed(list(range(-total_size-1, -1, batch_size)) + [-1]))
        else:
            result = list(range(-1, -total_size - 1, -batch_size)) + [-total_size - 1]
        return list(map(lambda s: slice(*s, order), zip(*(result[:-1], result[1:]))))


class OrderedGrid(object):

    def __init__(self, shape: Tuple, block_shape: Tuple, order: Tuple,
                 block_order=None):
        if block_order is not None:
            assert len(block_order) == len(shape)
        self.shape = tuple(shape)
        self.block_shape = tuple(np.min([shape, block_shape], axis=0))
        self.order = tuple(order)
        self.grid_shape = []
        self.grid_slices = []
        for i in range(len(self.shape)):
            dim = self.shape[i]
            block_dim = block_shape[i]
            axis_order = order[i]
            reverse_blocks = False
            if block_order is not None:
                reverse_blocks = block_order[i] == -1
            axis_slices = get_slices(dim, block_dim, axis_order, reverse_blocks)
            self.grid_slices.append(axis_slices)
            self.grid_shape.append(len(axis_slices))
        self.grid_shape = tuple(self.grid_shape)
        # Assumes C-style ordering.
        # We add len(shape) to allow for axis consisting of the actual slices.
        self.slices = np.array(list(itertools.product(*self.grid_slices)),
                               dtype=slice).reshape(tuple(list(self.grid_shape) + [len(shape)]))

    def index_iterator(self) -> Iterator[Tuple]:
        if 0 in self.shape:
            return []
        return itertools.product(*map(range, self.grid_shape))
