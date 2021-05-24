# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
from typing import Tuple, Iterator

import numpy as np
import scipy.special

from nums.core.settings import np_ufunc_map


# pylint: disable = no-member


def get_uop_output_type(op_name, dtype):
    a = np.array(1, dtype=dtype)
    result_dtype = np.__getattribute__(op_name)(a).dtype
    return np.__getattribute__(str(result_dtype))


def get_bop_output_type(op_name, dtype_a, dtype_b):
    a = np.array(1, dtype=dtype_a)
    b = np.array(2, dtype=dtype_b)
    op_name = np_ufunc_map.get(op_name, op_name)
    try:
        dtype = np.__getattribute__(op_name)(a, b).dtype
        return np.__getattribute__(str(dtype))
    except Exception as _:
        dtype = scipy.special.__getattribute__(op_name)(a, b).dtype
        return np.__getattribute__(str(dtype))


def is_uint(val, type_test=False):
    return is_type(type_test, val,
                   (np.uint, np.uint8, np.uint16, np.uint32, np.uint64))


def is_int(val, type_test=False):
    return is_type(type_test, val,
                   (int, np.int8, np.int16, np.int32, np.int64))


def is_float(val, type_test=False):
    return is_type(type_test, val,
                   (float, np.float16, np.float32, np.float64))


def is_complex(val, type_test=False):
    return is_type(type_test, val,
                   (np.complex64, np.complex128))


def is_type(type_test, val, types):
    return val in types if type_test else isinstance(val, types)


def get_reduce_output_type(op_name, dtype):
    a = np.array([0, 1], dtype=dtype)
    dtype = np.__getattribute__(op_name)(a).dtype
    return np.__getattribute__(str(dtype))


def shape_from_block_array(arr: np.ndarray):
    grid_shape = arr.shape
    num_axes = len(arr.shape)
    shape = np.zeros(num_axes, dtype=int)
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


def idx2addr(index: tuple, shape: tuple):
    strides = [np.product(shape[i:]) for i in range(1, len(shape))] + [1]
    addr: int = sum(np.array(index) * strides)
    return addr


def addr2idx(addr: int, shape: tuple):
    strides = [np.product(shape[i:]) for i in range(1, len(shape))] + [1]
    index = []
    val = addr
    for i in range(len(strides)):
        stride = strides[i]
        axis_index = int(val/stride)
        index.append(axis_index)
        val %= stride
    return tuple(index)


def slice_sel_to_index_list(slice_selection: tuple):
    slice_ranges = []
    for slice_or_index in slice_selection:
        if isinstance(slice_or_index, slice):
            slice_ranges.append(list(range(slice_or_index.start, slice_or_index.stop)))
        elif isinstance(slice_or_index, int):
            slice_ranges.append([slice_or_index])
    index_list = list(itertools.product(*slice_ranges))
    return index_list


def translate_index_list(from_index_list, from_shape, to_shape):
    to_index_list = []
    for src_index in from_index_list:
        addr = idx2addr(src_index, from_shape)
        to_index_list.append(addr2idx(addr, to_shape))
    return to_index_list
