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


from typing import Tuple, List, Iterator

import itertools
import numpy as np


def get_output_type(dtype_a, dtype_b):
    a = np.array(1, dtype=dtype_a)
    b = np.array(2, dtype=dtype_b)
    dtype = (a + b).dtype
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


def intersect(a_rect, b_rect):
    num_dims = len(a_rect)
    assert len(b_rect) == num_dims
    result = []
    for dim in range(num_dims):
        rd = [None, None]
        ad = a_rect[dim]
        bd = b_rect[dim]
        rd[0] = bd[0] if ad[0] < bd[0] else ad[0]
        rd[1] = ad[1] if ad[1] < bd[1] else bd[1]
        if rd[1] <= rd[0]:
            return None
        result.append(rd)
    return result


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


def shape_from_subscript(slices_or_indices, shape, drop_dims=True):
    ss_shape = []
    for i, entry in enumerate(slices_or_indices):
        if isinstance(entry, (np.intp, int)):
            if not drop_dims:
                ss_shape.append(1)
        else:
            slice_tuple = (0 if entry.start is None else entry.start,
                           shape[i] if entry.stop is None else entry.stop)
            ss_shape.append(slice_tuple[1] - slice_tuple[0])
    return ss_shape


def squeeze_like(slice_or_indices, shape, other_shape):
    ss_shape = shape_from_subscript(slice_or_indices, shape, drop_dims=False)
    ss_pos, other_pos = len(ss_shape) - 1, len(other_shape) - 1
    while True:
        if ss_pos < 0:
            pass
        elif other_pos < 0:
            pass
        ss_dim = ss_shape[ss_pos]
        other_dim = other_shape[other_pos]
        if ss_dim == other_dim:
            pass
        elif ss_dim == 1:
            assert other_dim == 1
        elif other_dim == 1:
            pass
        else:
            raise Exception("The shape %s can't match %s" % (ss_shape, other_shape))
        ss_pos -= 1
        other_pos -= 1


def squeeze_subscript(slices_or_indices, return_indices=False, invert=False):
    result = []
    for i, entry in enumerate(slices_or_indices):
        if isinstance(entry, (np.intp, int)) == invert:
            result.append(i if return_indices else entry)
    return result


def is_array_like(obj):
    return isinstance(obj, (tuple, list, np.ndarray))


def is_advanced_subscript(subscript: tuple):
    for obj in subscript:
        if is_array_like(obj):
            return True
    return False


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


def relative_subscript_rect(pos: np.ndarray, shape: np.ndarray):
    pairs = np.concatenate([pos.reshape(-1, 1), (pos+shape).reshape(-1, 1)], axis=1).tolist()
    subscript = tuple(map(lambda sparams: slice(*sparams), pairs))
    return subscript


def shape_from_slices(slices: Tuple[slice]):
    shape = []
    for s in slices:
        if s.step is None or s.step > 0:
            shape.append(s.stop - s.start)
        else:
            shape.append(s.start - s.stop)
    return tuple(shape)


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

    def ordered_iterator(self) -> Iterator[Tuple]:
        if 0 in self.shape:
            return []
        offset = np.zeros(len(self.order), dtype=np.intp)
        order = np.array(self.order, dtype=np.intp)
        for i, axis_order in enumerate(self.order):
            if axis_order < 0:
                offset[i] = self.grid_shape[i] - 1
        for index in self.index_iterator():
            r: List = (np.array(index, dtype=np.intp) * order + offset).tolist()
            yield tuple(r)

    def slice_iterator(self):
        # This can be made faster with numpy iterators.
        for grid_index in self.index_iterator():
            yield self.slices[grid_index]
