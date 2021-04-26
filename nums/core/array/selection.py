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


from typing import List, Tuple, Union

import numpy as np

import nums.core.array.utils as array_utils


def get_array_order(array, axis=0):
    # An axis is ordered if all diffs have same sign, and no diffs of 0 (no repetitions).
    if array.shape[axis] == 1:
        if array[axis] >= 0:
            return 1
        else:
            return -1
    diffs = np.diff(array, axis=axis)
    if np.all(diffs > 0):
        return 1
    elif np.all(diffs < 0):
        return -1
    else:
        # Order is undefined.
        return 0


def is_advanced_selection(subscript: tuple):
    assert isinstance(subscript, tuple)
    num_ordered_arrays = 0
    num_indexes = 0
    for obj in subscript:
        if array_utils.is_array_like(obj):
            if isinstance(obj, np.ndarray):
                array = obj
            else:
                # TODO (hme): This is inefficient.
                array = np.array(obj, dtype=np.intp)
            if len(array.shape) > 1:
                return True
            elif get_array_order(array, axis=0) == 0:
                return True
            num_ordered_arrays += 1
        elif isinstance(obj, (int, np.intp)):
            num_indexes += 1
    if num_ordered_arrays > 1:
        return True
    if num_ordered_arrays == 1:
        # In this case, indexes is considered an advanced index.
        return num_indexes > 0
    # We can compute ordered arrays fast, so don't consider ordered arrays as advanced.
    return False


def pos_step_slice_to_range(n, start_bound, stop_bound):
    if n < 0:
        n += stop_bound - start_bound
    n = np.clip(n, start_bound, stop_bound)
    return n


def neg_step_slice_to_range(n, start_bound, stop_bound):
    # Any start, stop above (start_bound - stop_bound)
    # is outside the bounds of the slice parameters
    # for backward steps.
    if 0 <= n:
        n -= start_bound - stop_bound
    # Similar to positive steps,
    # any start, stop below the stop bounds is clipped for negative steps.
    n = np.clip(n, stop_bound, start_bound)
    return n


def trim_slice_bounds(s: slice, size: int):
    start, stop, step = s.start, s.stop, s.step
    if step is None or step > 0:
        start_bound = 0
        stop_bound = size
        if start is None:
            start = start_bound
        if stop is None:
            stop = stop_bound
        # For positive steps, clip between -size and size, and compute size + n if n is < 0.
        # For positive slice parameters, Python slice semantics always clip stop by size,
        # so it makes sense to do the same in the negative direction.
        # We end up with start, stop >= 0 after adding size to n after clipping.
        # This matches NumPy's slice semantics for positive steps because
        # we simply select a sequence of elements in the forward direction.
        start = pos_step_slice_to_range(start, start_bound, stop_bound)
        stop = pos_step_slice_to_range(stop, start_bound, stop_bound)
    elif step < 0:
        # For negative steps, the start bound is -1,
        # and the stop bound is -size-1. The stop bound is offset by -1
        # to match a start bound of -1, so that the maximum number of items
        # touched is at most size == start_bound - stop_bound == -1 - (-size - 1).
        start_bound = -1
        stop_bound = -size - 1
        if start is None:
            start = start_bound
        if stop is None:
            stop = stop_bound
        start = neg_step_slice_to_range(start, start_bound, stop_bound)
        stop = neg_step_slice_to_range(stop, start_bound, stop_bound)
    else:
        raise ValueError("A step of 0 is invalid.")
    return slice(start, stop, step)


def slice_to_range(s: slice, size: int):
    st = trim_slice_bounds(s, size)
    return np.arange(st.start, st.stop, st.step)


class AxisSelection(object):
    def is_empty(self):
        raise NotImplementedError()

    def selector(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def order(self):
        raise NotImplementedError()


class AxisSlice(AxisSelection):
    @classmethod
    def from_size(cls, size: int, s: Union[slice, None] = None):
        if s is None:
            s = slice(None, None, None)
        st = trim_slice_bounds(s, size)
        return cls(st.start, st.stop, st.step)

    def __init__(
        self, start: Union[int, None], stop: Union[int, None], step: Union[int, None]
    ):
        self.start = start
        self.stop = stop
        self.step = step

    def is_empty(self):
        if self.order() > 0:
            return self.stop <= self.start
        else:
            return self.start <= self.stop

    def selector(self):
        return self.to_slice()

    def to_slice(self):
        return slice(self.start, self.stop, self.step)

    def shape(self):
        if self.step is None or self.step > 0:
            step_mag = 1 if self.step is None else abs(self.step)
            interval = max(0, self.stop - self.start)
        else:
            step_mag = abs(self.step)
            interval = max(0, self.start - self.stop)
        if interval == 0:
            return (0,)
        if step_mag == 1:
            return (interval,)
        if step_mag > interval:
            return (1,)
        # TODO (hme): Make this faster by not instantiating a range.
        return slice_to_range(slice(0, interval, step_mag), interval).shape

    def order(self):
        return np.sign(1 if self.step is None else self.step)


class AxisIndex(AxisSelection):
    @classmethod
    def from_index(cls, index: int, size: int):
        if 0 <= index < size:
            pass
        elif -size <= index < 0:
            index = pos_step_slice_to_range(index, 0, size)
        else:
            raise IndexError("Index %d is out of bounds for size %d" % (index, size))
        return cls(index)

    def __init__(self, index: int):
        self.index = index

    def to_index(self):
        return self.index

    def selector(self):
        return self.to_index()

    def is_empty(self):
        # Index can never be empty.
        return False

    def shape(self):
        return ()

    def order(self):
        return np.sign(self.index)


class AxisArray(AxisSelection):
    # See https://numpy.org/doc/stable/reference/arrays.indexing.html#integer-array-indexing
    def __init__(self, array_like):
        if isinstance(array_like, np.ndarray):
            assert array_like.dtype.kind == "i" or array_like.dtype.kind == "b"
            arr: np.ndarray = array_like
        else:
            # Must infer type.
            is_bool = True
            for entry in array_like:
                if not isinstance(entry, (np.intp, int, bool)):
                    raise Exception("Only integer or boolean arrays are valid indices.")
                if isinstance(entry, (int, np.intp)):
                    is_bool = False
            arr: np.ndarray = np.array(array_like, dtype=(bool if is_bool else np.intp))
        self.array = arr

    def is_empty(self):
        return len(self.array) == 0

    def selector(self):
        return self.array

    def shape(self):
        return self.array.shape

    def order(self):
        return get_array_order(self.array)


class AxisEmpty(AxisSelection):
    def selector(self):
        return slice(0, 0, None)

    def is_empty(self):
        return True

    def shape(self):
        return (0,)

    def order(self):
        return 0


class BasicSelection(object):
    @classmethod
    def from_shape(cls, shape: Tuple):
        return cls.from_subscript(shape, (...,))

    @classmethod
    def block_selection(
        cls, shape: Tuple, block_shape: Tuple, order=None, block_order=None
    ):
        if order is None:
            order = tuple([1] * len(shape))
        grid = array_utils.OrderedGrid(
            shape=shape, block_shape=block_shape, order=order, block_order=block_order
        )
        selection_grid = np.empty(grid.grid_shape, dtype=cls)
        for index in grid.index_iterator():
            slices: Tuple[slice] = tuple(grid.slices[index])
            # This must be shape, since slices are bounded by shape.
            selection_grid[index] = cls.from_subscript(shape, slices)
        return selection_grid

    @classmethod
    def from_subscript(cls, shape: Tuple, subscript: Tuple):
        axis_sels: List[AxisSelection] = []
        contains_ellipsis = False
        contains_array = False
        i = 0
        for axis_ss in subscript:
            if axis_ss is Ellipsis:
                if contains_ellipsis:
                    raise ValueError("Subscripts may contain at most 1 ellipsis.")
                contains_ellipsis = True
                ellipsis_length = len(shape) - (len(subscript) - 1)
                for j in list(range(i, i + ellipsis_length)):
                    # We instantiate a list, so okay to modify i.
                    axis_sels.append(AxisSlice.from_size(shape[j]))
                    i += 1
            elif isinstance(axis_ss, (int, np.intp)):
                axis_sels.append(AxisIndex.from_index(axis_ss, shape[i]))
                i += 1
            elif isinstance(axis_ss, slice):
                axis_sels.append(AxisSlice.from_size(shape[i], axis_ss))
                i += 1
            elif isinstance(axis_ss, (list, tuple, np.ndarray)):
                if contains_array:
                    raise ValueError(
                        "%s may contain at most 1 array." % BasicSelection.__name__
                    )
                contains_array = True
                axis_array = AxisArray(axis_ss)
                if len(axis_array.array.shape) > 1:
                    raise ValueError(
                        "%s does not support multi-axis"
                        " arrays." % BasicSelection.__name__
                    )
                if axis_array.array.dtype.kind == "i" and np.any(
                    axis_array.array >= shape[i]
                ):
                    raise ValueError("Some indices exceed dimension of axis.")
                if get_array_order(axis_array.array, axis=0) == 0:
                    # The sequence must be monotonic with strict order.
                    raise ValueError(
                        "Input array cannot be interpreted as ordered set."
                    )
                axis_sels.append(axis_array)
                i += 1
            else:
                raise Exception(
                    "Unknown subscript type %s for axis %d." % (type(axis_ss), i)
                )

        while len(axis_sels) < len(shape):
            size = shape[len(axis_sels)]
            # Select rest of axes.
            axis_sels.append(AxisSlice.from_size(size))

        assert len(axis_sels) <= len(
            shape
        ), "More objects in selection tuple than axes in array."
        return BasicSelection(tuple(axis_sels), shape)

    def __init__(self, axes: Tuple[AxisSelection], shape: tuple):
        self.axes = axes
        self.shape = shape
        # Compute this lazily.
        self._output_shape = None

    def get_broadcastable_shape(self):
        oshape = []
        for axis in self.axes:
            axis_shape = axis.shape()
            if len(axis_shape) == 0:
                oshape.append(1)
            elif len(axis_shape) == 1:
                oshape = oshape + list(axis_shape)
            else:
                raise ValueError("Invalid axis shape %s" % str(axis_shape))
        return tuple(oshape)

    def get_broadcastable_block_shape(self, partial_block_shape):
        # Construct a block shape equal in length to this selection's shape.
        # In what follows, a dimension is "empty" if the axis selector is an integer.
        # Starting from the last axis,
        # the resulting block shape is constructed as follows:
        # - If this dim is non-empty and partial is non-empty, emit partial's dimension.
        # - If this dim is non-empty and partial is empty, emit 1.
        # - If this dim is empty and partial is non-empty, then emit 1 if partial is 1.
        # - If this dim is empty and partial is empty, then emit 1.
        # All else raises an exception.
        # Partial may be longer than this shape, so long as the starting axis dims are 1.
        obs = []
        partial_len = len(partial_block_shape)
        sel_len = len(self.axes)
        # Maintain separate indices. We don't decrement partial if we encounter an empty dim.
        j = -1
        for i in range(-1, -sel_len - 1, -1):
            axis_shape = self.axes[i].shape()
            if len(axis_shape) == 0:
                obs.insert(0, 1)
            elif len(axis_shape) == 1:
                if j < -partial_len:
                    # The rest of the axes have to be empty or of dimension 1.
                    assert axis_shape[0] == 1
                    obs.insert(0, 1)
                else:
                    obs.insert(0, partial_block_shape[j])
                    j -= 1
            else:
                raise ValueError("Invalid axis shape %s" % str(axis_shape))
        for k in range(j, -partial_len - 1, -1):
            assert partial_block_shape[k] == 1
        return obs

    def get_output_shape(self, include_indexes=False):
        if self._output_shape is None or include_indexes:
            oshape = []
            for axis in self.axes:
                axis_shape = axis.shape()
                if len(axis_shape) == 0:
                    if include_indexes:
                        oshape.append(1)
                    else:
                        continue
                elif len(axis_shape) == 1:
                    oshape = oshape + list(axis_shape)
                else:
                    raise ValueError("Invalid axis shape %s" % str(axis_shape))
            if include_indexes:
                # Never store output shape that includes indexes.
                return tuple(oshape)
            self._output_shape = tuple(oshape)
        return self._output_shape

    def order(self):
        return np.array([axis.order() for axis in self.axes], dtype=np.intp)

    def is_empty(self):
        for i in range(len(self.axes)):
            axis: AxisSelection = self.axes[i]
            if axis.is_empty():
                return True
        return False

    def selector(self):
        sel = []
        for i in range(len(self.axes)):
            if isinstance(self[i], AxisEmpty):
                sel.append(slice(0, 0))
            else:
                sel.append(self.axes[i].selector())
        return tuple(sel)

    def position(self, compute_stop=False):
        return Position.from_selection(self, compute_stop)

    def basic_steps(self):
        for item in self.axes:
            if isinstance(item, AxisSlice):
                if item.step not in (None, 1):
                    return False
        return True

    def is_aligned(self, block_shape):
        start_pos: Position = self.position()
        stop_pos: Position = self.position(compute_stop=True)
        assert len(start_pos.value) == len(block_shape)
        aligned = True
        output_shape = self.get_output_shape(include_indexes=True)

        for i in range(len(start_pos.value)):
            start_aligned = (start_pos.value[i] % block_shape[i]) == 0
            shape_aligned = (output_shape[i] % block_shape[i]) == 0
            stop_aligned = stop_pos.value[i] == self.shape[i]
            # Check that (self - pos).get_output_shape() % block_shape == 0
            # OR self.end() == self.shape, which means the last block
            # may not be the same shape as block_shape.
            # In the latter case, when using this method to do a reference
            # copy it's enough for the block shapes to be equal,
            # because the source/destination shapes are broadcasted
            # at the block-level.
            # If all conditions are met for each axis, then the selection is block-aligned
            # with the provided block shape. This assumes the provided block shape
            # is associated with the shape originally provided to this selection.
            aligned = aligned and start_aligned and (shape_aligned or stop_aligned)
            if not aligned:
                return False
        return True

    def __repr__(self):
        return str(self.selector())

    def __getitem__(self, item):
        assert isinstance(item, (np.intp, int))
        return self.axes[item]

    def __or__(self, other):
        # Union.
        raise NotImplementedError()

    def __xor__(self, other):
        # Union - intersection.
        raise NotImplementedError()

    def __and__(self, other):
        # We think of each axis of a selection as a partially ordered set.
        # The intersection of axis selectors preserve partial order.
        # The notion of intersection over selection operations
        # preserves
        result_axes: List[AxisSelection] = []
        result_shape: List[int] = []
        assert len(self.axes) == len(other.axes)
        for i in range(len(self.axes)):
            self_axis, other_axis = self.axes[i], other.axes[i]
            self_size, other_size = self.shape[i], other.shape[i]
            result_shape.append(min(self_size, other_size))
            if isinstance(self_axis, AxisSlice) and isinstance(other_axis, AxisSlice):
                result_axes.append(
                    self._slice_and_slice(self_axis, other_axis, self_size, other_size)
                )
            elif isinstance(self_axis, AxisIndex) and isinstance(other_axis, AxisIndex):
                if self_axis.index == other_axis.index:
                    result_axis: AxisIndex = AxisIndex(self_axis.index)
                else:
                    result_axis: AxisEmpty = AxisEmpty()
                result_axes.append(result_axis)
            elif isinstance(self_axis, AxisArray) and isinstance(other_axis, AxisArray):
                result_axis: AxisArray = self._array_and_array(self_axis, other_axis)
                result_axes.append(result_axis)
                if len(result_axis.array.shape) != 1:
                    raise Exception(
                        "Unexpected output shape length "
                        "of %d for array intersection." % len(result_axis.array.shape)
                    )
            elif isinstance(self_axis, AxisEmpty) or isinstance(other_axis, AxisEmpty):
                result_axes.append(AxisEmpty())
            elif isinstance(self_axis, AxisIndex) or isinstance(other_axis, AxisIndex):
                # If one of the operands is an index, then the intersection is simple.
                # The result must be an index.
                res: AxisArray = self._array_and_array(
                    self._to_array_axis(self_axis, self_size),
                    self._to_array_axis(other_axis, other_size),
                )
                assert len(res.array) <= 1
                if len(res.array) == 1:
                    result_axes.append(AxisIndex(res.array[0]))
                else:
                    result_axes.append(AxisEmpty())
            elif isinstance(self_axis, AxisArray) or isinstance(other_axis, AxisArray):
                # If one of the operands is an array,
                # then the resulting intersection must be an array.
                result_axes.append(
                    self._array_and_array(
                        self._to_array_axis(self_axis, self_size),
                        self._to_array_axis(other_axis, other_size),
                    )
                )
            else:
                # This should be impossible, since we've dealt with all potential mixed types.
                raise NotImplementedError("Operation not supported.")
        return BasicSelection(tuple(result_axes), tuple(result_shape))

    def _to_array_axis(self, anything: AxisSelection, size):
        if isinstance(anything, AxisSlice):
            return AxisArray(slice_to_range(anything.to_slice(), size))
        elif isinstance(anything, AxisIndex):
            return AxisArray([anything.index])
        elif isinstance(anything, AxisEmpty):
            return AxisArray([])
        else:
            assert isinstance(anything, AxisArray)
            return anything

    def _slice_and_slice(self, a: AxisSlice, b: AxisSlice, a_size, b_size):
        a_step = 1 if a.step is None else a.step
        b_step = 1 if b.step is None else b.step
        if a_step == 1 and b_step == 1:
            start = b.start if a.start < b.start else a.start
            stop = a.stop if a.stop < b.stop else b.stop
            step = None if a.step is None and b.step is None else 1
        elif (a_step < 0 < b_step) or (b_step < 0 < a_step):
            # Steps running in opposite directions.
            # Based on intersection of partial orders, the order is undefined.
            start, stop, step = 0, 0, None
        elif 0 < a_step and 0 < b_step:
            assert a.start >= 0 and b.start >= 0 and a.stop >= 0 and b.stop >= 0
            # TODO(hme): Optimize this by avoiding array instantiation.
            arr = self._array_and_array(
                self._to_array_axis(a, a_size), self._to_array_axis(b, b_size)
            )
            if arr.array.size == 0:
                start, stop, step = 0, 0, None
            else:
                steps = np.diff(arr.array)
                if steps.size == 0:
                    # Occurs only if one item is selected.
                    # This could be from same start value,
                    # or slice(0, 4, 3)
                    #    slice(1, 4, 2),
                    # or slice(0, 10, 3)
                    #    slice(2, 10, 2).
                    # The safe thing to do is set step large enough
                    # so that the resulting slice only selects
                    # the first value.
                    start = int(arr.array[0])
                    stop = min(a.stop, b.stop)
                    step = stop
                else:
                    # Steps should be all eq.
                    assert not np.any(np.diff(steps))
                    step = int(steps[0])
                    start = int(arr.array[0])
                    stop = min(a.stop, b.stop)
        elif a_step < 0 and b_step < 0:
            assert a.start < 0 and b.start < 0 and a.stop < 0 and b.stop < 0
            arr = self._array_and_array(
                self._to_array_axis(a, a_size), self._to_array_axis(b, b_size)
            )
            if arr.array.size == 0:
                start, stop, step = 0, 0, None
            else:
                steps = np.diff(arr.array)
                if steps.size == 0:
                    # Same logic as with positive steps.
                    start = int(arr.array[0])
                    stop = max(a.stop, b.stop)
                    step = stop
                else:
                    # Steps should be all eq.
                    assert not np.any(np.diff(steps))
                    start = int(arr.array[0])
                    stop = max(a.stop, b.stop)
                    step = int(steps[0])
        else:
            raise Exception("Impossible.")
        return AxisSlice(start, stop, step)

    def _array_and_array(self, a: AxisArray, b: AxisArray):
        return AxisArray(self._np_array_and_array(a.array, b.array))

    def _get_order(self, arr: np.ndarray):
        assert len(arr.shape) == 1
        order = get_array_order(arr, axis=0)
        if order == 0:
            # Includes diffs containing 0,
            # which imply repetitions.
            # This is not allowed for basic selection.
            raise ValueError("Input array cannot be interpreted as ordered set.")
        return order

    def _np_array_and_array(self, a: np.ndarray, b: np.ndarray):
        # We can only support intersection of single-axis arrays,
        # which we interpret as ordered sets.
        # Consider intersection of a matrix. We may have rows of
        # different shape.
        a_order, b_order = self._get_order(a), self._get_order(b)
        if a_order != b_order:
            # Such an order exists, but the order
            # is undefined for elements x,y \in S,
            # where x \neq y.
            # For our purposes, this is undefined.
            return np.array([], dtype=np.intp)
        else:
            return np.sort(np.array(list(set(a) & set(b)), dtype=np.intp))[
                :: np.sign(a_order)
            ]


class AdvancedSelection(object):
    # Allow mixture of slices and advanced indexing.
    # Will require some basic intersection operations,
    # but not as general as BasicSelection.
    def __init__(self):
        raise NotImplementedError()


class Position(object):
    @classmethod
    def from_selection(cls, sel: BasicSelection, compute_stop=False):
        shape_dim = len(sel.shape)
        value: np.ndarray = np.empty(shape_dim, dtype=np.intp)
        for i in range(shape_dim):
            axsel: AxisSelection = sel[i]
            if isinstance(sel[i], AxisSlice):
                axsel: AxisSlice = sel[i]
                value[i] = axsel.stop if compute_stop else axsel.start
            elif isinstance(sel[i], AxisIndex):
                axsel: AxisIndex = sel[i]
                value[i] = axsel.index
            elif isinstance(sel[i], AxisArray):
                axsel: AxisArray = sel[i]
                if compute_stop:
                    value[i] = (
                        max(axsel.array) if axsel.order() > 0 else min(axsel.array)
                    )
                else:
                    value[i] = (
                        min(axsel.array) if axsel.order() > 0 else max(axsel.array)
                    )
            else:
                ValueError("Unexpected axis type %s" % type(sel[i]))
            if axsel.order() < 0:
                value[i] += 1
        return cls(value)

    @classmethod
    def from_dim(cls, dim: int):
        return cls(np.zeros(dim, dtype=np.intp))

    def __init__(self, value: np.ndarray):
        self.dim = len(value)
        assert value.dtype == np.intp
        self.value = value

    def __repr__(self):
        return str(self.value.tolist())

    def __add__(self, other):
        return self.bop(other, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self.bop(other, "sub")

    def __rsub__(self, other):
        return self.bop(other, "rsub")

    def bop(self, other, bop):
        if isinstance(other, Position):
            assert self.dim == other.dim
            result = Position.from_dim(self.dim)
            if bop == "add":
                result.value = self.value + other.value
            elif bop == "sub":
                result.value = self.value - other.value
            elif bop == "rsub":
                result.value = other.value - self.value
            return result
        elif isinstance(other, BasicSelection):
            if bop == "sub":
                # This is not allowed for selections.
                # The new subscript bounds are unclear in this case.
                raise ValueError("Cannot subtract BasicSelection from Position.")
            elif bop not in ("add", "rsub"):
                raise ValueError("Unsupported op encountered %s" % bop)
            assert self.dim == len(other.shape)
            # TODO (hme): Test shape addition / subtraction under various circumstances.
            #  This should work okay, but does it work because of prior assumptions
            #  on AxisSelection types? This is okay, but we should answer this
            #  question.
            shape: list = list(other.shape)
            sel: list = list(other.selector())
            for i in range(self.dim):
                if isinstance(sel[i], slice):
                    if bop == "add":
                        sel[i] = slice(
                            sel[i].start + self.value[i],
                            sel[i].stop + self.value[i],
                            sel[i].step,
                        )
                        shape[i] += self.value[i]
                    elif bop == "rsub":
                        shape[i] -= self.value[i]
                        sel[i] = slice(
                            sel[i].start - self.value[i],
                            sel[i].stop - self.value[i],
                            sel[i].step,
                        )
                elif isinstance(sel[i], (int, np.intp)):
                    if bop == "add":
                        sel[i] = sel[i] + self.value[i]
                    elif bop == "rsub":
                        sel[i] = sel[i] - self.value[i]
                elif isinstance(sel[i], np.ndarray):
                    if bop == "add":
                        sel[i] = sel[i].copy() + self.value[i]
                    elif bop == "rsub":
                        sel[i] = sel[i].copy() - self.value[i]
                else:
                    raise ValueError(
                        "Unsupported instance encountered %s" % type(sel[i])
                    )
            shape: tuple = tuple(shape)
            sel: tuple = tuple(sel)
            return BasicSelection.from_subscript(shape, sel)
