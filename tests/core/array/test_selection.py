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

import tqdm
import numpy as np
import pytest

from nums.core.array import utils as array_utils
import nums.core.array.selection as sel_module
from nums.core.array.selection import BasicSelection, \
    AxisSelection, \
    AxisArray, \
    AxisSlice, \
    AxisIndex, \
    AxisEmpty, \
    is_advanced_selection


# pylint: disable=protected-access, cell-var-from-loop


def get_slices(size, index_multiplier=1, limit=None):
    index_multiplier = [None] + list(range(-size * index_multiplier, size * index_multiplier + 1))
    items = list()
    for start, stop, step in itertools.product(index_multiplier, repeat=3):
        if step == 0:
            continue
        items.append(slice(start, stop, step))
    if limit is None:
        return items
    else:
        return subsample(items, limit)


def get_indices(size, limit=None):
    if limit is None:
        return list(range(size))
    else:
        return subsample(list(range(size)), limit)


def get_arrays(size, limit=None):
    items = []
    for arr in itertools.product(range(size), repeat=size):
        items.append(np.array(arr, dtype=int))
    if limit is None:
        return items
    else:
        return subsample(items, limit)


def subsample(items, max_items, seed=1337):
    rs = np.random.RandomState(seed)
    return np.array(items)[rs.choice(np.arange(len(items), dtype=int), max_items)].tolist()


def alt_compute_intersection(sel_a, sel_b, shape):
    # This may also be wrong,
    # but allows testing of more than just correctness of intersection.
    true_sel_c = []
    sel_a_selector = sel_a.selector()
    sel_b_selector = sel_b.selector()
    for i in range(min(len(sel_a_selector), len(sel_b_selector))):
        axis_a = sel_a_selector[i]
        axis_b = sel_b_selector[i]
        if isinstance(axis_a, slice) or isinstance(axis_b, slice):
            if isinstance(axis_b, slice):
                # The intersection op is commutative, so
                # just swap if the other axis is slice.
                axis_a, axis_b = axis_b, axis_a
            # Required check, because otherwise we expect an array.
            if isinstance(axis_b, slice):
                if (axis_b.step is None or axis_b.step == 1) \
                        and (axis_a.step is None or axis_a.step == 1):
                    axis_c = slice(max(axis_a.start, axis_b.start),
                                   min(axis_a.stop, axis_b.stop))
                else:
                    a_range = sel_module.slice_to_range(axis_a, shape[i])
                    b_range = sel_module.slice_to_range(axis_b, shape[i])
                    axis_c = sel_a._np_array_and_array(a_range, b_range)
            elif isinstance(axis_b, int):
                axis_c = axis_b if axis_a.start <= axis_b < axis_a.stop else slice(0, 0)
            elif isinstance(axis_b, np.ndarray):
                # Currently not supported (and may be impossible in general).
                raise ValueError("Cannot compute intersection of slice and array.")
            else:
                raise ValueError("Unexpected type %s" % type(axis_b))
            true_sel_c.append(axis_c)
        elif isinstance(axis_a, int) or isinstance(axis_b, int):
            if isinstance(axis_b, int):
                axis_a, axis_b = axis_b, axis_a
            if isinstance(axis_b, int):
                axis_c = axis_a if axis_a == axis_b else slice(0, 0)
            elif isinstance(axis_b, np.ndarray):
                axis_c = axis_a if axis_a in axis_b else slice(0, 0)
            else:
                raise ValueError("Unexpected type %s" % type(axis_b))
            true_sel_c.append(axis_c)
        elif isinstance(axis_a, np.ndarray) and isinstance(axis_b, np.ndarray):
            axis_c = sel_a._np_array_and_array(axis_a, axis_b)
            true_sel_c.append(axis_c)
        else:
            raise ValueError("Unexpected types %s, %s" % (type(axis_a), type(axis_b)))
    return tuple(true_sel_c)


def test_basics():
    assert is_advanced_selection((0, np.array([3, 1, 2])))


def test_basic_slice_selection():
    # Test basic ops with selections of AxisSlice.
    arr: np.ndarray = np.arange(3)
    size = arr.shape[0]
    index_range = [None] + list(range(-size*3, size*3 + 1))
    slice_params = list(itertools.product(index_range, repeat=2))
    pbar = tqdm.tqdm(total=len(slice_params))
    for start, stop in slice_params:
        pbar.update(1)
        slice_sel = slice(start, stop)
        ds_sel = AxisSlice.from_size(size, slice(start, stop)).to_slice()
        assert np.allclose(arr[slice_sel], arr[ds_sel]), (slice_sel, ds_sel)


def test_stepped_slice_selection():
    # Ensure that selections from slices of different step sizes are correct.
    # Thoroughly tested over a subset of the parameter space.
    arr: np.ndarray = np.arange(3)
    size = arr.shape[0]
    slice_params = list(get_slices(3, 3))
    pbar = tqdm.tqdm(total=len(slice_params))
    for slice_sel in slice_params:
        pbar.set_description(str(slice_sel))
        pbar.update(1)
        arr_sel: np.ndarray = sel_module.slice_to_range(slice_sel, arr.shape[0])
        assert np.all(arr_sel < size)
        assert len(arr[slice_sel]) == len(arr[arr_sel]), (slice_sel, arr_sel)
        assert np.allclose(arr[slice_sel], arr[arr_sel]), (slice_sel, arr_sel)

        sel: BasicSelection = BasicSelection.from_subscript(arr.shape, (slice_sel,))
        assert sel.get_output_shape() == arr[slice_sel].shape
        if isinstance(sel[0], AxisSlice):
            if sel[0].step is None:
                assert sel[0].start >= 0 and sel[0].stop >= 0
            else:
                assert (sel[0].step < 0) == (sel[0].step < 0) == (sel[0].stop < 0)
            ds_sel = sel.selector()
            assert np.allclose(arr[slice_sel], arr[ds_sel]), (slice_sel, ds_sel)
        elif isinstance(sel[0], AxisArray):
            assert slice_sel.step is not None and slice_sel.step != 1
            ds_arr = sel[0].array
            assert np.allclose(arr[slice_sel], arr[ds_arr]), (slice_sel, ds_arr)


def test_index_selection():
    arr: np.ndarray = np.arange(3)
    size = arr.shape[0]
    index_range = list(range(-size*3, size*3 + 1))
    pbar = tqdm.tqdm(total=len(index_range))
    for index in index_range:
        pbar.update(1)
        index_sel = index
        try:
            ds_sel = AxisIndex.from_index(index, size).to_index()
            assert np.allclose(arr[index_sel], arr[ds_sel]), (index_sel, ds_sel)
        except IndexError as _:
            with pytest.raises(IndexError):
                _ = arr[index_sel]
            with pytest.raises(IndexError):
                _ = AxisIndex.from_index(index, size)


def test_slice_intersection():
    # Basic intersection tests.
    # Test subset, partial overlap, and disjoint relations.
    shape = (100,)
    ss = BasicSelection.from_subscript
    test_parameters = [
        # Basic tests.
        (slice(None, None), slice(33, 77), slice(33, 77)),
        (slice(33, 77), slice(55, 88), slice(55, 77)),
        (slice(22, 33), slice(44, 55), slice(44, 33)),
        # Test empty slices. Empty slice => empty slice.
        (slice(40, 50), slice(45, 45), slice(45, 45)),
        (slice(40, 50), slice(55, 55), slice(55, 50)),
        (slice(40, 50), slice(35, 35), slice(40, 35)),
        # Test different variant of empty slice => empty slice.
        (slice(40, 50), slice(60, 30), slice(60, 30)),
        (slice(70, 40), slice(60, 30), slice(70, 30))
    ]
    for a_slice, b_slice, c_slice in test_parameters:
        size = shape[0]
        a = ss(shape, (a_slice,))
        b = ss(shape, (b_slice,))
        c = a & b
        assert isinstance(c[0], AxisSlice)
        assert c[0].to_slice() == c_slice, (a_slice, b_slice, c_slice)
        assert np.allclose(sel_module.slice_to_range(c[0].to_slice(), size),
                           sel_module.slice_to_range(c_slice, size))


@pytest.mark.skip
def test_stepped_slice_intersection():
    size = 4
    shape = size,
    num_axes = len(shape)
    # arr = np.random.random_sample(np.product(shape)).reshape(shape)
    arr = np.arange(np.product(shape)).reshape(shape)
    ss = BasicSelection.from_subscript

    num_axes_pairs = list(itertools.product(np.arange(1, num_axes+1), repeat=2))
    for num_axes_idx, (num_axes_a, num_axes_b) in enumerate(num_axes_pairs):
        # Index multiplier of 1 is okay,
        # since we test larger index ranges in the slice selection test.
        # TODO (hme): Update this to catch error in block slice intersection test.
        all_axis_selections = list(get_slices(size, index_multiplier=1))
        test_selections_a = list(itertools.product(all_axis_selections, repeat=num_axes_a))
        test_selections_b = list(itertools.product(all_axis_selections, repeat=num_axes_b))
        test_selections = list(itertools.product(test_selections_a, test_selections_b))
        pbar = tqdm.tqdm(total=len(all_axis_selections)**(num_axes_a + num_axes_b))
        for ss_a, ss_b in test_selections:
            pbar.set_description(str(("%.3f" % ((num_axes_idx+1)/len(num_axes_pairs)),
                                      ss_a,
                                      ss_b)))
            pbar.update(1)
            sel_a: BasicSelection = ss(arr.shape, ss_a)
            true_arr_sel_a = arr[ss_a]
            test_arr_sel_a = arr[sel_a.selector()]
            assert np.allclose(true_arr_sel_a, test_arr_sel_a)

            assert len(ss_b) == num_axes_b
            sel_b: BasicSelection = ss(arr.shape, ss_b)
            true_arr_sel_b = arr[ss_b]
            test_arr_sel_b = arr[sel_b.selector()]
            assert np.allclose(true_arr_sel_b, test_arr_sel_b)

            sel_c: BasicSelection = sel_a & sel_b

            a_range = sel_module.slice_to_range(ss_a[0], size)
            b_range = sel_module.slice_to_range(ss_b[0], size)
            if len(a_range) <= len(b_range):
                min_array = a_range
                max_array = b_range
            else:
                min_array = b_range
                max_array = a_range

            # We want a semantically equivalent array-based selector.
            # Perform an ordered-set intersection:
            # intersect the generated sets,
            # and define the order using a partial order intersection.
            step_a = 1 if ss_a[0].step is None else ss_a[0].step
            step_b = 1 if ss_b[0].step is None else ss_b[0].step
            if np.sign(step_a) != np.sign(step_b):
                # Such an order exists, but the order
                # is undefined for elements x,y \in S,
                # where x \neq y.
                # For our purposes, this is undefined.
                ab_intersection = np.array([], dtype=np.intp)
            else:
                ab_intersection = np.sort(np.array(list(set(min_array) & set(max_array)),
                                                   dtype=np.intp))[::np.sign(step_a)]

            # Quickly test array intersection here as well.
            assert np.allclose(sel_a._np_array_and_array(a_range, b_range),
                               ab_intersection)

            true_arr_sel_c = arr[ab_intersection]
            test_arr_sel_c = arr[sel_c.selector()]
            assert np.allclose(true_arr_sel_c, test_arr_sel_c)


def test_index_intersection():
    shape = (100,)
    ss = BasicSelection.from_subscript
    for j in range(shape[0]):
        axis_index = (j,)
        for i in range(shape[0]):
            a = ss(shape, axis_index)
            b = ss(shape, (i,))
            assert isinstance(b[0], AxisIndex)
            b_ds: AxisIndex = b[0]
            c: AxisSelection = a & b
            a_val: int = axis_index[0]
            a_ds: AxisIndex = a[0]
            true_bool = a_val == i
            axis_bool = a_ds.index == b_ds.index
            test_bool = not isinstance(c[0], AxisEmpty)
            assert true_bool == axis_bool == test_bool


def test_array_intersection():
    # Test several basic tests.
    ss = BasicSelection.from_subscript
    size = 4
    sizes = list(itertools.product(np.arange(1, size), repeat=2))
    for sizes_idx, (size_a, size_b) in enumerate(sizes):
        arr: np.ndarray = np.arange(size).reshape((size,))
        test_selections = list(itertools.product(get_arrays(size_a),
                                                 get_arrays(size_b)))
        pbar = tqdm.tqdm(total=len(test_selections))
        for ss_a, ss_b in test_selections:
            pbar.set_description(str((sizes_idx/len(sizes), ss_a, ss_b)))
            ss_a = (ss_a,)
            ss_b = (ss_b,)
            pbar.update(1)
            try:
                sel_a: BasicSelection = ss(arr.shape, ss_a)
            except ValueError as _:
                with pytest.raises(ValueError):
                    ss(arr.shape, ss_a)
                continue
            try:
                sel_b: BasicSelection = ss(arr.shape, ss_b)
            except ValueError as _:
                with pytest.raises(ValueError):
                    ss(arr.shape, ss_b)
                continue
            sel_c: BasicSelection = sel_a & sel_b
            true_arr_sel_a = arr[ss_a]
            true_arr_sel_b = arr[ss_b]
            test_arr_sel_a = arr[sel_a.selector()]
            test_arr_sel_b = arr[sel_b.selector()]
            assert np.allclose(true_arr_sel_a, test_arr_sel_a)
            assert np.allclose(true_arr_sel_b, test_arr_sel_b)

            a_order = sel_module.get_array_order(ss_a[0], axis=0)
            b_order = sel_module.get_array_order(ss_b[0], axis=0)
            if a_order != b_order:
                ss_c = np.array([], dtype=np.intp)
            else:
                ss_c = np.sort(np.array(list(set(ss_a[0]) & set(ss_b[0])),
                                        dtype=np.intp))[::a_order]
            assert np.allclose(sel_c[0].array, ss_c)
            true_arr_sel_c = arr[ss_c]
            test_arr_sel_c = arr[sel_c.selector()]
            assert np.allclose(test_arr_sel_c, true_arr_sel_c)


def test_multiselect_intersection():
    # Generate ranges for different types over 3 axes.
    def invalid_operands(a, b, num_axes_a, num_axes_b):
        # Types must be the same across axes.
        # At most 1 array_like selector.
        types_a, types_b = list(map(type, a)), list(map(type, b))
        num_arrays_a = 0
        num_arrays_b = 0
        for i in range(max(len(a), len(b))):
            if i >= len(a):
                if not isinstance(b[i], slice):
                    return True
            elif i >= len(b):
                if not isinstance(a[i], slice):
                    return True
            else:
                if array_utils.is_array_like(a[i]):
                    if sel_module.get_array_order(np.array(a[i], dtype=np.intp)) == 0:
                        return True
                    num_arrays_a += 1
                if array_utils.is_array_like(b[i]):
                    if sel_module.get_array_order(np.array(b[i], dtype=np.intp)) == 0:
                        return True
                    num_arrays_b += 1
                if types_a[i] != types_b[i]:
                    return True
        if num_axes_a > 1 or num_axes_b > 1:
            return True
        return False

    size = 3
    num_axes = 2
    limit = 4
    ss = BasicSelection.from_subscript
    arr: np.ndarray = np.arange(size**num_axes).reshape((size,)*num_axes)

    num_axes_pairs = list(itertools.product(np.arange(1, num_axes+1), repeat=2))
    for num_axes_idx, (num_axes_a, num_axes_b) in enumerate(num_axes_pairs):
        all_axis_selections = get_slices(size, limit=limit) + \
                              get_indices(size, limit=limit) + \
                              get_arrays(size, limit=limit)
        test_selections_a = itertools.product(all_axis_selections, repeat=num_axes_a)
        test_selections_b = itertools.product(all_axis_selections, repeat=num_axes_b)
        test_selections = itertools.product(test_selections_a, test_selections_b)
        pbar = tqdm.tqdm(total=len(all_axis_selections)**(num_axes_a+num_axes_b))
        for ss_a, ss_b in test_selections:
            pbar.set_description("%.3f" % ((num_axes_idx+1)/len(num_axes_pairs)))
            # pbar.set_description(str((num_axes_idx/len(num_axes_pairs), ss_a, ss_b)))
            pbar.update(1)
            # We only support intersection on operands of the same type.
            # If there's a type difference along some axis, then skip.
            # When one subscript has more objects than another,
            # make sure the difference consists of slices.
            if invalid_operands(ss_a, ss_b, num_axes_a, num_axes_b):
                continue
            sel_a: BasicSelection = ss(arr.shape, ss_a)
            true_arr_sel_a = arr[ss_a]
            test_arr_sel_a = arr[sel_a.selector()]
            assert np.allclose(true_arr_sel_a, test_arr_sel_a)

            assert len(ss_b) == num_axes_b
            sel_b: BasicSelection = ss(arr.shape, ss_b)
            true_arr_sel_b = arr[ss_b]
            test_arr_sel_b = arr[sel_b.selector()]
            assert np.allclose(true_arr_sel_b, test_arr_sel_b)

            sel_c: BasicSelection = sel_a & sel_b
            ss_c = alt_compute_intersection(sel_a, sel_b, arr.shape)
            true_arr_sel_c = arr[ss_c]
            test_arr_sel_c = arr[sel_c.selector()]
            assert np.allclose(true_arr_sel_c, test_arr_sel_c)


def test_ellipsis():
    shape = 3, 5, 7, 11
    arr: np.ndarray = np.arange(np.product(shape)).reshape(shape)
    test_params = [
        (...,),
        (slice(0, 2), ...),
        (slice(0, 2), slice(1, 4), ...),
        (slice(0, 2), slice(1, 4), slice(2, 6), ...),
        (slice(0, 2), slice(1, 4), slice(2, 6), slice(3, 8), ...),
        (..., slice(0, 2)),
        (..., slice(0, 2), slice(1, 4)),
        (..., slice(0, 2), slice(1, 4), slice(2, 6)),
        (..., slice(0, 2), slice(1, 4), slice(2, 6), slice(3, 8)),
        (slice(0, 2),),
        (slice(0, 2), slice(1, 4)),
        (slice(0, 2), slice(1, 4), slice(2, 6)),
        (slice(0, 2), slice(1, 4), slice(2, 6), slice(3, 8))
    ]
    for true_sel in test_params:
        sel = BasicSelection.from_subscript(shape, true_sel)
        test_sel = sel.selector()
        assert np.allclose(arr[true_sel], arr[test_sel])

    with pytest.raises(ValueError):
        BasicSelection.from_subscript(shape, (..., ...))


def test_advanced_indexing_broadcasting():
    # Basic test to check broadcasting semantics of advanced indexing.
    shape = 5, 6, 7, 8, 9
    num_axes = len(shape)
    x = np.random.random_sample(np.product(shape)).reshape(shape)
    aidx_1 = []
    for axis, dim in enumerate(shape):
        bshape = [1]*num_axes
        bshape[axis] = dim
        bshape = tuple(bshape)
        aidx_1.append(np.arange(dim).reshape(bshape))
    aidx_1 = tuple(aidx_1)
    assert np.allclose(x, x[aidx_1])

    aidx_2 = np.ix_(*tuple(np.arange(dim) for dim in shape))
    assert np.allclose(x, x[aidx_2])

    subshape = tuple(np.random.randint(1, dim) for dim in shape)
    aidx_2 = np.ix_(*tuple(np.arange(dim) for dim in subshape))
    assert x[aidx_2].shape == subshape
    assert np.allclose(x[tuple(slice(0, dim) for dim in subshape)], x[aidx_2])


def test_multitype():
    # TODO(hme): Need this.
    pass


def test_signed_batch_slicing():
    max_size = 123
    for size in range(1, max_size):
        arr = np.arange(size)
        for batch_size in range(1, size):
            for order in (-1, 1):
                if order == 1:
                    res = np.concatenate(list(map(lambda x: arr[x],
                                                  array_utils.get_slices(size, batch_size, order))))
                    assert np.allclose(arr, res)
                else:
                    res = np.concatenate(list(map(lambda x: arr[x],
                                                  array_utils.get_slices(size, batch_size, order))))
                    assert np.allclose(arr[::-1], res)


@pytest.mark.skip
def test_batch_slice_intersection():
    shape = (20, 9, 4)
    block_shape = (4, 3, 2)
    # shape = (5, 4, 3)
    # block_shape = (4, 2, 1)
    size = np.product(shape)
    arr: np.ndarray = np.random.random_sample(size).reshape(shape)

    # Test to ensure selections cover basic array.
    grid: array_utils.OrderedGrid = array_utils.OrderedGrid(shape=shape,
                                                            block_shape=block_shape,
                                                            order=(1, 1, 1))
    asgn_test: np.ndarray = np.random.random_sample(size).reshape(shape)
    for index in grid.index_iterator():
        selection = tuple(grid.slices[index])
        asgn_test[selection] = arr[selection]
    assert np.allclose(arr, asgn_test)

    # Test intersection of a larger selection with variable step sizes with
    # basic selections of blocks.
    arr: np.ndarray = np.arange(size).reshape(shape)
    slices_set = []
    for dim in shape:
        dim_steps = list(filter(lambda i: i != 0, range(-dim*2, dim*2)))
        dim_slices = list(
            map(lambda step: slice(0, dim, step) if step > 0 else slice(-1, -dim - 1, step),
                dim_steps)
        )
        slices_set.append(dim_slices)
    slices_set = list(itertools.product(*slices_set))
    pbar = tqdm.tqdm(total=len(slices_set)*np.product(grid.grid_shape))
    for slices in slices_set:
        big_sliced_arr: np.ndarray = arr[tuple(slices)]
        big_bs: BasicSelection = BasicSelection.from_subscript(shape, tuple(slices))
        assert np.allclose(big_sliced_arr, arr[big_bs.selector()])
        grid: array_utils.OrderedGrid = array_utils.OrderedGrid(shape=shape,
                                                                block_shape=block_shape,
                                                                order=big_bs.order())
        small_sliced_arr: np.ndarray = np.empty(grid.grid_shape, dtype=np.ndarray)
        for index in grid.index_iterator():
            small_slices = tuple(grid.slices[index])
            small_bs: BasicSelection = BasicSelection.from_subscript(shape, small_slices)
            res_bs = big_bs & small_bs
            assert arr[res_bs.selector()].shape == res_bs.get_output_shape()
            small_arr = arr[res_bs.selector()]
            small_sliced_arr[tuple(index)] = small_arr
            pbar.update(1)
        stitched_arr = np.block(small_sliced_arr.tolist())
        assert stitched_arr.shape == big_sliced_arr.shape, (stitched_arr.shape,
                                                            big_sliced_arr.shape)
        assert np.allclose(big_sliced_arr, stitched_arr)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_basics()
    # test_basic_slice_selection()
    # test_stepped_slice_selection()
    # test_index_selection()
    # test_slice_intersection()
    # test_stepped_slice_intersection()
    # test_index_intersection()
    # test_array_intersection()
    # test_multiselect_intersection()
    # test_ellipsis()
    # test_advanced_indexing_broadcasting()
    # test_signed_batch_slicing()
    # test_batch_slice_intersection()
