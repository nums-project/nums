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

from nums.core.array.application import ArrayApplication


def test_subscript(app_inst: ArrayApplication):
    shape = 12, 21
    npX = np.arange(np.product(shape)).reshape(*shape)
    X = app_inst.array(npX, block_shape=(6, 7))
    for i in range(12):
        assert np.allclose((X[:, i].T @ X[:, i]).get(),
                           npX[:, i].T @ npX[:, i])

    # Aligned tests.
    for i in range(0, 21, 7):
        sel = slice(i, i+7)
        assert np.allclose((X[:, sel].T @ X[:, sel]).get(),
                           npX[:, sel].T @ npX[:, sel])

    # More basic tests.
    X_shape = 12, 21, 16
    npX = np.arange(np.product(X_shape)).reshape(*X_shape)
    X = app_inst.array(npX, block_shape=(6, 7, 8))
    for i in range(12):
        assert np.allclose(X[i].get(), npX[i])
        assert np.allclose(X[i, 1].get(), npX[i, 1])
        assert np.allclose(X[i, :, 2].get(), npX[i, :, 2])
        assert np.allclose(X[:, 3, i].get(), npX[:, 3, i])
        assert np.allclose(X[:, :, i].get(), npX[:, :, i])
        assert np.allclose((X[i].T @ X[i]).get(), npX[i].T @ npX[i])


def test_assign_basic(app_inst: ArrayApplication):
    X_shape = 12, 21, 16
    y_shape = 12, 16, 16
    npX = np.arange(np.product(X_shape)).reshape(*X_shape)
    npy = np.zeros(np.product(y_shape)).reshape(*y_shape)
    X = app_inst.array(npX, block_shape=(6, 7, 8))
    y = app_inst.array(npy, block_shape=(6, 8, 8))
    assert np.allclose(npy, y.get())
    for i in range(12):
        y[i, :, :] = y[i, :, :] + X[i].T @ X[i]
        npy[i, :, :] = npy[i, :, :] + npX[i].T @ npX[i]
        assert np.allclose(npy, y.get()), i
    for i in range(12):
        y[i, 0, :] = y[i, 0:1, :] + X[i][0:1]
        npy[i, 0, :] = npy[i, 0:1, :] + npX[i][0:1]
        assert np.allclose(npy, y.get()), i


def test_assign_2dim_accesses(app_inst: ArrayApplication):
    Z_shape = 32, 36
    W_shape = 32, 36
    npZ = np.zeros(np.product(Z_shape)).reshape(*Z_shape)
    npW = np.random.random_sample(np.product(W_shape)).reshape(*W_shape)
    Z = app_inst.array(npZ, block_shape=(4, 3))
    W = app_inst.array(npW, block_shape=(4, 3))

    # Assert immediately after each assignment and after all assignments.
    # Make all addresses different in non-symmetric way.
    assert np.allclose(Z.get(), npZ)
    Z[0, 0] = W[31, 31]
    npZ[0, 0] = npW[31, 31]
    assert np.allclose(Z.get(), npZ)
    Z[1:2, 0] = W[1:2, 0]
    npZ[1:2, 0] = npW[1:2, 0]
    assert np.allclose(Z.get(), npZ)
    Z[3, 4:5] = W[3, 4:5]
    npZ[3, 4:5] = npW[3, 4:5]
    assert np.allclose(Z.get(), npZ)
    Z[6:7, 8:9] = W[6:7, 8:9]
    npZ[6:7, 8:9] = npW[6:7, 8:9]
    assert np.allclose(Z.get(), npZ)
    Z[10] = W[10]
    npZ[10] = npW[10]
    assert np.allclose(Z.get(), npZ)
    Z[11:12] = W[11:12]
    npZ[11:12] = npW[11:12]
    assert np.allclose(Z.get(), npZ)
    Z[:, 13] = W[:, 13]
    npZ[:, 13] = npW[:, 13]
    assert np.allclose(Z.get(), npZ)
    Z[:, 14:15] = W[:, 14:15]
    npZ[:, 14:15] = npW[:, 14:15]
    assert np.allclose(Z.get(), npZ)
    Z[16, :] = W[16, :]
    npZ[16, :] = npW[16, :]
    assert np.allclose(Z.get(), npZ)
    Z[17:18, :] = W[17:18, :]
    npZ[17:18, :] = npW[17:18, :]
    assert np.allclose(Z.get(), npZ)
    Z[19:20, 21] = W[19, 21:22]
    npZ[19:20, 21] = npW[19, 21:22]
    assert np.allclose(Z.get(), npZ)
    Z[22, 23:24] = W[22:23, 23]
    npZ[22, 23:24] = npW[22:23, 23]
    assert np.allclose(Z.get(), npZ)


def test_assign_dependencies(app_inst: ArrayApplication):
    # Non-trivial dependencies among tasks.
    C_shape = 8, 9
    D_shape = 8, 9
    npC = np.zeros(np.product(C_shape)).reshape(*C_shape)
    npD = np.random.random_sample(np.product(D_shape)).reshape(*D_shape)
    C = app_inst.array(npC, block_shape=(4, 3))
    D = app_inst.array(npD, block_shape=(4, 3))
    for i in range(0, C_shape[0]-3, 3):
        for j in range(0, C_shape[1]-2, 2):
            i1, i2 = i, i + 3
            j1, j2 = j, j + 2
            C[i1, j1:j2] = C[i1, j1:j2] + D[i1, j1:j2]
            D[i1:i2, j1] = D[i1:i2, j1] + C[i1:i2, j1]
            npC[i1, j1:j2] = npC[i1, j1:j2] + npD[i1, j1:j2]
            npD[i1:i2, j1] = npD[i1:i2, j1] + npC[i1:i2, j1]
    assert np.allclose(C.get(), npC)
    assert np.allclose(D.get(), npD)


@pytest.mark.skip
def test_complete_3dim_slices(app_inst: ArrayApplication):
    # All combinations of axes of the following.
    # Index: [a]
    # Axis Slice: [:]
    # Partial Axis Slice: [a1:], [:a1]
    # Slice: [a1:a2]

    shape = 7, 9, 11
    block_shape = (4, 5, 6)
    num_axes = 3
    shape_range = list(map(lambda x: list(range(1, x + 1)), shape))
    axis_accessors = []
    for i in range(num_axes):
        # TODO (hme): Change to a[0] <= a[1].
        accessor_pairs = list(filter(lambda a: a[0] < a[1],
                                     itertools.product(shape_range[i],
                                                       shape_range[i])))
        axis_accessors.append(accessor_pairs)
    accessor_iterator = tuple(itertools.product(*axis_accessors))
    access_modes = [
        lambda a1, a2: a1,
        lambda a1, a2: slice(None, None),
        lambda a1, a2: slice(a1, None),
        lambda a1, a2: slice(None, a1),
        lambda a1, a2: slice(a1, a2)
    ]
    mode_iterator = tuple(itertools.product(access_modes, repeat=num_axes))
    pbar = tqdm.tqdm(total=len(accessor_iterator) * len(mode_iterator),
                     desc="Testing 3dim slices.")

    def test_assignment(laccessor):
        npA = np.zeros(np.product(shape)).reshape(*shape)
        npB = np.random.random_sample(np.product(shape)).reshape(*shape)
        A = app_inst.array(npA, block_shape=block_shape)
        B = app_inst.array(npB, block_shape=block_shape)
        assert np.allclose(npA[laccessor], A[laccessor].get())
        assert np.allclose(npB[laccessor], B[laccessor].get())
        npA[laccessor] = npB[laccessor]
        A[laccessor] = B[laccessor]
        assert np.allclose(npA, A.get())

    for entry in accessor_iterator:
        for mode in mode_iterator:
            accessor = tuple(mode[i](*entry[i]) for i in range(num_axes))
            test_assignment(accessor)
            pbar.update(1)


@pytest.mark.skip
def test_assign_complete_2dim_slices(app_inst: ArrayApplication):
    # All 2-dim slice assignments.
    # i_1:i_2 = k_1:k_2
    A_shape = 4, 6
    B_shape = 4, 6
    A_shape_range = list(map(lambda x: list(range(1, x + 1)), A_shape))
    B_shape_range = list(map(lambda x: list(range(1, x + 1)), B_shape))
    A_block_shapes = list(itertools.product(*A_shape_range))
    B_block_shapes = list(itertools.product(*B_shape_range))

    pbar = tqdm.tqdm(total=np.product([
        len(A_block_shapes),
        len(B_block_shapes),
        len(A_block_shapes)**2,
        len(B_block_shapes)**2,
        ]))
    for A_block_shape in A_block_shapes:
        for B_block_shape in B_block_shapes:
            if A_block_shape != B_block_shape:
                # If array shapes are equal
                # then block shapes must be equal.
                pbar.update(len(A_block_shapes)**2 * len(B_block_shapes)**2)
                continue
            npA = np.zeros(np.product(A_shape)).reshape(*A_shape)
            npB = np.random.random_sample(np.product(B_shape)).reshape(*B_shape)
            A = app_inst.array(npA, block_shape=A_block_shape)
            B = app_inst.array(npB, block_shape=B_block_shape)
            for A_strt in A_block_shapes:
                for A_stp in A_block_shapes:
                    for B_strt in B_block_shapes:
                        for B_stp in B_block_shapes:
                            pbar.update(1)
                            if A_stp[0] <= A_strt[0] or A_stp[1] <= A_strt[1]:
                                continue
                            if (A_stp[0] - A_strt[0] != B_stp[0] - B_strt[0]
                                    or A_stp[1] - A_strt[1] != B_stp[1] - B_strt[1]):
                                continue
                            desc_A = "(%d, %d)[%d:%d, %d:%d]" % (A.block_shape[0], A.block_shape[1],
                                                                 A_strt[0], A_stp[0],
                                                                 A_strt[1], A_stp[1])
                            desc_B = "(%d, %d)[%d:%d, %d:%d]" % (B.block_shape[0], B.block_shape[1],
                                                                 B_strt[0], B_stp[0],
                                                                 B_strt[1], B_stp[1])
                            desc = "Testing 2dim slices. %s = %s" % (desc_A, desc_B)
                            pbar.set_description(desc=desc)
                            assert np.allclose(B[B_strt[0]:B_stp[0], B_strt[1]:B_stp[1]].get(),
                                               npB[B_strt[0]:B_stp[0], B_strt[1]:B_stp[1]])
                            npA[A_strt[0]:A_stp[0], A_strt[1]:A_stp[1]] = npB[B_strt[0]:B_stp[0],
                                                                              B_strt[1]:B_stp[1]]
                            A[A_strt[0]:A_stp[0], A_strt[1]:A_stp[1]] = B[B_strt[0]:B_stp[0],
                                                                          B_strt[1]:B_stp[1]]
                            assert np.allclose(A.get(), npA)
                            assert np.allclose(B.get(), npB)


@pytest.mark.skip
def test_basic_assignment_broadcasting(app_inst: ArrayApplication):
    # Test mixed-length broadcasting.
    def get_sel(num_entries, shape):
        r = []
        for i in range(num_entries):
            dim = shape[i]
            start = rs.random_integers(0, dim-1)
            stop = rs.random_integers(start, dim)
            r.append((start, stop))
        return r

    rs = np.random.RandomState(1337)
    a_shape = (6, 7, 2, 5)
    a_block_shape = (2, 4, 2, 3)
    b_shape = (6, 7, 2, 5)
    b_block_shape = (3, 2, 1, 2)
    num_axes = len(a_shape)
    access_modes = [
        lambda a1, a2: a1,
        lambda a1, a2: slice(None, None, None),
        lambda a1, a2: slice(a1, None, None),
        lambda a1, a2: slice(None, a1, None),
        lambda a1, a2: slice(a1, a2, None)
    ]
    for a_len in range(num_axes):
        for b_len in range(num_axes):
            a_mode_iterator = list(itertools.product(access_modes, repeat=a_len))
            b_mode_iterator = list(itertools.product(access_modes, repeat=b_len))
            pbar = tqdm.tqdm(total=len(a_mode_iterator)*len(b_mode_iterator),
                             desc="Testing assignment broadcasting %d/%d" % (a_len*num_axes+b_len,
                                                                             num_axes**2))
            # Create some valid intervals.
            for a_mode in a_mode_iterator:
                for b_mode in b_mode_iterator:
                    pbar.update(1)
                    a_sel = get_sel(a_len, a_shape)
                    b_sel = get_sel(b_len, b_shape)
                    a_accessor = tuple(a_mode[i](*a_sel[i]) for i in range(a_len))
                    b_accessor = tuple(b_mode[i](*b_sel[i]) for i in range(b_len))
                    arr_a = np.arange(np.product(a_shape)).reshape(a_shape)
                    arr_b = np.arange(np.product(b_shape)).reshape(b_shape)
                    ba_a = app_inst.array(arr_a, a_block_shape)
                    ba_b = app_inst.array(arr_b, b_block_shape)
                    try:
                        arr_a[a_accessor] = arr_b[b_accessor]
                        broadcasted = True
                    except ValueError as _:
                        broadcasted = False
                    if broadcasted:
                        ba_a[a_accessor] = ba_b[b_accessor]
                        assert np.allclose(arr_a, ba_a.get())
                        assert np.allclose(arr_b, ba_b.get())


def test_ref_accessor(app_inst: ArrayApplication):
    # Test case:
    # start aligned
    # or selection shape modulo block is 0
    # or end is equal to shape.
    shape = (9, 8, 7)
    block_shape = (1, 2, 4)

    sels = [
        # End is equal to shape.
        ((slice(3, None), slice(2, None), slice(4, None)),
         (slice(3, None), slice(2, None), slice(4, None))),

        # Selection shape modulo block is 0
        ((slice(3, 7), slice(2, 6), slice(0, 4)),
         (slice(2, 6), slice(0, 4), slice(0, 4))),

        # Start aligned
        ((slice(0, 1), slice(0, 2), slice(0, 4)), None),
        ((slice(0, 1), slice(2, 4)), None),
        ((slice(0, 1)), None),
        ((slice(None, None), slice(0, 4), slice(0, 4)), None),
        ((slice(None, None), slice(None, None), slice(0, 4)), None),
        ((slice(0, None), slice(2, 6), slice(0, 4)), None),
        ((slice(0, 1), slice(0, None)), None),
        ((slice(0, 1)), None),
        ((slice(None, None), slice(0, 4), slice(0, None)), None),
        ((slice(None, None), slice(None, None), slice(0, 4)), None),

        # Broadcast tests.
        ((2,           slice(4, 6), slice(0, 4)),
         (slice(3, 4), slice(2, 4), slice(0, 4))),
        ((6,           slice(6, 8), slice(0, 4)),
         (slice(2, 3), slice(2, 4), slice(0, 4))),
        ((slice(None, None), slice(0, 4), slice(None, None)),
         (slice(4, 5), slice(4, 8), slice(None, None))),

    ]
    for sel_a, sel_b in sels:
        if sel_b is None:
            sel_b = sel_a
        arr_a = np.arange(np.product(shape)).reshape(shape)
        arr_b = np.random.random_sample(np.product(shape)).reshape(shape)
        ba_a = app_inst.array(arr_a, block_shape)
        ba_b = app_inst.array(arr_b, block_shape)
        arr_r = arr_a[sel_a]
        ba_r = ba_a[sel_a]
        assert np.allclose(arr_r, ba_r.get())
        arr_a[sel_a] = arr_b[sel_b]
        ba_a[sel_a] = ba_b[sel_b]
        assert np.allclose(arr_a, ba_a.get())

    # Unequal array shape lengths.
    shape_a = 15, 20
    block_shape_a = 1, 3
    shape_b = 9, 7, 23
    block_shape_b = 1, 3, 1
    arr_a = np.arange(np.product(shape_a)).reshape(shape_a)
    arr_b = np.random.random_sample(np.product(shape_b)).reshape(shape_b)
    ba_a = app_inst.array(arr_a, block_shape_a)
    ba_b = app_inst.array(arr_b, block_shape_b)
    assert np.allclose(arr_b[4:8], ba_b[4:8].get())
    arr_a[1, 3:6] = arr_b[2, 0:3, 3]
    ba_a[1, 3:6] = ba_b[2, 0:3, 3]
    assert np.allclose(arr_a, ba_a.get())

    # Test different broadcasting behavior.
    arr_a = np.arange(np.product(shape_a)).reshape(shape_a)
    arr_b = np.random.random_sample(np.product(shape_b)).reshape(shape_b)
    ba_a = app_inst.array(arr_a, block_shape_a)
    ba_b = app_inst.array(arr_b, block_shape_b)
    assert np.allclose(arr_b[4:8, 0:3], ba_b[4:8, 0:3].get())
    arr_a[8:12, 3:6] = arr_b[4:8, 0:3, 4]
    ba_a[8:12, 3:6] = ba_b[4:8, 0:3, 4]
    assert np.allclose(arr_a, ba_a.get())
    arr_a[2, 3:6] = arr_b[8:9, 0:3, 4]
    ba_a[2, 3:6] = ba_b[8:9, 0:3, 4]
    assert np.allclose(arr_a, ba_a.get())
    arr_a[3:4, 3:6] = arr_b[3, 0:3, 4]
    ba_a[3:4, 3:6] = ba_b[3, 0:3, 4]
    assert np.allclose(arr_a, ba_a.get())


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_subscript(app_inst)
    test_assign_basic(app_inst)
    test_assign_2dim_accesses(app_inst)
    test_assign_dependencies(app_inst)
    # test_complete_3dim_slices(app_inst)
    # test_assign_complete_2dim_slices(app_inst)
    test_basic_assignment_broadcasting(app_inst)
    test_ref_accessor(app_inst)
