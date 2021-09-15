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


import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.grid.grid import ArrayGrid


def qr(app: ArrayApplication, X: BlockArray):
    return indirect_tsqr(app, X)


def _qr_tree_reduce(
    app: ArrayApplication, oid_list, result_grid_entry, result_grid_shape
):
    if len(oid_list) == 1:
        return oid_list[0][0]
    q = oid_list
    while len(q) > 1:
        a_oid, a_ge, a_gs = q.pop(0)
        b_oid, _, _ = q.pop(0)
        ge, gs = (result_grid_entry, result_grid_shape) if len(q) == 0 else (a_ge, a_gs)
        c_oid = app.cm.qr(
            a_oid,
            b_oid,
            mode="r",
            axis=0,
            syskwargs={
                "grid_entry": ge,
                "grid_shape": gs,
                "options": {"num_returns": 1},
            },
        )
        q.append((c_oid, ge, gs))
    r_oid, r_ge, r_gs = q.pop(0)
    assert r_ge == result_grid_entry
    assert r_gs == result_grid_shape
    return r_oid


def indirect_tsr(app: ArrayApplication, X: BlockArray, reshape_output=True):
    assert len(X.shape) == 2
    # TODO (hme): This assertion is temporary and ensures returned
    #  shape of qr of block is correct.
    assert X.block_shape[0] >= X.shape[1]
    # Compute R for each block.
    grid = X.grid
    grid_shape = grid.grid_shape
    shape = X.shape
    block_shape = X.block_shape
    R_oids = []
    # Assume no blocking along second dim.
    for i in range(grid_shape[0]):
        # Select a row according to block_shape.
        row = []
        for j in range(grid_shape[1]):
            row.append(X.blocks[i, j].oid)
        ge, gs = (i, 0), (grid_shape[0], 1)
        oid = app.cm.qr(
            *row,
            mode="r",
            axis=1,
            syskwargs={
                "grid_entry": ge,
                "grid_shape": gs,
                "options": {"num_returns": 1},
            }
        )
        R_oids.append((oid, ge, gs))

    # Construct R by summing over R blocks.
    # TODO (hme): Communication may be inefficient due to redundancy of data.
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    tsR = BlockArray(
        ArrayGrid(shape=R_shape, block_shape=R_shape, dtype=X.dtype.__name__), app.cm
    )
    tsR.blocks[0, 0].oid = _qr_tree_reduce(app, R_oids, (0, 0), (1, 1))

    # If blocking is "tall-skinny," then we're done.
    if R_shape != R_block_shape:
        if reshape_output:
            R = tsR.reshape(R_shape, block_shape=R_block_shape)
        else:
            R = tsR
    else:
        R = tsR
    return R


def indirect_tsqr(app: ArrayApplication, X: BlockArray, reshape_output=True):
    shape = X.shape
    block_shape = X.block_shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    tsR = indirect_tsr(app, X, reshape_output=False)

    # Compute inverse of R.
    tsR_inverse = inv(app, tsR)
    # If blocking is "tall-skinny," then we're done.
    if R_shape != R_block_shape:
        R_inverse = tsR_inverse.reshape(R_shape, block_shape=R_block_shape)
        if reshape_output:
            R = tsR.reshape(R_shape, block_shape=R_block_shape)
        else:
            R = tsR
    else:
        R_inverse = tsR_inverse
        R = tsR

    Q = X @ R_inverse
    return Q, R


def direct_tsqr(app: ArrayApplication, X, reshape_output=True):
    assert len(X.shape) == 2

    # Compute R for each block.
    shape = X.shape
    grid = X.grid
    grid_shape = grid.grid_shape
    block_shape = X.block_shape
    Q_oids = []
    R_oids = []
    QR_dims = []
    Q2_shape = [0, shape[1]]
    for i in range(grid_shape[0]):
        # Select a row according to block_shape.
        row = []
        for j in range(grid_shape[1]):
            row.append(X.blocks[i, j].oid)
        # We invoke "reduced", so q, r is returned with dimensions (M, K), (K, N), K = min(M, N)
        M = grid.get_block_shape((i, 0))[0]
        N = shape[1]
        K = min(M, N)
        QR_dims.append(((M, K), (K, N)))
        Q2_shape[0] += K
        # Run each row on separate nodes along first axis.
        # This maintains some data locality.
        Q_oid, R_oid = app.cm.qr(
            *row,
            mode="reduced",
            axis=1,
            syskwargs={
                "grid_entry": (i, 0),
                "grid_shape": (grid_shape[0], 1),
                "options": {"num_returns": 2},
            }
        )
        R_oids.append(R_oid)
        Q_oids.append(Q_oid)

    # TODO (hme): This pulls several order N^2 R matrices on a single node.
    #  A solution is the recursive extension to direct TSQR.
    Q2_oid, R2_oid = app.cm.qr(
        *R_oids,
        mode="reduced",
        axis=0,
        syskwargs={
            "grid_entry": (0, 0),
            "grid_shape": (1, 1),
            "options": {"num_returns": 2},
        }
    )

    Q2_shape = tuple(Q2_shape)
    Q2_block_shape = (QR_dims[0][1][0], shape[1])
    Q2 = app.vec_from_oids(
        [Q2_oid], shape=Q2_shape, block_shape=Q2_block_shape, dtype=X.dtype
    )
    # The resulting Q's from this operation are N^2 (same size as above R's).
    Q2_oids = list(map(lambda block: block.oid, Q2.blocks.flatten()))

    # Construct Q.
    Q = app.zeros(shape=shape, block_shape=(block_shape[0], shape[1]), dtype=X.dtype)
    for i, grid_entry in enumerate(Q.grid.get_entry_iterator()):
        Q.blocks[grid_entry].oid = app.cm.bop(
            "tensordot",
            Q_oids[i],
            Q2_oids[i],
            a1_T=False,
            a2_T=False,
            axes=1,
            syskwargs={"grid_entry": grid_entry, "grid_shape": Q.grid.grid_shape},
        )

    # Construct R.
    shape = X.shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    tsR = app.vec_from_oids([R2_oid], shape=R_shape, block_shape=R_shape, dtype=X.dtype)
    # If blocking is "tall-skinny," then we're done.
    if R_shape == R_block_shape or not reshape_output:
        R = tsR
    else:
        R = tsR.reshape(R_shape, block_shape=R_block_shape)

    if Q.shape != block_shape or not reshape_output:
        Q = Q.reshape(shape, block_shape=block_shape)

    return Q, R


def svd(app: ArrayApplication, X):
    # TODO(hme): Optimize by merging with direct qr to compute U directly,
    #  to avoid wasting space storing intermediate Q.
    #  This may not really help until we have operator fusion.
    assert len(X.shape) == 2
    block_shape = X.block_shape
    shape = X.shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    Q, R = direct_tsqr(app, X, reshape_output=False)
    assert R.shape == R.block_shape
    R_U, S, VT = app.cm.svd(
        R.blocks[(0, 0)].oid, syskwargs={"grid_entry": (0, 0), "grid_shape": (1, 1)}
    )
    R_U: BlockArray = app.vec_from_oids([R_U], R_shape, R_block_shape, X.dtype)
    S: BlockArray = app.vec_from_oids([S], R_shape[:1], R_block_shape[:1], X.dtype)
    VT = app.vec_from_oids([VT], R_shape, R_block_shape, X.dtype)
    U = Q @ R_U

    return U, S, VT


def inv(app: ArrayApplication, X: BlockArray):
    # TODO (hme): Implement scalable version.
    block_shape = X.block_shape
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1]
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    if single_block:
        result = X.copy()
    else:
        result = X.reshape(block_shape=X.shape)
    result.blocks[0, 0].oid = app.cm.inv(
        result.blocks[0, 0].oid, syskwargs={"grid_entry": (0, 0), "grid_shape": (1, 1)}
    )
    if not single_block:
        result = result.reshape(block_shape=block_shape)
    return result


def inv_uppertri(app: ArrayApplication, X: BlockArray):
    # Inversion of an Upper Triangular Matrix
    # Use the method described in https://www.cs.utexas.edu/users/flame/pubs/siam_spd.pdf
    assert X.shape[0] == X.shape[1], "This function only accepts square matrices"
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    nonsquare_block = X.block_shape[0] != X.block_shape[1]

    # If X is single block or block size is non-square, then reshape BS to row_size // 4
    if single_block or nonsquare_block:
        X = X.reshape(block_shape=(X.shape[0]//4, X.shape[0]//4))

    # Setup metadata
    full_shape = X.shape
    grid_shape = X.grid.grid_shape
    block_shape = X.block_shape

    R = X.copy()
    Zs = app.zeros(full_shape, block_shape, X.dtype)

    # Calculate R_11^-1
    r11_oid = R.blocks[(0,0)].oid
    r11_inv_oid = app.cm.inv(r11_oid, syskwargs={
                                                "grid_entry": (0, 0),
                                                "grid_shape": grid_shape
                                            })
    R.blocks[(0,0)].oid = r11_inv_oid
    R_tl_shape = block_shape

    # Continue while R_tl.shape != R.shape
    while R_tl_shape[0] != full_shape[0] and R_tl_shape[1] != full_shape[1]:
        # Calculate R11
        R11_block = (
            int(np.ceil(R_tl_shape[0] // block_shape[0])),
            int(np.ceil(R_tl_shape[1] // block_shape[1]))
            )
        R11_oid = R.blocks[R11_block].oid
        R11_shape = R.blocks[R11_block].shape

        R11_inv_oid = app.cm.inv(R11_oid, syskwargs={
                                                        "grid_entry": R11_block,
                                                        "grid_shape": grid_shape
                                                    })

        # Reset R11 inplace
        R.blocks[R11_block].oid = R11_inv_oid

        # Calculate R01
        R01_oids = []
        R01_shapes = []
        R01_grid_entries = []
        R01_sb_row, R01_sb_col = 0, R11_block[1] # sb -- start_block
        R01_num_blocks = R11_block[0]

        # Collect data for R01
        for inc in range(R01_num_blocks):
            R01_oids.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].oid)
            R01_shapes.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].shape)
            R01_grid_entries.append((R01_sb_row + inc, R01_sb_col))

        # Perform matrix multiplication: R01_1 = -R00 @ R01
        R01_1_oids = []
        for row_block in range(R01_num_blocks):
            sub_oids = []

            for col_block in range(R01_num_blocks):

                # Get data for R00
                R00_oid = R.blocks[(row_block, col_block)].oid
                Z_oid = Zs.blocks[(row_block, col_block)].oid

                R00_bs = R.blocks[(row_block, col_block)].shape

                # Calculate -R00 = 0 - R00
                neg_R00_oid = app.cm.bop("subtract", Z_oid, R00_oid, R00_bs, R00_bs,
                                                False, False, axes=1, syskwargs={
                                                    "grid_entry": (row_block, col_block),
                                                    "grid_shape": grid_shape
                                                })
                # Calculate -R00 @ R01
                sub_oids.append(app.cm.bop("tensordot", neg_R00_oid, R01_oids[col_block],
                                    R00_bs, R01_shapes[col_block], False, False, axes=1, syskwargs={
                                        "grid_entry": R01_grid_entries[col_block],
                                        "grid_shape": grid_shape
                                    }
                                ))

            # Finished with one blocked mult
            R01_1_oids.append(app.cm.sum_reduce(*sub_oids, syskwargs={
                "grid_entry": R01_grid_entries[row_block],
                "grid_shape": grid_shape
            }))

        # Perform matrix multiplication: R_01_2 = R_01_1 @ R_11_inv
        R01_2_oids = []
        for row_block in range(R01_num_blocks):
            R01_2_oids.append(app.cm.bop("tensordot", R01_1_oids[row_block], R11_inv_oid,
                                R01_shapes[row_block], R11_shape, False, False, axes=1, syskwargs={
                                    "grid_entry": R01_grid_entries[row_block],
                                    "grid_shape": grid_shape
                                }
                            ))

        # Reset R_01
        for i, entry in enumerate(R01_grid_entries):
            R.blocks[entry].oid = R01_2_oids[i]

        # Recompute R_tl.shape
        r11_r, r11_c = R11_shape
        old_r, old_c = R_tl_shape

        R_tl_shape = (old_r + r11_r, old_c + r11_c)

    # By the time we finish, R = R_inv
    return R


def inv_cholesky(app: ArrayApplication, X: BlockArray):
    # Matrix Inversion for X where X is a square positive definite matrix
    # Particularly useful for least-squares regression on a large data corpus

    # Step 1: Calculate the U using Cholesky, where X = U^TU
    U = app.cholesky_blocked(X)

    # Step 2: Compute U^-1
    U_inv = app.inv_uppertri(U)

    # Step 3: Compute inv(X) by U_inv @ U_inv.T
    return U_inv @ U_inv.T


def cholesky(app: ArrayApplication, X: BlockArray):
    # TODO (hme): Implement scalable version.
    # Note:
    # A = Q, R
    # A.T @ A = R.T @ R
    # A.T @ A = L @ L.T
    # => R == L.T
    block_shape = X.block_shape
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1]
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    if single_block:
        result = X.copy()
    else:
        result = X.reshape(block_shape=X.shape)
    result.blocks[0, 0].oid = app.cm.cholesky(
        result.blocks[0, 0].oid, syskwargs={"grid_entry": (0, 0), "grid_shape": (1, 1)}
    )
    if not single_block:
        result = result.reshape(block_shape=block_shape)
    return result


def cholesky_blocked(app: ArrayApplication, X: BlockArray):
    assert_string = "This function only accepts "
    assert X.shape[0] == X.shape[1], assert_string + "square matrices"
    assert X.block_shape[0] == X.block_shape[1], assert_string + "square blocks"
    assert X.shape[0] % X.block_shape[0] == 0, assert_string + "blocks divisible by size of matrix"
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]

    # Setup metadata
    full_shape = X.shape
    block_shape = X.block_shape

    n, b = full_shape[0], block_shape[0]
    if single_block:
        # only one block means we do regular cholesky
        A_TL = app.cholesky(X)
    else:
        # Must do blocked cholesky

        # cholesky on A_TL
        A_TL = BlockArray.from_blocks(X.blocks[:1, :1], (b,b), cm=app.cm)
        A_TL = app.cholesky(A_TL)

        # A_TR = inv(A_TL).T @ A_TR
        A_TR = BlockArray.from_blocks(X.blocks[:1, 1:], (b,n-b), cm=app.cm)
        A_TL_inv_T = app.inv(A_TL).T
        A_TR = A_TL_inv_T @ A_TR

        # A_BR = A_BR - A_TR.T @ A_TR
        A_BR = BlockArray.from_blocks(X.blocks[1:, 1:],(n-b,n-b), cm=app.cm)
        A_BR = A_BR - (A_TR.T @ A_TR)
        while A_TL.shape[0] < n:
            A_TL_size = A_TL.shape[0]
            A_00 = A_TL
            if A_TL.shape[0] == n-b:
                A_01 = A_TR
                A_11 = app.cholesky(A_BR)
                A_TL = app.zeros((A_TL_size+b,A_TL_size+b),block_shape)
                for i in range(A_TL_size // b):
                    for j in range(A_TL_size // b):
                        A_TL.blocks[i,j].oid = A_00.blocks[i,j].oid

                for i in range(A_TL_size // b):
                    A_TL.blocks[i,A_TL_size//b].oid = A_01.blocks[i,0].oid
                A_TL.blocks[A_TL_size//b,A_TL_size//b].oid = A_11.blocks[0,0].oid
            else:
                A_01 = BlockArray.from_blocks(A_TR.blocks[:, :1], (A_TL_size,b), cm=app.cm)
                A_02 = BlockArray.from_blocks(
                    A_TR.blocks[:, 1:],
                    (A_TL_size,n-A_TL_size-b),
                    cm=app.cm
                    )
                A_11 = BlockArray.from_blocks(A_BR.blocks[:1, :1], (b,b), cm=app.cm)
                A_12 = BlockArray.from_blocks(A_BR.blocks[:1, 1:], (b,n-A_TL_size-b), cm=app.cm)
                A_22 = BlockArray.from_blocks(
                    A_BR.blocks[1:, 1:],
                    (n-A_TL_size-b,n-A_TL_size-b),
                    cm=app.cm
                )

                A_11 = app.cholesky(A_11)

                A_11_inv_T = app.inv(A_11).T
                A_12 = A_11_inv_T @ A_12

                A_22 = A_22 - A_12.T @ A_12

                # Get new A_TL, A_TR, A_BR
                A_TL = app.zeros((A_TL_size+b,A_TL_size+b),block_shape)
                A_TR = app.zeros((A_TL_size+b,n-A_TL_size-b),block_shape)

                for i in range(A_TL_size // b):
                    for j in range(A_TL_size // b):
                        A_TL.blocks[i,j].oid = A_00.blocks[i,j].oid

                for i in range(A_TL_size // b):
                    A_TL.blocks[i,A_TL_size//b].oid = A_01.blocks[i,0].oid

                A_TL.blocks[A_TL_size//b,A_TL_size//b].oid = A_11.blocks[0,0].oid

                for i in range(A_TL_size // b):
                    for j in range((n-A_TL_size-b)//b):
                        A_TR.blocks[i,j].oid = A_02.blocks[i,j].oid

                for j in range((n-A_TL_size-b)//b):
                    A_TR.blocks[A_TL_size // b,j].oid = A_12.blocks[0,j].oid

                A_BR = A_22
    return A_TL


def fast_linear_regression(app: ArrayApplication, X: BlockArray, y: BlockArray):
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    block_shape = X.block_shape
    shape = X.shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    Q, R = indirect_tsqr(app, X, reshape_output=False)
    R_inv = inv(app, R)
    if R_shape != R_block_shape:
        R_inv = R_inv.reshape(R_shape, block_shape=R_block_shape)
    theta = R_inv @ (Q.T @ y)
    return theta


def linear_regression(app: ArrayApplication, X: BlockArray, y: BlockArray):
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    block_shape = X.block_shape
    shape = X.shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    Q, R = direct_tsqr(app, X, reshape_output=False)
    # Invert R.
    R_inv = inv(app, R)
    if R_shape != R_block_shape:
        R_inv = R_inv.reshape(R_shape, block_shape=R_block_shape)
    theta = R_inv @ (Q.T @ y)
    return theta


def ridge_regression(app: ArrayApplication, X: BlockArray, y: BlockArray, lamb: float):
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert lamb >= 0
    block_shape = X.block_shape
    shape = X.shape
    R_shape = (shape[1], shape[1])
    R_block_shape = (block_shape[1], block_shape[1])
    R = indirect_tsr(app, X)
    lamb_vec = app.array(lamb * np.eye(R_shape[0]), block_shape=R_block_shape)
    # TODO (hme): A better solution exists, which inverts R by augmenting X and y.
    #  See Murphy 7.5.2.
    theta = inv(app, lamb_vec + R.T @ R) @ (X.T @ y)
    return theta
