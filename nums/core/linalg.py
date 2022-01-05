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


def pca(app: ArrayApplication, X: BlockArray):
    C = app.cov(X, rowvar=False)
    U, _, _ = svd(app, C)
    return X @ U


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
    """
    Distributed algorithm for the inversion of an Upper Triangular Matrix.
    We use the method and notation described in -
    https://www.cs.utexas.edu/users/flame/pubs/siam_spd.pdf.

    The upper-triangular matrix X is partitioned as follows:

                                    ____________________________
             _________________      |  R_00  ‖  R_01  |  R_02  |
             | R_TL  |  R_TR |      |==========================|
    X = R -> |---------------|   -> |   0    ‖  R_11  |  R_12  |
             |  0    |  R_BR |      |--------------------------|
             ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻      |   0    ‖   0    |  R_22  |
                                    ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻

    where the double lines represent the boundaries between R_TL, R_TR, and R_BR.
    """

    # Check if X is a square matrix and if it is backed by a single block.
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1], "This function only accepts square matrices"
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    nonsquare_block = X.block_shape[0] != X.block_shape[1]

    # If X is represented by a single block or its block size is non-square,
    # then perform a naive inversion.
    if single_block or nonsquare_block:
        return inv(app, X)

    # Setup the metadata.
    full_shape = X.shape
    grid_shape = X.grid.grid_shape
    block_shape = X.block_shape

    R = X.copy()
    Zs = app.zeros(full_shape, block_shape, X.dtype)

    # Calculate the inverse for block R_11.
    r11_oid = R.blocks[(0, 0)].oid
    r11_inv_oid = app.cm.inv(
        r11_oid, syskwargs={"grid_entry": (0, 0), "grid_shape": grid_shape}
    )
    R.blocks[(0, 0)].oid = r11_inv_oid
    R_tl_shape = block_shape

    # This iterative algorithm continues until shape(X) == shape(R_TL)
    while R_tl_shape[0] != full_shape[0] and R_tl_shape[1] != full_shape[1]:

        # Calculate the block index for R_11.
        R11_block = (
            R_tl_shape[0] // block_shape[0],
            R_tl_shape[1] // block_shape[1],
        )
        R11_oid = R.blocks[R11_block].oid
        R11_shape = R.blocks[R11_block].shape

        # Calculate the inverse for block R_11.
        R11_inv_oid = app.cm.inv(
            R11_oid, syskwargs={"grid_entry": R11_block, "grid_shape": grid_shape}
        )

        # Replace R_11 with its inverse in R inplace.
        R.blocks[R11_block].oid = R11_inv_oid

        # Calculate the blocks within R_01.
        R01_oids = []
        R01_shapes = []
        R01_grid_entries = []
        R01_sb_row, R01_sb_col = 0, R11_block[1]  # sb -- start_block
        R01_num_blocks = R11_block[0]

        # Collect the block metadata.
        for inc in range(R01_num_blocks):
            R01_oids.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].oid)
            R01_shapes.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].shape)
            R01_grid_entries.append((R01_sb_row + inc, R01_sb_col))

        # Perform blocked matrix multiplication: R_01 = -R_00 @ R_01.
        R01_1_oids = []
        for row_block in range(R01_num_blocks):
            sub_oids = []

            for col_block in range(R01_num_blocks):

                # Get the data for the next block in R_00.
                R00_oid = R.blocks[(row_block, col_block)].oid
                Z_oid = Zs.blocks[(row_block, col_block)].oid

                # Calculate -R_00 by subtracting R_00 from zero.
                neg_R00_oid = app.cm.bop(
                    "subtract",
                    Z_oid,
                    R00_oid,
                    False,
                    False,
                    axes=1,
                    syskwargs={
                        "grid_entry": (row_block, col_block),
                        "grid_shape": grid_shape,
                    },
                )

                # Calculate -R00 @ R01.
                sub_oids.append(
                    app.cm.bop(
                        "tensordot",
                        neg_R00_oid,
                        R01_oids[col_block],
                        False,
                        False,
                        axes=1,
                        syskwargs={
                            "grid_entry": R01_grid_entries[col_block],
                            "grid_shape": grid_shape,
                        },
                    )
                )

            # Perform the summation step over all column blocks.
            R01_1_oids.append(
                app.cm.sum_reduce(
                    *sub_oids,
                    syskwargs={
                        "grid_entry": R01_grid_entries[row_block],
                        "grid_shape": grid_shape,
                    }
                )
            )

        # Perform a second blocked matrix multiplication: R_01 = R_01 @ inv(R_11).
        R01_2_oids = []
        for row_block in range(R01_num_blocks):
            R01_2_oids.append(
                app.cm.bop(
                    "tensordot",
                    R01_1_oids[row_block],
                    R11_inv_oid,
                    False,
                    False,
                    axes=1,
                    syskwargs={
                        "grid_entry": R01_grid_entries[row_block],
                        "grid_shape": grid_shape,
                    },
                )
            )

        # Replace the entries in R with the new object IDs.
        for i, entry in enumerate(R01_grid_entries):
            R.blocks[entry].oid = R01_2_oids[i]

        # Recompute shape(R_TL).
        r11_r, r11_c = R11_shape
        old_r, old_c = R_tl_shape

        R_tl_shape = (old_r + r11_r, old_c + r11_c)

    # By the time we finish the algorithm, R is now an inverse of X.
    return R.reshape(block_shape=block_shape)


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
    theta = R_inv @ (Q.transpose(defer=True) @ y)
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
    theta = R_inv @ (Q.transpose(defer=True) @ y)
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
    theta = inv(app, lamb_vec + R.transpose(defer=True) @ R) @ (
        X.transpose(defer=True) @ y
    )
    return theta
