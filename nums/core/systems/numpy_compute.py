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


import random
import secrets
import numpy as np
from numpy.random import PCG64
from numpy.random import Generator
import scipy.linalg
from scipy.linalg import lapack
import scipy.special

from nums.core.storage.storage import ArrayGrid
from nums.core.systems.interfaces import ComputeImp, RNGInterface


def block_rng(seed, jump_index):
    return Generator(PCG64(seed).jumped(jump_index))


class RNG(RNGInterface):
    # A naive approach to implementing a parallel random state is to simply
    # increment the random seed given by the user, but this
    # will lead to many collisions if the user is also incrementing the random seed,
    # so a more principled approach is needed.
    # In particular, our generator should work just as serial generators would
    # if a block array is generated with a random seed of 0, and then
    # a random seed of 1.
    # See: https://numpy.org/doc/stable/reference/random/parallel.html
    # The easiest parallel RNG approach for NumS is to jump the BitGenerator state.
    # The way this works is as follows:
    # The user provides a seed, which we use to seed all bit generators.
    # Whenever a new array is sampled, increment the jump index for each block in the array,
    # and in the remote function, sample the block shape using a newly initialized
    # random state with the user provided seed and the jump index created for the block.
    # Blocks are sampled for new arrays by incrementing the jump index as before.
    # This can be viewed simply as incrementing the jump index whenever a new block needs to be
    # sampled, regardless of the array the block belongs to.
    # A global random state is maintained, just like in numpy, so that
    # a random state is not required to sample numbers.
    # The seed can be set for the global random state, which
    # re-instantiates the global random state.

    # One issue is the following:
    # nrs1 = NPRandom(1337)
    # nrs2 = NPRandom(1337)
    # x1 = nrs1.sample(shape=(100,), block_shape=(10,))
    # x2 = nrs1.sample(shape=(100,), block_shape=(11,))
    # x1 != x2
    # x1 performs more jumps because it has more blocks.
    # One way to remedy this is to always sample using the default internal block shape,
    # and then reshape to the required block shape.
    # This is of course sub-optimal -- we could alternatively jump according to a small
    # block shape, and generate as many jumps needed to sample the required block shape,
    # but this is tedious, and since we won't expose block_shape as a parameter in the final
    # api, the proposed approach works fine as-is.

    def __init__(self, seed=None, jump_index=0):
        # pylint: disable=no-member
        if seed is None:
            try:
                seed = secrets.getrandbits(128)
            except Exception as _:
                seed = random.getrandbits(128)
        self.seed = seed
        self.rng = PCG64(seed)
        self.jump_index = jump_index

    def new_block_rng_params(self):
        params = self.seed, self.jump_index
        self.jump_index += 1
        return params


class ComputeCls(ComputeImp):
    # pylint: disable=no-member, unused-variable

    # I/O operations.
    def touch(self, arr):
        return isinstance(arr, np.ndarray)

    def empty(self, grid_entry, grid_meta):
        grid = ArrayGrid.from_meta(grid_meta)
        block_shape = grid.get_block_shape(grid_entry)
        return np.empty(block_shape, dtype=grid.dtype)

    def new_block(self, op_name, grid_entry, grid_meta):
        op_func = np.__getattribute__(op_name)
        grid = ArrayGrid.from_meta(grid_meta)
        block_shape = grid.get_block_shape(grid_entry)
        if op_name == "eye":
            assert np.all(np.diff(grid_entry) == 0)
            return op_func(block_shape[0], dtype=grid.dtype)
        else:
            return op_func(block_shape, dtype=grid.dtype)

    def random_block(self, rng_params, rfunc_name, rfunc_args, shape, dtype):
        rng: Generator = block_rng(*rng_params)
        op_func = rng.__getattribute__(rfunc_name)
        if rfunc_name == "multinomial" or rfunc_name == "dirichlet":
            assert isinstance(rng_params[0], list)
            shape = tuple(list(shape) + [len(rng_params[0])])
        result = op_func(*rfunc_args).reshape(shape)
        if rfunc_name not in ("random", "integers"):
            # Only random supports sampling of a specific type.
            result = result.astype(dtype)
        return result

    def create_block(self, *src_arrs, src_params, dst_params, dst_shape, dst_shape_bc):
        # TODO (hme): Test putting dst_shape as first param.
        result = np.empty(shape=dst_shape, dtype=src_arrs[0].dtype)
        assert len(src_params) == len(dst_params)
        for i in range(len(src_params)):
            src_arr: np.ndarray = src_arrs[i]
            src_sel, srcT = src_params[i]
            if srcT:
                src_arr = src_arr.T
            dst_sel, dstT = dst_params[i]
            if dst_shape_bc is not None:
                result.reshape(dst_shape_bc)[dst_sel] = src_arr[src_sel]
            else:
                result[dst_sel] = src_arr[src_sel]
        return result

    def update_block(self, dst_arr, *src_arrs, src_params, dst_params):
        assert len(src_params) == len(dst_params)
        # We need to copy here. If we modify this after a no-copy assignment
        # of a block from array A to B, modifying B will modify the contents of A.
        dst_arr = dst_arr.copy()
        _, dstT = dst_params[0]
        if dstT:
            dst_arr = dst_arr.T
        for i in range(len(src_params)):
            src_arr: np.ndarray = src_arrs[i]
            src_sel, src_shape_bc, srcT = src_params[i]
            if srcT:
                src_arr = src_arr.T
            dst_sel, dstT = dst_params[i]
            if src_shape_bc is not None:
                dst_arr[dst_sel] = src_arr.reshape(src_shape_bc)[src_sel]
            else:
                dst_arr[dst_sel] = src_arr[src_sel]
        return dst_arr

    def diag(self, arr):
        return np.diag(arr)

    def reduce_axis(self, op_name, arr, axis, keepdims, transposed):
        op_func = np.__getattribute__(op_name)
        if transposed:
            arr = arr.T
        return op_func(arr, axis=axis, keepdims=keepdims)

    # This is essentially a map.
    def ufunc(self, op_name, arr):
        ufunc = np.__getattribute__(op_name)
        return ufunc(arr)

    def xlogy(self, arr_x, arr_y):
        return scipy.special.xlogy(arr_x, arr_y)

    def astype(self, arr, dtype_str):
        dtype = getattr(np, dtype_str)
        return arr.astype(dtype)

    def sum_reduce(self, *arrs):
        return np.add.reduce(arrs)

    def transpose(self, arr):
        return arr.T

    def split(self, arr, indices_or_sections, axis, transposed):
        if transposed:
            arr = arr.T
        return np.split(arr, indices_or_sections, axis)

    def bop(self, op, a1, a2, a1_shape, a2_shape, a1_T, a2_T, axes):
        if a1_T:
            a1 = a1.T
        if a2_T:
            a2 = a2.T
        if a1.shape != a1_shape:
            a1 = a1.reshape(a1_shape)
        if a2.shape != a2_shape:
            a2 = a2.reshape(a2_shape)
        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "truediv": lambda a, b: a / b,
            "pow": lambda a, b: a ** b,
            "tensordot": lambda a, b: np.tensordot(a, b, axes=axes),
            # bool ops
            "ge": lambda a, b: a >= b,
            "gt": lambda a, b: a > b,
            "le": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
        }[op](a1, a2)

    def qr(self, *arrays, mode="reduced", axis=None):
        if len(arrays) > 1:
            assert axis is not None
            arr = np.concatenate(arrays, axis=axis)
        else:
            arr = arrays[0]
        return np.linalg.qr(arr, mode=mode)

    def cholesky(self, arr):
        return np.linalg.cholesky(arr)

    def svd(self, arr):
        u, sigma, vT = np.linalg.svd(arr)
        u = u[:sigma.shape[0]]
        return u, sigma, vT

    def inv(self, arr):
        return np.linalg.inv(arr)

    def inv_sym_psd(self, arr: np.ndarray):
        if arr.dtype == np.float32:
            lapack_func = lapack.strtri
        elif arr.dtype == np.float64:
            lapack_func = lapack.dtrtri
        else:
            raise ValueError("Unsupported dtype %s" % str(arr.dtype))
        L_inv, info = lapack_func(scipy.linalg.cholesky(np.asfortranarray(arr),
                                                        lower=True,
                                                        overwrite_a=True,
                                                        check_finite=False),
                                  lower=1,
                                  unitdiag=0,
                                  overwrite_c=1)
        return L_inv.T @ L_inv

    # Boolean

    def allclose(self, a: np.ndarray, b: np.ndarray, rtol, atol):
        return np.allclose(a, b, rtol, atol)

    # Lapack

    def lapack_dtrtri(self, arr, lower=0, unitdiag=0, overwrite_c=0):
        inv, info = lapack.dtrtri(arr, lower, unitdiag, overwrite_c)
        return inv

    def lapack_strtri(self, arr, lower=0, unitdiag=0, overwrite_c=0):
        inv, info = lapack.strtri(arr, lower, unitdiag, overwrite_c)
        return inv

    # Logic

    def logical_and(self, *bool_list):
        return np.all(bool_list)
