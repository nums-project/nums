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
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.grid.grid import ArrayGrid
from nums.core.grid.grid import DeviceID
from nums.core.storage.storage import BimodalGaussian
from nums.core.systems import utils as systems_utils

import common  # pylint: disable=import-error, wrong-import-order


def test_scalar_op(app_inst: ArrayApplication):
    app_inst.scalar(1)
    app_inst.scalar(False)
    app_inst.scalar(2.0)
    app_inst.scalar(np.float32(1))
    app_inst.scalar(np.complex64(1))

    with pytest.raises(ValueError):
        app_inst.scalar(np.array(1))


def test_device_id_hashing(app_inst: ArrayApplication):
    assert app_inst is not None
    d1 = DeviceID(0, "node:localhost1", "cpu", 0)
    d2 = DeviceID(1, "node:localhost2", "cpu", 0)
    x = {}
    x[d1] = "one"
    x[d2] = "two"
    assert x[d1] == "one"
    assert x[d2] == "two"


def test_array_integrity(app_inst: ArrayApplication):
    shape = 12, 21
    npX = np.arange(np.product(shape)).reshape(*shape)
    X = app_inst.array(npX, block_shape=(6, 7))
    common.check_block_integrity(X)


def test_transpose(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(100, 9)
    X = app_inst.array(real_X, block_shape=(100, 1))
    assert np.allclose(X.T.get(), real_X.T)
    # Identity.
    assert np.allclose(X.T.T.get(), X.get())
    assert np.allclose(X.T.T.get(), real_X)


def test_reshape(app_inst: ArrayApplication):
    real_X, _ = BimodalGaussian.get_dataset(1000, 9)
    X = app_inst.array(real_X, block_shape=(100, 9))
    X = X.reshape((1000, 9), block_shape=(1000, 1))
    assert np.allclose(X.get(), real_X)


def test_concatenate(app_inst: ArrayApplication):
    axis = 1
    real_X, _ = BimodalGaussian.get_dataset(1000, 9)
    real_ones = np.ones(shape=(1000, 1))
    X = app_inst.array(real_X, block_shape=(100, 9))
    ones = app_inst.ones((1000, 1), (100, 1), dtype=X.dtype)
    X_concated = app_inst.concatenate(
        [X, ones], axis=axis, axis_block_size=X.block_shape[axis]
    )
    common.check_block_integrity(X_concated)
    real_X_concated = np.concatenate([real_X, real_ones], axis=axis)
    assert np.allclose(X_concated.get(), real_X_concated)

    real_X2 = np.random.random_sample(1000 * 17).reshape(1000, 17)
    X2 = app_inst.array(real_X2, block_shape=(X.block_shape[0], 3))
    X_concated = app_inst.concatenate(
        [X, ones, X2], axis=axis, axis_block_size=X.block_shape[axis]
    )
    common.check_block_integrity(X_concated)
    real_X_concated = np.concatenate([real_X, real_ones, real_X2], axis=axis)
    assert np.allclose(X_concated.get(), real_X_concated)

    y1 = app_inst.zeros(shape=(50,), block_shape=(10,), dtype=int)
    y2 = app_inst.ones(shape=(50,), block_shape=(10,), dtype=int)
    y = app_inst.concatenate([y1, y2], axis=0)
    common.check_block_integrity(y)


def test_split(app_inst: ArrayApplication):
    # TODO (hme): Implement a split leveraging block_shape param in reshape op.
    x = app_inst.array(np.array([1.0, 2.0, 3.0, 4.0]), block_shape=(4,))
    syskwargs = {
        "grid_entry": x.blocks[0].grid_entry,
        "grid_shape": x.blocks[0].grid_shape,
        "options": {"num_returns": 2},
    }
    res1, res2 = x.cm.split(
        x.blocks[0].oid, 2, axis=0, transposed=False, syskwargs=syskwargs
    )
    ba = BlockArray(ArrayGrid((4,), (2,), x.dtype.__name__), x.cm)
    ba.blocks[0].oid = res1
    ba.blocks[1].oid = res2
    assert np.allclose([1.0, 2.0, 3.0, 4.0], ba.get())


def test_touch(app_inst: ArrayApplication):
    ones = app_inst.ones((123, 456), (12, 34))
    assert ones.touch() is ones


def test_num_cores(app_inst: ArrayApplication):
    assert np.allclose(app_inst.cm.num_cores_total(), systems_utils.get_num_cores())


def ideal_tall_skinny_shapes(size, dtype):
    assert dtype in (np.float32, np.float64)
    denom = 2 if dtype is np.float64 else 1
    num_cols = 2 ** 8
    if size == "1024GB":
        # Approximately 1 TB, 1024 blocks, 1 GB / block.
        num_rows = 2 ** 30 // denom
        grid_shape = (2 ** 10, 1)
    elif size == "512GB":
        # 512GB, 512 blocks, 1 GB / block.
        # Perfect fit on 8 nodes.
        num_rows = 2 ** 29 // denom
        grid_shape = (2 ** 9, 1)
    elif size == "256GB":
        # 256GB, 256 blocks, 1 GB / block.
        # Perfect fit on 4 nodes.
        num_rows = 2 ** 28 // denom
        grid_shape = (2 ** 8, 1)
    elif size == "128GB":
        # 128GB, 128 blocks, 1 GB / block.
        # Perfect fit on 2 nodes.
        num_rows = 2 ** 27 // denom
        grid_shape = (2 ** 7, 1)
    elif size == "64GB":
        # Approximately 64GB, 64 blocks, 1 GB / block.
        # Perfect fit on 1 nodes.
        num_rows = 2 ** 26 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "32GB":
        num_rows = 2 ** 25 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "16GB":
        num_rows = 2 ** 24 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "8GB":
        num_rows = 2 ** 23 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "4GB":
        num_rows = 2 ** 22 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "2GB":
        num_rows = 2 ** 21 // denom
        grid_shape = (2 ** 6, 1)
    elif size == "1GB":
        # Approximately 1GB, 64 blocks, 16 MB / block.
        num_rows = 2 ** 20 // denom
        grid_shape = (2 ** 6, 1)
    else:
        raise Exception()
    shape = (num_rows, num_cols)
    block_shape = (num_rows // grid_shape[0], num_cols // grid_shape[1])
    return shape, block_shape, grid_shape


def ideal_square_shapes(size, dtype):
    assert dtype in (np.float32, np.float64)
    denom = 2 if dtype is np.float64 else 1
    # Assume 4 bytes, and start with a 1GB square array.
    shape = np.array([2 ** 14, 2 ** 14], dtype=int)
    if size == "4GB":
        shape *= 1 // denom
        grid_shape = (8, 8)
    elif size == "16GB":
        shape *= 4 // denom
        grid_shape = (8, 8)
    elif size == "64GB":
        shape *= 8 // denom
        grid_shape = (8, 8)
    elif size == "256GB":
        shape *= 16 // denom
        grid_shape = (16, 16)
    elif size == "1024GB":
        shape *= 32 // denom
        grid_shape = (32, 32)
    else:
        raise Exception()
    block_shape = tuple(shape // grid_shape)
    shape = tuple(shape)
    return shape, block_shape, grid_shape


def test_compute_block_shape(app_inst: ArrayApplication):
    dtype = np.float32
    cores_per_node = 64
    # Tall-skinny.
    for size in [64, 128, 256, 512, 1024]:
        size_str = "%sGB" % size
        num_nodes = size // 64
        cluster_shape = (16, 1)
        shape, expected_block_shape, expected_grid_shape = ideal_tall_skinny_shapes(
            size_str, dtype
        )
        block_shape = app_inst.cm.compute_block_shape(
            shape, dtype, cluster_shape, num_nodes * cores_per_node
        )
        grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
        print(
            "tall-skinny",
            "cluster_shape=%s" % str(cluster_shape),
            "grid_shape=%s" % str(expected_grid_shape),
            "size=%s" % size_str,
            "bytes computed=%s" % (grid.nbytes() / 10 ** 9),
        )
        assert expected_grid_shape == grid.grid_shape
        assert expected_block_shape == block_shape

    # Square.
    for size in [4, 16, 64, 256, 1024]:
        size_str = "%sGB" % size
        num_nodes = 1 if size < 64 else size // 64
        cluster_shape = int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes))
        shape, expected_block_shape, expected_grid_shape = ideal_square_shapes(
            size_str, dtype
        )
        block_shape = app_inst.cm.compute_block_shape(
            shape, dtype, cluster_shape, num_nodes * cores_per_node
        )
        grid: ArrayGrid = ArrayGrid(shape, block_shape, dtype.__name__)
        print(
            "square",
            "cluster_shape=%s" % str(cluster_shape),
            "grid_shape=%s" % str(expected_grid_shape),
            "size=%s" % size_str,
            "bytes computed=%s" % (grid.nbytes() / 10 ** 9),
        )
        assert expected_grid_shape == grid.grid_shape, "%s != %s" % (
            expected_grid_shape,
            grid.grid_shape,
        )
        assert expected_block_shape == block_shape, "%s != %s" % (
            expected_block_shape,
            block_shape,
        )


if __name__ == "__main__":
    # pylint: disable=import-error, no-member
    import conftest

    app_inst = conftest.get_app("serial")
    test_scalar_op(app_inst)
    test_array_integrity(app_inst)
    test_concatenate(app_inst)
    test_touch(app_inst)
    test_split(app_inst)
    test_compute_block_shape(app_inst)
