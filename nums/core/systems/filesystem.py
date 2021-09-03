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


import os
import pickle
from typing import Any, AnyStr, Tuple, Dict

import numpy as np
from numpy.compat import asbytes, asstr, asunicode

from nums.core import settings
from nums.core.array.blockarray import BlockArray
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid
from nums.core.grid.grid import DeviceID
from nums.core.storage import utils as storage_utils
from nums.core.storage.storage import StoredArrayS3


################
# S3
################
def write_meta_s3(filename: AnyStr, grid_meta: Dict):
    sa: StoredArrayS3 = StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta))
    return np.array(sa.put_grid(sa.grid), dtype=dict)


def delete_meta_s3(filename: AnyStr):
    sa: StoredArrayS3 = StoredArrayS3(filename)
    sa.init_grid()
    return np.array(sa.delete_grid(), dtype=dict)


def write_block_s3(block: Any, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return np.array(
        StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).put(grid_entry, block),
        dtype=dict,
    )


def read_block_s3(filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).get(grid_entry)


def delete_block_s3(filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return np.array(
        StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).delete(grid_entry),
        dtype=dict,
    )


###################
# DFS
###################

ARRAY_FILETYPE = "pkl"


def write_meta_fs(meta: Dict, filename: AnyStr):
    """
    Write meta data to disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(filename, "meta.pkl")
    with open(filepath, "wb") as fh:
        return np.array(pickle.dump(meta, fh), dtype=object)


def read_meta_fs(filename: AnyStr):
    """
    Read meta data from disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(filename, "meta.pkl")
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def delete_meta_fs(filename: AnyStr):
    """
    Delete meta data from disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(filename, "meta.pkl")
    return np.array(os.remove(filepath), dtype=object)


def save(block, filepath):
    if filepath.split(".")[-1] == "npy":
        return np.save(filepath, block)
    elif filepath.split(".")[-1] == "pkl":
        with open(filepath, "wb") as fh:
            return pickle.dump(block, fh)


def load(filepath):
    if filepath.split(".")[-1] == "npy":
        return np.load(filepath)
    elif filepath.split(".")[-1] == "pkl":
        with open(filepath, "rb") as fh:
            return pickle.load(fh)


def write_block_fs(block: Any, filename: AnyStr, grid_entry: Tuple):
    """
    Write block to disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(filename, entry_name)
    return np.array(save(block, filepath), dtype=object)


def read_block_fs(filename, grid_entry: Tuple):
    """
    Read block from disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(filename, entry_name)
    return load(filepath)


def delete_block_fs(filename, grid_entry: Tuple):
    """
    Delete block from disk.
    """
    settings.Path(filename).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(filename, entry_name)
    return np.array(os.remove(filepath), dtype=object)


##############
# NumPy API
##############
def loadtxt_block(
    fname,
    dtype,
    comments,
    delimiter,
    converters,
    skiprows,
    usecols,
    unpack,
    ndmin,
    encoding,
    max_rows,
):
    return np.loadtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        converters=converters,
        skiprows=skiprows,
        usecols=usecols,
        unpack=unpack,
        ndmin=ndmin,
        encoding=encoding,
        max_rows=max_rows,
    )


##############
# CSV API
##############
def read_csv_block(filename, file_start, file_end, dtype, delimiter, has_header):
    def _getconv(_dtype):
        """Adapted from numpy/lib/npyio.py"""

        def floatconv(x):
            x.lower()
            if "0x" in x:
                return float.fromhex(x)
            return float(x)

        if issubclass(_dtype, np.bool_):
            return lambda x: bool(int(x))
        if issubclass(_dtype, np.uint64):
            return np.uint64
        if issubclass(_dtype, np.int64):
            return np.int64
        if issubclass(_dtype, np.integer) or _dtype is int:
            return lambda x: int(float(x))
        elif issubclass(_dtype, np.longdouble):
            return np.longdouble
        elif issubclass(_dtype, np.floating) or _dtype is float:
            return floatconv
        elif issubclass(_dtype, complex):
            return lambda x: complex(asstr(x).replace("+-", "-"))
        elif issubclass(_dtype, np.bytes_):
            return asbytes
        elif issubclass(_dtype, np.unicode_):
            return asunicode
        else:
            return asstr

    lines = []
    converter = _getconv(dtype)
    with open(filename, "r", encoding="utf-8") as fh:
        try:
            fh.seek(file_start)
            if file_start != 0:
                char = None
                while char != "\n":
                    char = fh.read(1)
            header_skipped = False
            while fh.tell() < file_end:
                line = fh.readline().strip("\r\n")
                if file_start == 0 and has_header and not header_skipped:
                    header_skipped = True
                    continue
                line = line.split(delimiter)
                if len(line) == 0:
                    continue
                line = list(map(converter, line))
                lines.append(line)
        except StopIteration:
            pass
    array = np.array(lines, dtype=dtype)
    return array, array.shape


class FileSystem(object):
    # pylint: disable=unused-argument
    # TODO (hme):
    #  - Idempotency for write/delete.
    #  - Write-constraints based on cluster disk capacity.
    #  - Replication of data.
    #  - Less stringent replication of meta-data.
    #  - Journaling?

    def __init__(self, cm: ComputeManager):
        self.cm: ComputeManager = cm
        for func in [
            write_meta_s3,
            delete_meta_s3,
            write_block_s3,
            read_block_s3,
            delete_block_s3,
            write_meta_fs,
            read_meta_fs,
            delete_meta_fs,
            write_block_fs,
            read_block_fs,
            delete_block_fs,
            loadtxt_block,
            read_csv_block,
        ]:
            self.cm.register(func.__name__, func, {})

    ##################################################
    # Block-level (remote) operations
    ##################################################

    def write_meta_s3(self, filename: AnyStr, grid_meta: Dict, syskwargs: Dict):
        return self.cm.call("write_meta_s3", filename, grid_meta, syskwargs=syskwargs)

    def delete_meta_s3(self, filename: AnyStr, syskwargs: Dict):
        return self.cm.call("delete_meta_s3", filename, syskwargs=syskwargs)

    def write_block_s3(
        self,
        block: Any,
        filename: AnyStr,
        grid_entry: Tuple,
        grid_meta: Dict,
        syskwargs: Dict,
    ):
        return self.cm.call(
            "write_block_s3",
            block,
            filename,
            grid_entry,
            grid_meta,
            syskwargs=syskwargs,
        )

    def read_block_s3(
        self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        return self.cm.call(
            "read_block_s3", filename, grid_entry, grid_meta, syskwargs=syskwargs
        )

    def delete_block_s3(
        self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        return self.cm.call(
            "delete_block_s3", filename, grid_entry, grid_meta, syskwargs=syskwargs
        )

    def loadtxt_block(
        self,
        fname,
        dtype,
        comments,
        delimiter,
        converters,
        skiprows,
        usecols,
        unpack,
        ndmin,
        encoding,
        max_rows,
        syskwargs: Dict,
    ):
        # TODO (hme): Invoke file_exists with options to determine which nodes to pull from.
        return self.cm.call(
            "loadtxt_block",
            fname,
            dtype,
            comments,
            delimiter,
            converters,
            skiprows,
            usecols,
            unpack,
            ndmin,
            encoding,
            max_rows,
            syskwargs=syskwargs,
        )

    def write_block_fs(
        self,
        block: Any,
        filename: AnyStr,
        grid_entry: Tuple,
        grid_meta: Dict,
        syskwargs: Dict,
    ):
        return self.cm.call(
            "write_block_fs", block, filename, grid_entry, syskwargs=syskwargs
        )

    def read_block_fs(
        self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        return self.cm.call("read_block_fs", filename, grid_entry, syskwargs=syskwargs)

    def delete_block_fs(
        self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        return self.cm.call(
            "delete_block_fs", filename, grid_entry, syskwargs=syskwargs
        )

    ##################################################
    # Array-level operations
    ##################################################

    def write_meta_fs(self, ba: BlockArray, filename: str):
        addresses: dict = {}
        for grid_entry in ba.grid.get_entry_iterator():
            device_id: DeviceID = self.cm.device_grid.get_device_id(
                grid_entry, ba.grid.grid_shape
            )
            addresses[grid_entry] = str(device_id)
        meta = {
            "filename": filename,
            "grid_meta": ba.grid.to_meta(),
            "addresses": addresses,
        }
        oids = []
        for grid_entry in ba.grid.get_entry_iterator():
            device_id: DeviceID = self.cm.device_grid.get_device_id(
                grid_entry, ba.grid.grid_shape
            )
            oid = self.cm.call(
                "write_meta_fs", meta, filename, syskwargs={"device_id": device_id}
            )
            oids.append(oid)
        return oids

    def read_meta_fs(self, filename: str):
        for device_id in self.cm.devices():
            oid = self.cm.call(
                "read_meta_fs", filename, syskwargs={"device_id": device_id}
            )
            result = self.cm.get(oid)
            if result is not None:
                return result
        raise Exception("failed to load metadata.")

    def delete_meta_fs(self, filename: str):
        oids = []
        for device_id in self.cm.devices():
            oid = self.cm.call(
                "delete_meta_fs", filename, syskwargs={"device_id": device_id}
            )
            oids.append(oid)
        return oids

    def repartition(self, filename: AnyStr, grid_meta: Dict, syskwargs):
        """
        Repartition a loaded array according to provided grid meta data.
        Implement as simple write then delete sequence.
        In order to delete, need old "grid_meta," which we can read from dfs.
        """
        raise NotImplementedError()

    def loadtxt(
        self,
        fname,
        dtype=float,
        comments="# ",
        delimiter=" ",
        converters=None,
        skiprows=0,
        usecols=None,
        unpack=False,
        ndmin=0,
        encoding="bytes",
        max_rows=None,
        num_workers=4,
    ) -> BlockArray:
        # pylint: disable=unused-variable
        (
            bytes_per_char,
            bytes_per_row,
            bytes_per_col,
            num_cols,
        ) = storage_utils.get_np_txt_info(fname, comments, delimiter)
        chars_per_row = bytes_per_row // bytes_per_char
        assert np.allclose(float(chars_per_row), bytes_per_row / bytes_per_char)
        comment_lines, trailing_newlines = storage_utils.get_np_comments(
            fname, comments
        )
        nonrow_chars = trailing_newlines
        for line in comment_lines:
            nonrow_chars += len(line)
        file_size = storage_utils.get_file_size(fname)
        file_chars = file_size // bytes_per_char
        assert np.allclose(float(file_chars), file_size / bytes_per_char)
        row_chars = file_chars - nonrow_chars
        num_rows = row_chars // chars_per_row
        assert np.allclose(float(num_rows), float(row_chars / chars_per_row))
        num_rows_final = num_rows - skiprows
        if max_rows is not None:
            num_rows_final = (num_rows_final, max_rows)
        row_batches: storage_utils.Batch = storage_utils.Batch.from_num_batches(
            num_rows_final, num_workers
        )
        grid = ArrayGrid(
            shape=(num_rows_final, num_cols),
            block_shape=(row_batches.batch_size, num_cols),
            dtype=np.float64.__name__ if dtype is float else dtype.__name__,
        )
        result: BlockArray = BlockArray(grid, cm=self.cm)
        for i, grid_entry in enumerate(grid.get_entry_iterator()):
            row_start, row_end = row_batches.batches[i]
            batch_skiprows = skiprows + row_start
            batch_max_rows = grid.get_block_shape(grid_entry)[0]
            assert batch_max_rows == row_end - row_start
            result.blocks[grid_entry].oid = self.loadtxt_block(
                fname,
                dtype=dtype,
                comments=comments,
                delimiter=delimiter,
                converters=converters,
                skiprows=batch_skiprows,
                usecols=usecols,
                unpack=unpack,
                ndmin=ndmin,
                encoding=encoding,
                max_rows=batch_max_rows,
                syskwargs={"grid_entry": grid_entry, "grid_shape": grid.grid_shape},
            )
        return result

    def read_csv(
        self, filename, dtype=float, delimiter=",", has_header=False, num_workers=4
    ):
        file_size = storage_utils.get_file_size(filename)
        file_batches: storage_utils.Batch = storage_utils.Batch.from_num_batches(
            file_size, num_workers
        )
        blocks = []
        shape_oids = []
        for i, batch in enumerate(file_batches.batches):
            file_start, file_end = batch
            block_oid, shape_oid = self.cm.call(
                "read_csv_block",
                filename,
                file_start,
                file_end,
                dtype,
                delimiter,
                has_header,
                syskwargs={
                    "grid_entry": (i,),
                    "grid_shape": (num_workers,),
                    "options": {"num_returns": 2},
                },
            )
            blocks.append(block_oid)
            shape_oids.append(shape_oid)
        shapes = self.cm.get(shape_oids)
        arrays = []
        for i in range(len(shapes)):
            shape = shapes[i]
            if shape[0] == 0:
                continue
            block = blocks[i]
            grid = ArrayGrid(shape=shape, block_shape=shape, dtype=dtype.__name__)
            arr = BlockArray(grid, self.cm)
            iter_one = True
            for grid_entry in grid.get_entry_iterator():
                assert iter_one
                iter_one = False
                arr.blocks[grid_entry].oid = block
            arrays.append(arr)
        return arrays
