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


from typing import Optional

from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray
import nums.core.array.utils as array_utils
from nums.core.array.application import ArrayApplication


def init(num_cpus: Optional[int] = None, cluster_shape: Optional[tuple] = None):
    # pylint: disable = import-outside-toplevel
    import nums.core.settings as settings

    if cluster_shape is not None:
        settings.cluster_shape = cluster_shape
        settings.num_cpus = num_cpus
    _instance()
    return None


def read(filename: str) -> BlockArray:
    """
    Args:
        filename: The name of the file to read. This must be the name of an array
            that was previously written using the nums.write command.

    Returns:
        A BlockArray instance.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.lower().split("s3://")[-1]
        return _instance().read_s3(filename)
    else:
        return _instance().read_fs(filename)


def write(filename: str, ba: BlockArray) -> BlockArray:
    """
    Args:
        filename: The name of the file to write. Supports the s3 protocol.
        ba: The BlockArray instance to write.

    Returns:
        A BlockArray indicating the outcome of this operation.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.lower().split("s3://")[-1]
        return _instance().write_s3(ba, filename)
    else:
        return _instance().write_fs(ba, filename)


def delete(filename: str) -> bool:
    """
    Args:
        filename: The name of the file to delete. This must be a file previously
            written to disk.

    Returns:
        A BlockArray indicating the outcome of this operation.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.lower().split("s3://")[-1]
        return _instance().delete_s3(filename)
    else:
        return _instance().delete_fs(filename)


def read_csv(filename, dtype=float, delimiter=",", has_header=False) -> BlockArray:
    """Read a csv text file.

    Args:
        filename: The filename of the csv.
        dtype: The data type of the csv file's entries.
        delimiter: The value delimiter for each row; usually a comma.
        has_header: Whether the csv file has a header. The header is discarded.

    Returns:
        A BlockArray instance.
    """
    return _instance().read_csv(filename, dtype, delimiter, has_header)


def from_modin(df):
    # pylint: disable = import-outside-toplevel, protected-access, unidiomatic-typecheck
    import numpy as np

    try:
        from modin.pandas.dataframe import DataFrame
        from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame
        from modin.engines.ray.pandas_on_ray.frame.partition import (
            PandasOnRayFramePartition,
        )
    except Exception as e:
        raise Exception(
            "Unable to import modin. Install modin with command 'pip install modin'"
        ) from e

    assert isinstance(df, DataFrame), "Unexpected dataframe type %s" % str(type(df))
    assert isinstance(
        df._query_compiler._modin_frame, PandasOnRayFrame
    ), "Unexpected dataframe type %s" % str(type(df._query_compiler._modin_frame))
    frame: PandasOnRayFrame = df._query_compiler._modin_frame

    app: ArrayApplication = _instance()
    system = app.cm

    # Make sure the partitions are numeric.
    dtype = frame.dtypes[0]
    if not array_utils.is_supported(dtype, type_test=True):
        raise TypeError("%s is not supported." % str(dtype))
    for dt in frame.dtypes:
        if dt != dtype:
            raise TypeError("Mixed types are not supported (%s != %s).")

    dtype = np.__getattribute__(str(dtype))

    # Convert from Pandas to NumPy.
    pd_parts = frame._partition_mgr_cls.map_partitions(
        frame._partitions, lambda df: np.array(df)
    )
    grid_shape = len(frame._row_lengths), len(frame._column_widths)

    shape = (np.sum(frame._row_lengths), np.sum(frame._column_widths))
    block_shape = app.get_block_shape(shape, dtype)
    rows = []
    for i in range(grid_shape[0]):
        cols = []
        for j in range(grid_shape[1]):
            curr_block_shape = (frame._row_lengths[i], frame._column_widths[j])
            part: PandasOnRayFramePartition = pd_parts[(i, j)]
            part.drain_call_queue()
            ba: BlockArray = BlockArray.from_oid(
                part.oid, curr_block_shape, dtype, system
            )
            cols.append(ba)
        if grid_shape[1] == 1:
            row_ba: BlockArray = cols[0]
        else:
            row_ba: BlockArray = app.concatenate(
                cols, axis=1, axis_block_size=block_shape[1]
            )
        rows.append(row_ba)
    result = app.concatenate(rows, axis=0, axis_block_size=block_shape[0])
    return result
