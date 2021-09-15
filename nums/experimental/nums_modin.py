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

from nums.core.application_manager import instance as _instance
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray


def from_modin(df):
    # pylint: disable = import-outside-toplevel, protected-access, unidiomatic-typecheck
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
    assert dtype in (
        float,
        np.float,
        np.float32,
        np.float64,
        int,
        np.int,
        np.int32,
        np.int64,
    )
    # Make sure dtypes are equal.
    for dt in frame.dtypes:
        if type(frame.dtypes.dtype) == np.dtype:
            continue
        assert dt == frame.dtypes
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


if __name__ == "__main__":
    from nums.core import settings
    import modin.pandas as mpd

    filename = settings.pj(
        settings.project_root, "tests", "core", "storage", "test.csv"
    )
    df = mpd.read_csv(filename)
    ba: BlockArray = from_modin(df)
    print(ba.get())

    df = mpd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    )
    ba: BlockArray = from_modin(df)
    print(ba.get())
