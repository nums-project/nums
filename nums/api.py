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


from nums.core.application_manager import instance as _instance
from nums.core.array.blockarray import BlockArray


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


def delete(filename: str) -> BlockArray:
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
