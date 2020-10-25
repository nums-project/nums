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


from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray
import numpy as np


def read(filename: str) -> BlockArray:
    """
    :param filename: The name of the file to read. This must be the name of an array
    that was previously written using the nums.write command.
    :return: An instance of BlockArray.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.split("s3://")[-1]
        return instance().read_s3(filename)
    else:
        return instance().read_fs(filename)


def write(filename: str, ba: BlockArray) -> BlockArray:
    """
    :param filename: The name of the file to write. Supports the s3 protocol.
    :param ba: The BlockArray instance to write.
    :return: A BlockArray indicating the outcome of this operation.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.split("s3://")[-1]
        return instance().write_s3(ba, filename)
    else:
        return instance().write_fs(ba, filename)


def delete(filename: str) -> BlockArray:
    """
    :param filename: The name of the file to delete. This must be a file previously
                     written to disk.
    :return: A BlockArray indicating the outcome of this operation.
    """
    if filename.lower().startswith("s3://"):
        filename = filename.split("s3://")[-1]
        return instance().delete_s3(filename)
    else:
        return instance().delete_fs(filename)


def read_csv(filename, dtype=np.float, delimiter=',', has_header=False) -> BlockArray:
    """
    Read a csv text file.
    :param filename: The filename of the csv.
    :param dtype: The data type of the csv file's entries.
    :param delimiter: The value delimiter for each row; usually a comma.
    :param has_header: Whether the csv file has a header. The header is discarded.
    :return: A BlockArray instance.
    """
    return instance().read_csv(filename, dtype, delimiter, has_header)
