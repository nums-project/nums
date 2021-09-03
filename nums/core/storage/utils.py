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
from typing import List

import numpy as np


class Batch(object):

    """
    Simple object for creating an object that can
    generate batches of sequential integers.
    """

    @classmethod
    def from_num_batches(cls, total_size, num_batches):
        batch_size = (total_size + num_batches - 1) // num_batches
        return cls(total_size, batch_size)

    def __init__(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        """
        self.total_size = total_size
        self.batch_size = batch_size
        self.batches = self.get_batches(total_size, batch_size)
        self.num_batches = len(self.batches)

    def get_batches(self, total_size, batch_size):
        """
        :param total_size: Total number of items to split into batches.
        :param batch_size: Size of each batch.
        :return: A list of 2-tuples.
                 Each 2-tuple is a segment of indices corresponding to items of size batch_size.
                 The size of the list is total_size / batch_size.
        """
        if total_size < batch_size:
            return [[0, total_size]]
        batches = list(range(0, total_size, batch_size))
        num_batches = int(total_size / batch_size)
        batches = [batches[i : i + 2] for i in range(0, num_batches, 1)]
        if len(batches[-1]) == 1:
            batches[-1].append(total_size)
        if batches[-1][1] != total_size:
            batches.append([batches[-1][1], total_size])
        return batches


def reverse_readline(filename, buf_size=8192):
    # https://stackoverflow.com/questions/2301789/how-to-read-a-file-in-reverse-order
    """A generator that returns the lines of a file in reverse order"""
    with open(filename, encoding="utf-8") as fh:
        segment = None
        offset = 0
        file_size = remaining_size = fh.seek(0, os.SEEK_END)
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split("\n")
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read.
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first.
                if buffer[-1] != "\n":
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty.
        if segment is not None:
            yield segment


def get_file_size(fname):
    with open(fname, "rt", encoding="utf-8") as fh:
        return fh.seek(0, os.SEEK_END)


def get_char_encoding(fname):
    # Compute the number of bytes used to encode a character for given file.
    fh = open(fname, "rt", encoding="utf-8")
    enc: str = fh.encoding
    fh.close()
    if enc.lower().startswith("utf"):
        bits_per_char = int(enc.split("-")[1])
    elif enc.lower() == "ascii":
        bits_per_char = 8
    else:
        raise Exception("Unsupported encoding.")
    assert np.allclose(float(bits_per_char // 8), (bits_per_char / 8))
    bytes_per_char = bits_per_char // 8
    return bytes_per_char


def get_np_txt_info(fname, comments: str, delimiter: str):
    bytes_per_char = get_char_encoding(fname)

    # Test encoding and extract various other details.
    fh = open(fname, "rt", encoding="utf-8")

    num_test_rows = 3
    rows_a: List[str] = []
    bytes_per_row = None
    for row in fh:
        if row.startswith(comments):
            continue
        if bytes_per_row is None:
            bytes_per_row = len(row) * bytes_per_char
        assert len(row) * bytes_per_char == bytes_per_row
        rows_a.append(row)
        if len(rows_a) == num_test_rows:
            break

    rows_b: List[str] = []
    fh.seek(0)
    for row in fh:
        if row.startswith(comments):
            continue
        rows_b.append(row)
        break
    while len(rows_b) < num_test_rows:
        rows_b.append(fh.read(bytes_per_row))

    bytes_per_col = None
    num_cols = None
    for i in range(len(rows_a)):
        assert rows_a[i] == rows_b[i]
        row_a_str = rows_a[i].strip("\n").split(delimiter)
        row_b_str = rows_b[i].strip("\n").split(delimiter)
        assert len(row_a_str) == len(row_b_str)
        if num_cols is None:
            num_cols = len(row_a_str)
        for j in range(len(row_a_str)):
            if bytes_per_col is None:
                bytes_per_col = len(row_a_str[j])
            assert row_a_str[j] == row_b_str[j]
        row_a = list(map(float, row_a_str))
        row_b = list(map(float, row_b_str))
        assert np.allclose(row_a, row_b)
    fh.close()
    return bytes_per_char, bytes_per_row, bytes_per_col, num_cols


def get_np_comments(fname, comments):
    fh = open(fname, "rt", encoding="utf-8")
    comment_lines = []
    trailing_newlines = 0
    for row in fh:
        if row.startswith(comments):
            comment_lines.append(row)
        break
    fh.close()
    for row in reverse_readline(fname):
        if row.startswith(comments):
            comment_lines.append(row)
        elif row == "\n":
            trailing_newlines += 1
        break
    return comment_lines, trailing_newlines
