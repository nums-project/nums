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

from collections import OrderedDict

def find_slices(blocks):
    src_params, dst_params, count = [], [], 0
    for key in blocks:
        length = len(blocks[key])
        slice_obj_dst = slice(count, count + length, 1)
        slice_obj_src = slice(0, length, 1)
        src_params.append((slice_obj_src, False))
        dst_params.append((slice_obj_dst, False))
        count += length
    return src_params, dst_params, count



