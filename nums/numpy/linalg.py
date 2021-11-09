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


import warnings

from nums.core.application_manager import instance as _instance
from nums.core import linalg
from nums.core.array.blockarray import BlockArray


def qr(a: BlockArray, mode="reduced"):
    if mode != "reduced":
        raise NotImplementedError("Only reduced QR decomposition is supported.")
    return linalg.qr(_instance(), a)


def svd(a: BlockArray, full_matrices=False, compute_uv=True, hermitian=False):
    if not (not full_matrices and compute_uv and not hermitian):
        raise NotImplementedError("SVD currently supported on default parameters only.")
    return linalg.svd(_instance(), a)


def inv(a: BlockArray):
    if not a.is_single_block():
        warnings.warn(
            "nums.numpy.linalg.inv is not a scalable implementation. "
            + ("Input array is %s bytes. " % a.nbytes)
            + "Abort this operation if input array is too large to "
            + "execute on a single node."
        )
    return linalg.inv(_instance(), a)


def pca(X: BlockArray):
    return linalg.pca(_instance(), X)
