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


from nums.api import read, write, delete, read_csv
from nums.core.version import __version__

__all__ = ["numpy", "read", "write", "delete", "read_csv"]


def init():
    # pylint: disable = import-outside-toplevel
    # Explicitly initialize application instance.
    from nums.core.application_manager import instance

    return instance()
