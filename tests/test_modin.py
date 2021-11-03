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


from nums.core.array.blockarray import BlockArray


# pylint: disable=import-outside-toplevel
def test_modin():
    import nums
    import nums.numpy as nps
    from nums.core import application_manager
    import modin.pandas as mpd

    from nums.core import settings

    settings.system_name = "ray"
    nps_app_inst = application_manager.instance()
    nps.random.reset()

    filename = settings.pj(
        settings.project_root, "tests", "core", "storage", "test.csv"
    )
    ba1 = nums.read_csv(filename, has_header=True)
    df = mpd.read_csv(filename)
    ba2: BlockArray = nums.from_modin(df)
    assert nps.allclose(ba1, ba2)
    application_manager.destroy()
