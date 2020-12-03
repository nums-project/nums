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


import boto3
from moto import mock_s3
import numpy as np

from nums.core.array.blockarray import BlockArray


# pylint: disable=import-outside-toplevel


def test_rwd():
    import nums
    from nums.core import application_manager
    from nums.core import settings
    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()

    array: np.ndarray = np.random.random(35).reshape(7, 5)
    ba: BlockArray = nps_app_inst.array(array, block_shape=(3, 4))
    filename = "/tmp/darrays/read_write_delete_array_test"
    write_result_ba: BlockArray = nums.write(filename, ba)
    write_result_np = write_result_ba.get()
    for grid_entry in write_result_ba.grid.get_entry_iterator():
        assert write_result_ba[grid_entry].get() == write_result_np[grid_entry]
    ba_read: BlockArray = nums.read(filename)
    assert nps_app_inst.get(nps_app_inst.allclose(ba, ba_read))
    delete_result_ba: BlockArray = nums.delete(filename)
    delete_result_np = delete_result_ba.get()
    for grid_entry in delete_result_ba.grid.get_entry_iterator():
        assert delete_result_ba[grid_entry].get() == delete_result_np[grid_entry]


@mock_s3
def test_rwd_s3():
    import nums
    from nums.core import application_manager
    from nums.core import settings
    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()

    conn = boto3.resource('s3', region_name='us-east-1')
    assert conn.Bucket('darrays') not in conn.buckets.all()
    conn.create_bucket(Bucket='darrays')

    array: np.ndarray = np.random.random(35).reshape(7, 5)
    ba: BlockArray = nps_app_inst.array(array, block_shape=(3, 4))
    filename = "s3://darrays/read_write_delete_array_test"
    write_result_ba: BlockArray = nums.write(filename, ba)
    write_result_np = write_result_ba.get()
    for grid_entry in write_result_ba.grid.get_entry_iterator():
        assert write_result_ba[grid_entry].get() == write_result_np[grid_entry]
    ba_read: BlockArray = nums.read(filename)
    assert nps_app_inst.get(nps_app_inst.allclose(ba, ba_read))
    delete_result_ba: BlockArray = nums.delete(filename)
    delete_result_np = delete_result_ba.get()
    for grid_entry in delete_result_ba.grid.get_entry_iterator():
        assert delete_result_ba[grid_entry].get() == delete_result_np[grid_entry]


def test_read_csv():
    import nums
    from nums.core import settings
    settings.system_name = "serial"

    filename = settings.pj(settings.project_root, "tests", "core", "storage", "test.csv")
    ba = nums.read_csv(filename, has_header=True)
    assert np.allclose(ba[0].get(), [123, 4, 5])
    assert np.allclose(ba[-1].get(), [1.2, 3.4, 5.6])


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    from nums.core import settings
    settings.system_name = "serial"
    nps_app_inst = application_manager.instance()
    # test_rwd()
    test_rwd_s3()
    # test_read_csv()
