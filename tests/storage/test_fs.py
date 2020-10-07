import os

import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray


def test_text_basic(app_inst: ArrayApplication):
    seed = 1337
    rs = np.random.RandomState(seed)

    fname = "test_text.out"
    header = ["field1", "field2", "field3"]
    data = rs.random_sample(99).reshape(33, 3)

    np.savetxt(fname=fname,
               X=data,
               fmt='%.18e',
               delimiter=',',
               newline='\n',
               header=",".join(header),
               footer='',
               comments='# ',
               encoding=None)

    np_loaded_data = np.loadtxt(
        fname, dtype=float, comments='# ', delimiter=',',
        converters=None, skiprows=0, usecols=None, unpack=False,
        ndmin=0, encoding='bytes', max_rows=None)

    assert np.allclose(data, np_loaded_data)

    nums_array = app_inst.loadtxt(
        fname, dtype=float, comments='# ', delimiter=',',
        converters=None, skiprows=0, usecols=None, unpack=False,
        ndmin=0, encoding='bytes', max_rows=None)

    np.allclose(data, nums_array.get())

    os.remove(fname)
    assert not os.path.exists(fname)


def test_rwd(app_inst: ArrayApplication):
    array: np.ndarray = np.random.random(35).reshape(7, 5)
    ba: BlockArray = app_inst.array(array, block_shape=(3, 4))
    filename = "darrays/read_write_delete_array_test"
    write_result_ba: BlockArray = app_inst.write_fs(ba, filename)
    write_result_np = write_result_ba.get()
    for grid_entry in write_result_ba.grid.get_entry_iterator():
        assert write_result_ba[grid_entry].get() == write_result_np[grid_entry]
        print(write_result_np[grid_entry])
    ba_read: BlockArray = app_inst.read_fs(filename)
    assert app_inst.get(app_inst.allclose(ba, ba_read))
    delete_result_ba: BlockArray = app_inst.delete_fs(filename)
    delete_result_np = delete_result_ba.get()
    for grid_entry in delete_result_ba.grid.get_entry_iterator():
        assert delete_result_ba[grid_entry].get() == delete_result_np[grid_entry]
        print(delete_result_np[grid_entry])


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_text_basic(app_inst)
    test_rwd(app_inst)
