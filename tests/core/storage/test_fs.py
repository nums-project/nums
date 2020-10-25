import os
import time

import numpy as np
import pytest

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray


def test_loadtxt(app_inst: ArrayApplication):
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
    filename = "/tmp/darrays/read_write_delete_array_test"
    write_result_ba: BlockArray = app_inst.write_fs(ba, filename)
    write_result_np = write_result_ba.get()
    for grid_entry in write_result_ba.grid.get_entry_iterator():
        assert write_result_ba[grid_entry].get() == write_result_np[grid_entry]
    ba_read: BlockArray = app_inst.read_fs(filename)
    assert app_inst.get(app_inst.allclose(ba, ba_read))
    delete_result_ba: BlockArray = app_inst.delete_fs(filename)
    delete_result_np = delete_result_ba.get()
    for grid_entry in delete_result_ba.grid.get_entry_iterator():
        assert delete_result_ba[grid_entry].get() == delete_result_np[grid_entry]


def _read_serially(filename, has_header):
    with open(filename) as fh:
        rows = []
        header_read = False
        for line in fh:
            if not header_read and has_header:
                header_read = True
                continue
            row = line.strip("\n\r")
            if row == "":
                continue
            row = row.split(",")
            row = list(map(np.float, row))
            rows.append(row)
        return np.array(rows)


def test_read_csv(app_inst: ArrayApplication):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    filename = os.path.join(dir_path, "test.csv")
    ba_data: BlockArray = app_inst.read_csv(filename, has_header=True)
    np_data = _read_serially(filename, has_header=True)
    assert np.allclose(ba_data.get(), np_data)


@pytest.mark.skip
def test_higgs(app_inst: ArrayApplication):
    from nums.core import settings
    filename = os.path.join(settings.data_dir, "HIGGS.csv")
    t = time.time()
    ba: BlockArray = app_inst.read_csv(filename, num_workers=12)
    ba.touch()
    print("HIGGS nums load time", time.time() - t, ba.shape, ba.block_shape)
    t = time.time()
    np_data = _read_serially(filename, has_header=False)
    print("HIGGS serial load time", time.time() - t, np_data.shape)
    assert np.allclose(ba.get(), np_data)


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("ray-cyclic")
    # test_loadtxt(app_inst)
    # test_rwd(app_inst)
    test_read_csv(app_inst)
    # test_higgs(app_inst)
