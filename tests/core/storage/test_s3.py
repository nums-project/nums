import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.storage.storage import StoredArrayS3, ArrayGrid


def test_rwd(app_inst: ArrayApplication):
    array: np.ndarray = np.random.random(35).reshape(7, 5)
    ba: BlockArray = app_inst.array(array, block_shape=(3, 4))
    filename = "darrays/read_write_delete_array_test"
    write_result: BlockArray = app_inst.write_s3(ba, filename)
    write_result_arr = app_inst.get(write_result)
    for grid_entry in write_result.grid.get_entry_iterator():
        assert 'ETag' in write_result_arr[grid_entry]
    ba_read: BlockArray = app_inst.read_s3(filename)
    assert app_inst.get(app_inst.allclose(ba, ba_read))
    delete_result: BlockArray = app_inst.delete_s3(filename)
    delete_result_arr = app_inst.get(delete_result)
    for grid_entry in delete_result.grid.get_entry_iterator():
        deleted_key = delete_result_arr[grid_entry]["Deleted"][0]["Key"]
        assert deleted_key == StoredArrayS3(filename, delete_result.grid).get_key(grid_entry)


def test_array_rwd():
    X: np.ndarray = np.random.random(3)
    stored_X = StoredArrayS3("darrays/%s_X" % "__test__")
    stored_X.put_grid(ArrayGrid(shape=X.shape,
                                block_shape=X.shape, dtype=np.float64.__name__))
    stored_X.init_grid()
    stored_X.put_array(X)
    assert np.allclose(X, stored_X.get_array())
    stored_X.del_array()
    stored_X.delete_grid()


def test_grid_copy():
    grid = ArrayGrid(shape=(1, 2),
                     block_shape=(1, 2), dtype=np.float64.__name__)
    assert grid.copy() is not grid


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_rwd(app_inst)
    test_array_rwd()
    test_grid_copy()
