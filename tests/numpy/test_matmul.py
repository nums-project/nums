def test_matmul(nps_app_inst):
    import numpy as np

    from nums.numpy import BlockArray
    from nums.core.storage.storage import ArrayGrid
    from nums.numpy import numpy_utils
    from nums import numpy as nps

    def check_matmul_op(_np_a, _np_b):
        _name = 'matmul'
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b)
        _ns_result = ns_ufunc(_ns_a, _ns_b)
        assert np.allclose(_np_result, _ns_result.get())

    def check_tensordot_op(_np_a, _np_b, axes):
        _name = 'tensordot'
        np_ufunc = np.__getattribute__(_name)
        ns_ufunc = nps.__getattribute__(_name)
        _ns_a = nps.array(_np_a)
        _ns_b = nps.array(_np_b)
        _np_result = np_ufunc(_np_a, _np_b, axes=axes)
        _ns_result = ns_ufunc(_ns_a, _ns_b, axes=axes)
        assert np.allclose(_np_result, _ns_result.get())

    np_v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    np_w = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    check_tensordot_op(np_v, np_w, axes=1)
    check_tensordot_op(np_v, np_v, axes=1)

    np_A = np.stack([np_v + i for i in range(20)])
    np_B = np.transpose(np.stack([np_w + i for i in range(20)]))
    check_matmul_op(np_A, np_B)
     

if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_matmul(nps_app_inst)
