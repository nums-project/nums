def test_shape(nps_app_inst):
    import numpy as np

    from nums.numpy import BlockArray
    from nums.core.storage.storage import ArrayGrid
    from nums.numpy import numpy_utils
    from nums import numpy as nps

    def check_expand_and_squeeze(_np_a, axes):
        _name = 'matmul'
        np_expand_dims = np.__getattribute__('expand_dims')
        ns_expand_dims = nps.__getattribute__('expand_dims')
        np_squeeze = np.__getattribute__('squeeze')
        ns_squeeze = nps.__getattribute__('squeeze')
        _ns_a = nps.array(_np_a)
        _np_result = np_expand_dims(_np_a, axes)
        _ns_result = ns_expand_dims(_ns_a, axes)
        assert np.allclose(_np_result, _ns_result.get())
        check_dim(_np_result, _ns_result)
        _np_result = np_squeeze(_np_a)
        _ns_result = ns_squeeze(_ns_a)
        assert np.allclose(_np_result, _ns_result.get())
        check_dim(_np_result, _ns_result)

    def check_dim(_np_a, _ns_a):
        np_ndim = np.__getattribute__('ndim')
        ns_ndim = nps.__getattribute__('ndim')
        assert np_ndim(_np_a) == np_ndim(_ns_a)

    np_A = np.ones((10, 20, 30, 40))
    check_expand_and_squeeze(np_A, axes=0)
    check_expand_and_squeeze(np_A, axes=2)
    check_expand_and_squeeze(np_A, axes=4)
    check_expand_and_squeeze(np_A, axes=(2, 3))
    check_expand_and_squeeze(np_A, axes=(0, 5))
    check_expand_and_squeeze(np_A, axes=(0, 5, 6))
    check_expand_and_squeeze(np_A, axes=(2, 3, 5, 6, 7))


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_shape(nps_app_inst)
