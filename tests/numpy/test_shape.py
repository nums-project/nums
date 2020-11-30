def test_shape(nps_app_inst):
    import numpy as np

    from nums.numpy import BlockArray
    from nums.core.storage.storage import ArrayGrid
    from nums.numpy import numpy_utils
    from nums import numpy as nps
    shape = (10, 20, 30, 40)
    block_shape = (10, 10, 10, 10)
    ns_ins = application_manager.instance()

    def check_expand_and_squeeze(_np_a, axes):
        np_expand_dims = np.__getattribute__('expand_dims')
        ns_expand_dims = nps.__getattribute__('expand_dims')
        np_squeeze = np.__getattribute__('squeeze')
        ns_squeeze = nps.__getattribute__('squeeze')

        _ns_a = nps.array(_np_a)
        _ns_ins_a = ns_ins.array(_np_a, block_shape=block_shape)

        _np_result = np_expand_dims(_np_a, axes)
        _ns_result = ns_expand_dims(_ns_a, axes)
        _ns_ins_result = ns_expand_dims(_ns_ins_a, axes)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())
        check_dim(_np_result, _ns_result)
        check_dim(_np_result, _ns_ins_result)

        _np_result = np_squeeze(_np_a)
        _ns_result = ns_squeeze(_ns_a)
        _ns_ins_result = ns_squeeze(_ns_ins_a)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())
        check_dim(_np_result, _ns_result)
        check_dim(_np_result, _ns_ins_result)

    def check_dim(_np_a, _ns_a):
        np_ndim = np.__getattribute__('ndim')
        ns_ndim = nps.__getattribute__('ndim')
        assert np_ndim(_np_a) == np_ndim(_ns_a)

    def check_swapaxes(_np_a, axis1, axis2):
        ns_ins = application_manager.instance()
        np_swapaxes = np.__getattribute__('swapaxes')
        ns_swapaxes = nps.__getattribute__('swapaxes')

        _ns_a = nps.array(_np_a)
        _ns_ins_a = ns_ins.array(_np_a, block_shape=block_shape)

        _np_result = np_swapaxes(_np_a, axis1, axis2)
        _ns_result = ns_swapaxes(_ns_a, axis1, axis2)
        _ns_ins_result = ns_swapaxes(_ns_ins_a, axis1, axis2)
        assert np.allclose(_np_result, _ns_result.get())
        assert np.allclose(_np_result, _ns_ins_result.get())

    np_A = np.ones(shape)
    check_expand_and_squeeze(np_A, axes=0)
    check_expand_and_squeeze(np_A, axes=2)
    check_expand_and_squeeze(np_A, axes=4)
    check_expand_and_squeeze(np_A, axes=(2, 3))
    check_expand_and_squeeze(np_A, axes=(0, 5))
    check_expand_and_squeeze(np_A, axes=(0, 5, 6))
    check_expand_and_squeeze(np_A, axes=(2, 3, 5, 6, 7))

    check_swapaxes(np_A, axis1=0, axis2=1)
    check_swapaxes(np_A, axis1=0, axis2=2)
    check_swapaxes(np_A, axis1=0, axis2=3)
    check_swapaxes(np_A, axis1=1, axis2=2)
    check_swapaxes(np_A, axis1=1, axis2=3)
    check_swapaxes(np_A, axis1=2, axis2=3)
    check_swapaxes(np_A, axis1=3, axis2=3)


if __name__ == "__main__":
    from nums.core import application_manager
    nps_app_inst = application_manager.instance()
    test_shape(nps_app_inst)
