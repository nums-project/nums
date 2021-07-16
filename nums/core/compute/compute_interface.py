from typing import Tuple, Dict

from nums.core.systems.utils import method_meta


class ComputeInterface(object):
    def touch(self, arr, syskwargs: Dict):
        """
        "Touch" the given array. This returns nothing and can be used to wait for
        a computation without pulling data to the head node.
        """
        raise NotImplementedError()

    def empty(self, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict):
        raise NotImplementedError()

    def new_block(
        self, op_name: str, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        raise NotImplementedError()

    def random_block(
        self, rng_params, rfunc_name, rfunc_args, shape, dtype, syskwargs: Dict
    ):
        raise NotImplementedError()

    def permutation(self, rng_params, size, syskwargs: Dict):
        raise NotImplementedError()

    def diag(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def arange(self, start, stop, step, dtype, syskwargs: Dict):
        raise NotImplementedError()

    def transpose(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def swapaxes(self, arr, axis1, axis2, syskwargs: Dict):
        raise NotImplementedError()

    def create_block(
        self,
        *src_arrs,
        src_params,
        dst_params,
        dst_shape,
        dst_shape_bc,
        syskwargs: Dict
    ):
        raise NotImplementedError()

    def update_block(self, dst_arr, *src_arrs, src_params, dst_params, syskwargs: Dict):
        raise NotImplementedError()

    def update_block_by_index(self, dst_arr, src_arr, index_pairs, syskwargs: Dict):
        raise NotImplementedError()

    def update_block_along_axis(
        self, dst_arr, src_arr, index_pairs, axis, syskwargs: Dict
    ):
        raise NotImplementedError()

    def bop(self, op, a1, a2, a1_T, a2_T, axes, syskwargs: Dict):
        raise NotImplementedError()

    def bop_reduce(self, op, a1, a2, a1_T, a2_T, syskwargs: Dict):
        raise NotImplementedError()

    def split(self, arr, indices_or_sections, axis, transposed, syskwargs: Dict):
        raise NotImplementedError()

    def size(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def select_median(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def weighted_median(self, *arr_and_weights, syskwargs: Dict):
        raise NotImplementedError()

    def pivot_partition(self, arr, pivot, op, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_returns=2)
    def qr(self, *arrays, mode="reduced", axis=None, syskwargs: Dict):
        raise NotImplementedError()

    def cholesky(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    @method_meta(num_returns=3)
    def svd(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def inv(self, arr, syskwargs: Dict):
        raise NotImplementedError()

    def array_compare(self, func_name, a, b, args, syskwargs: Dict):
        raise NotImplementedError()

    def map_uop(self, op_name, arr, args, kwargs, syskwargs: Dict):
        raise NotImplementedError()

    def where(self, arr, x, y, block_slice_tuples, syskwargs: Dict):
        raise NotImplementedError()

    def reduce_axis(self, op_name, arr, axis, keepdims, transposed, syskwargs: Dict):
        raise NotImplementedError()

    # Scipy

    def xlogy(self, arr_x, arr_y, syskwargs: Dict):
        raise NotImplementedError()

    def logical_and(self, *bool_list, syskwargs: Dict):
        raise NotImplementedError()

    def astype(self, arr, dtype_str, syskwargs: Dict):
        raise NotImplementedError()

    def arg_op(
        self, op_name, arr, block_slice, other_argoptima, other_optima, syskwargs: Dict
    ):
        raise NotImplementedError()

    def reshape(self, arr, shape, syskwargs: Dict):
        raise NotImplementedError()


class ComputeImp(object):
    pass


class RNGInterface(object):
    def new_block_rng_params(self):
        raise NotImplementedError()
