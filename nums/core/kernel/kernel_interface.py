import abc
from typing import Tuple, Dict

from nums.core.backends.utils import method_meta


class Kernel(abc.ABC):
    @abc.abstractmethod
    def touch(self, arr, syskwargs: Dict):
        """
        "Touch" the given array. This returns nothing and can be used to wait for
        a computation without pulling data to the head node.
        """
        pass

    @abc.abstractmethod
    def new_block(
        self, op_name: str, grid_entry: Tuple, grid_meta: Dict, syskwargs: Dict
    ):
        pass

    @abc.abstractmethod
    def random_block(
        self, rng_params, rfunc_name, rfunc_args, shape, dtype, syskwargs: Dict
    ):
        pass

    @abc.abstractmethod
    def permutation(self, rng_params, size, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def diag(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def arange(self, start, stop, step, dtype, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def sum_reduce(self, *arrs, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def transpose(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def swapaxes(self, arr, axis1, axis2, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def create_block(
        self,
        *src_arrs,
        src_params,
        dst_params,
        dst_shape,
        dst_shape_bc,
        syskwargs: Dict
    ):
        pass

    @abc.abstractmethod
    def update_block(self, dst_arr, *src_arrs, src_params, dst_params, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def update_block_by_index(self, dst_arr, src_arr, index_pairs, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def advanced_assign_block_along_axis(
        self,
        dst_arr,
        src_arr,
        ss,
        axis,
        dst_coord,
        src_coord,
        syskwargs: Dict,
    ):
        pass

    @abc.abstractmethod
    def advanced_select_block_along_axis(
        self,
        dst_arr,
        src_arr,
        ss,
        dst_axis,
        src_axis,
        dst_coord,
        src_coord,
        syskwargs: Dict,
    ):
        pass

    @abc.abstractmethod
    def bop(self, op, a1, a2, a1_T, a2_T, axes, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def bop_reduce(self, op, a1, a2, a1_T, a2_T, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def split(self, arr, indices_or_sections, axis, transposed, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def shape_dtype(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def size(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def tdigest_chunk(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def percentiles_from_tdigest(self, q, *digests, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def select_median(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def weighted_median(self, *arr_and_weights, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def pivot_partition(self, arr, pivot, op, syskwargs: Dict):
        pass

    @abc.abstractmethod
    @method_meta(num_returns=2)
    def qr(self, *arrays, mode="reduced", axis=None, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def cholesky(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    @method_meta(num_returns=3)
    def svd(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def inv(self, arr, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def array_compare(self, func_name, a, b, args, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def map_uop(self, op_name, arr, args, kwargs, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def where(self, arr, x, y, block_slice_tuples, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def reduce_axis(self, op_name, arr, axis, keepdims, transposed, syskwargs: Dict):
        pass

    # Scipy

    @abc.abstractmethod
    def xlogy(self, arr_x, arr_y, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def logical_and(self, *bool_list, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def astype(self, arr, dtype_str, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def arg_op(
        self, op_name, arr, block_slice, other_argoptima, other_optima, syskwargs: Dict
    ):
        pass

    @abc.abstractmethod
    def reshape(self, arr, shape, syskwargs: Dict):
        pass

    @abc.abstractmethod
    def identity(self, value):
        pass


class KernelImp:
    pass


class RNGInterface:
    def new_block_rng_params(self):
        raise NotImplementedError()
