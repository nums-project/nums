
import warnings
import itertools

import numpy as np

from nums.core.array import utils as array_utils
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array.view import ArrayView
from nums.core.grid.grid import ArrayGrid
from nums.core.compute.compute_manager import ComputeManager
from nums.core.array.base import Block


def atleast_nd_array(obj):
    if isinstance(obj, list):
        return np.array(obj)
    return obj


def tree_reduce(
        cm, op_name, blocks_or_oids, result_grid_entry, result_grid_shape
):
    """
    Basic tree reduce imp.
    Schedules op on same node as left operand.
    :param op_name: The reduction op.
    :param blocks_or_oids: A list of type Block or a list of tuples.
                           Tuples must be of the form
                           (oid, grid_entry, grid_shape, transposed)
    :param result_grid_entry: The grid entry of the result block. This will be used
                              to compute the final reduction step.
    :param result_grid_shape: The grid entry of the result block. This will be used
                              to compute the final reduction step.
    :return: The oid of the result.
    """
    oid_list = blocks_or_oids
    if isinstance(blocks_or_oids[0], Block):
        oid_list = [
            (b.oid, b.grid_entry, b.grid_shape, b.transposed)
            for b in blocks_or_oids
        ]
    if len(oid_list) == 1:
        return oid_list[0][0]
    q = oid_list
    while len(q) > 1:
        a_oid, a_ge, a_gs, a_T = q.pop(0)
        b_oid, _, _, b_T = q.pop(0)
        ge, gs = (
            (result_grid_entry, result_grid_shape) if len(q) == 0 else (a_ge, a_gs)
        )
        c_oid = cm.bop_reduce(
            op_name,
            a_oid,
            b_oid,
            a_T,
            b_T,
            syskwargs={
                "grid_entry": ge,
                "grid_shape": gs,
            },
        )
        q.append((c_oid, ge, gs, False))
    r_oid, r_ge, r_gs, _ = q.pop(0)
    assert r_ge == result_grid_entry
    assert r_gs == result_grid_shape
    return r_oid


def get_compatible_tensordot_operands(cm, left, right, axes: int):
    from nums.core.array.blockarray import BlockArray

    def right_from_left_ba(left, right):
        # Use this when left.size > right.size.
        if not isinstance(left, BlockArray):
            # Make left a block array if it's not already.
            left = BlockArray.to_block_array(left, cm)

        if isinstance(right, BlockArray):
            if left.block_shape[-axes:] == right.block_shape[:axes]:
                return left, right
            else:
                compatible_right_block_shape = tuple(left.block_shape[-axes:] + right.block_shape[axes:])
                return left, right.reshape(block_shape=compatible_right_block_shape)

        assert isinstance(right, np.ndarray)
        # Other is the right operand of a tensordot over k axes.
        # Set the first k dims of the right operand's block shape
        # equal to the last k dims of the left operand's block shape.
        # Set the dim of the rest of the axes to the corresponding dim of the right operand's shape.
        # This last step is a conservative solution to the partitioning choices we have for the
        # remaining dims.
        compatible_right_block_shape = tuple(left.block_shape[-axes:] + right.shape[axes:])
        return left, BlockArray.to_block_array(right, cm,
                                               block_shape=compatible_right_block_shape)

    def left_from_right_ba(left, right):
        # Use this when left.size <= right.size.
        if not isinstance(right, BlockArray):
            # Make right a block array if it's not already.
            right = BlockArray.to_block_array(right, cm)

        if isinstance(left, BlockArray):
            if left.block_shape[-axes:] == right.block_shape[:axes]:
                return left, right
            else:
                compatible_left_block_shape = tuple(left.block_shape[:-axes] + right.block_shape[:axes])
                return left.reshape(block_shape=compatible_left_block_shape), right

        assert isinstance(left, np.ndarray)
        compatible_left_block_shape = tuple(left.shape[:-axes] + right.block_shape[:axes])
        return BlockArray.to_block_array(left, cm, block_shape=compatible_left_block_shape), \
               right

    left = atleast_nd_array(left)
    right = atleast_nd_array(right)
    # Make sure we have enough dims for tensordot.
    assert axes <= len(left.shape)
    assert axes <= len(right.shape)

    if array_utils.np_tensordot_param_test(
            left.shape, left.ndim, right.shape, right.ndim, axes
    ):
        raise ValueError("shape-mismatch for tensordot.")

    # Four possibilities:
    # 1. Left is blockarray, right is ndarray.
    # 2. Left is ndarray, right is blockarray.
    # 3. Both left and right are ndarray.
    # 4. Both left and right are blockarray.
    if left.size > right.size:
        # Covers all 4 cases when we prefer to repartition right array based on left array.
        return right_from_left_ba(left, right)
    else:
        # Covers all 4 cases when we prefer to repartition left array based on right array.
        return left_from_right_ba(left, right)


def _compute_tensordot_syskwargs(self_block: Block, other_block: Block):
    # Schedule on larger block.
    if np.product(self_block.shape) >= np.product(other_block.shape):
        return self_block.true_grid_entry(), self_block.true_grid_shape()
    else:
        return other_block.true_grid_entry(), other_block.true_grid_shape()


def tensordot(cm, left, right, axes=2):
    if isinstance(axes, int):
        pass
    elif array_utils.is_array_like(axes):
        raise NotImplementedError("Non-integer axes is currently not supported.")
    else:
        raise TypeError(f"Unexpected axes type '{type(axes).__name__}'")

    left, right = get_compatible_tensordot_operands(cm, left, right, axes=axes)

    this_axes = left.grid.grid_shape[:-axes]
    this_sum_axes = left.grid.grid_shape[-axes:]
    other_axes = right.grid.grid_shape[axes:]
    other_sum_axes = right.grid.grid_shape[:axes]
    assert this_sum_axes == other_sum_axes
    result_shape = tuple(left.shape[:-axes] + right.shape[axes:])
    result_block_shape = tuple(left.block_shape[:-axes] + right.block_shape[axes:])
    result_grid = ArrayGrid(
        shape=result_shape,
        block_shape=result_block_shape,
        dtype=array_utils.get_bop_output_type(
            "tensordot", left.dtype, right.dtype
        ).__name__,
    )
    assert result_grid.grid_shape == tuple(this_axes + other_axes)
    result = BlockArray(result_grid, left.cm)
    this_dims = list(itertools.product(*map(range, this_axes)))
    other_dims = list(itertools.product(*map(range, other_axes)))
    sum_dims = list(itertools.product(*map(range, this_sum_axes)))
    for i in this_dims:
        for j in other_dims:
            grid_entry = tuple(i + j)
            result_block: Block = result.blocks[grid_entry]
            sum_oids = []
            for k in sum_dims:
                self_block: Block = left.blocks[tuple(i + k)]
                other_block: Block = right.blocks[tuple(k + j)]
                dot_grid_args = left._compute_tensordot_syskwargs(
                    self_block, other_block
                )
                dotted_oid = cm.bop(
                    "tensordot",
                    self_block.oid,
                    other_block.oid,
                    self_block.transposed,
                    other_block.transposed,
                    axes=axes,
                    syskwargs={
                        "grid_entry": dot_grid_args[0],
                        "grid_shape": dot_grid_args[1],
                    },
                )
                sum_oids.append(
                    (dotted_oid, dot_grid_args[0], dot_grid_args[1], False)
                )
            result_block.oid = tree_reduce(cm,
                                           "sum",
                                           sum_oids,
                                           result_block.grid_entry,
                                           result_block.grid_shape)
    return result


def get_compatible_elementwise_operands(left, right, cm):

    def truncated_block_shape(block_shape_a, shape_b):
        # Creates block shape for b based on b's available dims.
        # Example: If a has shape NxMxK and b has shape K, set b.block_shape = a.block_shape[:-1].
        assert len(block_shape_a) >= len(shape_b)
        block_shape_b = []
        for i in range(-1, len(shape_b) - 1, -1):
            block_shape_b.append(block_shape_a[i])
        return block_shape_b

    def broadcast_a_to_b(a, b):
        if not isinstance(a, BlockArray):
            # Make a a block array if it's not already.
            a = BlockArray.to_block_array(a, cm)

        if len(a.shape) == len(b.shape):
            right_block_shape = a.block_shape
        else:
            right_block_shape = truncated_block_shape(a.block_shape, b.shape)

        if isinstance(b, BlockArray):
            if a.block_shape == b.block_shape:
                return a, b
            else:
                return a, b.reshape(block_shape=right_block_shape)
        else:
            return a, BlockArray.to_block_array(b, a.cm, block_shape=right_block_shape)

    left = atleast_nd_array(left)
    right = atleast_nd_array(right)
    assert array_utils.can_broadcast_shapes(left.shape, right.shape)

    # Handle cases where ndims don't match.
    if left.ndims > right.ndims:
        left, right = broadcast_a_to_b(left, right)
    else:
        right, left = broadcast_a_to_b(right, left)
    return left, right


def fast_element_wise(op_name, left, right):
    """
    Implements fast scheduling for basic element-wise operations.
    """
    dtype = array_utils.get_bop_output_type(op_name, left.dtype, right.dtype)
    # Schedule the op first.
    blocks = np.empty(shape=left.grid.grid_shape, dtype=Block)
    for grid_entry in left.grid.get_entry_iterator():
        self_block: Block = left.blocks[grid_entry]
        other_block: Block = right.blocks[grid_entry]
        blocks[grid_entry] = block = Block(
            grid_entry=grid_entry,
            grid_shape=self_block.grid_shape,
            rect=self_block.rect,
            shape=self_block.shape,
            dtype=dtype,
            transposed=False,
            cm=left.cm,
        )
        block.oid = left.cm.bop(
            op_name,
            self_block.oid,
            other_block.oid,
            self_block.transposed,
            other_block.transposed,
            axes={},
            syskwargs={
                "grid_entry": grid_entry,
                "grid_shape": left.grid.grid_shape,
            },
        )
    return BlockArray(
        ArrayGrid(left.shape, left.block_shape, dtype.__name__),
        left.cm,
        blocks=blocks,
    )


def elementwise(op_name, left, right, cm):
    left, right = get_compatible_elementwise_operands(left, right, cm)
    if left.shape == right.shape and left.block_shape == right.block_shape:
        return fast_element_wise(op_name, left, right)
    blocks_op = left.blocks.__getattribute__("__%s__" % op_name)
    return BlockArray.from_blocks(
        blocks_op(right.blocks), result_shape=None, cm=left.cm
    )


def inequality(op, left, right, cm):
    left, right = get_compatible_elementwise_operands(left, right, cm)
    right = left.convert_other_elementwise(right)
    assert (
            right.shape == () or right.shape == left.shape
    ), "Currently supports comparison with scalars only."
    shape = array_utils.broadcast(left.shape, right.shape).shape
    block_shape = array_utils.broadcast_block_shape(
        left.shape, right.shape, left.block_shape
    )
    dtype = bool.__name__
    grid = ArrayGrid(shape, block_shape, dtype)
    result = BlockArray(grid, cm)
    for grid_entry in result.grid.get_entry_iterator():
        if right.shape == ():
            other_block: Block = right.blocks.item()
        else:
            other_block: Block = right.blocks[grid_entry]
        result.blocks[grid_entry] = left.blocks[grid_entry].bop(
            op, other_block, args={}
        )

    return result
