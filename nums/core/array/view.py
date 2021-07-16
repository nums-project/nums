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


from typing import Tuple

import numpy as np

from nums.core.array import selection
from nums.core.array import utils as array_utils
from nums.core.array.base import Block, BlockArrayBase
from nums.core.array.selection import BasicSelection
from nums.core.compute.compute_manager import ComputeManager
from nums.core.grid.grid import ArrayGrid


class ArrayView(object):
    @classmethod
    def from_block_array(cls, bab):
        assert isinstance(bab, BlockArrayBase)
        return cls(source=bab, block_shape=bab.block_shape)

    @classmethod
    def from_subscript(cls, bab, subscript):
        assert isinstance(bab, BlockArrayBase)
        return cls(source=bab, sel=BasicSelection.from_subscript(bab.shape, subscript))

    def __init__(self, source, sel: BasicSelection = None, block_shape: tuple = None):
        self._source: BlockArrayBase = source
        self._cm: ComputeManager = self._source.cm

        if sel is None:
            sel = BasicSelection.from_shape(self._source.shape)
        # Currently, this is all we support.
        assert len(sel.axes) == len(self._source.shape)
        self.sel = sel

        self.shape: tuple = self.sel.get_output_shape()
        if block_shape is None:
            block_shape: tuple = array_utils.block_shape_from_subscript(
                self.sel.selector(), self._source.block_shape
            )
        self.block_shape = block_shape
        assert len(self.block_shape) == len(self.shape)
        self.grid: ArrayGrid = ArrayGrid(
            self.shape, self.block_shape, dtype=self._source.dtype.__name__
        )

    def __getitem__(self, item):
        if isinstance(item, tuple):
            for val in item:
                assert isinstance(val, (slice, int, np.intp))
            return self.select(item)
        elif isinstance(item, (slice, int, np.intp)):
            return self.select((item,))
        else:
            raise Exception("getitem failed", item)

    def select(self, subscript: tuple):
        if selection.is_advanced_selection(subscript):
            # This is not optimized.
            return self.advanced_select(subscript)
        else:
            # This is optimized.
            return self.basic_select(subscript)

    def basic_select(self, subscript: tuple):
        # No support for subscripts of subscripts.
        # We create new block arrays to deal with nested subscripts.
        assert self.shape == self._source.shape
        assert self.block_shape == self._source.block_shape
        sel: BasicSelection = BasicSelection.from_subscript(self.shape, subscript)
        result: ArrayView = ArrayView(self._source, sel)
        return result

    def create(self, concrete_cls=None) -> BlockArrayBase:
        if self.sel.basic_steps():
            if self.sel.is_aligned(self._source.block_shape):
                # Assertion below should form a conjunction with the above condition.
                # This isn't currently an issue but an assumption that
                # may not always hold true, depending on how the ArrayView
                # is constructed.
                assert array_utils.can_broadcast_shape_to(
                    self.sel.get_broadcastable_block_shape(self.block_shape),
                    self._source.block_shape,
                )
                return self.create_references(concrete_cls)
            else:
                return self.create_basic_single_step(concrete_cls)
        else:
            return self.create_basic_multi_step(concrete_cls)

    def create_references(self, concrete_cls) -> BlockArrayBase:
        # TODO (hme): Double check this.
        array_cls = BlockArrayBase if concrete_cls is None else concrete_cls
        dst_ba: BlockArrayBase = array_cls(self.grid, self._cm)
        if 0 in self.shape:
            return dst_ba
        grid_offset = self.sel.position().value // np.array(
            self._source.block_shape, dtype=np.int
        )
        dst_inflated_shape = self.sel.get_broadcastable_shape()
        dst_inflated_block_shape = self.sel.get_broadcastable_block_shape(
            self.block_shape
        )
        dst_inflated_grid: ArrayGrid = ArrayGrid(
            dst_inflated_shape, dst_inflated_block_shape, self.grid.dtype.__name__
        )
        dst_grid_entry_iterator = list(dst_ba.grid.get_entry_iterator())
        for dst_index, dst_inflated_grid_entry in enumerate(
            dst_inflated_grid.get_entry_iterator()
        ):
            dst_grid_entry = dst_grid_entry_iterator[dst_index]
            src_grid_entry = tuple(
                (np.array(dst_inflated_grid_entry, dtype=np.int) + grid_offset).tolist()
            )
            dst_ba.blocks[dst_grid_entry].oid = self._source.blocks[src_grid_entry].oid
            dst_ba.blocks[dst_grid_entry].transposed = self._source.blocks[
                src_grid_entry
            ].transposed
        return dst_ba

    def create_basic_single_step(self, concrete_cls) -> BlockArrayBase:
        array_cls = BlockArrayBase if concrete_cls is None else concrete_cls
        dst_ba: BlockArrayBase = array_cls(self.grid, self._cm)
        if 0 in self.shape:
            return dst_ba

        src_sel_arr: np.ndarray = selection.BasicSelection.block_selection(
            self._source.shape, self._source.block_shape
        )
        # TODO(hme): The following op is very slow for integer subscripts of large arrays.
        src_sel_clipped: np.ndarray = src_sel_arr & self.sel
        assert src_sel_clipped.shape == self._source.grid.grid_shape

        broadcast_shape = self.sel.get_broadcastable_shape()
        broadcast_block_shape = self.sel.get_broadcastable_block_shape(
            dst_ba.block_shape
        )
        dst_grid_bc: ArrayGrid = ArrayGrid(
            broadcast_shape, broadcast_block_shape, self.grid.dtype.__name__
        )
        dst_sel_arr: np.ndarray = selection.BasicSelection.block_selection(
            broadcast_shape, broadcast_block_shape
        )
        dst_sel_offset: np.ndarray = dst_sel_arr + self.sel.position()
        dst_entry_iterator = list(dst_ba.grid.get_entry_iterator())
        for dst_index, dst_grid_entry_bc in enumerate(dst_grid_bc.get_entry_iterator()):
            dst_sel_offset_block: BasicSelection = dst_sel_offset[dst_grid_entry_bc]
            if dst_sel_offset_block.is_empty():
                continue
            src_dst_intersection_arr = src_sel_clipped & dst_sel_offset_block
            cm: ComputeManager = self._cm
            src_oids = []
            src_params = []
            dst_params = []
            for _, src_grid_entry in enumerate(self._source.grid.get_entry_iterator()):
                src_dst_intersection_block: BasicSelection = src_dst_intersection_arr[
                    src_grid_entry
                ]
                if src_dst_intersection_block.is_empty():
                    continue
                src_block: Block = self._source.blocks[src_grid_entry]
                src_oids.append(src_block.oid)
                src_sel_block: BasicSelection = src_sel_arr[src_grid_entry]
                src_dep_sel_loc = src_dst_intersection_block - src_sel_block.position()
                src_params.append((src_dep_sel_loc.selector(), src_block.transposed))
                dst_block_sel_loc = (
                    src_dst_intersection_block - dst_sel_offset_block.position()
                )
                dst_params.append((dst_block_sel_loc.selector(), False))
            dst_block: Block = dst_ba.blocks.reshape(dst_grid_bc.grid_shape)[
                dst_grid_entry_bc
            ]
            dst_block.oid = cm.create_block(
                *src_oids,
                src_params=src_params,
                dst_params=dst_params,
                dst_shape=dst_block.shape,
                dst_shape_bc=dst_sel_offset_block.get_output_shape(),
                syskwargs={
                    "grid_entry": dst_entry_iterator[dst_index],
                    "grid_shape": self.grid.grid_shape,
                }
            )
        return dst_ba

    def create_basic_multi_step(self, concrete_cls) -> BlockArrayBase:
        # Create each entry one by one.
        raise NotImplementedError("Positive steps of size 1 are currently supported.")

    def advanced_select(self, subscript: tuple):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            for entry in key:
                assert isinstance(entry, (slice, int, np.intp, type(...)))
            return self.assign(key, value)
        elif isinstance(key, (slice, int, np.intp, type(...))):
            return self.assign((key,), value)
        else:
            raise Exception("setitem failed", key)

    def assign(self, subscript: Tuple, value):
        if selection.is_advanced_selection(subscript):
            # This is not optimized.
            return self.advanced_assign(subscript, value)
        else:
            # This is optimized.
            return self.basic_assign(subscript, value)

    def basic_assign(self, subscript: tuple, value):
        # No support for subscripts of subscripts.
        # We create new block arrays to deal with nested subscripts.
        assert self.shape == self._source.shape
        assert self.block_shape == self._source.block_shape
        dst_ba: BlockArrayBase = self._source
        dst_sel = selection.BasicSelection.from_subscript(dst_ba.shape, subscript)
        if 0 in dst_sel.get_output_shape():
            # Nothing to do.
            return
        if dst_sel.basic_steps():
            value_is_aligned = True
            if isinstance(value, ArrayView):
                value_is_aligned = value.sel.is_aligned(value._source.block_shape)
            if (
                value_is_aligned
                and dst_sel.is_aligned(self._source.block_shape)
                and self.block_shape == value.block_shape
            ):
                # TODO (hme): Sometimes self.block_shape != value.block_shape
                #  when it is in fact equal. This happens when value
                #  is created from the last block of its source,
                #  and is being assigned to the last block of this view's source.
                #  This is a minor issue.
                return self.assign_references(dst_sel, value)
            else:
                return self.basic_assign_single_step(dst_sel, value)
        else:
            return self.basic_assign_multi_step(dst_sel, value)

    def assign_references(self, dst_sel: BasicSelection, value):
        # TODO (hme): This seems overly complicated, but correct. Double check it.
        #  Also, revisit some of the variable names. They will likely
        #  be confusing in the future.
        # The destination has same block shape as value,
        # but the destination selection may not have the same shape as value.
        # May need to broadcast value to destination selection output shape.
        dst_offset = dst_sel.position().value // np.array(
            self._source.block_shape, dtype=np.int
        )
        # Do we need to broadcast?
        if isinstance(value, ArrayView) and (
            dst_sel.get_output_shape() != value.sel.get_output_shape()
        ):
            value = value.create()
        if isinstance(value, ArrayView):
            # This is the best case.
            # We don't need to create value to perform the reference copy.
            # No broadcasting required, so this should be okay.
            src_offset = value.sel.position().value // np.array(
                value._source.block_shape, dtype=np.int
            )
            src_inflated_shape = dst_sel.get_broadcastable_shape()
            src_inflated_block_shape = dst_sel.get_broadcastable_block_shape(
                value.block_shape
            )
            src_inflated_grid: ArrayGrid = ArrayGrid(
                src_inflated_shape, src_inflated_block_shape, self.grid.dtype.__name__
            )
            for src_grid_entry_inflated in src_inflated_grid.get_entry_iterator():
                # Num axes in value grid may be too small.
                dst_grid_entry = tuple(
                    (
                        np.array(src_grid_entry_inflated, dtype=np.int) + dst_offset
                    ).tolist()
                )
                src_grid_entry = tuple(
                    (
                        np.array(src_grid_entry_inflated, dtype=np.int) + src_offset
                    ).tolist()
                )
                # This is a reference assignment, and the grid properties between the
                # two blocks may differ, so retain those properties in the copy.
                dst_block: Block = self._source.blocks[dst_grid_entry]
                src_block_copy: Block = value._source.blocks[src_grid_entry].copy()
                src_block_copy.grid_entry = dst_block.grid_entry
                src_block_copy.grid_shape = dst_block.grid_shape
                src_block_copy.rect = dst_block.rect
                self._source.blocks[dst_grid_entry] = src_block_copy
        elif isinstance(value, BlockArrayBase):
            # The value has already been created, so just leverage value's existing grid iterator.
            if value.shape != dst_sel.get_output_shape():
                # Need to broadcast.
                src_ba: BlockArrayBase = value.broadcast_to(dst_sel.get_output_shape())
            else:
                src_ba: BlockArrayBase = value
            src_inflated_shape = dst_sel.get_broadcastable_shape()
            src_inflated_block_shape = dst_sel.get_broadcastable_block_shape(
                src_ba.block_shape
            )
            src_inflated_grid: ArrayGrid = ArrayGrid(
                src_inflated_shape, src_inflated_block_shape, self.grid.dtype.__name__
            )
            src_grid_entry_iterator = list(src_ba.grid.get_entry_iterator())
            for src_index, src_grid_entry_inflated in enumerate(
                src_inflated_grid.get_entry_iterator()
            ):
                src_grid_entry = src_grid_entry_iterator[src_index]
                dst_grid_entry = tuple(
                    (
                        np.array(src_grid_entry_inflated, dtype=np.int) + dst_offset
                    ).tolist()
                )
                # This is a reference assignment, and the grid properties between the
                # two blocks may differ, so retain those properties in the copy.
                dst_block: Block = self._source.blocks[dst_grid_entry]
                src_block_copy: Block = src_ba.blocks[src_grid_entry].copy()
                src_block_copy.grid_entry = dst_block.grid_entry
                src_block_copy.grid_shape = dst_block.grid_shape
                src_block_copy.rect = dst_block.rect
                self._source.blocks[dst_grid_entry] = src_block_copy

    def basic_assign_single_step(self, dst_sel: BasicSelection, value):
        assert isinstance(value, (ArrayView, BlockArrayBase))

        dst_ba: BlockArrayBase = self._source
        dst_sel_arr: np.ndarray = selection.BasicSelection.block_selection(
            dst_ba.shape, dst_ba.block_shape
        )

        dst_sel_clipped: np.ndarray = dst_sel_arr & dst_sel
        assert dst_sel_clipped.shape == self._source.grid.grid_shape

        # We create value's block array, in case we need to broadcast.
        # This may not be necessary, but alternative solutions are extremely tedious.
        # The result is a block array with replicated blocks,
        # which match the output shape of dst_sel.
        if isinstance(value, ArrayView):
            src_ba_bc: BlockArrayBase = value.create().broadcast_to(
                dst_sel.get_output_shape()
            )
        elif isinstance(value, BlockArrayBase):
            src_ba_bc: BlockArrayBase = value.broadcast_to(dst_sel.get_output_shape())
        else:
            raise Exception("Unexpected value type %s." % type(value))
        # Different lengths occur when an index is used to perform
        # a selection on an axis. Numpy semantics drops such axes. To allow operations
        # between source and destination selections, dropped axes are restored with dimension 1
        # so that selections are of equal length.
        # We restore the dropped dimensions of the destination selection, because
        # the source selection must be broadcastable to the destination selection
        # for the assignment to be valid.
        src_inflated_shape = dst_sel.get_broadcastable_shape()
        # The block shapes need not be equal, but the broadcast source block shape must
        # match the block shape we obtain below, so that there's a 1-to-1 correspondence
        # between the grid entries.
        src_inflated_block_shape = dst_sel.get_broadcastable_block_shape(
            src_ba_bc.block_shape
        )
        src_inflated_grid: ArrayGrid = ArrayGrid(
            src_inflated_shape, src_inflated_block_shape, self.grid.dtype.__name__
        )
        src_sel_arr: np.ndarray = selection.BasicSelection.block_selection(
            src_inflated_shape, src_inflated_block_shape
        )
        src_sel_offset: np.ndarray = src_sel_arr + dst_sel.position()
        # The enumeration of grid entries is identical if the broadcast source grid and
        # inflated grid have the same number of blocks.
        src_grid_entry_iterator = list(src_ba_bc.grid.get_entry_iterator())
        for dst_grid_entry in dst_ba.grid.get_entry_iterator():
            dst_sel_block: BasicSelection = dst_sel_arr[dst_grid_entry]
            dst_sel_block_clipped: BasicSelection = dst_sel_clipped[dst_grid_entry]
            if dst_sel_block_clipped.is_empty():
                continue
            src_intersection_arr = src_sel_offset & dst_sel_block_clipped
            src_oids = []
            src_params = []
            dst_params = []
            dst_block: Block = dst_ba.blocks[dst_grid_entry]
            for src_index, src_grid_entry_bc in enumerate(
                src_inflated_grid.get_entry_iterator()
            ):
                src_intersection_block: BasicSelection = src_intersection_arr[
                    src_grid_entry_bc
                ]
                if src_intersection_block.is_empty():
                    continue

                src_grid_entry = src_grid_entry_iterator[src_index]
                src_block: Block = src_ba_bc.blocks[src_grid_entry]
                src_oids.append(src_block.oid)

                src_sel_block_offset: BasicSelection = src_sel_offset[src_grid_entry_bc]
                src_dep_sel_loc = (
                    src_intersection_block - src_sel_block_offset.position()
                )
                src_params.append(
                    (
                        src_dep_sel_loc.selector(),
                        src_sel_block_offset.get_output_shape(),
                        src_block.transposed,
                    )
                )
                # We're looking at intersection of dst block and src block, so the
                # location to which we assign must be offset by dst_sel_block.
                dst_block_sel_loc: BasicSelection = (
                    src_intersection_block - dst_sel_block.position()
                )
                dst_params.append((dst_block_sel_loc.selector(), dst_block.transposed))
            if len(src_oids) == 0:
                continue
            dst_block.oid = self._cm.update_block(
                dst_block.oid,
                *src_oids,
                src_params=src_params,
                dst_params=dst_params,
                syskwargs={
                    "grid_entry": dst_block.grid_entry,
                    "grid_shape": dst_block.grid_shape,
                }
            )

    def basic_assign_multi_step(self, dst_sel: BasicSelection, value):
        # Update each entry in subscript shape, one by one.
        raise NotImplementedError()

    def advanced_assign(self, subscript: tuple, value):
        raise NotImplementedError()
