from nums.core.array.application import BlockArray


def check_block_integrity(arr: BlockArray):
    for grid_entry in arr.grid.get_entry_iterator():
        assert arr.blocks[grid_entry].grid_entry == grid_entry
        assert arr.blocks[grid_entry].grid_shape == arr.grid.grid_shape
        assert arr.blocks[grid_entry].rect == arr.grid.get_slice_tuples(grid_entry)
        assert arr.blocks[grid_entry].shape == arr.grid.get_block_shape(grid_entry)
        assert arr.blocks[grid_entry].dtype == arr.dtype
