import zarr
import fsspec

from nums.core.grid.grid import ArrayGrid
from nums.core.array.blockarray import BlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.filesystem import FileSystem


class ZarrGroup:
    def __init__(
        self,
        url: str,
        km: KernelManager,
        fs: FileSystem,
        consolidated: bool = True,
        mode: str = "r",
        **kwargs
    ):
        self.km = km
        self.fs = fs
        self.url = url
        assert consolidated, "Only consolidated mode is currently supported."
        assert mode == "r", "Only read mode is currently supported."
        assert "mode" not in kwargs
        self.consolidated = consolidated
        self.kwargs = kwargs.copy()
        self.kwargs["mode"] = mode

        self.zarr_group = zarr.open_consolidated(fsspec.get_mapper(url), **kwargs)

    def __getattr__(self, item):
        return self.zarr_group.__getattribute__(item)

    def blockarray(self, field_name):
        zarr_arr = self.zarr_group[field_name]
        grid = ArrayGrid(
            zarr_arr.shape,
            block_shape=zarr_arr.chunks,
            dtype=str(zarr_arr.dtype),
        )
        ba = BlockArray(grid, self.km)
        for grid_entry in grid.get_entry_iterator():
            ba.blocks[grid_entry].oid = self.fs.read_block_zarr(
                self.url,
                field_name,
                self.consolidated,
                self.kwargs,
                grid_entry,
                grid.get_block_shape(grid_entry),
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": grid.grid_shape,
                },
            )
        return ba
