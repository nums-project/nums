import zarr
import fsspec

from nums.core.grid.grid import ArrayGrid
from nums.core.array.blockarray import BlockArray
from nums.core.kernel.kernel_manager import KernelManager
from nums.core.backends.filesystem import FileSystem


class ZarrGroup:
    def __init__(self, url: str, km: KernelManager, fs: FileSystem):
        self.km = km
        self.fs = fs
        self.url = url
        self.zarr_group = zarr.open_consolidated(fsspec.get_mapper(url), mode="r")

    def __getattr__(self, item):
        return self.zarr_group.__getattribute__(item)

    def blockarray(self, field_name):
        self.zarr_arr = self.zarr_group[field_name]
        self.grid = ArrayGrid(
            self.zarr_arr.shape,
            block_shape=self.zarr_arr.chunks,
            dtype=str(self.zarr_arr.dtype),
        )
        ba = BlockArray(self.grid, self.km)
        for grid_entry in self.grid.get_entry_iterator():
            ba.blocks[grid_entry].oid = self.fs.read_block_zarr(
                self.url,
                field_name,
                grid_entry,
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": self.grid.grid_shape,
                },
            )
        return ba
