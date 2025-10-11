import zarr
import numpy as np
from pathlib import Path


class ZarrArchiveManager:
    def __init__(self, meta, archive_path):
        """
        meta: dict with keys 'shape', 'dtype', 'chunks', 'fill_value' (optional)
        archive_path: path to zarr archive
        """
        self.meta = meta
        self.archive_path = Path(archive_path)
        self.zarr_array = self._init_or_expand_archive()

    def _init_or_expand_archive(self):
        shape = tuple(self.meta["shape"])
        dtype = self.meta["dtype"]
        chunks = self.meta.get("chunks", None)
        fill_value = self.meta.get("fill_value", np.nan)

        if not self.archive_path.exists():
            # Create new archive
            zarr_array = zarr.open(
                self.archive_path.as_posix(),
                mode="w",
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                fill_value=fill_value,
            )
        else:
            # Open existing archive
            zarr_array = zarr.open(self.archive_path.as_posix(), mode="a")
            # Expand if needed
            new_shape = []
            expand = False
            for i, dim in enumerate(shape):
                if dim > zarr_array.shape[i]:
                    new_shape.append(dim)
                    expand = True
                else:
                    new_shape.append(zarr_array.shape[i])
            if expand:
                zarr_array.resize(tuple(new_shape))
                # Fill new region with nans
                slices = tuple(
                    slice(
                        zarr_array.shape[i] - (shape[i] - zarr_array.shape[i]),
                        zarr_array.shape[i],
                    )
                    if shape[i] > zarr_array.shape[i]
                    else slice(0, 0)
                    for i in range(len(shape))
                )
                if any(s.stop > s.start for s in slices):
                    zarr_array[slices] = fill_value
        return zarr_array

    def write(self, data, index):
        """
        Write data to the archive at the given index.
        data: numpy array matching the shape of the slice
        index: tuple of indices/slices
        """
        self.zarr_array[index] = data


# Example usage:
# meta = {'shape': (100, 200), 'dtype': 'float32', 'chunks': (10, 10)}
# manager = ZarrArchiveManager(meta, '/tmp/myarchive.zarr')
# manager.write(np.ones((10, 20)), (slice(0, 10), slice(0, 20)))

import xarray as xr
import numpy as np
from pathlib import Path


class XarrayZarrManager:
    def __init__(self, template, archive_path):
        """
        template: xarray.DataArray or xarray.Dataset with desired shape, coords, and dtype
        archive_path: path to zarr archive
        """
        self.archive_path = Path(archive_path)
        self.template = template

        if not self.archive_path.exists():
            # Create new archive from template
            self.template.to_zarr(self.archive_path, mode="w")
            self.ds = xr.open_zarr(self.archive_path, mode="a")
        else:
            # Open existing archive
            self.ds = xr.open_zarr(self.archive_path, mode="a")
            # Expand if needed
            self._expand_if_needed()

    def _expand_if_needed(self):
        # Only supports expanding along dimensions present in template
        for dim in self.template.dims:
            tmpl_len = self.template.sizes[dim]
            ds_len = self.ds.sizes[dim]
            if tmpl_len > ds_len:
                # Expand dimension and fill with NaNs
                new_coords = list(self.ds.coords[dim].values) + list(
                    self.template.coords[dim].values[ds_len:]
                )
                self.ds = self.ds.reindex({dim: new_coords}, fill_value=np.nan)
                self.ds.to_zarr(self.archive_path, mode="w", overwrite=True)
                self.ds = xr.open_zarr(self.archive_path, mode="a")

    def write(self, data, indexers):
        """
        Write data to the archive at the given indexers.
        data: xarray.DataArray or numpy array
        indexers: dict of {dim: index/slice}
        """
        # Select the region and assign
        self.ds.loc[indexers] = data
        self.ds.to_zarr(self.archive_path, mode="a")


# Example usage:
# template = xr.DataArray(
#     np.full((100, 200), np.nan, dtype="float32"),
#     dims=("time", "y"),
#     coords={"time": np.arange(100), "y": np.arange(200)},
#     name="myvar"
# )
# manager = XarrayZarrManager(template, "/tmp/myarchive.zarr")
# manager.write(np.ones((10, 20)), {"time": slice(0, 10), "y": slice(0, 20)})
