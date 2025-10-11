# New cell: build xarray DataArray with (time, y, x) and lat/lon coords
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import xarray as xr

path = Path("/home/jam/git/radar/data/dk.com.202508081630.500_max.h5")


def open_h5(path: str | Path) -> xr.DataArray:
    """
    Open a DMI radar HDF5 file and return an xarray.DataArray with (time, y, x) dims and lat/lon coordinates.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    xr.DataArray
        DataArray containing the radar data, with time, y, x dimensions and lat/lon coordinates.
    """
    with h5py.File(path, "r") as f:
        # Read raw data array
        data: np.ndarray = f["dataset1"]["data1"]["data"][...]  # (ny, nx) uint8

        # Get quantity name
        quantity: str = f["dataset1"]["data1"].attrs.get("quantity", b"var")
        if isinstance(quantity, bytes):
            quantity = quantity.decode()

        # Read metadata
        what: dict = f["what"].attrs
        where: dict = f["where"].attrs

        # Helper to decode bytes to str
        def _val(a: bytes | str) -> str:
            return a.decode() if isinstance(a, bytes | bytearray) else a

        # Parse timestamp from metadata
        date_str: str = _val(what["date"])  # 'YYYYMMDD'
        time_str: str = _val(what["time"])  # 'HHMMSS'
        ts: datetime = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # noqa: DTZ007

        # Read scaling and mask info
        gain: float = float(what.get("gain", 1.0))
        offset: float = float(what.get("offset", 0.0))
        nodata = what.get("nodata")
        undetect = what.get("undetect")

        # Scale data
        scaled: np.ndarray = data.astype("float32") * gain + offset

        # Build mask for nodata/undetect values
        mask: np.ndarray = np.zeros_like(scaled, dtype=bool)
        if nodata is not None:
            mask |= data == nodata
        if undetect is not None:
            mask |= data == undetect
        scaled = np.ma.masked_array(scaled, mask=mask)

        ny, nx = data.shape

        # Read corner coordinates for bilinear interpolation
        ul_lat, ul_lon = float(where["UL_lat"]), float(where["UL_lon"])
        ur_lat, ur_lon = float(where["UR_lat"]), float(where["UR_lon"])
        ll_lat, ll_lon = float(where["LL_lat"]), float(where["LL_lon"])
        lr_lat, lr_lon = float(where["LR_lat"]), float(where["LR_lon"])

        # Interpolate lat/lon for each pixel
        wx = np.linspace(0.0, 1.0, nx)  # left->right
        wy = np.linspace(0.0, 1.0, ny)  # top->bottom

        top_lat = ul_lat + wx * (ur_lat - ul_lat)
        top_lon = ul_lon + wx * (ur_lon - ul_lon)
        bottom_lat = ll_lat + wx * (lr_lat - ll_lat)
        bottom_lon = ll_lon + wx * (lr_lon - ll_lon)

        lat2d = (top_lat * (1 - wy[:, None])) + (bottom_lat * wy[:, None])
        lon2d = (top_lon * (1 - wy[:, None])) + (bottom_lon * wy[:, None])

        # Build DataArray with time, y, x dims and lat/lon coords
        da = xr.DataArray(
            scaled[np.newaxis, ...],  # add time dim
            dims=("time", "y", "x"),
            coords=dict(
                time=[np.datetime64(ts)],
                lat=(("y", "x"), lat2d),
                lon=(("y", "x"), lon2d),
            ),
            name=quantity,
            attrs={
                "gain": gain,
                "offset": offset,
                "nodata": nodata,
                "undetect": undetect,
                "projection": _val(where.get("projdef", "")),
                "description": quantity,
                "source": _val(what.get("source", "")),
                "product": _val(what.get("product", "")),
            },
        )

    return da


def select_region(
    da: xr.DataArray | xr.Dataset,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> xr.DataArray | xr.Dataset | None:
    """Select a geographic region from 2D coordinate data"""
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Create mask
    mask = (
        (da.lat >= lat_min)
        & (da.lat <= lat_max)
        & (da.lon >= lon_min)
        & (da.lon <= lon_max)
    )

    # Find bounding box
    y_idx, x_idx = np.where(mask)

    if len(y_idx) == 0:
        return None

    # Get min/max indices with some padding
    y_slice = slice(max(0, y_idx.min() - 1), min(da.sizes["y"], y_idx.max() + 2))
    x_slice = slice(max(0, x_idx.min() - 1), min(da.sizes["x"], x_idx.max() + 2))

    return da.isel(y=y_slice, x=x_slice)


class GetH5:
    da: Iterable[xr.DataArray] | xr.DataArray

    def __init__(self, path: Iterable[str | Path] | str | Path) -> None:
        if isinstance(path, Iterable):
            self.da = []
            for p in path:
                self.da.append(open_h5(p))
        else:
            self.da = open_h5(path)

    def set_region(
        self, lat_range: tuple[float, float], lon_range: tuple[float, float]
    ) -> None:
        if isinstance(self.da, list):
            for i, d in enumerate(self.da):
                self.da[i] = select_region(d, lon_range=lon_range, lat_range=lat_range)
        else:
            self.da = select_region(self.da, lon_range=lon_range, lat_range=lat_range)

    def get_tensor(self) -> torch.Tensor:
        if isinstance(self.da, list):
            return torch.concat([torch.from_numpy(d.values) for d in self.da], dim=0)
        else:
            return torch.from_numpy(self.da.values)
