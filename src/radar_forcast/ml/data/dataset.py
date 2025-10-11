from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from radar_forcast.data import GetH5

TRAIN = [
    pd.Timestamp("2025-08-01T00:025:00", tz="UTC"),
    pd.Timestamp("2025-09-13T23:59:00", tz="UTC"),
]
VAL = [
    pd.Timestamp("2025-09-13T23:59:00", tz="UTC"),
    pd.Timestamp("2025-09-21T00:025:00", tz="UTC"),
]
TEST = [
    pd.Timestamp("2025-09-21T00:025:00", tz="UTC"),
    pd.Timestamp("2025-09-30T23:59:00", tz="UTC"),
]
DEBUG = [
    pd.Timestamp("2025-08-21T00:025:00", tz="UTC"),
    pd.Timestamp("2025-08-22T23:59:00", tz="UTC"),
]

REGION_AA = {"lat_range": (56.4, 57.4), "lon_range": (9.4, 10.4)}

RADAR_FILE_NAME = "{year}/dk_{time}.h5"


class RadarSequenceDataset(Dataset):
    """
    PyTorch dataset for radar forecasting with temporal sequences.

    Uses two consecutive timesteps (k-1, k) as input to predict the next timestep (k+1).
    """

    def __init__(
        self,
        data_dir: str,
        region: Literal["aa", "dk"] = "aa",
        time_slice: Literal["train", "val", "test", "debug"] = "debug",
        root_path: str | Path = Path("/media/jam/HDD/dmi/radar"),
        # transform: Callable | None = None,
        # target_transform: Callable | None = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing timestamped data files
            file_pattern: Pattern to match data files (e.g., "*.npy", "*.pt")
            transform: Optional transform to apply to input sequences
            target_transform: Optional transform to apply to target
        """
        self.data_dir = data_dir
        self.region = region
        self.time_slice = time_slice
        self.root_path = Path(root_path)
        # self.transform = transform
        # self.target_transform = target_transform

        match self.time_slice:
            case "train":
                self.start_time = TRAIN[0]
                self.end_time = TRAIN[1]
            case "val":
                self.start_time = VAL[0]
                self.end_time = VAL[1]
            case "test":
                self.start_time = TEST[0]
                self.end_time = TEST[1]

        self.total_intervals = self._get_5min_intervals(self.start_time, self.end_time)

    def _get_5min_intervals(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Calculate number of 5-minute intervals between two timestamps."""
        time_diff = end - start
        total_minutes = time_diff.total_seconds() / 60
        return int(total_minutes / 5)

    def __len__(self) -> int:
        # We can create sequences starting from index 1 up to len-2
        # (need k-1, k, k+1)
        return self.total_intervals - 2

    def idx_to_date(self, idx: int) -> pd.Timestamp:
        return self.start_time + (idx * pd.Timedelta(5, "m"))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_sequence: Tensor of shape [2, ...] containing timesteps k-1 and k
            target: Tensor containing timestep k+1
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        idx_date = self.idx_to_date(idx)
        idx_date_m1 = idx_date - pd.Timedelta(5, "m")
        target_date = idx_date + pd.Timedelta(5, "m")
        input_files = [
            self.root_path
            / RADAR_FILE_NAME.format(idx_date.year, idx_date.strftime("%Y%m%d%H%M%S")),
            self.root_path
            / RADAR_FILE_NAME.format(
                idx_date_m1.year, idx_date_m1.strftime("%Y%m%d%H%M%S")
            ),
        ]
        target_file = self.root_path / RADAR_FILE_NAME.format(
            target_date.year, target_date.strftime("%Y%m%d%H%M%S")
        )
        input = GetH5(input_files)
        target = GetH5(target_file)

        if self.region == "aa":
            input.set_region(REGION_AA)
            target.set_region(REGION_AA)

        # Load three consecutive timesteps
        # timestep_k_minus_1 = self._load_file(self.file_paths[idx])  # k-1
        # timestep_k = self._load_file(self.file_paths[idx + 1])  # k
        # timestep_k_plus_1 = self._load_file(self.file_paths[idx + 2])  # k+1 (target)

        # Stack input sequence [k-1, k]

        return input.get_tensor(), target.get_tensor()

    def _load_file(self, file_path: str | Path) -> torch.Tensor:
        """Load a single data file and convert to torch tensor."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext == ".npy":
            data = np.load(file_path)
            return torch.from_numpy(data).float()
        elif file_ext == ".pt":
            return torch.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def get_file_info(self, idx: int) -> tuple[str, str, str]:
        """Get the file paths for a given sequence index."""
        return (
            self.file_paths[idx],  # k-1
            self.file_paths[idx + 1],  # k
            self.file_paths[idx + 2],  # k+1
        )
