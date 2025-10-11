import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from requests import HTTPError

from radar_forcast.data import request_downloade

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(
    api_key: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    interval_min: int,
) -> None:
    interval = pd.Timedelta(interval_min, "m")
    root_save_path = Path("/media/jam/HDD/dmi/radar")
    partial_url = "https://dmigw.govcloud.dk/v1/radardata/download/dk.com.{date_time}.500_max.h5?api-key={api_key}"
    time = start_time
    while time < end_time:
        time_str = time.strftime("%Y%m%d%H%M")
        url = partial_url.format(date_time=time_str, api_key=api_key)
        (root_save_path / f"{time.year}").mkdir(parents=True, exist_ok=True)
        out_path = root_save_path / f"{time.year}/dk_{time_str}.h5"
        try:
            if not out_path.exists():
                request_downloade(url, out_path=out_path)
                logger.info(f"Downloaded to {out_path}")
            else:
                logger.info(f"File already exist: {out_path}")
        except HTTPError:
            logger.warning(f"Failed to load: {time_str}")
        time += interval


if __name__ == "__main__":
    parser = ArgumentParser(description="Download DMI radar data")
    parser.add_argument(
        "-st",
        "--start_time",
        default=None,
        help="Time to downloade firsr file, must match a 5 min intervalif None expected to be in config.yaml.",
    )
    parser.add_argument(
        "-et",
        "--end_time",
        default=None,
        help="Time to downloade last fileif None expected to be in config.yaml.",
    )
    parser.add_argument(
        "-rsp",
        "--root_save_path",
        default=None,
        help="Path to folder containing year folderif None expected to be in config.yaml.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="DMI api key, if None expected to be in config.yaml",
    )
    parser.add_argument(
        "-cp",
        "--config_path",
        default="../config.yaml",
        help="Path to config.yaml containing API_KEY",
    )
    args = parser.parse_args()

    with Path(args.config_path).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    for arg_k in vars(args):
        arg_v = getattr(args, arg_k)
        if arg_v is not None:
            cfg[arg_k] = arg_v

    main(
        api_key=cfg["api_key"],
        start_time=cfg["start_time"],
        end_time=cfg["end_time"],
        interval_min=cfg["interval_min"],
    )
