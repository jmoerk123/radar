import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def request_downloade(
    url: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
