"""
Download the Alibaba PAI GPU trace (cluster-trace-gpu-v2020) task table.

The archive is large (hundreds of MB), so it isn't committed. After running
this, pass ``--pai data/pai_task_table.csv`` to run_trace_experiments.py.

    python scripts/fetch_traces.py
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz"
DEST = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DEST.mkdir(exist_ok=True)
    archive = DEST / "pai_task_table.tar.gz"
    if not archive.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DEST, filter="data")
    print(f"Extracted to {DEST}. The CSV has no header row in some mirrors; see the trace README for column order.")


if __name__ == "__main__":
    main()
