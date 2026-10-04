"""Download the pinned Satlas Sentinel-2 RGB checkpoint outside git."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/allenai/satlas-pretrain/resolve/main/"
    "sentinel2_swinb_si_rgb.pth?download=true"
)
DEFAULT_OUTPUT = Path("models/satlas/sentinel2_swinb_si_rgb.pth")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, args.output)
    else:
        print(f"Using existing {args.output}")

    print(f"SHA256 {sha256(args.output)}")
    print(
        "Copy this SHA256 into the M3b issue/results before treating the "
        "checkpoint as fully pinned."
    )


if __name__ == "__main__":
    main()
