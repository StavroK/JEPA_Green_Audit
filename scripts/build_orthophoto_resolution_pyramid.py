"""Create coarser RGB versions of a georeferenced 1 m orthophoto.

The same image/date/geography is resampled to 2 m, 5 m, and 10 m using area
averaging. This isolates spatial resolution from temporal change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

DEFAULT_SOURCE = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.tif")
DEFAULT_DIR = Path("data/interim/inegi_orthophoto/resolution_pyramid")
RESOLUTIONS = (1, 2, 5, 10)


def resample_rgb(src, target_resolution_m: float):
    src_res_x = abs(src.transform.a)
    src_res_y = abs(src.transform.e)
    scale_x = src_res_x / target_resolution_m
    scale_y = src_res_y / target_resolution_m
    width = max(1, int(round(src.width * scale_x)))
    height = max(1, int(round(src.height * scale_y)))

    transform = Affine(
        target_resolution_m,
        0.0,
        src.transform.c,
        0.0,
        -target_resolution_m,
        src.transform.f,
    )

    out = np.zeros((3, height, width), dtype=np.uint8)
    for band in range(1, 4):
        reproject(
            source=rasterio.band(src, band),
            destination=out[band - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.average,
        )
    return out, transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.source),
        "purpose": "same-date resolution sensitivity benchmark",
        "resolutions_m": [],
    }

    with rasterio.open(args.source) as src:
        if src.count != 3:
            raise ValueError("source must be 3-band RGB")
        for resolution in RESOLUTIONS:
            if resolution == 1:
                arr = src.read([1, 2, 3])
                transform = src.transform
            else:
                arr, transform = resample_rgb(src, float(resolution))

            tif_path = args.output_dir / f"fundidora_2007_{resolution}m_rgb.tif"
            png_path = args.output_dir / f"fundidora_2007_{resolution}m_rgb.png"

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=arr.shape[1],
                width=arr.shape[2],
                transform=transform,
                count=3,
                dtype="uint8",
                compress="deflate",
                photometric="RGB",
            )
            with rasterio.open(tif_path, "w", **profile) as dst:
                dst.write(arr)

            Image.fromarray(np.moveaxis(arr, 0, -1), mode="RGB").save(png_path)
            manifest["resolutions_m"].append(
                {
                    "resolution_m": resolution,
                    "tif": str(tif_path),
                    "png": str(png_path),
                    "shape_chw": list(arr.shape),
                }
            )
            print(
                f"{resolution} m: {arr.shape[2]} x {arr.shape[1]} pixels -> {png_path}"
            )

    manifest_path = args.output_dir / "resolution_pyramid.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
