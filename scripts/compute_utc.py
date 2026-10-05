"""Compute auditable Urban Tree Canopy (UTC) metrics from a validated canopy mask.

UTC is defined here as:
    tree-canopy area / valid analysis area * 100

This script does NOT create the canopy mask. It only computes metrics from a
binary canopy raster that has already been validated for the target date/AOI.

Input raster conventions:
- canopy pixels: value 1
- non-canopy valid pixels: value 0
- optional nodata: excluded from denominator

The raster geotransform is used to calculate pixel area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def utc_metrics(mask: np.ndarray, pixel_area_m2: float, nodata=None) -> dict[str, float | int]:
    if mask.ndim != 2:
        raise ValueError("Canopy mask must be a 2D raster")

    if nodata is None:
        valid = np.isfinite(mask)
    else:
        valid = np.isfinite(mask) & (mask != nodata)

    if not np.any(valid):
        raise ValueError("Canopy mask contains no valid pixels")

    values = mask[valid]
    invalid_values = values[(values != 0) & (values != 1)]
    if invalid_values.size:
        sample = np.unique(invalid_values)[:10].tolist()
        raise ValueError(f"Expected binary canopy mask values 0/1; found {sample}")

    valid_pixels = int(valid.sum())
    canopy_pixels = int(np.sum(values == 1))
    total_area_m2 = valid_pixels * pixel_area_m2
    canopy_area_m2 = canopy_pixels * pixel_area_m2
    utc_pct = 100.0 * canopy_area_m2 / total_area_m2

    return {
        "valid_pixels": valid_pixels,
        "canopy_pixels": canopy_pixels,
        "pixel_area_m2": float(pixel_area_m2),
        "analysis_area_m2": float(total_area_m2),
        "analysis_area_ha": float(total_area_m2 / 10000.0),
        "canopy_area_m2": float(canopy_area_m2),
        "canopy_area_ha": float(canopy_area_m2 / 10000.0),
        "utc_pct": float(utc_pct),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--date-label", required=True)
    p.add_argument("--source", required=True, help="Canopy-mask provenance/model/human source")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    with rasterio.open(args.mask) as src:
        mask = src.read(1)
        transform = src.transform
        nodata = src.nodata
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(
                "UTC area metrics require a projected raster CRS with linear units"
            )
        pixel_area_m2 = abs(transform.a * transform.e)

    metrics = utc_metrics(mask, pixel_area_m2, nodata=nodata)

    result = {
        "metric": "Urban Tree Canopy",
        "definition": "tree-canopy area / valid analysis area * 100",
        "site": args.site,
        "date_label": args.date_label,
        "canopy_mask": str(args.mask),
        "source": args.source,
        "crs": str(src.crs),
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "interpretation_limits": [
            "UTC measures horizontal crown cover, not tree count.",
            "UTC accuracy depends on the validity of the input canopy mask.",
            "Do not compare UTC values across dates unless imagery, masks, and AOI are temporally and spatially comparable.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"UTC: {metrics['utc_pct']:.2f}%")
    print(f"Canopy area: {metrics['canopy_area_ha']:.3f} ha")
    print(f"Analysis area: {metrics['analysis_area_ha']:.3f} ha")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
