"""Generate a semi-automatic RGB vegetation/canopy candidate mask for UTC review.

This is a *proposal* layer only. It is intentionally not reported as UTC.
RGB alone cannot reliably distinguish tree crowns from grass, shrubs, green
roofs, shadows, or other confounders. The mask is meant to reduce annotation
effort before human validation/correction.

The heuristic uses robustly stretched RGB plus:
- normalized excess green (ExG);
- green-channel dominance;
- brightness bounds;
- a simple local texture term to favor crown-like heterogeneous regions.

No network download or paid service is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import sieve


def robust_rgb(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3 or data.shape[0] < 3:
        raise ValueError("Expected raster with at least 3 bands")
    rgb = data[:3].astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for b in range(3):
        band = rgb[b]
        finite = np.isfinite(band)
        vals = band[finite]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[b] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return np.moveaxis(out, 0, -1)


def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Fast edge-padded square-window mean using integral images."""
    if radius <= 0:
        return a.astype(np.float32)
    k = 2 * radius + 1
    padded = np.pad(a.astype(np.float32), radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    total = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return total / float(k * k)


def canopy_candidate(
    rgb: np.ndarray,
    *,
    exg_threshold: float = 0.08,
    green_margin: float = 0.015,
    min_brightness: float = 0.08,
    max_brightness: float = 0.82,
    texture_radius: int = 2,
    min_texture: float = 0.018,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return an RGB-derived candidate mask and diagnostic feature layers."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB array")

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    denom = r + g + b + 1e-6
    rn, gn, bn = r / denom, g / denom, b / denom
    exg = 2.0 * gn - rn - bn
    brightness = (r + g + b) / 3.0
    green_dominance = g - np.maximum(r, b)

    # Texture from local luminance standard deviation. It is only a weak
    # tree-crown cue; grass and built edges can still pass this filter.
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mean = box_mean(lum, texture_radius)
    mean2 = box_mean(lum * lum, texture_radius)
    texture = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

    candidate = (
        (exg >= exg_threshold)
        & (green_dominance >= green_margin)
        & (brightness >= min_brightness)
        & (brightness <= max_brightness)
        & (texture >= min_texture)
    )

    return candidate, {
        "exg": exg,
        "green_dominance": green_dominance,
        "brightness": brightness,
        "texture": texture,
    }


def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out = base.copy()
    # Candidate pixels get a semi-transparent green emphasis.
    out[mask, 0] = (0.45 * out[mask, 0]).astype(np.uint8)
    out[mask, 1] = np.clip(0.55 * out[mask, 1] + 115, 0, 255).astype(np.uint8)
    out[mask, 2] = (0.45 * out[mask, 2]).astype(np.uint8)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output-mask", type=Path, required=True)
    p.add_argument("--output-preview", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--exg-threshold", type=float, default=0.08)
    p.add_argument("--green-margin", type=float, default=0.015)
    p.add_argument("--min-brightness", type=float, default=0.08)
    p.add_argument("--max-brightness", type=float, default=0.82)
    p.add_argument("--texture-radius", type=int, default=2)
    p.add_argument("--min-texture", type=float, default=0.018)
    p.add_argument(
        "--min-region-pixels",
        type=int,
        default=8,
        help="Remove connected candidate regions smaller than this many pixels.",
    )
    args = p.parse_args()

    with rasterio.open(args.image) as src:
        data = src.read()
        if src.count < 3:
            raise ValueError("Input image must contain at least 3 bands")
        rgb = robust_rgb(data)
        mask, features = canopy_candidate(
            rgb,
            exg_threshold=args.exg_threshold,
            green_margin=args.green_margin,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
            texture_radius=args.texture_radius,
            min_texture=args.min_texture,
        )

        cleaned = sieve(
            mask.astype(np.uint8),
            size=max(1, args.min_region_pixels),
            connectivity=8,
        ).astype(bool)

        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")

        pixel_area = abs(src.transform.a * src.transform.e)
        candidate_pixels = int(cleaned.sum())
        valid_pixels = int(cleaned.size)
        candidate_area_m2 = candidate_pixels * pixel_area

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_mask, "w", **profile) as dst:
        dst.write(cleaned.astype(np.uint8), 1)

    overlay = make_overlay(rgb, cleaned)
    args.output_preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(args.output_preview)

    diagnostics = {
        "method": "rgb_canopy_candidate_v1",
        "status": "proposal_not_validated_utc",
        "source_image": str(args.image),
        "parameters": {
            "exg_threshold": args.exg_threshold,
            "green_margin": args.green_margin,
            "min_brightness": args.min_brightness,
            "max_brightness": args.max_brightness,
            "texture_radius": args.texture_radius,
            "min_texture": args.min_texture,
            "min_region_pixels": args.min_region_pixels,
        },
        "candidate_pixels": candidate_pixels,
        "valid_pixels": valid_pixels,
        "candidate_fraction": candidate_pixels / valid_pixels,
        "candidate_area_m2": candidate_area_m2,
        "candidate_area_ha": candidate_area_m2 / 10000.0,
        "feature_summary": {
            k: {
                "p05": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
            }
            for k, v in features.items()
        },
        "warning": (
            "This RGB-derived layer is an annotation aid. Do not report its "
            "candidate fraction as Urban Tree Canopy until human validation."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Candidate fraction: {100*diagnostics['candidate_fraction']:.2f}%")
    print(f"Candidate area: {diagnostics['candidate_area_ha']:.3f} ha")
    print(f"Wrote {args.output_mask}")
    print(f"Wrote {args.output_preview}")
    print(f"Wrote {args.output_json}")
    print("STATUS: proposal only — not validated UTC")


if __name__ == "__main__":
    main()
