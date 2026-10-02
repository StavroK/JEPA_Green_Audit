"""Ingest INEGI high-resolution elevation products for an AOI.

Supports:
- MDS / surface model raster (vegetation + buildings + terrain)
- optional MDT / terrain model raster
- optional normalized DSM (nDSM = MDS - MDT)

Inputs may be GeoTIFF files or INEGI ZIP archives containing a GeoTIFF.
This is structural context, not RGB imagery.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject, transform_geom


def load_aoi(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        if not payload.get("features"):
            raise ValueError("AOI FeatureCollection is empty")
        return payload["features"][0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def resolve_raster(path: Path, temp_dir: Path) -> Path:
    """Return a GeoTIFF path from either a TIF or an INEGI ZIP archive."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError("source must be .tif/.tiff or .zip")

    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith((".tif", ".tiff"))
            and not name.lower().endswith(".ovr")
        ]
        if not names:
            raise ValueError(f"No GeoTIFF found inside {path}")
        if len(names) > 1:
            # Prefer the actual dataset rather than auxiliary/preview products.
            names.sort(key=lambda n: ("/conjunto_de_datos/" not in f"/{n.lower()}", len(n)))
        selected = names[0]
        archive.extract(selected, temp_dir)
        return temp_dir / selected


def crop_single_band(source: Path, aoi_wgs84: dict):
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError(f"{source} has no CRS")
        if src.count < 1:
            raise ValueError(f"{source} contains no raster bands")

        geom = transform_geom("EPSG:4326", src.crs, aoi_wgs84)
        cropped, transform = mask(src, [geom], crop=True, indexes=1, filled=True)
        arr = cropped.astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            height=arr.shape[0],
            width=arr.shape[1],
            transform=transform,
            nodata=np.nan,
        )
        return arr, profile


def align_to_reference(source: Path, reference_profile: dict) -> np.ndarray:
    with rasterio.open(source) as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan

        dst = np.full(
            (reference_profile["height"], reference_profile["width"]),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return dst


def write_float_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def preview_uint8(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(arr[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = np.round(scaled[valid] * 255).astype(np.uint8)
    return out


def stats(arr: np.ndarray) -> dict:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(valid.size),
        "min_m": float(valid.min()),
        "max_m": float(valid.max()),
        "mean_m": float(valid.mean()),
        "p02_m": float(np.percentile(valid, 2)),
        "p50_m": float(np.percentile(valid, 50)),
        "p98_m": float(np.percentile(valid, 98)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True, help="INEGI MDS ZIP or GeoTIFF")
    parser.add_argument("--terrain", type=Path, help="Optional matching MDT ZIP or GeoTIFF")
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/inegi_elevation"),
    )
    parser.add_argument(
        "--prefix",
        default="aoi",
        help="Output filename prefix, e.g. fundidora or la_pastora",
    )
    args = parser.parse_args()

    aoi = load_aoi(args.aoi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        surface_path = resolve_raster(args.surface, tmp_dir / "surface")
        surface, surface_profile = crop_single_band(surface_path, aoi)

        surface_tif = args.output_dir / f"{args.prefix}_mds.tif"
        surface_png = args.output_dir / f"{args.prefix}_mds_preview.png"
        write_float_raster(surface_tif, surface, surface_profile)
        Image.fromarray(preview_uint8(surface), mode="L").save(surface_png)

        report = {
            "surface_source": str(args.surface),
            "aoi": str(args.aoi),
            "surface_stats": stats(surface),
            "note": (
                "MDS is elevation/structure data, not RGB imagery. "
                "Do not pass the grayscale preview to an RGB model as if it were an orthophoto."
            ),
        }

        if args.terrain:
            terrain_path = resolve_raster(args.terrain, tmp_dir / "terrain")
            terrain = align_to_reference(terrain_path, surface_profile)
            ndsm = surface - terrain
            ndsm[~np.isfinite(surface) | ~np.isfinite(terrain)] = np.nan

            terrain_tif = args.output_dir / f"{args.prefix}_mdt_aligned.tif"
            ndsm_tif = args.output_dir / f"{args.prefix}_ndsm.tif"
            ndsm_png = args.output_dir / f"{args.prefix}_ndsm_preview.png"

            write_float_raster(terrain_tif, terrain, surface_profile)
            write_float_raster(ndsm_tif, ndsm, surface_profile)
            Image.fromarray(preview_uint8(ndsm), mode="L").save(ndsm_png)

            report.update(
                {
                    "terrain_source": str(args.terrain),
                    "terrain_stats": stats(terrain),
                    "ndsm_stats": stats(ndsm),
                    "ndsm_definition": "MDS minus MDT; approximate above-ground object height where products are co-registered and temporally compatible",
                }
            )

    report_path = args.output_dir / f"{args.prefix}_elevation_provenance.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {surface_tif}")
    print(f"Wrote {surface_png}")
    if args.terrain:
        print(f"Wrote {args.output_dir / f'{args.prefix}_mdt_aligned.tif'}")
        print(f"Wrote {args.output_dir / f'{args.prefix}_ndsm.tif'}")
        print(f"Wrote {args.output_dir / f'{args.prefix}_ndsm_preview.png'}")
    print(f"Wrote {report_path}")
    print("Surface stats:", report["surface_stats"])
    if args.terrain:
        print("nDSM stats:", report["ndsm_stats"])


if __name__ == "__main__":
    main()
