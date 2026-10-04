"""Run SatlasPretrain Sentinel-2 RGB land-cover inference for an AOI.

Important:
- uses the Earth Search 'visual' asset (ESA-style 8-bit TCI), not the project's
  percentile-stretched display PNG;
- loads only the land_cover head from the full Satlas checkpoint;
- outputs the native Satlas land-cover classes plus an auditable project-level
  composition summary;
- predictions are QC/model evidence, not human labels or ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from pystac_client import Client
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import torch
import torch.nn.functional as F

from jepa_green_audit.sentinel2 import EARTH_SEARCH, COLLECTION

SATLAS_COMMIT = "c8b9aa5d4acdd3e4f58eb7cbb28ac18bb12c985f"
DEFAULT_VENDOR = Path("vendor/satlas")
DEFAULT_WEIGHTS = Path("models/satlas/sentinel2_swinb_si_rgb.pth")
DEFAULT_CONFIG = Path("vendor/satlas/configs/sentinel2/swinb_si_rgb.txt")

LAND_COVER_CLASSES = [
    "background",
    "water",
    "developed",
    "tree",
    "shrub",
    "grass",
    "crop",
    "bare",
    "snow",
    "wetland",
    "mangroves",
    "moss",
]

PROJECT_GROUPS = {
    "tree_canopy": {"tree"},
    "other_vegetation": {"shrub", "grass", "crop", "wetland", "mangroves", "moss"},
    "developed": {"developed"},
    "bare_ground": {"bare"},
    "water": {"water"},
    "uncertain": {"background", "snow"},
}

PROJECT_COLORS = {
    "tree_canopy": (0, 100, 0),
    "other_vegetation": (60, 180, 75),
    "developed": (200, 60, 60),
    "bare_ground": (150, 150, 150),
    "water": (50, 100, 220),
    "uncertain": (30, 30, 30),
}


def load_feature(path: Path) -> tuple[dict, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError("Expected one AOI feature")
        feature = features[0]
    elif payload.get("type") == "Feature":
        feature = payload
    else:
        raise ValueError("Expected GeoJSON Feature or FeatureCollection")
    coords = feature["geometry"]["coordinates"][0]
    xs = [float(p[0]) for p in coords]
    ys = [float(p[1]) for p in coords]
    return feature, [min(xs), min(ys), max(xs), max(ys)]


def scene_ids(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        year: payload["observations"][year]["scene"]["item_id"]
        for year in ("2025", "2026")
    }


def fetch_item(item_id: str):
    catalog = Client.open(EARTH_SEARCH)
    items = list(
        catalog.search(
            collections=[COLLECTION],
            ids=[item_id],
            max_items=1,
        ).items()
    )
    if not items:
        raise RuntimeError(f"Sentinel-2 item not found: {item_id}")
    return items[0]


def read_visual_aoi(item, bbox: list[float]) -> np.ndarray:
    if "visual" not in item.assets:
        raise KeyError(
            f"{item.id} has no 'visual' TCI asset; available={sorted(item.assets)}"
        )
    href = item.assets["visual"].href
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    ):
        with rasterio.open(href) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *bbox, densify_pts=21
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            count = min(src.count, 3)
            arr = src.read(list(range(1, count + 1)), window=window, boundless=True)
    if arr.shape[0] != 3:
        raise RuntimeError(f"Expected 3-band TCI; got shape {arr.shape}")
    return np.moveaxis(arr, 0, -1).astype(np.uint8)


def center_pad_512(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w, _ = rgb.shape
    if h > 512 or w > 512:
        raise ValueError(f"AOI TCI {h}x{w} exceeds 512; tiling is required")
    top = (512 - h) // 2
    left = (512 - w) // 2
    bottom = 512 - h - top
    right = 512 - w - left
    padded = np.pad(
        rgb,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded, (top, bottom, left, right)


def load_landcover_model(vendor: Path, config_path: Path, weights: Path, device: str):
    sys.path.insert(0, str(vendor.resolve()))
    import satlas.model.dataset as satlas_dataset
    import satlas.model.model as satlas_model

    config = json.loads(config_path.read_text(encoding="utf-8"))
    idx = next(
        i for i, spec in enumerate(config["Tasks"]) if spec["Name"] == "land_cover"
    )
    task_spec = dict(config["Tasks"][idx])
    task_spec["Task"] = satlas_dataset.tasks["land_cover"]

    model_cfg = dict(config["Model"])
    model_cfg["Backbone"] = dict(model_cfg["Backbone"])
    model_cfg["Backbone"]["Pretrained"] = False
    model_cfg["Heads"] = [dict(config["Model"]["Heads"][idx])]

    model = satlas_model.Model(
        {
            "config": model_cfg,
            "channels": config["Channels"],
            "tasks": [task_spec],
        }
    )

    checkpoint = torch.load(weights, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    selected = {}
    for key, value in checkpoint.items():
        if key.startswith("backbone.") or key.startswith("intermediates."):
            selected[key] = value
        elif key.startswith(f"heads.{idx}."):
            selected["heads.0." + key[len(f"heads.{idx}."):]] = value

    missing, unexpected = model.load_state_dict(selected, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    # No task-specific keys should be missing after remapping.
    meaningful_missing = [
        key for key in missing
        if key.startswith("backbone.")
        or key.startswith("intermediates.")
        or key.startswith("heads.0.")
    ]
    if meaningful_missing:
        raise RuntimeError(f"Missing checkpoint keys: {meaningful_missing[:10]}")

    model.to(device)
    model.eval()
    return model


def project_map(native_names: np.ndarray) -> np.ndarray:
    groups = list(PROJECT_GROUPS)
    out = np.full(native_names.shape, groups.index("uncertain"), dtype=np.uint8)
    for idx, group in enumerate(groups):
        out[np.isin(native_names, list(PROJECT_GROUPS[group]))] = idx
    return out


def composition(mask: np.ndarray, names: list[str]) -> dict[str, float]:
    total = mask.size
    return {
        name: round(100.0 * float(np.sum(mask == i)) / total, 3)
        for i, name in enumerate(names)
    }


def save_project_png(mask: np.ndarray, output: Path) -> None:
    names = list(PROJECT_GROUPS)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, name in enumerate(names):
        rgb[mask == idx] = PROJECT_COLORS[name]
    Image.fromarray(rgb, mode="RGB").save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _, bbox = load_feature(args.aoi)
    model = load_landcover_model(
        args.vendor, args.config, args.weights, args.device
    )
    ids = scene_ids(args.baseline_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": {
            "name": "SatlasPretrain Sentinel-2 Swin-v2-Base single-image RGB",
            "upstream_commit": SATLAS_COMMIT,
            "checkpoint": str(args.weights),
            "task": "land_cover",
        },
        "input": {
            "asset": "Earth Search visual (TCI)",
            "normalization": "uint8 RGB / 255",
            "aoi_bbox_wgs84": bbox,
        },
        "project_taxonomy": {k: sorted(v) for k, v in PROJECT_GROUPS.items()},
        "years": {},
        "limitations": [
            "10 m Sentinel-2 land cover does not delineate individual tree crowns.",
            "Model predictions are not ground truth and require local validation.",
            "Developed combines multiple urban surfaces at this resolution.",
        ],
    }

    for year, item_id in ids.items():
        item = fetch_item(item_id)
        rgb = read_visual_aoi(item, bbox)
        h, w, _ = rgb.shape
        padded, (top, bottom, left, right) = center_pad_512(rgb)

        tensor = (
            torch.from_numpy(padded)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(args.device)
        )

        with torch.no_grad():
            outputs, _ = model([tensor], selected_task="land_cover")
            probs = outputs[0][0]
            probs = F.interpolate(
                probs[None],
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )[0]
            native = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

        native = native[top : top + h, left : left + w]
        native_names = np.asarray(LAND_COVER_CLASSES, dtype=object)[native]
        project = project_map(native_names)
        project_names = list(PROJECT_GROUPS)

        tci_path = args.output_dir / f"{args.site}_{year}_satlas_tci.png"
        Image.fromarray(rgb, mode="RGB").save(tci_path)
        semantic_path = args.output_dir / f"{args.site}_{year}_semantic.png"
        save_project_png(project, semantic_path)
        np.save(args.output_dir / f"{args.site}_{year}_landcover_native.npy", native)

        summary["years"][year] = {
            "item_id": item_id,
            "shape": [h, w],
            "native_composition_pct": composition(native, LAND_COVER_CLASSES),
            "project_composition_pct": composition(project, project_names),
            "tci_png": str(tci_path),
            "semantic_png": str(semantic_path),
        }

        print(f"{year}: {summary['years'][year]['project_composition_pct']}")

    out_json = args.output_dir / f"{args.site}_satlas_landcover.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
