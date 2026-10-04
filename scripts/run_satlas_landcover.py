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
from rasterio.windows import Window, from_bounds
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


def context_window_for_aoi(
    aoi_window: Window, context_size: int = 512
) -> tuple[Window, tuple[slice, slice]]:
    """Return a fixed-size context window and AOI slices within it.

    Satlas was trained on image tiles with real spatial context. Padding a small
    50-100 px AOI to 512 with black pixels can dominate the model response.
    Instead, infer on a real 512x512 Sentinel context tile and crop predictions
    back to the target AOI.
    """
    if aoi_window.width > context_size or aoi_window.height > context_size:
        raise ValueError(
            f"AOI window {aoi_window.width:.1f}x{aoi_window.height:.1f} "
            f"exceeds {context_size}; tiling is required"
        )

    col0 = int(np.floor(aoi_window.col_off + aoi_window.width / 2 - context_size / 2))
    row0 = int(np.floor(aoi_window.row_off + aoi_window.height / 2 - context_size / 2))
    context = Window(col0, row0, context_size, context_size)

    aoi_col_start = int(np.floor(aoi_window.col_off)) - col0
    aoi_row_start = int(np.floor(aoi_window.row_off)) - row0
    aoi_col_end = int(np.ceil(aoi_window.col_off + aoi_window.width)) - col0
    aoi_row_end = int(np.ceil(aoi_window.row_off + aoi_window.height)) - row0

    slices = (
        slice(aoi_row_start, aoi_row_end),
        slice(aoi_col_start, aoi_col_end),
    )
    return context, slices


def read_visual_context(
    item, bbox: list[float], context_size: int = 512
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
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
            aoi_window = from_bounds(left, bottom, right, top, src.transform)
            context_window, aoi_slices = context_window_for_aoi(
                aoi_window, context_size=context_size
            )
            count = min(src.count, 3)
            arr = src.read(
                list(range(1, count + 1)),
                window=context_window,
                boundless=True,
                fill_value=0,
            )
    if arr.shape[0] != 3:
        raise RuntimeError(f"Expected 3-band TCI; got shape {arr.shape}")

    context_rgb = np.moveaxis(arr, 0, -1).astype(np.uint8)
    rows, cols = aoi_slices
    aoi_rgb = context_rgb[rows, cols]
    return context_rgb, aoi_rgb, aoi_slices


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


def confidence_diagnostics(probs: torch.Tensor) -> dict[str, float]:
    """Summarize predictive confidence for a CxHxW probability tensor."""
    max_prob = probs.max(dim=0).values
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=0)
    entropy = entropy / np.log(probs.shape[0])
    return {
        "mean_max_probability": round(float(max_prob.mean().item()), 4),
        "median_max_probability": round(float(max_prob.median().item()), 4),
        "fraction_max_probability_ge_0_5": round(float((max_prob >= 0.5).float().mean().item()), 4),
        "mean_normalized_entropy": round(float(entropy.mean().item()), 4),
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
            "Inference uses a real 512x512 Sentinel context tile and crops predictions to the AOI.",
            "Developed combines multiple urban surfaces at this resolution.",
        ],
    }

    for year, item_id in ids.items():
        item = fetch_item(item_id)
        context_rgb, rgb, aoi_slices = read_visual_context(item, bbox)
        h, w, _ = rgb.shape

        tensor = (
            torch.from_numpy(context_rgb)
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
                size=context_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0]
            native_context = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

        rows, cols = aoi_slices
        aoi_probs = probs[:, rows, cols].cpu()
        native = native_context[rows, cols]
        native_names = np.asarray(LAND_COVER_CLASSES, dtype=object)[native]
        project = project_map(native_names)
        project_names = list(PROJECT_GROUPS)

        context_path = args.output_dir / f"{args.site}_{year}_satlas_context.png"
        Image.fromarray(context_rgb, mode="RGB").save(context_path)
        tci_path = args.output_dir / f"{args.site}_{year}_satlas_tci.png"
        Image.fromarray(rgb, mode="RGB").save(tci_path)
        semantic_path = args.output_dir / f"{args.site}_{year}_semantic.png"
        save_project_png(project, semantic_path)
        np.save(args.output_dir / f"{args.site}_{year}_landcover_native.npy", native)
        np.save(
            args.output_dir / f"{args.site}_{year}_landcover_probabilities.npy",
            aoi_probs.numpy().astype(np.float32),
        )

        summary["years"][year] = {
            "item_id": item_id,
            "shape": [h, w],
            "context_shape": list(context_rgb.shape[:2]),
            "context_png": str(context_path),
            "native_composition_pct": composition(native, LAND_COVER_CLASSES),
            "project_composition_pct": composition(project, project_names),
            "confidence": confidence_diagnostics(aoi_probs),
            "tci_png": str(tci_path),
            "semantic_png": str(semantic_path),
        }

        print(f"{year}: {summary['years'][year]['project_composition_pct']}")
        print(f"{year} confidence: {summary['years'][year]['confidence']}")

    out_json = args.output_dir / f"{args.site}_satlas_landcover.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
