"""Generate RGB + canopy-mask QC overlays for a dense UTC annotation dataset.

The script samples representative patches from train/val/test and writes
side-by-side panels:
1. RGB patch
2. binary canopy mask
3. RGB with semi-transparent canopy overlay

This is a human QC utility only. It does not alter labels or train models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def overlay_rgb_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be HxWx3")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Mask shape must match RGB spatial shape")
    base = rgb.astype(np.float32)
    out = base.copy()
    positive = mask > 0
    # Green overlay keeps canopy easy to inspect without changing geometry.
    overlay = np.zeros_like(base)
    overlay[..., 1] = 255.0
    out[positive] = (1.0 - alpha) * base[positive] + alpha * overlay[positive]
    return np.clip(out, 0, 255).astype(np.uint8)


def choose_records(records: list[dict], per_split: int) -> list[dict]:
    selected = []
    for split in ("train", "val", "test"):
        rows = [r for r in records if r.get("split") == split]
        if not rows:
            continue
        # Deterministic spread across the split instead of just taking the first N.
        if len(rows) <= per_split:
            picks = rows
        else:
            idx = np.linspace(0, len(rows) - 1, per_split, dtype=int)
            picks = [rows[int(i)] for i in idx]
        selected.extend(picks)
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-split", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.45)
    args = p.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1")

    data = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        raise ValueError("Dataset manifest has no records")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_records(records, args.per_split)
    index_rows = []

    for rec in selected:
        rgb_path = args.images_dir / rec["file"]
        mask_path = args.masks_dir / rec["mask_file"]
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)

        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"Shape mismatch for {rec['file']}: RGB={rgb.shape[:2]} mask={mask.shape}")

        mask_rgb = np.repeat(mask[..., None], 3, axis=2)
        overlay = overlay_rgb_mask(rgb, mask, alpha=args.alpha)

        panel = np.concatenate([rgb, mask_rgb, overlay], axis=1)
        out_name = f"{rec['split']}_{int(rec['id']):03d}_qc.png"
        Image.fromarray(panel, mode="RGB").save(args.output_dir / out_name)

        index_rows.append({
            "id": rec["id"],
            "split": rec["split"],
            "file": rec["file"],
            "mask_file": rec["mask_file"],
            "canopy_fraction": rec.get("canopy_fraction"),
            "qc_file": out_name,
        })

    out_manifest = args.output_dir / "qc_index.json"
    out_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "dense UTC annotation visual QC",
                "dataset_manifest": str(args.dataset_manifest),
                "per_split": args.per_split,
                "alpha": args.alpha,
                "panel_order": ["rgb", "binary_mask", "rgb_plus_canopy_overlay"],
                "records": index_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(index_rows)} QC panels to {args.output_dir}")
    for split in ("train", "val", "test"):
        n = sum(r["split"] == split for r in index_rows)
        print(f"{split}: {n} panels")
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
