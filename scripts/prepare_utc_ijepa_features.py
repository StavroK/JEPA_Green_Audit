"""Precompute frozen I-JEPA patch-token grids for the UTC benchmark.

Uses the already validated FrozenIJEPAEncoder and official local ViT-H/14
checkpoint. Each RGB patch is deterministically resized to 224x224 and encoded
as a 16x16x1280 frozen feature grid. Features are cached once so the lightweight
segmentation decoder can be trained efficiently on CPU.

I-JEPA upstream is CC BY-NC 4.0; this path is research/prototype-only unless
separate commercial rights are obtained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jepa_green_audit.ijepa_encoder import FrozenIJEPAEncoder


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--upstream",type=Path,default=Path("vendor/ijepa"))
    p.add_argument("--checkpoint",type=Path,default=Path("models/IN1K-vit.h.14-300e.pth.tar"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    encoder=FrozenIJEPAEncoder(
        upstream_dir=args.upstream,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    features={}
    records=[]
    total=len(dm["records"])
    for i,r in enumerate(dm["records"],start=1):
        image_path=args.dataset_dir/"images"/r["file"]
        rgb=np.asarray(Image.open(image_path).convert("RGB"))
        grid=encoder.encode_rgb_patch_grid(rgb)
        key=f"patch_{int(r['id']):03d}"
        features[key]=grid
        records.append({
            "id":int(r["id"]),
            "file":r["file"],
            "split":r["split"],
            "feature_key":key,
            "shape":list(grid.shape),
        })
        print(f"{i}/{total} {r['file']} -> {grid.shape}")

    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**features)
    meta={
        "task":"utc_ijepa_frozen_patch_features",
        "dataset_manifest":str(args.dataset_dir/"dataset_manifest.json"),
        "feature_file":str(args.output),
        "encoder":encoder.metadata(),
        "records":records,
        "note":"Frozen features only; no test labels used during feature extraction."
    }
    meta_path=args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")


if __name__=="__main__":
    main()
