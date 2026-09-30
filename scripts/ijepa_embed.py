"""Extract frozen I-JEPA embeddings from one or two RGB image tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jepa_green_audit.change import cosine_embedding_change
from jepa_green_audit.ijepa_encoder import FrozenIJEPAEncoder


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ijepa_embeddings.json"),
    )
    args = parser.parse_args()

    encoder = FrozenIJEPAEncoder(
        upstream_dir=args.upstream,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    before = encoder.encode_rgb(read_rgb(args.before))

    payload = {
        "encoder": encoder.metadata(),
        "before": {
            "path": str(args.before),
            "embedding": before.tolist(),
        },
    }

    if args.after is not None:
        after = encoder.encode_rgb(read_rgb(args.after))
        payload["after"] = {
            "path": str(args.after),
            "embedding": after.tolist(),
        }
        payload["cosine_embedding_change"] = cosine_embedding_change(before, after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    if "cosine_embedding_change" in payload:
        print(
            "Cosine embedding change: "
            f"{payload['cosine_embedding_change']:.6f}"
        )


if __name__ == "__main__":
    main()
