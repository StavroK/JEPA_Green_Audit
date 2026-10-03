"""Build embeddings for the 2007 same-date resolution benchmark.

For each manually labeled geographic cell, extract the corresponding RGB patch
from the 1 m, 2 m, 5 m and 10 m resolution pyramid and compute:

- supervised ImageNet ResNet18 frozen embeddings
- frozen I-JEPA embeddings

The label is created once from the native 1 m RGB and reused for all resolutions,
so only spatial resolution changes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jepa_green_audit.ijepa_encoder import FrozenIJEPAEncoder

DEFAULT_LABELS = Path("data/labels/fundidora_2007_resolution/fundidora_2007_labels.csv")
DEFAULT_PYRAMID = Path("data/interim/inegi_orthophoto/resolution_pyramid")
DEFAULT_OUTPUT = Path("data/processed/fundidora_2007_resolution_embeddings.npz")
DEFAULT_META = Path("data/processed/fundidora_2007_resolution_embeddings.json")
RESOLUTIONS = (1, 2, 5, 10)


def load_labeled_cells(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label", "").strip()
            if label not in {"vegetation", "non_vegetation", "uncertain"}:
                raise ValueError(
                    f"Missing/invalid label at row={row.get('row')} col={row.get('col')}: {label!r}"
                )
            if label == "uncertain":
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No usable labeled rows found")
    return rows


def cell_box(
    width: int,
    height: int,
    row: int,
    col: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    x_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    return (
        int(x_edges[col]),
        int(y_edges[row]),
        int(x_edges[col + 1]),
        int(y_edges[row + 1]),
    )


class FrozenResNet18Encoder:
    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/torchvision are required. Install requirements-jepa.txt."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        arr = np.asarray(rgb).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        tensor = (tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            embedding = self.model(tensor)
        return embedding.detach().cpu().numpy()[0].astype(np.float32)

    def metadata(self) -> dict:
        return {
            "family": "supervised ImageNet",
            "model": "ResNet18",
            "weights": "IMAGENET1K_V1",
            "embedding_dim": 512,
            "frozen": True,
            "input_size": 224,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--pyramid-dir", type=Path, default=DEFAULT_PYRAMID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ijepa"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/IN1K-vit.h.14-300e.pth.tar"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = load_labeled_cells(args.labels)
    resnet = FrozenResNet18Encoder(device=args.device)
    ijepa = FrozenIJEPAEncoder(
        upstream_dir=args.upstream,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    y = np.asarray([row["label"] == "vegetation" for row in rows], dtype=bool)
    grid_rows = np.asarray([int(row["grid_rows"]) for row in rows], dtype=int)
    grid_cols = np.asarray([int(row["grid_cols"]) for row in rows], dtype=int)
    sample_rows = np.asarray([int(row["row"]) for row in rows], dtype=int)
    sample_cols = np.asarray([int(row["col"]) for row in rows], dtype=int)

    payload = {
        "y": y,
        "row": sample_rows,
        "col": sample_cols,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }

    resolution_meta = {}
    for resolution in RESOLUTIONS:
        image_path = args.pyramid_dir / f"fundidora_2007_{resolution}m_rgb.png"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        resnet_features = []
        ijepa_features = []

        print(f"{resolution} m: encoding {len(rows)} labeled cells")
        for index, row in enumerate(rows, start=1):
            gr = int(row["grid_rows"])
            gc = int(row["grid_cols"])
            rr = int(row["row"])
            cc = int(row["col"])

            patch = image.crop(cell_box(width, height, rr, cc, gr, gc))
            rgb = np.asarray(patch)

            resnet_features.append(resnet.encode_rgb(rgb))
            ijepa_features.append(ijepa.encode_rgb(rgb))

            if index % 25 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)}", end="\r")
        print()

        payload[f"X_resnet18_{resolution}m"] = np.asarray(
            resnet_features, dtype=np.float32
        )
        payload[f"X_ijepa_{resolution}m"] = np.asarray(
            ijepa_features, dtype=np.float32
        )
        resolution_meta[str(resolution)] = {
            "image": str(image_path),
            "shape_wh": [width, height],
            "samples": len(rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    meta = {
        "labels": str(args.labels),
        "resolutions_m": list(RESOLUTIONS),
        "samples": len(rows),
        "vegetation": int(y.sum()),
        "non_vegetation": int((~y).sum()),
        "resnet18": resnet.metadata(),
        "ijepa": ijepa.metadata(),
        "resolution_sources": resolution_meta,
        "experimental_control": (
            "Same 2007 acquisition, same geographic cells and labels at all "
            "resolutions; only spatial resolution changes."
        ),
    }
    args.metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    print(
        f"Samples: {len(rows)} | vegetation={int(y.sum())} | "
        f"non-vegetation={int((~y).sum())}"
    )


if __name__ == "__main__":
    main()
