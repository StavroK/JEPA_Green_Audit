"""Partial fine-tuning supervised baseline for the 2007 resolution benchmark.

For each resolution and geographic fold:
- start from ImageNet-pretrained ResNet18
- freeze stem/layer1/layer2/layer3
- fine-tune layer4 + a new binary classifier head
- train only on the geographic training folds
- evaluate once on the held-out geographic fold

This is intentionally stronger than the frozen-feature linear probe while still
appropriate for a small labeled dataset. No test-fold labels are used for model
selection or early stopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from jepa_green_audit.benchmark import binary_segmentation_metrics
try:
    from scripts.build_2007_resolution_embeddings import (
        DEFAULT_LABELS,
        DEFAULT_PYRAMID,
        RESOLUTIONS,
        cell_box,
        load_labeled_cells,
    )
    from scripts.run_2007_resolution_cv import geographic_column_folds, summarize
except ModuleNotFoundError:
    from build_2007_resolution_embeddings import (
        DEFAULT_LABELS,
        DEFAULT_PYRAMID,
        RESOLUTIONS,
        cell_box,
        load_labeled_cells,
    )
    from run_2007_resolution_cv import geographic_column_folds, summarize

DEFAULT_OUTPUT = Path("outputs/fundidora_2007_resnet18_finetune_cv.json")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


class CellDataset:
    def __init__(self, image, rows, indices, train: bool):
        import torchvision.transforms as T
        self.image = image
        self.rows = rows
        self.indices = np.asarray(indices, dtype=int)
        self.train = train
        common = [
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                *common,
            ])
        else:
            self.transform = T.Compose(common)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        row = self.rows[idx]
        gr, gc = int(row["grid_rows"]), int(row["grid_cols"])
        rr, cc = int(row["row"]), int(row["col"])
        patch = self.image.crop(
            cell_box(self.image.width, self.image.height, rr, cc, gr, gc)
        )
        x = self.transform(patch)
        y = 1 if row["label"] == "vegetation" else 0
        return x, y


def make_model(device: str):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)

    return model.to(torch.device(device))


def train_and_predict(image, rows, train_idx, test_idx, device, epochs, batch_size, lr, seed):
    import torch
    from torch.utils.data import DataLoader

    seed_everything(seed)
    train_ds = CellDataset(image, rows, train_idx, train=True)
    test_ds = CellDataset(image, rows, test_idx, train=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_model(device)
    dev = torch.device(device)

    y_train = np.asarray([
        1 if rows[int(i)]["label"] == "vegetation" else 0 for i in train_idx
    ])
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=dev)
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    model.train()
    for _epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    pred, truth, scores = [], [], []
    with torch.inference_mode():
        for x, y in test_loader:
            logits = model(x.to(dev))
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred.extend((prob >= 0.5).cpu().numpy().tolist())
            scores.extend(prob.cpu().numpy().tolist())
            truth.extend(y.numpy().tolist())

    return (
        np.asarray(truth, dtype=bool),
        np.asarray(pred, dtype=bool),
        np.asarray(scores, dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--pyramid-dir", type=Path, default=DEFAULT_PYRAMID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_labeled_cells(args.labels)
    y = np.asarray([row["label"] == "vegetation" for row in rows], dtype=bool)
    cols = np.asarray([int(row["col"]) for row in rows], dtype=int)
    grid_cols = int(rows[0]["grid_cols"])
    folds = geographic_column_folds(cols, grid_cols)

    payload = {
        "benchmark_type": "partial_finetune_supervised_geographic_cv",
        "model": "ImageNet ResNet18 layer4+fc fine-tuned",
        "year": 2007,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "results": {},
    }

    for resolution in RESOLUTIONS:
        image_path = args.pyramid_dir / f"fundidora_2007_{resolution}m_rgb.png"
        image = Image.open(image_path).convert("RGB")
        runs = []
        print(f"{resolution} m:")
        for fold_index, (train_mask, test_mask, col_range) in enumerate(folds):
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            truth, pred, scores = train_and_predict(
                image,
                rows,
                train_idx,
                test_idx,
                args.device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.seed + fold_index,
            )
            metrics = binary_segmentation_metrics(truth, pred)
            metrics.update({
                "fold": fold_index,
                "test_columns": list(col_range),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "test_vegetation": int(truth.sum()),
                "predicted_vegetation": int(pred.sum()),
                "mean_positive_probability": float(scores.mean()),
            })
            runs.append(metrics)
            print(
                f"  fold {fold_index}: IoU={metrics['iou']:.3f} "
                f"F1={metrics['f1_dice']:.3f} "
                f"veg={int(truth.sum())}/{len(truth)}"
            )

        summary = {"runs": runs}
        summary.update(summarize(runs))
        payload["results"][str(resolution)] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print("Aggregate:")
    for resolution in RESOLUTIONS:
        result = payload["results"][str(resolution)]
        print(
            f"  {resolution:>2} m | IoU={result['iou_mean']:.3f} ±{result['iou_std']:.3f} | "
            f"F1={result['f1_dice_mean']:.3f} ±{result['f1_dice_std']:.3f}"
        )


if __name__ == "__main__":
    main()
