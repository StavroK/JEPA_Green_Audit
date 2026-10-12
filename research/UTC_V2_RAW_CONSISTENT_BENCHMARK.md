# UTC v2 raw-consistent benchmark

## Purpose

Remove the main preprocessing caveat from the v1 Fundidora → La Pastora benchmark.

The v1 Fundidora patches were exported with a per-patch 2–98% percentile stretch.
La Pastora is available as native uint8 1 m RGB. A scientifically cleaner benchmark
should therefore train and evaluate both sites from raw-consistent RGB imagery.

## Design

The v2 dataset keeps the exact Fundidora human labels and spatial split geometry:

- same patch IDs;
- same source coordinates;
- same train/validation/test split;
- same dense human masks;
- no reannotation.

Only the model input changes: RGB is re-extracted directly from the original
Fundidora orthophoto with **no per-patch percentile stretch**.

## Step 1 — rebuild Fundidora v2

```bash
python scripts/rebuild_utc_dataset_raw_v2.py \
  --v1-dataset-dir outputs/utc/fundidora_2007/segmentation_patches_v1 \
  --source-image data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.tif \
  --output-dir outputs/utc/fundidora_2007/segmentation_patches_v2_raw
```

If the source TIFF lives elsewhere, pass the actual Fundidora 2007 crop used to
create the v1 patch coordinates.

## Step 2 — precompute v2 I-JEPA features

```bash
python scripts/prepare_utc_ijepa_features.py \
  --dataset-dir outputs/utc/fundidora_2007/segmentation_patches_v2_raw \
  --upstream vendor/ijepa \
  --checkpoint models/IN1K-vit.h.14-300e.pth.tar \
  --output outputs/utc/fundidora_2007/segmentation_patches_v2_raw/ijepa_features.npz
```

## Step 3 — train architecture-matched multi-seed v2 models

```bash
python scripts/run_utc_label_efficiency_multiseed.py \
  --dataset-dir outputs/utc/fundidora_2007/segmentation_patches_v2_raw \
  --output-root outputs/utc/fundidora_2007/v2_raw_multiseed \
  --ijepa-features outputs/utc/fundidora_2007/segmentation_patches_v2_raw/ijepa_features.npz \
  --seeds 42 43 44 45 46 \
  --budgets 25 50 100 \
  --models unet segformer dinov2 ijepa
```

## Step 4 — zero-shot La Pastora evaluation

Use the same 240 model-blind La Pastora point labels, but set
`--preprocessing raw` during target inference. Thresholds must remain selected
from Fundidora validation only.

The v2 result should be reported separately from the legacy v1 benchmark rather
than silently replacing it.

## Acceptance criterion

The v2 benchmark is complete when:

1. the Fundidora raw dataset reproduces the original coordinates/splits/masks;
2. all four model families are trained under 25%, 50%, and 100% budgets across seeds 42–46;
3. La Pastora cross-site point evaluation uses raw RGB with no target calibration;
4. v1 vs v2 cross-site metrics are compared explicitly;
5. conclusions are revised only if the v2 evidence supports them.
