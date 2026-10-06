# UTC segmentation benchmark roadmap

## Goal

Move beyond the Random Forest diagnostic baseline toward a modern semantic
segmentation benchmark for Urban Tree Canopy (UTC) mapping.

The Random Forest remains valuable as a lightweight, interpretable baseline,
but the production UTC candidate should be selected from models designed for
dense semantic segmentation.

## Benchmark order

1. **Random Forest RGB baseline**
   - freeze after a clean independent V4 validation;
   - report balanced accuracy, precision, recall, F1, IoU, UTC area bias.

2. **U-Net**
   - practical reference segmentation model;
   - strong small/medium-data baseline;
   - train on high-resolution RGB patches.

3. **DeepLabV3 / DeepLabV3+**
   - high-quality semantic segmentation comparator;
   - use a pretrained ResNet backbone where licensing permits.

4. **SegFormer-B0/B2**
   - transformer segmentation comparator;
   - useful efficiency/accuracy tradeoff.

5. **Self-supervised/foundation representation + decoder**
   - DINOv2/DINOv3-style frozen or lightly adapted encoder;
   - I-JEPA representation branch where technically appropriate;
   - compare transfer and label efficiency rather than assuming SSL wins.

## Required evaluation protocol

Use spatially disjoint train/validation/test areas. Random pixel splits are not
acceptable for final reporting because adjacent urban pixels are highly
correlated.

Primary metrics:

- tree precision;
- tree recall;
- tree F1;
- tree IoU;
- balanced accuracy;
- UTC area bias in percentage points;
- inference cost/time.

Suggested research targets for the project:

- precision >= 0.85;
- recall >= 0.85;
- F1 >= 0.85;
- IoU >= 0.75;
- balanced accuracy >= 0.90;
- absolute UTC area bias <= 3 percentage points.

These are project acceptance targets, not universal standards.

## Labeling requirement

Sparse point labels are sufficient for the Random Forest calibration experiment,
but they are not enough to train a conventional dense segmentation model.

Before the deep-learning benchmark, create a dense human-reviewed reference set
of representative patches containing:

- dense crowns;
- isolated street trees;
- grass and low vegetation;
- water;
- pavement;
- roofs;
- shadows;
- mixed canopy/building edges.

Keep a geographically separate test set untouched until the final comparison.

## Structural branch

For a current/future Monterrey UTC product, add same-period nDSM/canopy-height
features when available.

Do **not** fuse Fundidora 2007 RGB with 2024 MDS/MDT for a claimed 2007 UTC map.
The 2024 nDSM remains a structural-method example until imagery and elevation
are temporally compatible.

Future benchmark:

```text
RGB-only segmentation
vs
RGB + contemporaneous nDSM
```

This comparison directly tests whether structural information reduces
grass/water/pavement confusion while retaining tree canopy.

## Decision gates

### Gate A — freeze RF baseline
Complete a leakage-free independent V4 holdout.

### Gate B — dense annotation set
Build spatially distributed human-reviewed patch masks.

### Gate C — model benchmark
Train/evaluate U-Net, DeepLabV3(+), SegFormer, and SSL/foundation variants using
the same splits.

### Gate D — production candidate
Select the model based on independent test performance, UTC area bias,
computational cost, reproducibility, and cross-site transfer.
