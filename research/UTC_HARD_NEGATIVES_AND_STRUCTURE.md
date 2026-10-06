# Hard-negative calibration and structural context

## Why this round exists

The first human-calibrated RGB classifier recovered substantially more visible
canopy than the simple green/texture heuristic, but visual QC still showed
false positives over dark pavement, water, low vegetation/grass, roofs, and
shadows.

The correction strategy is **hard-negative calibration**, not merely increasing
the probability threshold.

## Hard-negative categories

The point labeler now distinguishes:

- pavement;
- water;
- grass / low vegetation;
- roof / building;
- shadow;
- bare soil;
- generic non-tree;
- tree canopy;
- uncertain.

All non-tree subtypes collapse to the binary non-tree class during model
training, while subtype counts are retained in the output metadata. This makes
the training set auditable and lets reviewers deliberately target systematic
confounders.

## Added RGB features

The classifier now includes:

- RGB and normalized RGB;
- Excess Green;
- brightness and green dominance;
- chroma and saturation;
- circular hue coordinates;
- local luminance means and texture;
- local RGB/color variance.

These features are intended to reduce reliance on brightness alone.

## Structural evidence from INEGI

Fundidora also has INEGI MDS + MDT products from 2024, allowing:

```text
nDSM = MDS - MDT
```

This is valuable because water, pavement, grass, and bare ground should
generally remain near terrain height, while trees and buildings are elevated.

However, the RGB orthophoto used for the historical UTC pilot is from **2007**.
Therefore the 2024 nDSM must **not** be fused pixel-by-pixel into the 2007
classifier or used to claim a 2007 UTC result. Seventeen years of structural
change would create temporal leakage/misalignment.

For the 2007 pilot, use 2024 nDSM only as:
- a methodological example of the structural branch;
- a diagnostic for what a future contemporaneous RGB+nDSM UTC product should do.

For a current Monterrey UTC product, the preferred architecture is:

```text
same-period high-resolution RGB
        +
same-period nDSM / canopy-height evidence
        +
human hard negatives / independent validation
        ↓
tree-canopy probability
        ↓
validated UTC
```

A structural veto should also not blindly remove all high objects because
buildings are elevated too. RGB/semantic evidence and nDSM must be used jointly.
