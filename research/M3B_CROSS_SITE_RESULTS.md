# M3b — Cross-site UTC validation: Fundidora → La Pastora

## Objective

Test whether the Fundidora urban-tree-canopy (UTC) findings generalize to a second, denser, structurally different Monterrey site using independent high-resolution human labels.

The target site is **La Pastora, 2007**, derived from a 1 m INEGI orthophoto. This experiment is historical and should not be interpreted as current canopy condition.

## Validation design

- Source/training site: **Fundidora, 2007**
- Target/validation site: **La Pastora, 2007**
- Target RGB size: 612 × 612 px
- Human reference: **240 model-blind point annotations**
- Reference class counts: 162 tree / 78 non-tree
- Sampling: deterministic 15 × 16 jittered grid, seed 42
- No La Pastora labels were used for model fitting, threshold selection, or calibration
- Thresholds were selected from Fundidora validation only
- Legacy v1 inference reproduces the per-tile 2–98% percentile RGB stretch used by the Fundidora training pipeline
- Reported IoU is **point-classification Jaccard**, not dense-mask IoU
- Reported prevalence bias is **point-sample prevalence bias**, not UTC area bias

## Architecture-matched multi-seed result

Five seeds were evaluated (42–46) at 25% and 50% Fundidora training-label budgets.

| Model | Fundidora labels | Balanced accuracy | Tree F1 | Point IoU | Point prevalence bias |
|---|---:|---:|---:|---:|---:|
| U-Net | 25% | 0.520 ± 0.012 | 0.078 ± 0.042 | 0.041 ± 0.023 | -64.75 ± 1.58 pp |
| U-Net | 50% | 0.520 ± 0.032 | 0.087 ± 0.101 | 0.048 ± 0.059 | -64.00 ± 3.83 pp |
| SegFormer-B0 | 25% | 0.560 ± 0.049 | 0.211 ± 0.160 | 0.125 ± 0.106 | -58.83 ± 7.55 pp |
| SegFormer-B0 | 50% | 0.525 ± 0.015 | 0.097 ± 0.057 | 0.052 ± 0.032 | -63.92 ± 2.22 pp |
| **DINOv2 frozen + decoder** | **25%** | **0.804 ± 0.004** | **0.847 ± 0.018** | **0.734 ± 0.028** | **-6.92 ± 4.99 pp** |
| **DINOv2 frozen + decoder** | **50%** | **0.824 ± 0.021** | **0.874 ± 0.012** | **0.776 ± 0.019** | **-3.25 ± 4.52 pp** |
| I-JEPA frozen + decoder | 25% | 0.570 ± 0.038 | 0.361 ± 0.244 | 0.242 ± 0.184 | -45.25 ± 19.64 pp |
| I-JEPA frozen + decoder | 50% | 0.633 ± 0.067 | 0.530 ± 0.181 | 0.377 ± 0.169 | -35.50 ± 14.90 pp |

## Interpretation

The strongest result is not simply that DINOv2 ranks first. It is that the DINOv2 branch remains **both accurate and stable under a site shift and reduced source-site labeling budgets**.

At 25% of the Fundidora training labels, frozen DINOv2 retains:

- balanced accuracy 0.804 ± 0.004;
- tree F1 0.847 ± 0.018;
- point IoU 0.734 ± 0.028.

At 50%, performance increases to:

- balanced accuracy 0.824 ± 0.021;
- tree F1 0.874 ± 0.012;
- point IoU 0.776 ± 0.019.

U-Net and SegFormer-B0 largely collapse toward non-tree predictions at their Fundidora-selected thresholds. I-JEPA transfers useful signal, especially at the 50% label budget, but remains substantially below DINOv2 and has much larger cross-seed variance.

## Supported claim

> On the Fundidora → La Pastora cross-site UTC benchmark, frozen DINOv2 features provide substantially stronger and more stable label-efficient transfer than U-Net, SegFormer-B0, and frozen I-JEPA. This advantage persists with only 25% and 50% of the Fundidora training labels.

This is a **cross-site point-classification claim**. It is not a claim about dense canopy-area estimation.

## Single-run full-label reference

A previously completed full-label DINOv2 zero-shot run on the same 240 La Pastora points achieved:

- balanced accuracy: 0.818
- precision: 0.904
- recall: 0.815
- F1: 0.857
- point IoU: 0.750
- point prevalence bias: -6.67 pp

The 50% multi-seed mean is numerically above this single full-label run. That should **not** be interpreted as evidence that 50% labels outperform 100%, because the full-label condition does not yet have a comparable multi-seed transfer baseline.

## Limitations

1. The human validation target is point-based rather than a dense canopy mask.
2. Point IoU is not directly comparable with dense pixel IoU reported in the Fundidora segmentation benchmark.
3. The reference prevalence of 67.5% is the sampled-point prevalence, not a UTC area estimate.
4. The current models are legacy-v1 models trained from per-patch percentile-stretched RGB; a final publication-quality v2 benchmark should use raw/consistent preprocessing at both sites.
5. La Pastora is one external site. Additional geographically distinct sites would be required to claim broad citywide or regional generalization.
6. I-JEPA upstream is CC BY-NC 4.0 and remains research/prototype-only unless separately licensed.

## Reproducibility

Primary result file:

`outputs/utc/la_pastora_2007/cross_site/multimodel_multiseed/multimodel_multiseed_zero_shot_summary.json`

Human point labels:

`data/labels/la_pastora_2007_points/la_pastora_2007_point_labels_clean.json`

Cross-site runner:

`scripts/run_utc_cross_site_multimodel_multiseed.py`

Per-model point inference:

`scripts/evaluate_utc_cross_site_points.py`

## M3b decision

**M3b acceptance criterion: satisfied.**

The second-site experiment demonstrates that the strongest Fundidora representation-learning finding—DINOv2 label efficiency—survives an independent site shift under the locked-source-threshold protocol.

Future work should move to:
- raw-consistent v2 preprocessing;
- additional external sites;
- dense-mask or block-level UTC area validation;
- prediction/error visualizations;
- adaptation experiments using La Pastora train/validation labels while preserving an untouched test region.
