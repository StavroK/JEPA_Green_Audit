# M3d — Niños Héroes third-site validation setup

## Source QC

The candidate AOI is covered by INEGI orthophoto archive `889463305439_b.zip` (sheet G14C26A1).

Ingestion metadata:
- acquisition year: 2007
- spatial resolution: 1.0 m
- RGB bands: 3
- dtype: uint8
- CRS: EPSG:6369
- crop shape: 616 × 617 × 3

The preview contains a heterogeneous urban park scene with visible tree canopy, grass/open fields, roads, roofs/buildings, paved areas and strong shadow structure. This makes it suitable as a third cross-site point-validation target.

## Validation protocol

Use the same model-blind deterministic design as La Pastora:

- 15 × 16 grid
- one jittered point per cell
- 240 total points
- seed 42
- labels created before model evaluation
- tree/non-tree/uncertain primary labels
- detail categories retained for error analysis
- no target threshold tuning or fitting

## Commands

Generate point plan:

```bash
python scripts/prepare_utc_point_validation.py \
  --image outputs/utc/ninos_heroes_2007/ninos_heroes_rgb_preview.png \
  --output data/labels/ninos_heroes_2007_points/ninos_heroes_2007_point_plan.json \
  --rows 15 \
  --cols 16 \
  --seed 42 \
  --site ninos_heroes \
  --date-label 2007
```

Then open `tools/utc_canopy_validation_point_labeler.html` in a browser and load:
1. `outputs/utc/ninos_heroes_2007/ninos_heroes_rgb_preview.png`
2. `data/labels/ninos_heroes_2007_points/ninos_heroes_2007_point_plan.json`

Export the reviewed JSON as `ninos_heroes_2007_point_labels.json`.

Validate:

```bash
python scripts/validate_utc_point_labels.py \
  --input data/labels/ninos_heroes_2007_points/ninos_heroes_2007_point_labels.json
```

Do not run the Fundidora-trained models on Niños Héroes until the point labels are complete. This preserves model-blind target validation.
