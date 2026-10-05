# Semi-automatic UTC canopy candidate workflow

## Purpose

Manual crown tracing is not scalable for Fundidora or Monterrey. The first
automation step is therefore an **RGB-derived candidate mask** that reduces
human annotation effort.

The candidate is deliberately *not* an authoritative UTC layer.

## Why it is only a candidate

A 1 m RGB orthophoto has enough spatial detail to see many crowns, but RGB
alone cannot reliably distinguish all of:

- tree canopy;
- grass and shrubs;
- shadows;
- green roofs/materials;
- mixed crown/ground pixels.

The current heuristic combines normalized Excess Green, green-channel dominance,
brightness, local texture, and small-region removal. It is intended to propose
likely crown pixels for review.

## Validation path

1. generate the candidate mask and overlay;
2. inspect false positives and false negatives across built, park, road, and
   mixed-edge areas;
3. create human-reviewed positive/negative samples;
4. calibrate or train a segmentation/classification model;
5. reserve independent validation samples;
6. only after validation rename the accepted binary layer a UTC canopy mask;
7. compute UTC with `scripts/compute_utc.py`.

## Command

```bash
python scripts/propose_utc_canopy_mask.py \
  --image outputs/utc/fundidora_2007/fundidora_2007_rgb_crop.tif \
  --output-mask outputs/utc/fundidora_2007/fundidora_2007_canopy_candidate.tif \
  --output-preview outputs/utc/fundidora_2007/fundidora_2007_canopy_candidate_overlay.png \
  --output-json outputs/utc/fundidora_2007/fundidora_2007_canopy_candidate.json
```

The JSON reports `candidate_fraction`, not UTC.
