# Human-calibrated RGB canopy classifier

The first RGB heuristic recovered only a small subset of visible canopy and
also highlighted some grass/other vegetation. It is useful as a diagnostic, but
not sufficient for UTC.

The next stage uses sparse human point labels instead of full crown polygons.

## Labeling protocol

Open `tools/utc_canopy_point_labeler.html` and load the original
`fundidora_2007_rgb_preview.png`.

Create at least **100 tree** and **100 non-tree** clicks, distributed across the
whole AOI.

Tree examples should include:
- dense crowns;
- isolated crowns;
- bright and dark crowns;
- crowns next to roads/buildings;
- crown edges.

Non-tree examples should include:
- grass;
- bare soil;
- asphalt;
- roofs;
- shadow without visible crown structure;
- water;
- landscaped low vegetation where distinguishable.

Use **Uncertain** rather than forcing ambiguous pixels.

## Training

```bash
python scripts/train_utc_rgb_classifier.py \
  --image outputs/utc/fundidora_2007/fundidora_2007_rgb_crop.tif \
  --labels outputs/utc/fundidora_2007/utc_canopy_point_labels.json \
  --output-probability outputs/utc/fundidora_2007/fundidora_2007_canopy_probability.tif \
  --output-mask outputs/utc/fundidora_2007/fundidora_2007_canopy_rf_candidate.tif \
  --output-preview outputs/utc/fundidora_2007/fundidora_2007_canopy_rf_overlay.png \
  --output-json outputs/utc/fundidora_2007/fundidora_2007_canopy_rf_metrics.json
```

The script reports cross-validated balanced accuracy on the clicked samples, but
this is **not independent spatial validation**. A separate spatial validation
sample remains required before the mask can be accepted for UTC reporting.
