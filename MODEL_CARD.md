# Model Card — JEPA Green Audit

## Intended use

Research and prototype decision support for urban vegetation monitoring from satellite or aerial imagery.

## Intended outputs

- vegetation coverage estimate;
- vegetation segmentation mask;
- temporal gain/loss estimate;
- learned-representation change score;
- candidate zones for human review.

## Non-goals

The system is **not** intended to:

- diagnose tree disease from RGB imagery alone;
- replace certified arborists or environmental professionals;
- infer causality from visual change;
- make regulatory compliance determinations autonomously;
- identify individuals or perform surveillance.

## Terminology

Use **Vegetation Condition Signal** instead of “tree health” unless the model is trained and validated against appropriate biological ground truth.

Use **estimated vegetation loss** instead of “trees removed” unless imagery resolution and validation support that conclusion.

## Risks

### Resolution
Sentinel-2 spatial resolution is too coarse for reliable individual-tree auditing in many urban contexts.

### Seasonality
Dry/wet seasons can look like degradation or recovery.

### Atmospheric effects
Clouds, haze, shadows, and sensor differences may create false change.

### Domain shift
A model adapted to one city, sensor, or season may not transfer without validation.

### Representation ambiguity
Embedding distance signals change but does not by itself explain what changed.

## Human oversight

High-change areas should be reviewed against:
- source imagery;
- acquisition dates;
- cloud/quality metadata;
- conventional vegetation indices;
- available municipal or field observations.

## Provenance requirements

Each production inference should preserve:
- area-of-interest identifier;
- acquisition date;
- imagery source;
- preprocessing version;
- encoder/model version;
- downstream head version;
- thresholds;
- output confidence/quality flags.
