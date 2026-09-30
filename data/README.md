# Data

Do not commit large raw imagery or proprietary geospatial datasets to this repository.

Recommended structure when running locally:

```text
data/
  raw/
    sentinel2/
    aerial/
  interim/
    clipped/
    aligned/
    tiles/
  processed/
    masks/
    embeddings/
    features/
  labels/
```

## Phase 1

Use open Sentinel-2 imagery for neighborhood and park-scale experiments.

Useful bands include:
- B02 blue;
- B03 green;
- B04 red;
- B08 near infrared.

NDVI uses:

```text
NDVI = (NIR - RED) / (NIR + RED)
```

## Data governance

For every scene preserve:
- source;
- license;
- acquisition timestamp;
- cloud metadata;
- spatial reference system;
- processing history.

## Local pilot

A future Monterrey pilot should define a small set of representative AOIs rather than attempting the entire metropolitan area immediately. Include parks, dense urban blocks, industrial areas, and mixed vegetation conditions.
