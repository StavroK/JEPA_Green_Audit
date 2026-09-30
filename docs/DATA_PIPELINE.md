# Sentinel-2 Baseline Pipeline

## Pilot AOI

The first reproducible pilot is a bounding area around **Parque Fundidora,
Monterrey, Nuevo León, Mexico**.

The repository AOI is intentionally documented as a **pilot bounding box**, not
an official property or legal boundary.

File: `config/aoi_fundidora.geojson`

## Why Fundidora

It provides a compact mixed urban environment with substantial green space,
built surfaces, water, paths and industrial structures. That is useful for
testing both vegetation segmentation and false-positive behavior.

## Source

Catalog:

`https://earth-search.aws.element84.com/v1`

Collection:

`sentinel-2-l2a`

Earth Search provides public STAC metadata and Sentinel-2 Level-2A
Cloud-Optimized GeoTIFF assets. The MVP uses:

| Earth Search asset | Sentinel-2 band | Purpose | Native resolution |
|---|---|---|---:|
| `red` | B04 | Red reflectance | 10 m |
| `nir` | B08 | Near-infrared reflectance | 10 m |
| `scl` | SCL | Scene classification / quality | 20 m |

The SCL layer is resampled with nearest-neighbor onto the 10 m analysis grid.

## Comparable dates

The baseline searches the same seasonal window in consecutive years:

- 2025-08-01 through 2025-09-29
- 2026-08-01 through 2026-09-29

This does not eliminate phenological or rainfall differences, but it is more
defensible than comparing unrelated seasons.

## Scene selection

A scene can have low cloud cover overall while still being cloudy over a small
AOI. Therefore the workflow:

1. queries scenes below a configurable scene-cloud threshold;
2. reads only the Fundidora AOI;
3. evaluates the local SCL usable-pixel fraction;
4. selects the candidate with the highest local usable fraction;
5. uses scene cloud cover as a tie-break.

Default candidate count: 5.

## Quality mask

The first MVP excludes SCL classes:

- 0 — no data;
- 1 — saturated / defective;
- 3 — cloud shadow;
- 7 — unclassified / low-probability cloud;
- 8 — medium-probability cloud;
- 9 — high-probability cloud;
- 10 — thin cirrus;
- 11 — snow / ice.

Classes 2, 4, 5 and 6 remain valid for analysis because the NDVI threshold,
rather than SCL's semantic vegetation class, defines the vegetation mask.

## NDVI and coverage

```text
NDVI = (NIR - RED) / (NIR + RED)
```

For the initial baseline:

```text
vegetation = valid_pixel AND NDVI >= 0.30
coverage = vegetation_pixels / usable_pixels
```

The threshold is an explicit experiment parameter, not a universal biological
definition.

## Reproduce

```bash
pip install -r requirements.txt
pip install -e .
python scripts/fetch_sentinel2.py
```

Output:

`outputs/fundidora_sentinel2_baseline.json`

The JSON preserves:
- STAC item ID;
- acquisition datetime;
- platform;
- scene cloud cover;
- MGRS tile when available;
- source asset URLs;
- AOI;
- CRS and grid transform;
- usable-pixel fraction;
- NDVI statistics;
- vegetation coverage;
- before/after change;
- scene-selection candidates.

## What is deliberately not committed

Large GeoTIFF, JP2, NumPy and derived raster files remain outside Git. The
project should commit code, configuration and small provenance/summary JSON,
not duplicate public imagery archives.
