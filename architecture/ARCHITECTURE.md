# Architecture

## Goal

Create a low-cost, reproducible urban vegetation monitoring system that separates **representation learning**, **remote-sensing analytics**, and **business interpretation**.

## Logical architecture

```text
Data Sources
  ├─ Sentinel-2 multispectral imagery
  ├─ aerial orthophotos / drone imagery
  └─ optional municipal boundaries / field observations
            |
            v
Ingestion & Preprocessing
  ├─ reprojection / cloud masking
  ├─ AOI clipping
  ├─ temporal alignment
  └─ image tiling
            |
            +-------------------------+
            |                         |
            v                         v
   JEPA representation path       Spectral path
   ├─ context encoder            ├─ NDVI
   ├─ target encoder             ├─ EVI/SAVI (later)
   └─ embedding extraction       └─ simple vegetation masks
            |                         |
            +------------+------------+
                         |
                         v
                 Feature fusion
                         |
              +----------+----------+
              |                     |
              v                     v
       segmentation head      change detector
              |                     |
              +----------+----------+
                         |
                         v
               Geo indicators layer
               ├─ vegetation coverage
               ├─ gain / loss
               ├─ embedding anomaly
               └─ confidence / provenance
                         |
                         v
              Dashboard / API / reports
```

## MVP boundaries

The first browser release uses sample values and simulated spatial cells so that the user experience can be demonstrated on GitHub Pages with no cloud bill.

The analytical package provides real formulas and interfaces for:

- NDVI computation;
- vegetation masks;
- coverage percentage;
- before/after change;
- cosine-distance embedding change.

It does **not** pretend to execute a heavy JEPA model in the browser.

## Future production architecture

A production implementation can separate responsibilities:

1. object storage for source imagery;
2. batch geospatial preprocessing;
3. GPU inference for JEPA embeddings;
4. geospatial database for zones and observations;
5. API serving computed indicators;
6. browser dashboard for audit and exploration.

## Design principles

- **Auditable:** every score should trace to imagery date, AOI, model version, and formula.
- **Model-agnostic:** JEPA is the primary research path but baselines remain first-class.
- **Low-cost first:** the public demo must not require a paid backend.
- **Human validation:** high-risk findings become candidates for field inspection, not automatic factual claims.
- **Temporal:** change through time is more valuable than a one-off “green/not green” map.
