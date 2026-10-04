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
            +--------------------+--------------------+-------------------+
            |                    |                    |                   |
            v                    v                    v                   v
   JEPA representation       Spectral path      Semantic path      Structural path
   ├─ frozen encoder         ├─ NDVI            ├─ tree            ├─ nDSM / height
   ├─ embedding change      ├─ EVI/SAVI        ├─ shrub/grass     └─ later field data
   └─ anomaly features      └─ vegetation      ├─ developed
                                                  ├─ bare
                                                  └─ water
            |                    |                    |                   |
            +--------------------+--------------------+-------------------+
                                      |
                                      v
                              Evidence fusion
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
               ├─ semantic land-cover composition
               ├─ tree / other-vegetation separation
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


## Semantic perception branch

M3 showed that I-JEPA should not be treated as the sole semantic perception
model. The operational architecture therefore separates four evidence types:

- **semantic:** what is present (tree, grass/shrub, developed, bare, water);
- **spectral:** vegetation-related physical signal (NDVI and related indices);
- **structural:** vertical structure / height (nDSM where available);
- **representation:** learned visual similarity/change (I-JEPA and baselines).

The first semantic baseline selected for Sentinel-2 is **SatlasPretrain**, using
its Sentinel-2 model family and land-cover taxonomy. SatlasPretrain explicitly
includes a land-cover segmentation task with water, developed, tree, shrub,
grass, crop, bare and related classes, plus a tree-cover regression task.

For high-resolution orthophotos, object-level segmentation (for example SAMGeo
or another aerial semantic model) may be evaluated separately. SAM-style
segmentation is not treated as a substitute for Sentinel-2 land-cover semantics
because Sentinel's 10 m pixels do not support reliable individual-tree crown
delineation.

Semantic outputs are evidence, not ground truth. They require local validation
before being used in audit indicators.
