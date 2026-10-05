# Urban Tree Canopy (UTC) measurement plan — Monterrey

## Product goal

The project is being re-centered around a municipal-scale outcome:

> Measure and track urban tree canopy across Monterrey with auditable UTC %,
> canopy area, canopy gain/loss, and related vegetation indicators.

UTC is defined as the horizontal area covered by tree crowns divided by the
valid analysis area.

```text
UTC % = tree-canopy area / valid analysis area × 100
```

UTC is not tree count, generic vegetation coverage, or NDVI coverage.

## Measurement architecture

The project should separate four evidence layers:

1. **Canopy measurement layer — primary UTC source**
   - validated high-resolution RGB/aerial orthophoto segmentation;
   - target spatial resolution approximately 0.5–1.5 m where feasible;
   - outputs a binary/probabilistic tree-canopy mask.

2. **Structural confirmation layer**
   - nDSM / canopy-height evidence from MDS - MDT;
   - used to distinguish elevated crowns from grass/low vegetation;
   - must not be used alone as UTC because buildings are also elevated.

3. **Frequent spectral monitoring layer**
   - Sentinel-2 NDVI/EVI/SAVI and similar indicators;
   - used for vegetation condition/change screening;
   - not reported as UTC.

4. **Representation/change layer**
   - I-JEPA / DINO / supervised encoders;
   - useful for change detection and transfer experiments;
   - not the authoritative UTC source.

## Current source reality

### Fundidora

Available and pinned:
- 2007 INEGI RGB orthophoto, 1 m, sheet G14C26A3;
- 2024 INEGI MDS 1.5 m;
- 2024 INEGI MDT 1.5 m;
- derived nDSM workflow.

These sources are **not temporally aligned**. Therefore they must not be fused
to claim a current or historical UTC measurement for one date.

The 2007 RGB can support a historical canopy-segmentation benchmark.
The 2024 nDSM can support structural experiments.
A same-period high-resolution RGB source is still required for a defensible
2024 UTC measurement.

### La Pastora

Current Sentinel observations support vegetation monitoring but not defensible
individual crown/UTC measurement at 10 m.

The verified G14C26A4 MDT catalog record exists, but a matched MDS and
high-resolution RGB source still need to be pinned.

## First UTC pilot

The first defensible pilot should use one AOI with:

- high-resolution RGB for date T;
- canopy mask created from that RGB and independently reviewed;
- optional same-period nDSM for structural confirmation;
- fixed projected AOI;
- UTC computed with `scripts/compute_utc.py`.

Required outputs:

- UTC %;
- canopy area (ha);
- total valid analysis area (ha);
- canopy mask;
- imagery/source provenance;
- validation notes;
- uncertainty / limitations.

## Temporal tracking

For dates T1 and T2, report:

- UTC % at T1;
- UTC % at T2;
- UTC change in percentage points;
- canopy gain area (ha);
- canopy loss area (ha).

Do not attribute gain/loss causally without independent evidence.

## Scale-up path for Monterrey

1. validate one high-resolution pilot AOI;
2. replicate on several heterogeneous urban neighborhoods/parks;
3. calibrate a tree-canopy segmentation model;
4. tile metropolitan imagery;
5. aggregate UTC to neighborhoods, municipalities, parks, census units, and
   other reporting geographies;
6. use Sentinel-2 as a frequent change-screening layer between high-resolution
   canopy refreshes.
