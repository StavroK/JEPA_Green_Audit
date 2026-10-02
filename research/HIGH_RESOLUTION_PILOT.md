# M3b — High-resolution RGB spatial-resolution sensitivity pilot

## Question

Does the weak/unstable frozen I-JEPA result at Parque Fundidora primarily reflect
the representation itself, or the coarse and heterogeneous 10 m Sentinel-2 RGB
input?

This experiment adds a second environment and a higher-resolution RGB source.
It does **not** replace or overwrite the Fundidora benchmark.

## Sites

### Site A — Parque Fundidora

Role: difficult mixed urban/vegetation environment.

The existing Sentinel-2 benchmark remains frozen for traceability.

INEGI's 1:10,000 Monterrey sheet layout places the Fundidora area within
**G14C26A3** (1 m-class orthophoto family / historical coverage).

### Site B — La Pastora

Role: denser green urban environment.

The repository AOI is `config/aoi_la_pastora.geojson`. It is intentionally an
approximate research bounding box and is **not** an official park or protected
area boundary.

The La Pastora area falls within INEGI sheet **G14C26A4**. Public metadata and
published use of the Monterrey orthophoto series confirm historical orthophoto
coverage for the G14C26 sub-sheets. INEGI also currently exposes 2024-derived
1.5 m elevation products for G14C26A4, demonstrating current high-resolution
photogrammetric source coverage, but that elevation product is not itself RGB.

## Data-source constraint

For the open GitHub benchmark, use only imagery whose redistribution/use terms
are compatible with the experiment.

The historical INEGI orthophoto is appropriate for a **spatial-resolution /
segmentation** experiment, but it must not be compared against 2025/2026
Sentinel-2 as if both images describe the same vegetation condition.

Treat imagery dates as separate observations.

## Experiment matrix

| Site | Source | Approx. GSD | Spectral channels | Purpose |
|---|---|---:|---|---|
| Fundidora | Sentinel-2 L2A | 10 m | RGB + NIR | Current spectral baseline |
| Fundidora | INEGI orthophoto | ~1 m | RGB | High-resolution visual benchmark |
| La Pastora | Sentinel-2 L2A | 10 m | RGB + NIR | Dense-green spectral baseline |
| La Pastora | INEGI orthophoto | ~1 m | RGB | Dense-green high-resolution visual benchmark |

## Models

Run the same geographically separated evaluation for:

1. NDVI (only where NIR is available)
2. frozen I-JEPA RGB
3. supervised ImageNet RGB baseline
4. non-JEPA self-supervised RGB baseline
5. fusion where inputs are legitimately co-registered and contemporaneous

Do **not** fuse a historical orthophoto with a modern NDVI acquisition as though
the two were one observation.

## Labels

Use independent RGB-only human visible-cover labels.

At higher resolution, define review cells in ground units rather than forcing the
same 8x8 count. The target cell should be large enough to contain meaningful
canopy/grass context while avoiding the 10 m mixed-pixel problem.

Recommended first pass: approximately 20–30 m ground cells, then sensitivity at
10 m and 50 m if time permits.

## Primary comparisons

1. Same site, Sentinel RGB vs high-resolution RGB:
   isolates spatial-resolution effect on RGB representation quality.
2. Fundidora vs La Pastora at the same source/resolution:
   tests environmental heterogeneity.
3. NDVI vs RGB representation at Sentinel resolution:
   tests spectral advantage.
4. RGB supervised vs RGB SSL vs I-JEPA at high resolution:
   tests representation family independently of NIR.

## Interpretation guardrails

- Do not select the greener site because it makes JEPA look better.
- Keep Fundidora results in the report.
- Report class prevalence and trivial baselines for every geographic fold.
- Report imagery date and ground sample distance for every result.
- A historical 1 m result is evidence about spatial detail / representation,
  not current vegetation condition.
- Visible-cover labels are not biological tree-health diagnoses.
