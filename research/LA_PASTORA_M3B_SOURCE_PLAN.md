# M3b La Pastora source validation

## Site role

La Pastora is the dense-green replication site for M3b. Fundidora remains the
heterogeneous mixed-urban site.

The configured AOI is intentionally an approximate research polygon rather than
an official park/protected-area boundary.

## AOI

Configured file:

`config/aoi_la_pastora.geojson`

WGS84 bbox:

- west: -100.2545
- south: 25.6615
- east: -100.2405
- north: 25.6745

## INEGI sheet validation

INEGI catalog product UPC **794551182970** identifies sheet **G14C26A4**:

- product: Modelo digital de elevación de tipo terreno
- resolution: 1.5 m
- scale: 1:10 000
- edition: 2026
- temporal coverage: 2024
- west/east: 100°16'43.52"W to 100°13'16.63"W
- south/north: 25°37'27.32"N to 25°41'17.71"N

The configured La Pastora AOI lies fully inside this extent.

Run:

```bash
python scripts/validate_la_pastora_sources.py
```

## Sentinel-2 replication

Use the same acquisition windows and selection policy as Fundidora first, so the
sites are directly comparable:

```bash
python scripts/fetch_sentinel2.py \
  --aoi config/aoi_la_pastora.geojson \
  --output outputs/la_pastora_sentinel2_baseline.json
```

The script ranks candidate scenes by usable SCL fraction over the local AOI and
uses scene cloud cover only as a tie-break.

Do not assume the selected dates will exactly match Fundidora. Record the chosen
scene IDs and acquisition dates and interpret site comparisons with seasonality
in mind.

## High-resolution RGB status

A verified downloadable G14C26A4 RGB orthophoto has **not yet been pinned**.
Do not substitute the 2007 G14C26A3 Fundidora orthophoto or infer a direct file
URL from the sheet key.

If a historical G14C26A4 orthophoto is found, use it only for a same-image
representation/resolution experiment with labels created from that historical
RGB itself. Do not compare it directly against 2025/2026 labels as current truth.

## Structural source status

The G14C26A4 MDT catalog record is verified. The matching direct download,
checksum, and matching MDS product still need to be pinned before structural
ingestion.

Once both MDS and MDT are verified:

1. add both URLs and SHA256 values to `data/sources/inegi_products.json`;
2. download reproducibly with `scripts/download_inegi_products.py`;
3. derive nDSM using the existing elevation ingestion workflow;
4. aggregate structural features to the La Pastora review grid.

## Gate to labeling

Do not start manual labeling until:

1. Sentinel 2025 and 2026 scenes are selected and provenance is saved;
2. RGB review images are exported;
3. the AOI is visually inspected for dense-green character and major edge
   contamination;
4. the labeling grid/cell size is chosen from the actual imagery.
