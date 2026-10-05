"""Aggregate Urban Tree Canopy (UTC) metrics by management zones.

This converts a validated binary canopy raster into actionable area-level metrics
for municipalities, parks, campuses, industrial sites, neighborhoods, or other
user-defined polygons.

Inputs
------
--mask
    Projected binary canopy GeoTIFF (1=tree canopy, 0=non-canopy, optional nodata).
--zones
    Polygon GeoJSON containing management/reporting areas.
--target-utc
    Optional user-defined UTC target percentage. When supplied, the output
    includes UTC deficit in percentage points and canopy-area deficit required
    to reach that target, assuming the target is physically achievable.

Outputs
-------
- GeoJSON with per-zone UTC metrics and optional deficit metrics.
- CSV table for dashboards/resource planning.

Important
---------
A low UTC value is a prioritization signal, not proof that an area is suitable
for planting. Planting feasibility requires separate land-use, ownership,
infrastructure, utility, soil, safety, and field-validation layers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


def zone_metrics(
    canopy: np.ndarray,
    valid: np.ndarray,
    zone_mask: np.ndarray,
    pixel_area_m2: float,
    target_utc: float | None = None,
) -> dict:
    inside = zone_mask & valid
    valid_pixels = int(inside.sum())
    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "analysis_area_ha": 0.0,
            "canopy_area_ha": 0.0,
            "utc_pct": None,
        }

    canopy_pixels = int(np.sum((canopy == 1) & inside))
    analysis_area_m2 = valid_pixels * pixel_area_m2
    canopy_area_m2 = canopy_pixels * pixel_area_m2
    utc_pct = 100.0 * canopy_area_m2 / analysis_area_m2

    out = {
        "valid_pixels": valid_pixels,
        "canopy_pixels": canopy_pixels,
        "analysis_area_ha": analysis_area_m2 / 10000.0,
        "canopy_area_ha": canopy_area_m2 / 10000.0,
        "utc_pct": utc_pct,
    }

    if target_utc is not None:
        deficit_pp = max(0.0, target_utc - utc_pct)
        required_canopy_m2 = max(
            0.0,
            (target_utc / 100.0) * analysis_area_m2 - canopy_area_m2,
        )
        out["target_utc_pct"] = target_utc
        out["utc_deficit_pp"] = deficit_pp
        out["canopy_deficit_ha"] = required_canopy_m2 / 10000.0

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--zones", type=Path, required=True)
    p.add_argument("--id-field", default="id")
    p.add_argument("--name-field", default="name")
    p.add_argument("--target-utc", type=float)
    p.add_argument("--output-geojson", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()

    zones = json.loads(args.zones.read_text(encoding="utf-8"))
    if zones.get("type") != "FeatureCollection":
        raise ValueError("Zones must be a GeoJSON FeatureCollection")

    with rasterio.open(args.mask) as src:
        canopy = src.read(1)
        if src.crs is None or not src.crs.is_projected:
            raise ValueError("UTC aggregation requires a projected canopy raster")
        nodata = src.nodata
        valid = np.isfinite(canopy) if nodata is None else (
            np.isfinite(canopy) & (canopy != nodata)
        )
        values = canopy[valid]
        bad = values[(values != 0) & (values != 1)]
        if bad.size:
            raise ValueError("Canopy raster must contain only 0/1 plus optional nodata")

        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        raster_crs = src.crs.to_string()

        out_features = []
        rows = []

        for idx, feature in enumerate(zones["features"]):
            props = dict(feature.get("properties") or {})
            zone_id = props.get(args.id_field, idx)
            zone_name = props.get(args.name_field, str(zone_id))

            geom = feature["geometry"]
            # GeoJSON is conventionally WGS84 unless a source CRS is explicitly
            # managed externally. Transform into the raster CRS for masking.
            projected_geom = transform_geom("EPSG:4326", src.crs, geom)

            mask_inside = geometry_mask(
                [projected_geom],
                out_shape=canopy.shape,
                transform=src.transform,
                invert=True,
            )
            metrics = zone_metrics(
                canopy,
                valid,
                mask_inside,
                pixel_area_m2,
                target_utc=args.target_utc,
            )

            enriched = dict(props)
            enriched.update({
                "zone_id": zone_id,
                "zone_name": zone_name,
                **{
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                },
            })

            out_features.append({
                "type": "Feature",
                "properties": enriched,
                "geometry": geom,
            })
            rows.append(enriched)

    output = {
        "type": "FeatureCollection",
        "properties": {
            "metric": "Urban Tree Canopy",
            "source_mask": str(args.mask),
            "raster_crs": raster_crs,
            "target_utc_pct": args.target_utc,
            "warning": (
                "Priority/deficit metrics indicate where canopy is low; they do "
                "not establish planting feasibility without additional constraints."
            ),
        },
        "features": out_features,
    }

    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_geojson.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output_geojson}")
    print(f"Wrote {args.output_csv}")
    print(f"Zones processed: {len(rows)}")


if __name__ == "__main__":
    main()
