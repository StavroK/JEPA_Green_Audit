"""Validate whether a GeoJSON AOI lies inside a known INEGI sheet extent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Verified from INEGI catalog UPC 794551182970 (G14C26A4 MDT).
G14C26A4_BOUNDS = {
    "west": -100.2787555556,   # 100°16'43.52"W
    "east": -100.2212861111,   # 100°13'16.63"W
    "south": 25.6242555556,    # 25°37'27.32"N
    "north": 25.6882527778,    # 25°41'17.71"N
}


def load_bbox(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError("Expected a single-feature FeatureCollection")
        geometry = features[0]["geometry"]
    elif payload.get("type") == "Feature":
        geometry = payload["geometry"]
    else:
        raise ValueError("Expected GeoJSON Feature or FeatureCollection")

    coords = geometry["coordinates"][0]
    xs = [float(p[0]) for p in coords]
    ys = [float(p[1]) for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def contains(sheet: dict[str, float], bbox: list[float]) -> bool:
    west, south, east, north = bbox
    return (
        west >= sheet["west"]
        and east <= sheet["east"]
        and south >= sheet["south"]
        and north <= sheet["north"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aoi",
        type=Path,
        default=Path("config/aoi_la_pastora.geojson"),
    )
    args = parser.parse_args()

    bbox = load_bbox(args.aoi)
    ok = contains(G14C26A4_BOUNDS, bbox)

    print(f"AOI bbox: {bbox}")
    print(f"G14C26A4 bounds: {G14C26A4_BOUNDS}")
    print(f"AOI fully inside G14C26A4: {ok}")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
