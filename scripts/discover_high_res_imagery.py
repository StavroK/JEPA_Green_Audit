"""Discover free/open higher-resolution imagery for an AOI.

This utility probes two independent sources:

1. INEGI's legacy orthophoto WMS endpoint, if it is still reachable.
2. OpenAerialMap's public STAC catalog (CC BY 4.0 imagery).

It only discovers metadata. It does not download or redistribute imagery.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DEFAULT_AOI = Path("config/aoi_la_pastora.geojson")
DEFAULT_OUTPUT = Path("outputs/high_res_imagery_discovery.json")

INEGI_WMS_CANDIDATES = (
    "https://antares.inegi.org.mx/cgi-bin/map4/mapserv_orto",
    "http://antares.inegi.org.mx/cgi-bin/map4/mapserv_orto",
)
OAM_STAC_SEARCH = "https://api.imagery.hotosm.org/stac/search"


def load_geometry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        if not payload.get("features"):
            raise ValueError("AOI FeatureCollection is empty")
        return payload["features"][0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def bbox_from_geometry(geometry: dict[str, Any]) -> list[float]:
    coords = geometry.get("coordinates")
    if not coords:
        raise ValueError("AOI geometry has no coordinates")

    points: list[tuple[float, float]] = []

    def walk(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, list):
            for child in value:
                walk(child)

    walk(coords)
    if not points:
        raise ValueError("Could not derive AOI bbox")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 20,
) -> Any:
    body = None
    headers = {"User-Agent": "JEPA-Green-Audit/1.0"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = 15) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "JEPA-Green-Audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def probe_inegi_wms() -> dict[str, Any]:
    attempts = []

    for base in INEGI_WMS_CANDIDATES:
        query = urllib.parse.urlencode(
            {
                "SERVICE": "WMS",
                "REQUEST": "GetCapabilities",
                "VERSION": "1.1.1",
            }
        )
        url = f"{base}?{query}"

        try:
            content = fetch_bytes(url)
            root = ET.fromstring(content)
            layers = []

            for layer in root.findall(".//Layer"):
                name_el = layer.find("Name")
                title_el = layer.find("Title")
                name = name_el.text.strip() if name_el is not None and name_el.text else None
                title = title_el.text.strip() if title_el is not None and title_el.text else None
                if name or title:
                    layers.append({"name": name, "title": title})

            return {
                "status": "reachable",
                "endpoint": base,
                "capabilities_url": url,
                "layer_count": len(layers),
                "layers": layers[:100],
            }
        except Exception as exc:  # discovery should report, not crash
            attempts.append(
                {
                    "endpoint": base,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "status": "unreachable",
        "attempts": attempts,
        "note": (
            "The legacy INEGI WMS may have been retired, blocked, or require "
            "interactive/manual access through the INEGI map catalog."
        ),
    }


def search_openaerialmap(bbox: list[float], limit: int = 100) -> dict[str, Any]:
    payload = {
        "collections": ["openaerialmap"],
        "bbox": bbox,
        "limit": limit,
    }

    try:
        result = http_json(
            OAM_STAC_SEARCH,
            method="POST",
            payload=payload,
            timeout=30,
        )
    except Exception as exc:
        return {
            "status": "error",
            "endpoint": OAM_STAC_SEARCH,
            "error": f"{type(exc).__name__}: {exc}",
        }

    features = result.get("features", [])
    items = []

    for feature in features:
        props = feature.get("properties", {})
        assets = feature.get("assets", {})

        visual = assets.get("visual", {})
        data_asset = assets.get("data", {})
        image_asset = assets.get("image", {})

        href = (
            visual.get("href")
            or data_asset.get("href")
            or image_asset.get("href")
        )

        items.append(
            {
                "id": feature.get("id"),
                "datetime": props.get("datetime")
                or props.get("start_datetime")
                or props.get("end_datetime"),
                "gsd_m": props.get("gsd"),
                "platform": props.get("platform"),
                "provider": props.get("provider"),
                "title": props.get("title"),
                "bbox": feature.get("bbox"),
                "asset_href": href,
                "asset_keys": sorted(assets.keys()),
                "license": props.get("license"),
            }
        )

    return {
        "status": "ok",
        "endpoint": OAM_STAC_SEARCH,
        "count": len(items),
        "items": items,
        "license_note": (
            "OpenAerialMap documentation states imagery in the Open Imagery "
            "Network is CC BY 4.0; verify item-level metadata before reuse."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oam-limit", type=int, default=100)
    args = parser.parse_args()

    geometry = load_geometry(args.aoi)
    bbox = bbox_from_geometry(geometry)

    print(
        "AOI bbox: "
        f"{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}"
    )

    print("\nProbing INEGI orthophoto WMS...")
    inegi = probe_inegi_wms()
    print(f"INEGI WMS: {inegi['status']}")
    if inegi["status"] == "reachable":
        print(
            f"  endpoint={inegi['endpoint']} | "
            f"layers={inegi['layer_count']}"
        )
    else:
        for attempt in inegi.get("attempts", []):
            print(f"  {attempt['endpoint']}: {attempt['error']}")

    print("\nSearching OpenAerialMap...")
    oam = search_openaerialmap(bbox, limit=args.oam_limit)
    print(f"OpenAerialMap: {oam['status']}")
    if oam["status"] == "ok":
        print(f"  matches={oam['count']}")
        for item in oam["items"][:20]:
            print(
                "  - "
                f"{item['id']} | date={item['datetime']} | "
                f"gsd={item['gsd_m']} m | "
                f"asset={'yes' if item['asset_href'] else 'no'}"
            )
    else:
        print(f"  {oam.get('error')}")

    report = {
        "aoi": str(args.aoi),
        "bbox_wgs84": bbox,
        "inegi_wms": inegi,
        "openaerialmap": oam,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
