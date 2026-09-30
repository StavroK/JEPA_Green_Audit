"""Sentinel-2 Level-2A ingestion through the public Earth Search STAC API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import rasterio
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# SCL classes excluded for the MVP. This follows the conservative invalid-class
# set used by Copernicus' Sentinel-2 mosaic workflow.
INVALID_SCL = {0, 1, 3, 7, 8, 9, 10, 11}


@dataclass(frozen=True)
class SceneSummary:
    item_id: str
    datetime: str
    cloud_cover: float | None
    platform: str | None
    mgrs_tile: str | None
    red_href: str
    nir_href: str
    scl_href: str


def search_scenes(
    bbox: list[float],
    datetime_range: str,
    cloud_cover_lt: float = 60.0,
    limit: int = 30,
) -> list[Item]:
    """Return candidate Sentinel-2 L2A scenes ordered by scene cloud cover."""
    catalog = Client.open(EARTH_SEARCH)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
        max_items=limit,
    )
    items = list(search.items())
    return sorted(items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))


def summarize_item(item: Item) -> SceneSummary:
    """Extract the assets and provenance fields used by this project."""
    missing = [key for key in ("red", "nir", "scl") if key not in item.assets]
    if missing:
        raise KeyError(f"STAC item {item.id} is missing required assets: {missing}")
    return SceneSummary(
        item_id=item.id,
        datetime=str(item.datetime or item.properties.get("datetime")),
        cloud_cover=_as_float(item.properties.get("eo:cloud_cover")),
        platform=item.properties.get("platform"),
        mgrs_tile=item.properties.get("s2:mgrs_tile") or item.properties.get("mgrs:tile"),
        red_href=item.assets["red"].href,
        nir_href=item.assets["nir"].href,
        scl_href=item.assets["scl"].href,
    )


def _as_float(value) -> float | None:
    return None if value is None else float(value)


def _read_band_to_grid(
    href: str,
    bbox_wgs84: Iterable[float],
    *,
    dst_crs=None,
    dst_transform=None,
    dst_width=None,
    dst_height=None,
    resampling: Resampling = Resampling.nearest,
):
    """Read only the AOI window from a COG and optionally warp to a target grid."""
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    ):
        with rasterio.open(href) as src:
            if dst_crs is None:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21
                )
                window = from_bounds(left, bottom, right, top, src.transform)
                data = src.read(1, window=window, boundless=True)
                transform = src.window_transform(window)
                return data, src.crs, transform

            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                resampling=resampling,
            ) as vrt:
                return vrt.read(1), dst_crs, dst_transform


def load_aoi(item: Item, bbox_wgs84: list[float]) -> dict:
    """Load red, NIR and SCL for the AOI on the 10 m red-band grid."""
    red, crs, transform = _read_band_to_grid(item.assets["red"].href, bbox_wgs84)
    height, width = red.shape

    nir, _, _ = _read_band_to_grid(
        item.assets["nir"].href,
        bbox_wgs84,
        dst_crs=crs,
        dst_transform=transform,
        dst_width=width,
        dst_height=height,
        resampling=Resampling.bilinear,
    )
    scl, _, _ = _read_band_to_grid(
        item.assets["scl"].href,
        bbox_wgs84,
        dst_crs=crs,
        dst_transform=transform,
        dst_width=width,
        dst_height=height,
        resampling=Resampling.nearest,
    )

    return {
        "red": red.astype(np.float32),
        "nir": nir.astype(np.float32),
        "scl": scl.astype(np.uint8),
        "crs": str(crs),
        "transform": tuple(transform),
    }


def valid_pixel_mask(scl: np.ndarray) -> np.ndarray:
    """Return pixels considered valid for the MVP vegetation calculation."""
    arr = np.asarray(scl)
    return ~np.isin(arr, list(INVALID_SCL))
