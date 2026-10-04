from pathlib import Path

from scripts.fetch_sentinel2 import read_aoi
from scripts.validate_la_pastora_sources import G14C26A4_BOUNDS, contains, load_bbox


def test_la_pastora_aoi_is_inside_g14c26a4():
    path = Path("config/aoi_la_pastora.geojson")
    bbox = load_bbox(path)
    assert bbox == [-100.2545, 25.6615, -100.2405, 25.6745]
    assert contains(G14C26A4_BOUNDS, bbox)


def test_sentinel_reader_accepts_feature_collection_aoi():
    feature, bbox = read_aoi(Path("config/aoi_la_pastora.geojson"))
    assert feature["properties"]["name"] == "La Pastora dense-green pilot AOI"
    assert bbox == [-100.2545, 25.6615, -100.2405, 25.6745]
