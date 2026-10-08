import json
from pathlib import Path

def test_auto_la_pastora_aoi_bbox():
    p=Path("config/aoi_la_pastora_green_core_auto.geojson")
    payload=json.loads(p.read_text(encoding="utf-8"))
    f=payload["features"][0]
    pts=f["geometry"]["coordinates"][0]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    bbox=[min(xs),min(ys),max(xs),max(ys)]
    assert bbox==[-100.252112409,25.6659435255,-100.246078709,25.6714115577]
    assert f["properties"]["id"]=="la_pastora_green_core_auto"
