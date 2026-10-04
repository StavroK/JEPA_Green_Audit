import json
from pathlib import Path

from scripts.run_dynamic_world_semantic import CLASSES, PROJECT_GROUPS, load_geometry


def test_dynamic_world_taxonomy_is_complete():
    covered = set().union(*[set(v) for v in PROJECT_GROUPS.values()])
    assert covered == set(CLASSES)


def test_dynamic_world_loads_la_pastora_geometry():
    geometry = load_geometry(Path("config/aoi_la_pastora.geojson"))
    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"][0]) >= 4
