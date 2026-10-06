import json
from pathlib import Path

def test_placeholder_for_retired_validation_schema(tmp_path: Path):
    payload={"points":[
        {"x":1,"y":2,"human_label":"tree"},
        {"x":3,"y":4,"human_label":"non"},
        {"x":5,"y":6,"human_label":"uncertain"},
    ]}
    p=tmp_path/"reviewed.json"
    p.write_text(json.dumps(payload),encoding="utf-8")
    loaded=json.loads(p.read_text(encoding="utf-8"))
    definite=[x for x in loaded["points"] if x.get("human_label") in {"tree","non"}]
    assert len(definite)==2
