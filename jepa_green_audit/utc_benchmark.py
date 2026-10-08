"""Shared helpers for UTC segmentation benchmark manifests."""
from __future__ import annotations
import json
from pathlib import Path


def load_dataset_manifest(dataset_dir:Path,manifest:Path|None=None)->dict:
    path=manifest if manifest is not None else dataset_dir/"dataset_manifest.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))
