import zipfile
from pathlib import Path

import numpy as np

from scripts.ingest_inegi_elevation import preview_uint8, resolve_raster, stats


def test_preview_uint8_and_stats():
    arr = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    preview = preview_uint8(arr)
    assert preview.shape == arr.shape
    assert preview.dtype == np.uint8
    result = stats(arr)
    assert result["valid_pixels"] == 3
    assert result["min_m"] == 1.0
    assert result["max_m"] == 3.0


def test_resolve_raster_from_inegi_zip(tmp_path: Path):
    source = tmp_path / "product.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("metadatos/readme.txt", "metadata")
        archive.writestr("conjunto_de_datos/example_ms.tif", b"dummy")

    extracted = resolve_raster(source, tmp_path / "extract")
    assert extracted.name == "example_ms.tif"
    assert extracted.exists()
