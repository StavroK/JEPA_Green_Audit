from pathlib import Path
import zipfile

import numpy as np

from scripts.generate_2007_resolution_labeling_pack import (
    GRID_COLS,
    GRID_ROWS,
    TARGET_SAMPLE_COUNT,
    sampled_cells,
)
from scripts.ingest_inegi_orthophoto_bil import extract_bil_dataset


def test_extract_bil_dataset_excludes_aux(tmp_path: Path):
    archive_path = tmp_path / "ortho.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("g14c26a3.bil", b"rgb")
        archive.writestr("g14c26a3.hdr", b"NBANDS 3")
        archive.writestr("g14c26a3.blw", b"world")
        archive.writestr("g14c26a3.prj", b"projection")
        archive.writestr("g14c26a3.aux", b"incorrect one-band metadata")

    bil = extract_bil_dataset(archive_path, tmp_path / "extract")
    assert bil.exists()
    assert (bil.parent / "g14c26a3.hdr").exists()
    assert not (bil.parent / "g14c26a3.aux").exists()


def test_resolution_label_grid_size_and_sampling():
    assert GRID_COLS * GRID_ROWS == 2304
    cells = sampled_cells()
    assert len(cells) == TARGET_SAMPLE_COUNT == 384
    assert len(set(cells)) == 384
    assert min(r for r, _ in cells) == 0
    assert max(r for r, _ in cells) == GRID_ROWS - 1
    assert min(c for _, c in cells) == 0
    assert max(c for _, c in cells) >= GRID_COLS - 6
