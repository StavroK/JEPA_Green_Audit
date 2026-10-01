import csv
from pathlib import Path

from scripts.label_m3_patches import first_unlabeled_index, label_counts, load_rows, save_rows


FIELDNAMES = ["year", "row", "col", "patch_path", "label", "reviewer", "notes"]


def test_label_state_helpers(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    rows = [
        {
            "year": "2025",
            "row": "0",
            "col": "0",
            "patch_path": "a.png",
            "label": "vegetation",
            "reviewer": "",
            "notes": "",
        },
        {
            "year": "2025",
            "row": "0",
            "col": "1",
            "patch_path": "b.png",
            "label": "",
            "reviewer": "",
            "notes": "",
        },
    ]
    save_rows(csv_path, rows, FIELDNAMES)
    loaded, fieldnames = load_rows(csv_path)

    assert fieldnames == FIELDNAMES
    assert first_unlabeled_index(loaded) == 1
    counts = label_counts(loaded)
    assert counts["vegetation"] == 1
    assert counts["unlabeled"] == 1


def test_save_rows_is_valid_csv(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    rows = [{
        "year": "2026",
        "row": "1",
        "col": "2",
        "patch_path": "c.png",
        "label": "uncertain",
        "reviewer": "BK",
        "notes": "mixed",
    }]
    save_rows(csv_path, rows, FIELDNAMES)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))

    assert saved == rows
