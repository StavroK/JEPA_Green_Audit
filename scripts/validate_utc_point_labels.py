"""Validate and optionally normalize independent UTC point-label JSON."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VALID_HUMAN = {"tree", "non", "uncertain"}
VALID_DETAIL = {"tree", "non", "grass", "pavement", "roof", "bare", "shadow", "water", "uncertain"}


def summarize(payload: dict) -> dict:
    points = payload.get("points", [])
    human = Counter()
    detail = Counter()
    reviewers = Counter()
    invalid = []
    duplicate_cells = set()
    seen_cells = set()

    for p in points:
        h = p.get("human_label", "")
        d = p.get("detail_label", "")
        human[h or "unlabeled"] += 1
        detail[d or "unlabeled"] += 1
        reviewers[p.get("reviewer", "")] += 1
        if h and h not in VALID_HUMAN:
            invalid.append({"id": p.get("id"), "field": "human_label", "value": h})
        if d and d not in VALID_DETAIL:
            invalid.append({"id": p.get("id"), "field": "detail_label", "value": d})
        cell = (p.get("grid_row"), p.get("grid_col"))
        if cell in seen_cells:
            duplicate_cells.add(cell)
        seen_cells.add(cell)

    return {
        "n_points": len(points),
        "human_counts": dict(human),
        "detail_counts": dict(detail),
        "unique_reviewers": len(reviewers),
        "duplicate_grid_cells": sorted(duplicate_cells),
        "invalid_labels": invalid,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--reviewer", default=None, help="If supplied, overwrite reviewer metadata for all labeled points.")
    args = p.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = summarize(payload)

    if args.reviewer is not None:
        for point in payload.get("points", []):
            if point.get("human_label"):
                point["reviewer"] = args.reviewer
        payload["reviewer_normalization"] = {
            "applied": True,
            "reviewer": args.reviewer,
            "note": "Reviewer metadata normalized after UI shortcut-focus bug; labels and coordinates unchanged.",
        }
        if args.output is None:
            raise ValueError("--output is required when --reviewer is used")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote normalized labels: {args.output}")

    print(f"Points: {report['n_points']}")
    print(f"Human labels: {report['human_counts']}")
    print(f"Detail labels: {report['detail_counts']}")
    print(f"Unique reviewer strings: {report['unique_reviewers']}")
    print(f"Duplicate grid cells: {len(report['duplicate_grid_cells'])}")
    print(f"Invalid labels: {len(report['invalid_labels'])}")


if __name__ == "__main__":
    main()
