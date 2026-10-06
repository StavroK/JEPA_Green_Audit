"""Convert reviewed UTC validation points into a new training-label batch.

Use this only AFTER the validation round has been evaluated and explicitly
retired as a holdout. Once these labels are promoted into training, that
validation set is no longer independent and a fresh holdout must be sampled.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reviewed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()

    payload=json.loads(args.reviewed.read_text(encoding="utf-8"))
    labels=[]
    skipped=0
    for item in payload.get("points",[]):
        h=item.get("human_label")
        if h not in {"tree","non"}:
            skipped+=1
            continue
        labels.append({
            "x":int(item["x"]),
            "y":int(item["y"]),
            "label":h,
            "source":"retired_independent_validation",
            "original_id":item.get("id"),
        })

    out={
        "schema_version":1,
        "task":"utc_canopy_point_labels",
        "image":payload.get("image","fundidora_2007_rgb_preview.png"),
        "width_px":payload.get("image_width_px"),
        "height_px":payload.get("image_height_px"),
        "labels":labels,
        "provenance":{
            "source_reviewed_validation":str(args.reviewed),
            "note":"This validation round is now retired from holdout use. Sample a new independent validation set after retraining."
        }
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Promoted labels: {len(labels)}")
    print(f"Skipped uncertain/unreviewed: {skipped}")
    print("IMPORTANT: this validation set is no longer independent after promotion.")


if __name__=="__main__": main()
