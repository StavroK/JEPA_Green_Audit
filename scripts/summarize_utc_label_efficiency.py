"""Summarize UTC label-efficiency benchmark metrics into JSON and CSV."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

FILES={
 "unet":"unet_metrics.json",
 "segformer":"segformer_b0_metrics.json",
 "dinov2":"dinov2_decoder_metrics.json",
 "ijepa":"ijepa_decoder_metrics.json",
}

def extract(model,payload):
    if model=="unet":
        # Full-label U-Net predates dual-threshold output. Label-eff runs retain
        # fixed 0.5 in "test"; use it consistently.
        m=payload.get("test") or payload.get("fixed_threshold_test")
        threshold=m.get("threshold",0.5)
    else:
        m=payload["test_at_locked_threshold"]
        threshold=m["threshold"]
    return {
      "threshold":threshold,
      "precision":m["precision"],"recall":m["recall"],"f1":m["f1"],
      "iou":m["iou"],"balanced_accuracy":m["balanced_accuracy"],
      "utc_area_bias_pp":m["utc_area_bias_pp"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--budgets",nargs="+",type=int,default=[25,50])
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for budget in args.budgets:
        for model,name in FILES.items():
            path=args.root/f"{model}_train_{budget}"/name
            if not path.exists():
                print(f"Missing: {path}")
                continue
            payload=json.loads(path.read_text(encoding="utf-8"))
            rows.append({"budget_pct":budget,"model":model,**extract(model,payload)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    csv_path=args.output.with_suffix(".csv")
    if rows:
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")

if __name__=="__main__":main()
