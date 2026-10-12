"""Run multi-seed zero-shot Fundidora→La Pastora point validation for DINOv2.

For each requested seed and Fundidora label budget, this script:
- loads the corresponding Fundidora-trained DINOv2 decoder checkpoint,
- reads the Fundidora-locked validation-selected threshold from its metrics JSON,
- runs zero-shot inference on the La Pastora RGB image,
- samples the same independent human point labels,
- writes one result JSON per seed/budget,
- writes a summary JSON with mean ± sample SD across seeds.

No La Pastora labels are used for threshold selection or model fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


METRIC_KEYS = (
    "balanced_accuracy",
    "precision_tree",
    "recall_tree",
    "f1_tree",
    "iou_tree",
    "point_prevalence_bias_pp",
)


def locked_threshold(metrics_path: Path) -> float:
    payload=json.loads(metrics_path.read_text(encoding="utf-8"))
    if "test_at_locked_threshold" in payload:
        return float(payload["test_at_locked_threshold"]["threshold"])
    if "best_validation" in payload and "threshold" in payload["best_validation"]:
        return float(payload["best_validation"]["threshold"])
    raise KeyError(f"No locked threshold found in {metrics_path}")


def summarize_results(rows: list[dict]) -> dict:
    grouped={}
    for row in rows:
        grouped.setdefault(str(row["label_budget"]),[]).append(row)
    out={}
    for budget,items in grouped.items():
        metrics={}
        for key in METRIC_KEYS:
            vals=[float(x["metrics"][key]) for x in items]
            metrics[key]={
                "mean":statistics.fmean(vals),
                "sd":statistics.stdev(vals) if len(vals)>1 else 0.0,
                "values":vals,
            }
        out[budget]={
            "n_seeds":len(items),
            "seeds":[int(x["seed"]) for x in items],
            "metrics":metrics,
        }
    return out


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("outputs/utc/fundidora_2007/label_efficiency_multiseed"))
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44,45,46])
    p.add_argument("--budgets",type=int,nargs="+",default=[25,50])
    p.add_argument("--tile-size",type=int,default=256)
    p.add_argument("--stride",type=int,default=128)
    args=p.parse_args()

    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]

    for seed in args.seeds:
        for budget in args.budgets:
            run_dir=args.root/f"seed_{seed}"/"runs"/f"dinov2_train_{budget}"
            checkpoint=run_dir/"dinov2_decoder_best.pt"
            metrics_json=run_dir/"dinov2_decoder_metrics.json"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            if not metrics_json.exists():
                raise FileNotFoundError(metrics_json)

            threshold=locked_threshold(metrics_json)
            out=args.output_dir/f"dinov2_train_{budget}_seed_{seed}_zero_shot_points.json"
            cmd=[
                sys.executable,
                "scripts/evaluate_utc_cross_site_points.py",
                "--model-type","dinov2",
                "--checkpoint",str(checkpoint),
                "--threshold",str(threshold),
                "--image",str(args.image),
                "--labels",str(args.labels),
                "--output",str(out),
                "--tile-size",str(args.tile_size),
                "--stride",str(args.stride),
                "--preprocessing","percentile",
            ]
            print(f"\n=== seed={seed} budget={budget}% threshold={threshold:.2f} ===",flush=True)
            subprocess.run(cmd,check=True)

            result=json.loads(out.read_text(encoding="utf-8"))
            rows.append({
                "seed":seed,
                "label_budget":budget,
                "threshold":threshold,
                "checkpoint":str(checkpoint),
                "fundidora_metrics":str(metrics_json),
                "result_file":str(out),
                "metrics":result["metrics"],
            })

    summary={
        "status":"multi_seed_zero_shot_cross_site_point_validation",
        "model":"dinov2_frozen_decoder",
        "source_training_site":"fundidora_2007",
        "target_validation_site":"la_pastora_2007",
        "seeds":args.seeds,
        "label_budgets_percent":args.budgets,
        "threshold_policy":"Each seed/budget uses its own Fundidora validation-selected locked threshold. No target-site calibration.",
        "preprocessing":"legacy Fundidora v1 per-tile 2-98 percentile stretch",
        "rows":rows,
        "aggregate":summarize_results(rows),
        "note":"Point-based cross-site metrics; not dense UTC area estimates.",
    }
    summary_path=args.output_dir/"dinov2_multiseed_zero_shot_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print(f"\nWrote {summary_path}")
    for budget in args.budgets:
        agg=summary["aggregate"][str(budget)]["metrics"]
        print(
            f"{budget}% labels | "
            f"BA={agg['balanced_accuracy']['mean']:.3f}±{agg['balanced_accuracy']['sd']:.3f} | "
            f"F1={agg['f1_tree']['mean']:.3f}±{agg['f1_tree']['sd']:.3f} | "
            f"IoU={agg['iou_tree']['mean']:.3f}±{agg['iou_tree']['sd']:.3f} | "
            f"bias={agg['point_prevalence_bias_pp']['mean']:+.2f}±{agg['point_prevalence_bias_pp']['sd']:.2f} pp"
        )


if __name__=="__main__":
    main()
