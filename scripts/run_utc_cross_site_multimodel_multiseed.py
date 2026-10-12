"""Run architecture-matched multi-seed zero-shot cross-site UTC validation.

Models:
- U-Net
- SegFormer-B0
- frozen DINOv2 + decoder
- frozen I-JEPA ViT-H/14 + decoder

For each model, seed and 25%/50% Fundidora label budget, the script evaluates
the same independent La Pastora human points. Thresholds are selected only from
Fundidora validation:
- SegFormer/DINOv2/I-JEPA: read each run's stored validation-selected threshold.
- U-Net: select threshold on the common Fundidora validation split for that
  checkpoint because the original U-Net trainer stored only the fixed-0.5 result.

No La Pastora label is used for model fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_utc_unet import PatchDataset, UNetSmall, evaluate


MODEL_FILES={
    "unet":("unet_best.pt","unet_metrics.json"),
    "segformer":("segformer_b0_best.pt","segformer_b0_metrics.json"),
    "dinov2":("dinov2_decoder_best.pt","dinov2_decoder_metrics.json"),
    "ijepa":("ijepa_decoder_best.pt","ijepa_decoder_metrics.json"),
}
METRIC_KEYS=("balanced_accuracy","precision_tree","recall_tree","f1_tree","iou_tree","point_prevalence_bias_pp")


def stored_locked_threshold(metrics_path:Path)->float:
    payload=json.loads(metrics_path.read_text(encoding="utf-8"))
    if "test_at_locked_threshold" in payload:
        return float(payload["test_at_locked_threshold"]["threshold"])
    if "best_validation" in payload and "threshold" in payload["best_validation"]:
        return float(payload["best_validation"]["threshold"])
    raise KeyError(f"No validation-selected threshold in {metrics_path}")


def select_unet_threshold(dataset_dir:Path, checkpoint:Path, batch_size:int=4)->float:
    dm=json.loads((dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    val=[r for r in dm["records"] if r["split"]=="val"]
    if not val:
        raise ValueError("Fundidora validation split is empty")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    model=UNetSmall(int(ck.get("base_channels",32))).to(device)
    model.load_state_dict(ck["model_state"])
    loader=DataLoader(PatchDataset(dataset_dir,val,False),batch_size=batch_size,shuffle=False)
    rows=[]
    for t in np.arange(.2,.801,.05):
        m=evaluate(model,loader,device,float(t))
        rows.append((float(t),m))
    best=max(rows,key=lambda x:(x[1]["iou"],x[1]["f1"],x[1]["balanced_accuracy"]))
    return round(best[0],2)


def summarize(rows:list[dict])->dict:
    out={}
    for model in sorted({r["model"] for r in rows}):
        out[model]={}
        for budget in sorted({r["label_budget"] for r in rows if r["model"]==model}):
            items=[r for r in rows if r["model"]==model and r["label_budget"]==budget]
            metrics={}
            for key in METRIC_KEYS:
                vals=[float(x["metrics"][key]) for x in items]
                metrics[key]={
                    "mean":statistics.fmean(vals),
                    "sd":statistics.stdev(vals) if len(vals)>1 else 0.0,
                    "values":vals,
                }
            out[model][str(budget)]={
                "n_seeds":len(items),
                "seeds":[x["seed"] for x in items],
                "metrics":metrics,
            }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fundidora-dataset",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("outputs/utc/fundidora_2007/label_efficiency_multiseed"))
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44,45,46])
    p.add_argument("--budgets",type=int,nargs="+",default=[25,50])
    p.add_argument("--models",nargs="+",choices=("unet","segformer","dinov2","ijepa"),
                   default=["unet","segformer","dinov2","ijepa"])
    p.add_argument("--ijepa-upstream",type=Path,default=Path("vendor/ijepa"))
    p.add_argument("--ijepa-encoder-checkpoint",type=Path,default=Path("models/IN1K-vit.h.14-300e.pth.tar"))
    args=p.parse_args()

    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in args.models:
        for seed in args.seeds:
            for budget in args.budgets:
                run_dir=args.root/f"seed_{seed}"/"runs"/f"{model}_train_{budget}"
                ck_name,metrics_name=MODEL_FILES[model]
                checkpoint=run_dir/ck_name
                metrics_path=run_dir/metrics_name
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)

                if model=="unet":
                    threshold=select_unet_threshold(args.fundidora_dataset,checkpoint)
                    threshold_source="Fundidora validation sweep performed post hoc on original checkpoint"
                else:
                    if not metrics_path.exists():
                        raise FileNotFoundError(metrics_path)
                    threshold=stored_locked_threshold(metrics_path)
                    threshold_source="Stored Fundidora validation-selected threshold"

                out=args.output_dir/f"{model}_train_{budget}_seed_{seed}_zero_shot_points.json"
                cmd=[
                    sys.executable,"scripts/evaluate_utc_cross_site_points.py",
                    "--model-type",model,
                    "--checkpoint",str(checkpoint),
                    "--threshold",str(threshold),
                    "--image",str(args.image),
                    "--labels",str(args.labels),
                    "--output",str(out),
                    "--preprocessing","percentile",
                ]
                if model=="ijepa":
                    cmd += [
                        "--ijepa-upstream",str(args.ijepa_upstream),
                        "--ijepa-encoder-checkpoint",str(args.ijepa_encoder_checkpoint),
                    ]
                print(f"\n=== {model} seed={seed} budget={budget}% threshold={threshold:.2f} ===",flush=True)
                subprocess.run(cmd,check=True)
                result=json.loads(out.read_text(encoding="utf-8"))
                rows.append({
                    "model":model,
                    "seed":seed,
                    "label_budget":budget,
                    "threshold":threshold,
                    "threshold_source":threshold_source,
                    "checkpoint":str(checkpoint),
                    "metrics":result["metrics"],
                })

    aggregate=summarize(rows)
    summary={
        "status":"architecture_matched_multiseed_zero_shot_cross_site_point_validation",
        "source_training_site":"fundidora_2007",
        "target_validation_site":"la_pastora_2007",
        "models":args.models,
        "seeds":args.seeds,
        "label_budgets_percent":args.budgets,
        "rows":rows,
        "aggregate":aggregate,
        "note":"All target metrics are point-based. No La Pastora calibration or fitting.",
    }
    out=args.output_dir/"multimodel_multiseed_zero_shot_summary.json"
    out.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"\nWrote {out}")
    for model in args.models:
        for budget in args.budgets:
            m=aggregate[model][str(budget)]["metrics"]
            print(
                f"{model:9s} {budget:>2}% | "
                f"BA={m['balanced_accuracy']['mean']:.3f}±{m['balanced_accuracy']['sd']:.3f} | "
                f"F1={m['f1_tree']['mean']:.3f}±{m['f1_tree']['sd']:.3f} | "
                f"IoU={m['iou_tree']['mean']:.3f}±{m['iou_tree']['sd']:.3f} | "
                f"bias={m['point_prevalence_bias_pp']['mean']:+.2f}±{m['point_prevalence_bias_pp']['sd']:.2f} pp"
            )


if __name__=="__main__":
    main()
