"""Generate publication-style M3c cross-site UTC figures and point-error overlays.

Inputs are the already-computed raw-consistent multi-seed summary and per-run
point JSON files. This script does not retrain models or tune thresholds.

Outputs:
- publication_f1_vs_budget.png
- publication_balanced_accuracy_vs_budget.png
- publication_prevalence_bias_vs_budget.png
- point_error_<model>_<budget>.png
- publication_results_table.csv
- publication_results_table.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


MODEL_ORDER=["unet","segformer","dinov2","ijepa"]
MODEL_LABELS={
    "unet":"U-Net",
    "segformer":"SegFormer-B0",
    "dinov2":"DINOv2 frozen",
    "ijepa":"I-JEPA frozen",
}


def aggregate_rows(summary:dict)->list[dict]:
    rows=[]
    for model in MODEL_ORDER:
        if model not in summary["aggregate"]:
            continue
        budgets=summary["aggregate"][model]
        for budget_s in sorted(budgets,key=lambda x:int(x)):
            m=budgets[budget_s]["metrics"]
            rows.append({
                "model":model,
                "model_label":MODEL_LABELS[model],
                "budget":int(budget_s),
                "ba_mean":float(m["balanced_accuracy"]["mean"]),
                "ba_sd":float(m["balanced_accuracy"]["sd"]),
                "f1_mean":float(m["f1_tree"]["mean"]),
                "f1_sd":float(m["f1_tree"]["sd"]),
                "iou_mean":float(m["iou_tree"]["mean"]),
                "iou_sd":float(m["iou_tree"]["sd"]),
                "bias_mean":float(m["point_prevalence_bias_pp"]["mean"]),
                "bias_sd":float(m["point_prevalence_bias_pp"]["sd"]),
            })
    return rows


def write_tables(rows:list[dict],out_dir:Path)->None:
    fields=["model_label","budget","ba_mean","ba_sd","f1_mean","f1_sd","iou_mean","iou_sd","bias_mean","bias_sd"]
    with (out_dir/"publication_results_table.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r[k] for k in fields})

    lines=[
        "| Model | Labels | Balanced accuracy | Tree F1 | Point IoU | Point prevalence bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['budget']}% | "
            f"{r['ba_mean']:.3f} ± {r['ba_sd']:.3f} | "
            f"{r['f1_mean']:.3f} ± {r['f1_sd']:.3f} | "
            f"{r['iou_mean']:.3f} ± {r['iou_sd']:.3f} | "
            f"{r['bias_mean']:+.2f} ± {r['bias_sd']:.2f} pp |"
        )
    (out_dir/"publication_results_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def metric_plot(rows:list[dict],mean_key:str,sd_key:str,ylabel:str,title:str,path:Path,zero_line:bool=False)->None:
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for model in MODEL_ORDER:
        rs=sorted([r for r in rows if r["model"]==model],key=lambda r:r["budget"])
        if not rs:
            continue
        x=[r["budget"] for r in rs]
        y=[r[mean_key] for r in rs]
        e=[r[sd_key] for r in rs]
        ax.errorbar(x,y,yerr=e,marker="o",capsize=4,label=MODEL_LABELS[model])
    if zero_line:
        ax.axhline(0,linewidth=1,linestyle="--")
    ax.set_xlabel("Fundidora training-label budget (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({r["budget"] for r in rows}))
    ax.grid(True,alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path,dpi=220,bbox_inches="tight")
    plt.close(fig)


def load_point_runs(result_dir:Path,model:str,budget:int)->list[dict]:
    paths=sorted(result_dir.glob(f"{model}_train_{budget}_seed_*_zero_shot_points.json"))
    if not paths:
        raise FileNotFoundError(f"No point result files for {model} {budget}% in {result_dir}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def consensus_points(runs:list[dict])->list[dict]:
    by_id={}
    for run in runs:
        for p in run["points"]:
            by_id.setdefault(str(p["id"]),[]).append(p)
    out=[]
    for pid,items in by_id.items():
        labels={x["human_label"] for x in items}
        coords={(int(x["x"]),int(x["y"])) for x in items}
        if len(labels)!=1 or len(coords)!=1:
            raise ValueError(f"Inconsistent point metadata across runs for point {pid}")
        votes=[1 if x["model_class"]=="tree" else 0 for x in items]
        human=next(iter(labels))
        majority="tree" if fmean(votes)>=.5 else "non"
        correct=majority==human
        agreement=max(fmean(votes),1-fmean(votes))
        x,y=next(iter(coords))
        out.append({
            "id":pid,"x":x,"y":y,"human_label":human,
            "majority_class":majority,"correct":correct,
            "agreement":agreement,
        })
    return out


def draw_point_overlay(image_path:Path,points:list[dict],title:str,out:Path)->None:
    img=Image.open(image_path).convert("RGB")
    draw=ImageDraw.Draw(img)
    # Encoding: correct = circle; error = X. Line width/size increases as seed agreement increases.
    for p in points:
        x,y=int(p["x"]),int(p["y"])
        agreement=float(p["agreement"])
        r=4+int(round(4*agreement))
        width=2 if agreement<.8 else 3
        if p["correct"]:
            draw.ellipse((x-r,y-r,x+r,y+r),outline=(255,255,255),width=width)
            draw.ellipse((x-r+2,y-r+2,x+r-2,y+r-2),outline=(0,0,0),width=1)
        else:
            draw.line((x-r,y-r,x+r,y+r),fill=(255,255,255),width=width)
            draw.line((x-r,y+r,x+r,y-r),fill=(255,255,255),width=width)
            draw.line((x-r+1,y-r,x+r+1,y+r),fill=(0,0,0),width=1)
            draw.line((x-r+1,y+r,x+r+1,y-r),fill=(0,0,0),width=1)

    # Add a compact header band without obscuring the AOI.
    band_h=34
    canvas=Image.new("RGB",(img.width,img.height+band_h),(255,255,255))
    canvas.paste(img,(0,band_h))
    d=ImageDraw.Draw(canvas)
    n_err=sum(not p["correct"] for p in points)
    d.text((8,10),f"{title} | majority-vote errors: {n_err}/{len(points)} | circle=correct, X=error",fill=(0,0,0))
    canvas.save(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--result-dir",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--overlay-budgets",type=int,nargs="+",default=[25,100])
    args=p.parse_args()

    summary=json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=aggregate_rows(summary)
    write_tables(rows,args.output_dir)

    metric_plot(rows,"f1_mean","f1_sd","Tree F1","Raw-consistent cross-site label efficiency",args.output_dir/"publication_f1_vs_budget.png")
    metric_plot(rows,"ba_mean","ba_sd","Balanced accuracy","Cross-site balanced accuracy",args.output_dir/"publication_balanced_accuracy_vs_budget.png")
    metric_plot(rows,"bias_mean","bias_sd","Point prevalence bias (pp)","Cross-site point prevalence bias",args.output_dir/"publication_prevalence_bias_vs_budget.png",zero_line=True)

    for budget in args.overlay_budgets:
        for model in MODEL_ORDER:
            runs=load_point_runs(args.result_dir,model,budget)
            pts=consensus_points(runs)
            draw_point_overlay(
                args.image,pts,
                f"{MODEL_LABELS[model]} — {budget}% labels — {len(runs)} seeds",
                args.output_dir/f"point_error_{model}_{budget}.png",
            )

    print(f"Wrote publication assets to {args.output_dir}")
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__=="__main__":
    main()
