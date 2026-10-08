"""Aggregate multi-seed UTC label-efficiency results as mean ± SD."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path

FILES={
 "unet":"unet_metrics.json",
 "segformer":"segformer_b0_metrics.json",
 "dinov2":"dinov2_decoder_metrics.json",
 "ijepa":"ijepa_decoder_metrics.json",
}

METRICS=("precision","recall","f1","iou","balanced_accuracy","utc_area_bias_pp")


def extract(model:str,payload:dict)->dict:
    if model=="unet":
        m=payload.get("test") or payload.get("fixed_threshold_test")
        threshold=m.get("threshold",0.5)
    else:
        m=payload["test_at_locked_threshold"]
        threshold=m["threshold"]
    return {"threshold":threshold,**{k:float(m[k]) for k in METRICS}}


def mean_sd(xs:list[float])->tuple[float,float]:
    if not xs:
        return math.nan,math.nan
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs)>1 else 0.0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--budgets",nargs="+",type=int,default=[25,50])
    p.add_argument("--models",nargs="+",choices=list(FILES),default=list(FILES))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()

    per_run=[]
    for seed in args.seeds:
        for budget in args.budgets:
            for model in args.models:
                path=args.root/f"seed_{seed}"/"runs"/f"{model}_train_{budget}"/FILES[model]
                if not path.exists():
                    print(f"Missing: {path}")
                    continue
                payload=json.loads(path.read_text(encoding="utf-8"))
                per_run.append({"seed":seed,"budget_pct":budget,"model":model,**extract(model,payload)})

    summary=[]
    for budget in args.budgets:
        for model in args.models:
            rows=[r for r in per_run if r["budget_pct"]==budget and r["model"]==model]
            if not rows:
                continue
            out={"budget_pct":budget,"model":model,"n_seeds":len(rows)}
            for metric in METRICS:
                mu,sd=mean_sd([r[metric] for r in rows])
                out[f"{metric}_mean"]=mu
                out[f"{metric}_sd"]=sd
            out["thresholds"]=[r["threshold"] for r in rows]
            summary.append(out)

    payload={"per_run":per_run,"summary":summary}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")

    csv_path=args.output.with_suffix(".csv")
    if summary:
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            fields=[k for k in summary[0].keys() if k!="thresholds"]
            w=csv.DictWriter(f,fieldnames=fields)
            w.writeheader()
            for r in summary:
                w.writerow({k:v for k,v in r.items() if k!="thresholds"})

    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    for r in summary:
        print(
            f"{r['budget_pct']:>3}% {r['model']:<9} n={r['n_seeds']} "
            f"F1={r['f1_mean']:.3f}±{r['f1_sd']:.3f} "
            f"IoU={r['iou_mean']:.3f}±{r['iou_sd']:.3f} "
            f"BalAcc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_sd']:.3f} "
            f"UTCbias={r['utc_area_bias_pp_mean']:+.2f}±{r['utc_area_bias_pp_sd']:.2f} pp"
        )


if __name__=="__main__":
    main()
