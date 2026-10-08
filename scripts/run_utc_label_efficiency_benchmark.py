"""Run the UTC label-efficiency benchmark across nested training budgets.

Uses pre-generated manifests and the same validation/test sets for every model.
By default runs only 25% and 50% because the 100% full-label results already
exist from the main benchmark. Pass --budgets 25 50 100 to reproduce all.

Models:
- unet
- segformer
- dinov2
- ijepa

The I-JEPA model consumes precomputed frozen feature grids.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def command_for(model:str,budget:int,args)->list[str]:
    manifest=args.manifest_dir/f"dataset_manifest_train_{budget}.json"
    out=args.output_root/f"{model}_train_{budget}"
    py=sys.executable

    common=["--dataset-dir",str(args.dataset_dir),"--manifest",str(manifest),"--output-dir",str(out),"--seed",str(args.seed)]
    if model=="unet":
        return [py,"scripts/train_utc_unet.py",*common,
                "--epochs","40","--batch-size","4","--base-channels","32","--patience","8"]
    if model=="segformer":
        return [py,"scripts/train_utc_segformer.py",*common,
                "--epochs","25","--batch-size","2","--lr","0.0001","--patience","6"]
    if model=="dinov2":
        return [py,"scripts/train_utc_dinov2.py",*common,
                "--epochs","30","--batch-size","2","--lr","0.0005","--patience","7"]
    if model=="ijepa":
        return [py,"scripts/train_utc_ijepa_decoder.py",*common,
                "--features",str(args.ijepa_features),
                "--epochs","40","--batch-size","4","--lr","0.0005","--patience","8","--hidden","256"]
    raise ValueError(model)


def metric_file(model:str,out:Path)->Path:
    names={
        "unet":"unet_metrics.json",
        "segformer":"segformer_b0_metrics.json",
        "dinov2":"dinov2_decoder_metrics.json",
        "ijepa":"ijepa_decoder_metrics.json",
    }
    return out/names[model]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--manifest-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--ijepa-features",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=["unet","segformer","dinov2","ijepa"],
                   default=["unet","segformer","dinov2","ijepa"])
    p.add_argument("--budgets",nargs="+",type=int,choices=[25,50,100],default=[25,50])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--force",action="store_true")
    args=p.parse_args()

    args.output_root.mkdir(parents=True,exist_ok=True)
    for budget in args.budgets:
        manifest=args.manifest_dir/f"dataset_manifest_train_{budget}.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        for model in args.models:
            out=args.output_root/f"{model}_train_{budget}"
            metrics=metric_file(model,out)
            if metrics.exists() and not args.force:
                print(f"SKIP {model} {budget}%: {metrics} exists")
                continue
            cmd=command_for(model,budget,args)
            print("\nRUN",model,f"{budget}%")
            print(" ".join(cmd))
            subprocess.run(cmd,check=True)

    print("\nLabel-efficiency runs complete.")


if __name__=="__main__":
    main()
