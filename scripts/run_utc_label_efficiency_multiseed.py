"""Run a resumable multi-seed UTC label-efficiency benchmark.

For each seed:
1. creates deterministic nested 25%/50% manifests;
2. runs selected models on identical validation/test sets;
3. stores outputs under seed_<N>/runs.

Default seeds are 42-46. Existing metric files are skipped by the underlying
runner, so interrupted studies can be resumed safely.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd:list[str]):
    print("\n$ "+" ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--ijepa-features",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--budgets",nargs="+",type=int,choices=[25,50,100],default=[25,50])
    p.add_argument("--models",nargs="+",choices=["unet","segformer","dinov2","ijepa"],
                   default=["unet","segformer","dinov2","ijepa"])
    p.add_argument("--force",action="store_true")
    args=p.parse_args()

    py=sys.executable
    for seed in args.seeds:
        seed_root=args.output_root/f"seed_{seed}"
        manifest_dir=seed_root/"manifests"
        runs_dir=seed_root/"runs"

        run([
            py,"scripts/prepare_utc_label_efficiency_manifests.py",
            "--dataset-dir",str(args.dataset_dir),
            "--output-dir",str(manifest_dir),
            "--seed",str(seed),
        ])

        cmd=[
            py,"scripts/run_utc_label_efficiency_benchmark.py",
            "--dataset-dir",str(args.dataset_dir),
            "--manifest-dir",str(manifest_dir),
            "--output-root",str(runs_dir),
            "--ijepa-features",str(args.ijepa_features),
            "--budgets",*[str(x) for x in args.budgets],
            "--models",*args.models,
            "--seed",str(seed),
        ]
        if args.force:
            cmd.append("--force")
        run(cmd)

    print("\nMulti-seed label-efficiency benchmark complete.")


if __name__=="__main__":
    main()
