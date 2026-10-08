"""Create deterministic nested UTC label-efficiency manifests.

The 25%, 50%, and 100% budgets use exactly the same validation/test records.
Training subsets are nested and spatially balanced across all training blocks:
for this 36-patch benchmark, 25% selects one patch per train block, 50% selects
two, and 100% selects all four. Within each block the order is deterministically
shuffled from the supplied seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


def stable_block_seed(seed:int,br:int,bc:int)->int:
    raw=f"{seed}:{br}:{bc}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16],16)


def nested_train_order(records:list[dict],seed:int)->list[dict]:
    groups={}
    for r in records:
        if r["split"]!="train":
            continue
        key=(int(r["block_row"]),int(r["block_col"]))
        groups.setdefault(key,[]).append(r)

    ordered_groups={}
    for key,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:int(r["id"]))
        rng=random.Random(stable_block_seed(seed,*key))
        rng.shuffle(rows)
        ordered_groups[key]=rows

    out=[]
    depth=0
    while True:
        added=False
        for key in sorted(ordered_groups):
            rows=ordered_groups[key]
            if depth<len(rows):
                out.append(rows[depth]);added=True
        if not added:
            break
        depth+=1
    return out


def build_manifest(dm:dict,budget:float,seed:int)->dict:
    train=[r for r in dm["records"] if r["split"]=="train"]
    val=[r for r in dm["records"] if r["split"]=="val"]
    test=[r for r in dm["records"] if r["split"]=="test"]
    order=nested_train_order(train,seed)

    n=max(1,round(len(train)*budget))
    selected=order[:n]
    records=selected+val+test
    canopy=sum(r["canopy_pixels"] for r in selected)
    total=sum(r["total_pixels"] for r in selected)
    blocks=sorted({(r["block_row"],r["block_col"]) for r in selected})

    out=copy.deepcopy(dm)
    out["records"]=records
    out["label_efficiency"]={
        "budget_fraction":budget,
        "train_patches":len(selected),
        "full_train_patches":len(train),
        "seed":seed,
        "nested":True,
        "spatially_balanced_by_train_block":True,
        "train_patch_ids":[int(r["id"]) for r in selected],
        "train_blocks":[list(x) for x in blocks],
        "train_canopy_fraction":canopy/total if total else None,
        "validation_patches":len(val),
        "test_patches":len(test),
        "validation_and_test_unchanged":True,
    }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True,exist_ok=True)

    for pct,budget in ((25,0.25),(50,0.50),(100,1.0)):
        out=build_manifest(dm,budget,args.seed)
        path=args.output_dir/f"dataset_manifest_train_{pct}.json"
        path.write_text(json.dumps(out,indent=2),encoding="utf-8")
        meta=out["label_efficiency"]
        print(
            f"{pct}%: train={meta['train_patches']} "
            f"blocks={len(meta['train_blocks'])} "
            f"canopy_fraction={meta['train_canopy_fraction']:.3f} -> {path}"
        )


if __name__=="__main__":
    main()
