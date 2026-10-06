"""Diagnostic threshold sweep for a reviewed UTC validation file.

This is for diagnosis only. If a threshold is selected using this file, the
file becomes a calibration set and cannot be reported as the final independent
validation. A new holdout must then be sampled.
"""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,jaccard_score,balanced_accuracy_score


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reviewed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    payload=json.loads(args.reviewed.read_text(encoding="utf-8"))

    y=[];prob=[]
    for item in payload.get("points",[]):
        h=item.get("human_label")
        if h not in {"tree","non"}: continue
        y.append(1 if h=="tree" else 0)
        prob.append(float(item["model_probability"]))
    y=np.asarray(y,dtype=np.uint8);prob=np.asarray(prob,dtype=float)

    rows=[]
    for t in np.arange(0.50,0.91,0.05):
        pred=(prob>=t).astype(np.uint8)
        rows.append({
            "threshold":round(float(t),2),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "precision_tree":float(precision_score(y,pred,zero_division=0)),
            "recall_tree":float(recall_score(y,pred,zero_division=0)),
            "f1_tree":float(f1_score(y,pred,zero_division=0)),
            "iou_tree":float(jaccard_score(y,pred,zero_division=0)),
            "predicted_tree_count":int(pred.sum()),
        })

    best=max(rows,key=lambda r:(r["f1_tree"],r["iou_tree"],r["balanced_accuracy"]))
    out={
        "status":"diagnostic_only_not_final_validation",
        "n":int(len(y)),
        "rows":rows,
        "best_by_f1":best,
        "warning":"If you choose a threshold using this file, retire this validation set from final reporting and sample a new independent holdout."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("threshold  bal_acc  precision  recall  f1  iou")
    for r in rows:
        print(f"{r['threshold']:.2f}       {r['balanced_accuracy']:.3f}    {r['precision_tree']:.3f}      {r['recall_tree']:.3f}  {r['f1_tree']:.3f} {r['iou_tree']:.3f}")
    print(f"Best by F1: threshold={best['threshold']:.2f}, F1={best['f1_tree']:.3f}, IoU={best['iou_tree']:.3f}")
    print("STATUS: diagnostic only; new holdout required after threshold tuning.")


if __name__=="__main__": main()
