"""Evaluate UTC canopy predictions against independent human validation points."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reviewed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    payload=json.loads(args.reviewed.read_text(encoding="utf-8"))
    y=[];pred=[];probs=[]
    uncertain=0
    for item in payload.get("points",[]):
        h=item.get("human_label")
        if h=="uncertain" or h is None:
            uncertain+=1;continue
        if h not in {"tree","non"}: continue
        y.append(1 if h=="tree" else 0)
        pred.append(1 if item.get("model_class")=="tree" else 0)
        probs.append(float(item.get("model_probability",0)))
    y=np.asarray(y);pred=np.asarray(pred)
    if len(y)<20 or len(np.unique(y))<2:
        raise ValueError("Need at least 20 definite independent labels including both classes")
    cm=confusion_matrix(y,pred,labels=[0,1])
    tn,fp,fn,tp=cm.ravel()
    result={
      "status":"independent_spatial_validation",
      "n_definite":int(len(y)),"n_uncertain_or_unreviewed":int(uncertain),
      "tree_reference":int((y==1).sum()),"non_tree_reference":int((y==0).sum()),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "precision_tree":float(precision_score(y,pred,zero_division=0)),
      "recall_tree":float(recall_score(y,pred,zero_division=0)),
      "f1_tree":float(f1_score(y,pred,zero_division=0)),
      "iou_tree":float(jaccard_score(y,pred,zero_division=0)),
      "confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
      "acceptance_note":"Use these independent metrics, plus visual QC and uncertainty review, to decide whether the canopy mask is suitable for UTC reporting."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Independent labels: {len(y)} (tree={(y==1).sum()}, non-tree={(y==0).sum()})")
    print(f"Balanced accuracy: {result['balanced_accuracy']:.3f}")
    print(f"Tree precision: {result['precision_tree']:.3f}")
    print(f"Tree recall: {result['recall_tree']:.3f}")
    print(f"Tree F1: {result['f1_tree']:.3f}")
    print(f"Tree IoU: {result['iou_tree']:.3f}")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Wrote {args.output}")


if __name__=="__main__":main()
