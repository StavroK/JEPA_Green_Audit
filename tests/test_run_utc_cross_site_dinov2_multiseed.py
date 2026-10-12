from scripts.run_utc_cross_site_dinov2_multiseed import summarize_results

def test_summarize_results_groups_budgets():
    rows=[
        {"seed":42,"label_budget":25,"metrics":{"balanced_accuracy":.8,"precision_tree":.9,"recall_tree":.7,"f1_tree":.79,"iou_tree":.65,"point_prevalence_bias_pp":-5}},
        {"seed":43,"label_budget":25,"metrics":{"balanced_accuracy":.82,"precision_tree":.91,"recall_tree":.72,"f1_tree":.80,"iou_tree":.67,"point_prevalence_bias_pp":-3}},
        {"seed":42,"label_budget":50,"metrics":{"balanced_accuracy":.84,"precision_tree":.92,"recall_tree":.75,"f1_tree":.82,"iou_tree":.70,"point_prevalence_bias_pp":-2}},
    ]
    s=summarize_results(rows)
    assert s["25"]["n_seeds"]==2
    assert s["50"]["n_seeds"]==1
    assert round(s["25"]["metrics"]["balanced_accuracy"]["mean"],3)==.81
