from scripts.run_utc_cross_site_multimodel_multiseed import summarize

def test_summarize_groups_model_and_budget():
    rows=[
      {"model":"unet","seed":42,"label_budget":25,"metrics":{"balanced_accuracy":.5,"precision_tree":1.0,"recall_tree":.1,"f1_tree":.18,"iou_tree":.1,"point_prevalence_bias_pp":-60}},
      {"model":"unet","seed":43,"label_budget":25,"metrics":{"balanced_accuracy":.6,"precision_tree":.8,"recall_tree":.2,"f1_tree":.32,"iou_tree":.19,"point_prevalence_bias_pp":-50}},
      {"model":"dinov2","seed":42,"label_budget":25,"metrics":{"balanced_accuracy":.8,"precision_tree":.9,"recall_tree":.8,"f1_tree":.85,"iou_tree":.74,"point_prevalence_bias_pp":-5}},
    ]
    s=summarize(rows)
    assert s["unet"]["25"]["n_seeds"]==2
    assert s["dinov2"]["25"]["n_seeds"]==1


def test_resolve_fundidora_dataset_direct(tmp_path):
    from scripts.run_utc_cross_site_multimodel_multiseed import resolve_fundidora_dataset
    d=tmp_path/"fundidora_dense"
    (d/"images").mkdir(parents=True)
    (d/"masks").mkdir()
    (d/"dataset_manifest.json").write_text('{"records":[]}',encoding="utf-8")
    assert resolve_fundidora_dataset(d)==d
