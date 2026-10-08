from scripts.prepare_utc_label_efficiency_manifests import nested_train_order,build_manifest

def _records():
    rows=[]
    i=1
    for br,bc in [(0,2),(0,3),(1,0)]:
        for j in range(4):
            rows.append({"id":i,"split":"train","block_row":br,"block_col":bc,"canopy_pixels":j+1,"total_pixels":10})
            i+=1
    rows += [
        {"id":100,"split":"val","block_row":0,"block_col":1,"canopy_pixels":1,"total_pixels":10},
        {"id":101,"split":"test","block_row":0,"block_col":0,"canopy_pixels":1,"total_pixels":10},
    ]
    return rows

def test_nested_order_balances_blocks_by_depth():
    rows=_records()
    order=nested_train_order(rows,42)
    first=order[:3]
    assert len({(r["block_row"],r["block_col"]) for r in first})==3
    first_two=order[:6]
    counts={}
    for r in first_two:
        k=(r["block_row"],r["block_col"]);counts[k]=counts.get(k,0)+1
    assert set(counts.values())=={2}

def test_manifests_are_nested_and_holdouts_unchanged():
    dm={"records":_records()}
    m25=build_manifest(dm,.25,42)
    m50=build_manifest(dm,.50,42)
    m100=build_manifest(dm,1.0,42)
    s25=set(m25["label_efficiency"]["train_patch_ids"])
    s50=set(m50["label_efficiency"]["train_patch_ids"])
    s100=set(m100["label_efficiency"]["train_patch_ids"])
    assert s25 <= s50 <= s100
    for m in (m25,m50,m100):
        ids={r["id"] for r in m["records"] if r["split"]!="train"}
        assert ids=={100,101}
