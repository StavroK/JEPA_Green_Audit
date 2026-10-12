import json
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from scripts.rebuild_utc_dataset_raw_v2 import rebuild

def test_rebuild_preserves_masks_and_raw_pixels(tmp_path):
    src=tmp_path/"src.tif"
    arr=np.zeros((3,20,20),dtype=np.uint8)
    arr[0]=10;arr[1]=20;arr[2]=30
    with rasterio.open(src,"w",driver="GTiff",width=20,height=20,count=3,dtype="uint8",transform=from_origin(0,20,1,1)) as dst:
        dst.write(arr)

    v1=tmp_path/"v1";(v1/"masks").mkdir(parents=True)
    Image.fromarray(np.full((8,8),255,dtype=np.uint8)).save(v1/"masks"/"patch_001_train_b00.png")
    dm={"patch_size":8,"records":[{"id":1,"file":"patch_001_train_b00.png","mask_file":"patch_001_train_b00.png","x":2,"y":3,"width":8,"height":8,"split":"train","block_row":0,"block_col":0,"canopy_pixels":64,"total_pixels":64,"canopy_fraction":1.0}]}
    (v1/"dataset_manifest.json").write_text(json.dumps(dm),encoding="utf-8")

    out=tmp_path/"v2"
    rebuilt=rebuild(v1,src,out)
    rgb=np.asarray(Image.open(out/"images"/"patch_001_train_b00.png"))
    assert tuple(rgb[0,0])==(10,20,30)
    assert rebuilt["preprocessing"]=="raw_rgb_no_per_patch_stretch"
    assert (out/"masks"/"patch_001_train_b00.png").exists()
