#!/usr/bin/env python3
"""Create deterministic episode-safe window references; copies no image data."""
import hashlib,json,random
from pathlib import Path
import h5py,yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/train/phase5c_world_model_train_smoke_v1.yaml"; OUT=ROOT/"outputs/reference/phase5c_world_model_train256_val128_fixture_v1_seed20260831"; MANIFEST=OUT/"manifest.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    c=yaml.safe_load(CONFIG.read_text()); contract=yaml.safe_load((ROOT/"configs/dataset/phase5a_libero_world_model_contract_v1.yaml").read_text()); h=c["fixture"]["horizon"]; rng=random.Random(c["seed"]); pools={}
    with h5py.File(ROOT/c["dataset"]["path"],"r") as f:
        for split,key,nwant in [("train","train_episodes",c["fixture"]["train_windows"]),("validation","validation_episodes",c["fixture"]["validation_windows"])]:
            refs=[]
            for ep in contract["split"][key]:
                n=len(f[f"data/demo_{ep}/actions"]); refs.extend({"episode":ep,"start":t,"horizon":h} for t in range(n-h))
            rng.shuffle(refs); pools[split]=refs[:nwant]
    OUT.mkdir(parents=True); manifest={"fixture":"phase5c_world_model_train256_val128_fixture_v1_seed20260831","seed":c["seed"],"dataset_path":c["dataset"]["path"],"dataset_sha256":c["dataset"]["sha256"],"image_source":c["dataset"]["image_source"],"action_source":c["dataset"]["action_source"],"horizon":h,"train_windows":pools["train"],"validation_windows":pools["validation"],"boundary":"Reference-only deterministic windows; no images copied, no CUDA/model/environment work."}; MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps({"classification":"PASS","manifest":str(MANIFEST.relative_to(ROOT)),"sha256":sha(MANIFEST),"train":len(pools["train"]),"validation":len(pools["validation"])},indent=2))
if __name__=="__main__": main()
