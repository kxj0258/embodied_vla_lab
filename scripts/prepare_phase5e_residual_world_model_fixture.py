#!/usr/bin/env python3
import hashlib,json,random
from pathlib import Path
import h5py,yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml"; SPLIT=ROOT/"configs/train/phase5d_residual_world_model_static_v1.yaml"; OUT=ROOT/"outputs/reference/phase5e_residual_world_model_train256_dev128_fixture_v1_seed20260902"; MANIFEST=OUT/"manifest.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
 c=yaml.safe_load(CONFIG.read_text()); s=yaml.safe_load(SPLIT.read_text()); rng=random.Random(c["seed"]); h=c["fixture"]["horizon"]; pools={}
 with h5py.File(ROOT/c["dataset"]["path"],"r") as f:
  for name,episodes,count in [("train",s["split"]["train_episodes"],c["fixture"]["train_windows"]),("development",s["split"]["development_episodes"],c["fixture"]["development_windows"])]:
   refs=[]
   for ep in episodes:
    n=len(f[f"data/demo_{ep}/actions"]); refs.extend({"episode":ep,"start":t,"horizon":h} for t in range(n-h))
   rng.shuffle(refs); pools[name]=refs[:count]
 OUT.mkdir(parents=True); m={"fixture":"phase5e_residual_world_model_train256_dev128_fixture_v1_seed20260902","seed":c["seed"],"dataset_path":c["dataset"]["path"],"dataset_sha256":c["dataset"]["sha256"],"image_source":c["dataset"]["image_source"],"action_source":c["dataset"]["action_source"],"horizon":h,"train_windows":pools["train"],"development_windows":pools["development"],"prior_exposed_confirmation_included":False,"boundary":"Reference-only development fixture; no images copied or model/CUDA/environment work."}; MANIFEST.write_text(json.dumps(m,indent=2)+"\n"); print(json.dumps({"classification":"PASS","sha256":sha(MANIFEST),"train":len(pools["train"]),"development":len(pools["development"])},indent=2))
if __name__=="__main__": main()
