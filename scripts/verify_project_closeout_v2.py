#!/usr/bin/env python3
"""Read-only verifier for the current project closeout index."""
import hashlib,re
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/"results/project_closeout_v2_phase5e_manifest.yaml"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 m=yaml.safe_load(MANIFEST.read_text()); checks={"manifest_classification":m["classification"]=="PASS"}
 for i,item in enumerate(m["documents"],1):
  p=ROOT/item["path"]; checks[f"document_{i}"]=p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"]
 for i,path in enumerate(m["canonical_manifests"],1): checks[f"manifest_{i}"]=(ROOT/path).is_file()
 for i,path in enumerate(m["read_only_verifiers"],1): checks[f"verifier_{i}"]=(ROOT/path).is_file()
 text=(ROOT/"docs/repro_index.md").read_text(); refs=sorted(set(re.findall(r"`((?:results|configs|scripts|docs)/[^` ;]+)`",text))); missing=[x for x in refs if not (ROOT/x).exists()]; checks["repro_refs_exist"]=not missing
 payload={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"repro_refs_checked":len(refs),"missing_refs":missing,"cuda_initialized":False,"environment_steps":0}; print(yaml.safe_dump(payload,sort_keys=False)); raise SystemExit(0 if payload["classification"]=="PASS" else 1)
if __name__=="__main__": main()
