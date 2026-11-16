#!/usr/bin/env python3
"""Read-only verifier for the Phase 6A project index."""
import hashlib,re
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/"results/project_closeout_v3_phase6a_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=yaml.safe_load(MANIFEST.read_text()); checks={"classification":m["classification"]=="PASS"}
 for i,x in enumerate(m["documents"],1):
  p=ROOT/x["path"]; checks[f"document_{i}"]=p.is_file() and p.stat().st_size==x["bytes"] and sha(p)==x["sha256"]
 for i,p in enumerate(m["canonical_manifests"],1): checks[f"evidence_{i}"]=(ROOT/p).is_file()
 for i,p in enumerate(m["read_only_verifiers"],1): checks[f"verifier_{i}"]=(ROOT/p).is_file()
 refs=sorted(set(re.findall(r"`((?:results|configs|scripts|docs)/[^` ;]+)`",(ROOT/"docs/repro_index.md").read_text()))); missing=[p for p in refs if not (ROOT/p).exists()]; checks["references"]=not missing
 out={"classification":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"repro_refs_checked":len(refs),"missing_refs":missing,"cuda_initialized":False,"environment_steps":0}; print(yaml.safe_dump(out,sort_keys=False)); raise SystemExit(0 if out["classification"]=="PASS" else 1)
if __name__=="__main__": main()
