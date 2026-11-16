#!/usr/bin/env python3
"""Read-only verification of frozen Phase 4B development evidence."""
from pathlib import Path
import hashlib,json,yaml
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result_path=ROOT/"results/phase4b_openvla_viewpoint_dev30_v1.yaml"; r=yaml.safe_load(result_path.read_text())
    assert r["classification"]=="PASS" and r["development_decision"]=="NO_GO_DIRECT_EYE_IN_HAND_SUBSTITUTION"
    assert r["by_view"]["agentview"]["success_count"]==10 and r["by_view"]["eye_in_hand"]["success_count"]==0
    assert r["paired_outcome_counts"]=={"both_success":0,"agentview_only":10,"eye_in_hand_only":0,"neither_success":5}
    assert r["total_environment_steps"]==r["model_predictions"]==13663
    trajectory=ROOT/r["trajectory"]["path"]; assert digest(trajectory)==r["trajectory"]["sha256"]
    log=ROOT/r["host"]["raw_log"]["path"]; assert log.stat().st_size==r["host"]["raw_log"]["bytes"] and digest(log)==r["host"]["raw_log"]["sha256"]
    manifest_path=ROOT/"results/phase4b_viewpoint_dev_summary_v2_manifest.yaml"; m=yaml.safe_load(manifest_path.read_text())
    assert m["classification"]=="PASS" and m["environment_steps"]==m["model_predictions"]==0 and m["source"]["sha256"]==digest(result_path)
    for item in m["outputs"]:
        p=ROOT/item["path"]; assert p.stat().st_size==item["bytes"] and digest(p)==item["sha256"]
    print(json.dumps({"classification":"PASS","result_sha256":digest(result_path),"trajectory_lines":sum(1 for _ in trajectory.open()),"files_verified":5,"environment_steps_added":0,"model_predictions_added":0},indent=2))
if __name__=="__main__": main()
