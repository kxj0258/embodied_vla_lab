#!/usr/bin/env python3
"""Build a CPU-only quantitative figure from the frozen Phase 4B result."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/phase4b_openvla_viewpoint_dev30_v1.yaml"
OUT=ROOT/"outputs/presentation/phase4b_viewpoint_dev_summary_v2_20260828"
MANIFEST=ROOT/"results/phase4b_viewpoint_dev_summary_v2_manifest.yaml"
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for p in [OUT,MANIFEST]:
        if p.exists(): raise FileExistsError(f"refusing overwrite: {p}")
    r=yaml.safe_load(SOURCE.read_text()); OUT.mkdir(parents=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    views=["Agent view","Eye in hand"]; successes=[r["by_view"]["agentview"]["success_count"],r["by_view"]["eye_in_hand"]["success_count"]]
    bars=axes[0].bar(views,successes,color=["#33a6d8","#ef8354"],width=.58); axes[0].set_ylim(0,15.8); axes[0].set_ylabel("Strict successes / 15"); axes[0].set_title("Frozen OpenVLA base by viewpoint")
    for b,v in zip(bars,successes): axes[0].text(b.get_x()+b.get_width()/2,v+.35,f"{v}/15",ha="center",weight="bold",fontsize=13)
    labels=["Both","Agent only","Wrist only","Neither"]; counts=[r["paired_outcome_counts"][x] for x in ["both_success","agentview_only","eye_in_hand_only","neither_success"]]
    bars=axes[1].bar(labels,counts,color=["#55a868","#33a6d8","#ef8354","#777777"]); axes[1].set_ylim(0,10.8); axes[1].set_ylabel("Paired task/init states / 15"); axes[1].set_title("Paired closed-loop outcomes")
    for b,v in zip(bars,counts): axes[1].text(b.get_x()+b.get_width()/2,v+.25,str(v),ha="center",weight="bold",fontsize=12)
    fig.suptitle("Phase 4B development: direct wrist-view substitution is NO-GO",fontsize=15,weight="bold",y=.96)
    fig.subplots_adjust(left=.07,right=.98,bottom=.19,top=.82,wspace=.24)
    fig.text(.5,.045,"Development only · exact paired two-sided p = 0.001953 · final tasks untouched",ha="center",fontsize=10)
    png=OUT/"phase4b_agentview_vs_eye_in_hand.png"; svg=OUT/"phase4b_agentview_vs_eye_in_hand.svg"; fig.savefig(png,dpi=180); fig.savefig(svg); plt.close(fig)
    manifest={"manifest":"phase4b_viewpoint_dev_summary_v2_20260828","classification":"PASS","generation_mode":"offline_from_frozen_result","environment_steps":0,"model_predictions":0,"supersedes_noncanonical":"results/phase4b_viewpoint_dev_summary_v1_manifest.yaml","source":{"path":SOURCE.relative_to(ROOT).as_posix(),"sha256":digest(SOURCE)},"outputs":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p)} for p in [png,svg]],"boundary":"Development-only visualization; it does not add episodes or authorize final evaluation."}
    MANIFEST.write_text(yaml.safe_dump(manifest,sort_keys=False)); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
