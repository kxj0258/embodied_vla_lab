#!/usr/bin/env python3
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/presentation/phase4_viewpoint_closeout_v1_20260828"
PNG=OUT/"phase4_viewpoint_evidence_chain.png"; SVG=OUT/"phase4_viewpoint_evidence_chain.svg"
MANIFEST=ROOT/"results/phase4_viewpoint_closeout_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists() or MANIFEST.exists(): raise SystemExit("REFUSING_TO_OVERWRITE")
    OUT.mkdir(parents=True); fig,ax=plt.subplots(figsize=(15,5.6)); ax.set_xlim(0,15); ax.set_ylim(0,6); ax.axis("off")
    cards=[
      (0.35,"4A · Camera gate","128/640 RGB valid\nMAE 76.46 / 77.50","PIPELINE PASS","#dbeafe","#1d4ed8"),
      (3.35,"4B · Direct substitution","Agent 10/15\nWrist 0/15","NO-GO","#fee2e2","#b91c1c"),
      (6.35,"4C/4D · Wrist QLoRA","Loss −45.77%\nAcc. +7.73 pp\nL1 −32.43%","OFFLINE PASS","#dcfce7","#15803d"),
      (9.35,"4E · Closed loop","Base 0/3\nAdapter 0/3\n3×600 timeout","RECOVERY NO-GO","#fee2e2","#b91c1c"),
      (12.35,"Frozen conclusion","Offline fit ≠\nclosed-loop transfer","FINAL UNTOUCHED","#f1f5f9","#334155")]
    for i,(x,title,body,status,face,edge) in enumerate(cards):
        box=FancyBboxPatch((x,.9),2.3,3.8,boxstyle="round,pad=.08,rounding_size=.12",facecolor=face,edgecolor=edge,linewidth=2); ax.add_patch(box)
        ax.text(x+1.15,4.15,title,ha="center",va="center",fontsize=12,weight="bold",color=edge)
        ax.text(x+1.15,2.85,body,ha="center",va="center",fontsize=11,linespacing=1.45)
        ax.text(x+1.15,1.35,status,ha="center",va="center",fontsize=10,weight="bold",color=edge)
        if i<len(cards)-1: ax.add_patch(FancyArrowPatch((x+2.35,2.8),(cards[i+1][0]-.08,2.8),arrowstyle="-|>",mutation_scale=16,linewidth=1.8,color="#64748b"))
    ax.text(7.5,5.45,"Phase 4 · OpenVLA viewpoint robustness evidence chain",ha="center",fontsize=17,weight="bold")
    ax.text(7.5,.3,"Development-only, preregistered gates · failures preserved · offline metrics never treated as success/reward",ha="center",fontsize=11)
    fig.tight_layout(); fig.savefig(PNG,dpi=180,bbox_inches="tight"); fig.savefig(SVG,bbox_inches="tight"); plt.close(fig)
    MANIFEST.write_text(f'''asset: phase4_viewpoint_closeout_v1\nclassification: PASS\nsources:\n  - results/phase4a_openvla_viewpoint_camera_gate_v2.yaml\n  - results/phase4b_openvla_viewpoint_dev30_v1.yaml\n  - results/phase4d_openvla_wrist_step20_vs_base_fullval889_paired.yaml\n  - results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml\nartifacts:\n  png: {{path: {PNG.relative_to(ROOT)}, bytes: {PNG.stat().st_size}, sha256: {sha(PNG)}}}\n  svg: {{path: {SVG.relative_to(ROOT)}, bytes: {SVG.stat().st_size}, sha256: {sha(SVG)}}}\nvisual_qa: PENDING\nboundary: CPU-only synthesis of frozen development evidence; not final-set or general viewpoint-robustness evidence.\n''')
    print(PNG)
if __name__=="__main__": main()
