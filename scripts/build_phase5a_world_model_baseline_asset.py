#!/usr/bin/env python3
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/"results/phase5a_libero_world_model_data_contract_v1.yaml"; OUT=ROOT/"outputs/presentation/phase5a_world_model_data_v1_20260828"; PNG=OUT/"copy_frame_baseline_by_horizon.png"; SVG=OUT/"copy_frame_baseline_by_horizon.svg"; MANIFEST=ROOT/"results/phase5a_world_model_data_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists() or MANIFEST.exists(): raise SystemExit("REFUSING_TO_OVERWRITE")
    r=yaml.safe_load(SOURCE.read_text()); hs=[1,2,4,8]; OUT.mkdir(parents=True)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8)); colors={"agentview":"#2563eb","eye_in_hand":"#dc2626"}
    for v,label in [("agentview","Agentview"),("eye_in_hand","Wrist / eye-in-hand")]:
        rows=r["copy_current_frame_validation"][v]; a.plot(hs,[rows[h]["pixel_mae"] for h in hs],marker="o",linewidth=2.5,label=label,color=colors[v]); b.plot(hs,[rows[h]["psnr_db"] for h in hs],marker="o",linewidth=2.5,label=label,color=colors[v])
    a.set_title("Copy-current-frame MAE",weight="bold"); a.set_ylabel("Pixel MAE ↓"); b.set_title("Copy-current-frame PSNR",weight="bold"); b.set_ylabel("PSNR (dB) ↑")
    for ax in (a,b): ax.set_xlabel("Prediction horizon (control steps)"); ax.set_xticks(hs); ax.grid(alpha=.22); ax.legend()
    fig.suptitle("Phase 5A · Frozen validation baseline difficulty",fontsize=15,weight="bold"); fig.text(.5,.01,"40/10 episode split · no model training · copy-frame baseline is not learned world-model performance",ha="center"); fig.tight_layout(rect=[0,.06,1,.92]); fig.savefig(PNG,dpi=180); fig.savefig(SVG); plt.close(fig)
    MANIFEST.write_text(f'''asset: phase5a_world_model_data_v1\nclassification: PASS\nsource: {{path: {SOURCE.relative_to(ROOT)}, sha256: {sha(SOURCE)}}}\nartifacts:\n  png: {{path: {PNG.relative_to(ROOT)}, bytes: {PNG.stat().st_size}, sha256: {sha(PNG)}}}\n  svg: {{path: {SVG.relative_to(ROOT)}, bytes: {SVG.stat().st_size}, sha256: {sha(SVG)}}}\nvisual_qa: PENDING\nboundary: CPU-only visualization of copy-frame validation difficulty; no learned model or environment evidence.\n'''); print(PNG)
if __name__=="__main__": main()
