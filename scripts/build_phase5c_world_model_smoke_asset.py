#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/"outputs/train/phase5c_world_model_train100_pair_v1_seed20260831/result.json"; OUT=ROOT/"outputs/presentation/phase5c_world_model_smoke_v1_20260829"; PNG=OUT/"phase5c_training_and_validation.png"; SVG=OUT/"phase5c_training_and_validation.svg"; MANIFEST=ROOT/"results/phase5c_world_model_smoke_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists() or MANIFEST.exists(): raise SystemExit("REFUSING_TO_OVERWRITE")
 r=json.loads(SOURCE.read_text()); OUT.mkdir(parents=True); fig,(a,b)=plt.subplots(1,2,figsize=(13,4.8)); colors={"action_free":"#2563eb","action_conditioned":"#dc2626"}
 for name,label in [("action_free","Action-free"),("action_conditioned","Action-conditioned")]: a.plot([x["step"] for x in r["models"][name]["history"]],[x["loss"] for x in r["models"][name]["history"]],label=label,color=colors[name],alpha=.9)
 a.set_title("Paired 100-update training smoke",weight="bold"); a.set_xlabel("Update"); a.set_ylabel("Reconstruction + prediction L1"); a.grid(alpha=.2); a.legend()
 hs=[1,2,3,4]; b.plot(hs,[r["copy_frame_validation"][str(h)]["pixel_mae_0_255"] for h in hs],marker="o",label="Copy frame",color="#16a34a",linewidth=2.5)
 for name,label in [("action_free","Action-free"),("action_conditioned","Action-conditioned")]: b.plot(hs,[r["models"][name]["post_validation"][str(h)]["pixel_mae_0_255"] for h in hs],marker="o",label=label,color=colors[name],linewidth=2.5)
 b.set_title("Frozen validation after smoke",weight="bold"); b.set_xlabel("Horizon"); b.set_ylabel("Pixel MAE ↓"); b.set_xticks(hs); b.grid(alpha=.2); b.legend()
 fig.suptitle("Phase 5C · Optimization works, but learned baselines do not beat copy-frame",fontsize=14,weight="bold"); fig.text(.5,.01,"Engineering PASS · offline baseline NO-GO · no checkpoint selected",ha="center"); fig.tight_layout(rect=[0,.06,1,.92]); fig.savefig(PNG,dpi=180); fig.savefig(SVG); plt.close(fig)
 MANIFEST.write_text(f'''asset: phase5c_world_model_smoke_v1\nclassification: PASS\nsource: {{path: {SOURCE.relative_to(ROOT)}, sha256: {sha(SOURCE)}}}\nartifacts:\n  png: {{path: {PNG.relative_to(ROOT)}, bytes: {PNG.stat().st_size}, sha256: {sha(PNG)}}}\n  svg: {{path: {SVG.relative_to(ROOT)}, bytes: {SVG.stat().st_size}, sha256: {sha(SVG)}}}\nvisual_qa: PENDING\ndecision: ENGINEERING_PASS_OFFLINE_BASELINE_NO_GO\nboundary: CPU-only visualization of frozen smoke evidence; no checkpoint selection or world-model/planning/environment claim.\n'''); print(PNG)
if __name__=="__main__": main()
