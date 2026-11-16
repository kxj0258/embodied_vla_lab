#!/usr/bin/env python3
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/presentation/phase4e_wrist_no_go_summary_v1_20260828"; PNG=OUT/"phase4e_offline_gain_closed_loop_no_go.png"; MANIFEST=ROOT/"results/phase4e_wrist_no_go_summary_v1_manifest.yaml"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists() or MANIFEST.exists(): raise SystemExit("REFUSING_TO_OVERWRITE")
    OUT.mkdir(parents=True); fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8))
    a.bar(["Loss\n↓45.77%","Token acc.\n+7.73 pp","Continuous L1\n↓32.43%"],[45.77,7.73,32.43],color="#16a34a"); a.set_title("Teacher-forced wrist fullval889",weight="bold"); a.set_ylabel("Improvement magnitude"); a.grid(axis="y",alpha=.2); a.set_axisbelow(True)
    b.bar(["Frozen base","Wrist QLoRA\nstep-20"],[0,0],color=["#64748b","#dc2626"]); b.set_ylim(0,3.3); b.set_yticks([0,1,2,3]); b.set_ylabel("Strict successes / 3"); b.set_title("Task-7 wrist closed loop",weight="bold"); b.grid(axis="y",alpha=.2); b.set_axisbelow(True)
    for x in [0,1]: b.text(x,.08,"0/3\n3 × 600-step timeout",ha="center",va="bottom",weight="bold")
    fig.suptitle("Offline adaptation improved; closed-loop recovery did not",fontsize=15,weight="bold"); fig.text(.5,.015,"Same frozen step-20 adapter · development task only · final tasks untouched",ha="center"); fig.tight_layout(rect=[0,.06,1,.91]); fig.savefig(PNG,dpi=180); plt.close(fig)
    MANIFEST.write_text(f'''asset: phase4e_wrist_no_go_summary_v1\nclassification: PASS\nsource: results/phase4e_openvla_wrist_adapter_task7_dev3_v1.yaml\nartifact: {{path: {PNG.relative_to(ROOT)}, bytes: {PNG.stat().st_size}, sha256: {sha(PNG)}}}\nvisual_qa: PENDING\nboundary: CPU-only comparison; offline metrics are not environment success or reward.\n'''); print(PNG)
if __name__=="__main__": main()
