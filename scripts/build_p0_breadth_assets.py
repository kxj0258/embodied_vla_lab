#!/usr/bin/env python3
"""Build CPU-only P0-Breadth quantitative assets from the frozen 10-task result."""
from pathlib import Path
import hashlib, json
import matplotlib.pyplot as plt
import yaml
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"results/p0_breadth_openvla_base_10task30_v2.yaml"; OUT=ROOT/"outputs/presentation/p0_breadth_summary_v1_20260828"; MAN=ROOT/"results/p0_breadth_summary_v1_manifest.yaml"
def sha(p): h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 if OUT.exists() or MAN.exists(): raise FileExistsError("refusing overwrite")
 OUT.mkdir(parents=True); r=yaml.safe_load(SRC.read_text()); rows=r["task_summaries"]; labels=[f"T{x['task_id']}" for x in rows]; values=[x["success_count"] for x in rows]
 fig,ax=plt.subplots(figsize=(12,5.8)); colors=["#38bdf8" if v==3 else "#fbbf24" if v==2 else "#fb7185" for v in values]; bars=ax.bar(labels,values,color=colors,width=.72)
 ax.set_ylim(0,3.45); ax.set_yticks([0,1,2,3]); ax.set_ylabel("Strict successes / 3 fixed init states"); ax.set_xlabel("LIBERO-Goal task ID"); ax.set_title("Frozen OpenVLA-7B breadth gate: 20/30 successes, 10/10 task coverage",weight="bold"); ax.grid(axis="y",alpha=.22)
 [ax.text(b.get_x()+b.get_width()/2,b.get_height()+.08,f"{v}/3",ha="center",weight="bold") for b,v in zip(bars,values)]
 ax.text(.01,-.23,"T0 drawer | T1 bowl→stove | T2 bottle→cabinet | T3 drawer+bowl | T4 bowl→cabinet\nT5 push plate | T6 cream cheese→bowl | T7 stove switch | T8 bowl→plate | T9 bottle→rack",transform=ax.transAxes,fontsize=9); fig.tight_layout()
 png=OUT/"openvla_10task_success.png"; svg=OUT/"openvla_10task_success.svg"; fig.savefig(png,dpi=180); fig.savefig(svg); plt.close(fig)
 summary=OUT/"summary.json"; summary.write_text(json.dumps({"success":"20/30","task_coverage":"10/10","wilson_95_percent":[48.78,80.77],"timeouts":10,"per_task":{str(x['task_id']):f"{x['success_count']}/3" for x in rows},"boundary":r['boundary']},indent=2)+"\n")
 items=[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in (png,svg,summary)]
 m={"manifest":"p0_breadth_summary_v1_20260828","classification":"PASS","generation_mode":"offline_from_frozen_structured_result","environment_steps":0,"model_predictions":0,"source":{"path":SRC.relative_to(ROOT).as_posix(),"sha256":sha(SRC)},"artifacts":items,"boundary":"Quantitative presentation only; does not change the frozen 20/30 result."}; MAN.write_text(yaml.safe_dump(m,sort_keys=False)); print(json.dumps(m,indent=2))
if __name__=="__main__": main()
