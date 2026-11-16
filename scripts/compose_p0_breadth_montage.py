#!/usr/bin/env python3
"""Repair Task-6 presentation and concatenate the ten frozen HD success demos."""
from pathlib import Path
import hashlib,json,subprocess,yaml
ROOT=Path(__file__).resolve().parents[1]; FF=Path('/home/kxj/miniforge3/envs/evla-core/bin/ffmpeg'); FP=Path('/home/kxj/miniforge3/envs/evla-core/bin/ffprobe'); OUT=ROOT/'outputs/presentation/p0_breadth_10task_montage_v1_20260828'; RES=ROOT/'results/p0_breadth_10task_montage_v1_manifest.yaml'; FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def sha(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def probe(p):return json.loads(subprocess.check_output([str(FP),'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration','-of','json',str(p)]))['streams'][0]
def main():
 if OUT.exists() or RES.exists():raise FileExistsError('refusing overwrite')
 OUT.mkdir(parents=True); source=yaml.safe_load((ROOT/'results/p0_breadth_hd_10task_v1_manifest.yaml').read_text()); caps=source['captures']
 clean6=ROOT/caps[6]['clean_video']['path']; repair=OUT/'task06_presentation_recomposed_v2.mp4'
 vf=("scale=1080:1080:flags=lanczos,pad=1920:1080:0:0:color=0x0d121c,"
 f"drawtext=fontfile={FONT}:text='OpenVLA-7B · frozen base · Task 6':fontcolor=0x64d2ff:fontsize=40:x=1140:y=70,"
 f"drawtext=fontfile={FONT}:text='put the cream cheese in the bowl':fontcolor=white:fontsize=31:x=1140:y=190,"
 f"drawtext=fontfile={FONT}:text='STRICT SUCCESS · 108 steps':fontcolor=0x7ee2b8:fontsize=38:x=1140:y=900")
 subprocess.run([str(FF),'-hide_banner','-loglevel','error','-i',str(clean6),'-vf',vf,'-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(repair)],check=True)
 videos=[]
 for i,c in enumerate(caps):videos.append(repair if i==6 else ROOT/c['presentation_video']['path'])
 montage=OUT/'openvla_10task_success_montage_hd.mp4'; cmd=[str(FF),'-hide_banner','-loglevel','error'];
 for p in videos:cmd+=['-i',str(p)]
 graph=''.join(f'[{i}:v]setpts=PTS-STARTPTS[v{i}];' for i in range(10))+''.join(f'[v{i}]' for i in range(10))+'concat=n=10:v=1:a=0[outv]'
 cmd+=['-filter_complex',graph,'-map','[outv]','-r','20','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(montage)];subprocess.run(cmd,check=True)
 items=[]
 for p in (repair,montage):items.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'probe':probe(p)})
 m={'manifest':'p0_breadth_10task_montage_v1_20260828','classification':'PASS','generation_mode':'offline_from_frozen_hd_captures','environment_steps':0,'model_predictions':0,'source_manifest':'results/p0_breadth_hd_10task_v1_manifest.yaml','task6_original_presentation_status':'PRESERVED_NONCANONICAL_VISUAL_CORRUPTION','outputs':items,'montage_task_order':list(range(10)),'montage_expected_frames':sum(c['expected_and_observed_steps'] for c in caps),'boundary':'Presentation-only recomposition; frozen 20/30 aggregate is unchanged.'};RES.write_text(yaml.safe_dump(m,sort_keys=False));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
