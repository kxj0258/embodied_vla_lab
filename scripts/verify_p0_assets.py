#!/usr/bin/env python3
"""Read-only verification of P0 presentation assets and frozen source hashes."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
FFPROBE=Path("/home/kxj/miniforge3/envs/evla-core/bin/ffprobe")
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=ROOT/"results/p0_mvp_v2_manifest.yaml"); a=p.parse_args()
    manifest=yaml.safe_load(a.manifest.read_text()); checked=0
    for group in ["source_artifacts","videos","figures"]:
        for item in manifest[group]:
            path=ROOT/item["path"]; assert path.is_file(),path; assert path.stat().st_size==item["bytes"],path; assert digest(path)==item["sha256"],path; checked+=1
            if path.suffix==".mp4":
                info=json.loads(subprocess.check_output([str(FFPROBE),"-v","error","-show_entries","stream=codec_name,width,height,nb_frames","-of","json",str(path)])); stream=info["streams"][0]
                assert stream["codec_name"]=="h264" and stream["width"]==1280 and stream["height"]==720 and int(stream["nb_frames"])>0
    hd_manifest=ROOT/"results/p0_hd_comparisons_v1_manifest.yaml"
    if hd_manifest.is_file():
        hd=yaml.safe_load(hd_manifest.read_text())
        assert hd["classification"]=="PASS" and hd["environment_steps"]==0 and hd["model_predictions"]==0
        for item in hd["sources"]:
            path=ROOT/item["path"]; assert path.is_file(),path; assert digest(path)==item["sha256"],path; checked+=1
        for item in hd["outputs"]:
            path=ROOT/item["path"]; assert path.is_file(),path; assert path.stat().st_size==item["bytes"],path; assert digest(path)==item["sha256"],path
            info=json.loads(subprocess.check_output([str(FFPROBE),"-v","error","-show_entries","stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames","-of","json",str(path)]))["streams"][0]
            expected=item["probe"]
            assert info["codec_name"]=="h264" and info["width"]==1920 and info["height"]==1080 and info["pix_fmt"]=="yuv420p"
            assert info["r_frame_rate"]=="20/1" and info["nb_frames"]==expected["nb_frames"]
            checked+=1
    breadth_hd_path=ROOT/"results/p0_breadth_hd_10task_v1_manifest.yaml"
    if breadth_hd_path.is_file():
        breadth=yaml.safe_load(breadth_hd_path.read_text())
        assert breadth["classification"]=="PASS" and breadth["policy_success"]=="10/10"
        assert breadth["total_environment_steps"]==breadth["total_model_predictions"]==1296
        assert len(breadth["captures"])==10
        for capture in breadth["captures"]:
            result=ROOT/capture["result"]
            assert result.is_file() and digest(result)==capture["result_sha256"],result
            result_data=yaml.safe_load(result.read_text())
            assert result_data["classification"]=="PASS"
            assert result_data["steps"]==capture["expected_and_observed_steps"]
            for key in ["presentation_video","clean_video"]:
                item=capture[key]; path=ROOT/item["path"]
                assert path.is_file() and path.stat().st_size==item["bytes"] and digest(path)==item["sha256"],path
                checked+=1
            if "canonical_presentation_video" in capture:
                item=capture["canonical_presentation_video"]; path=ROOT/item["path"]
                assert path.is_file() and path.stat().st_size==item["bytes"] and digest(path)==item["sha256"],path
                checked+=1
            checked+=1
    montage_path=ROOT/"results/p0_breadth_10task_montage_v1_manifest.yaml"
    if montage_path.is_file():
        montage=yaml.safe_load(montage_path.read_text())
        assert montage["classification"]=="PASS" and montage["environment_steps"]==0 and montage["model_predictions"]==0
        assert montage["visual_qa"]["classification"]=="PASS"
        for item in montage["outputs"]:
            path=ROOT/item["path"]
            assert path.is_file() and path.stat().st_size==item["bytes"] and digest(path)==item["sha256"],path
            info=json.loads(subprocess.check_output([str(FFPROBE),"-v","error","-show_entries","stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames","-of","json",str(path)]))["streams"][0]
            expected=item["probe"]
            assert info["codec_name"]=="h264" and info["width"]==1920 and info["height"]==1080 and info["pix_fmt"]=="yuv420p"
            assert info["r_frame_rate"]=="20/1" and info["nb_frames"]==expected["nb_frames"]
            checked+=1
    print(json.dumps({"classification":"PASS","manifest":str(a.manifest),"files_verified":checked,"environment_steps":0,"model_inferences":0},indent=2))
if __name__=="__main__": main()
