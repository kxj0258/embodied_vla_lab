#!/usr/bin/env python3
"""CPU-only static contract gate for the Phase 5E residual smoke."""
import hashlib,json,sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/train/phase5e_residual_world_model_dev_smoke_v1.yaml"
TRAINER=ROOT/"scripts/train_phase5e_residual_world_model_smoke.py"
MODEL=ROOT/"scripts/phase5_residual_world_model.py"
RESULT=ROOT/"results/phase5e_residual_world_model_dev100_pair_v1_static_v2.yaml"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    config=yaml.safe_load(CONFIG.read_text())
    manifest_path=ROOT/config["fixture"]["manifest"]
    manifest=json.loads(manifest_path.read_text())
    train_refs=manifest["train_windows"]
    dev_refs=manifest["development_windows"]
    phase5d=yaml.safe_load((ROOT/"results/phase5d_residual_world_model_static_v1.yaml").read_text())
    frozen_train=set(phase5d["split"]["train"])
    frozen_dev=set(phase5d["split"]["development"])
    confirmation=set(phase5d["split"]["prior_exposed_confirmation"])
    train_eps={ref["episode"] for ref in train_refs}
    dev_eps={ref["episode"] for ref in dev_refs}
    checks={
        "phase5d_static_pass": phase5d["classification"]=="PASS",
        "fixture_hash": sha(manifest_path)=="6a5da23a0c1e3179ba24749eaadb9f91a1e34df6187275f44aedd69088a0d6b4",
        "fixture_counts": len(train_refs)==256 and len(dev_refs)==128,
        "episode_safe_h4": all(ref["horizon"]==4 and ref["start"]>=0 for ref in train_refs+dev_refs),
        "nested_split_exact": train_eps<=frozen_train and dev_eps<=frozen_dev and not train_eps&dev_eps,
        "confirmation_excluded": not confirmation&(train_eps|dev_eps) and manifest["prior_exposed_confirmation_included"] is False,
        "paired_models": config["models"]==["residual_action_free","residual_action_conditioned"],
        "paired_protocol": config["training"]["identical_batch_order"] and config["training"]["identical_encoder_decoder_initialization"],
        "smoke_budget": config["training"]["updates_per_model"]==100 and config["training"]["batch_size"]==8,
        "development_horizons": config["development_evaluation"]["horizons"]==[1,2,3,4],
        "go_rule_exact": config["development_evaluation"]["conditioned_go_requires_h4_better_than"]==["copy_current_frame","residual_action_free"],
        "no_selection_or_confirmation": config["development_evaluation"]["checkpoint_selection"] is False and config["development_evaluation"]["prior_exposed_confirmation_execution"] is False,
        "formal_training_forbidden": config["development_evaluation"]["formal_training_authorized"] is False,
        "fresh_targets": not (ROOT/config["output_dir"]).exists() and not (ROOT/config["structured_result"]).exists() and not (ROOT/config["raw_log"]).exists(),
        "trainer_frozen": sha(TRAINER)=="644adf4bef6fe63de1375d41b58f967dfc8543c757974a3a4e701d4d11507927",
        "model_frozen": sha(MODEL)=="692399bf18c356523ddfea5d7ef7a7553e7bf0d529c154975821c54858c65b83",
        "config_frozen": sha(CONFIG)=="f46a74b2d1c96621ff3bd87f9d12bd1a7a7ec914dc062fbe0c5ee75bfbea897d",
        "no_torch_or_cuda_initialized": "torch" not in sys.modules,
    }
    classification="PASS" if all(checks.values()) else "FAIL"
    result={
        "classification":classification,
        "experiment":"phase5e_residual_world_model_dev100_pair_v1_static_v2",
        "checks":checks,
        "checks_passed":sum(checks.values()),
        "checks_total":len(checks),
        "fixture_sha256":sha(manifest_path),
        "source_hashes":{"config":sha(CONFIG),"trainer":sha(TRAINER),"model":sha(MODEL)},
        "split":{"fixture_train_episodes":sorted(train_eps),"fixture_development_episodes":sorted(dev_eps),"confirmation_episodes":sorted(confirmation)},
        "cuda_initialized":False,"training_steps":0,"optimizer_steps":0,"environment_steps":0,"model_predictions":0,
        "boundary":"Static validation only; no torch import, CUDA, training, model forward, confirmation evaluation or environment work.",
    }
    if RESULT.exists(): raise FileExistsError(f"refusing overwrite: {RESULT}")
    RESULT.write_text(yaml.safe_dump(result,sort_keys=False))
    print(yaml.safe_dump(result,sort_keys=False))
    raise SystemExit(0 if classification=="PASS" else 1)

if __name__=="__main__": main()
