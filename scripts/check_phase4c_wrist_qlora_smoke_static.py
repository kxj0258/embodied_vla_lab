#!/usr/bin/env python3
"""CUDA-free static contract check for Phase 4C wrist QLoRA smoke."""
from pathlib import Path
import ast,hashlib,json,yaml
ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/'configs/train/openvla_phase4c_wrist_qlora_train_smoke_v1.yaml'; ENTRY=ROOT/'scripts/train_phase4c_openvla_wrist_qlora_smoke.py'; TRAINER=ROOT/'scripts/train_phase3b_openvla_qlora_smoke_v3.py'; RESULT=ROOT/'results/phase4c_openvla_wrist_qlora_train_smoke_v2_static.yaml'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=yaml.safe_load(CONFIG.read_text()); fixture=ROOT/c['smoke_fixture']; m=json.loads(fixture.read_text()); ast.parse(ENTRY.read_text()); ast.parse(TRAINER.read_text())
 checks={'new_experiment':c['experiment_id'].startswith('phase4c_'),'canonical_revision':c['checkpoint_revision']=='fa5ae1e7509348889295bba8e08621d8b55e9baf','wrist_source':c['dataset']['image_source']=='obs/eye_in_hand_rgb' and m['image_source']=='obs/eye_in_hand_rgb','fixture_hash':digest(fixture)==c['smoke_fixture_sha256'],'fixture_counts':m['train_samples']==20 and m['validation_smoke_samples']==10 and len(m['items'])==30,'all_items_wrist':all(x['view']=='eye_in_hand' for x in m['items']),'split_exact':len(c['split']['train_episodes'])==40 and len(c['split']['validation_episodes'])==10 and not set(c['split']['train_episodes'])&set(c['split']['validation_episodes']),'rank8_contract':c['lora']=={'rank':8,'alpha':8,'dropout':0.0,'target_modules':'all-linear','initialization':'gaussian'},'optimizer_contract':c['training_steps']==20 and c['batch_size']==1 and c['optimizer']['lr']==1e-4 and c['max_grad_norm']==1.0,'fresh_targets':not (ROOT/c['output_dir']).exists() and not (ROOT/c['raw_log']).exists(),'base_trainer_frozen':digest(TRAINER)=='e58f91aa0f4d687ee9e14ac57d8421a46a00c60c609558452f3887a40d15a178','v1_static_failure_preserved':(ROOT/'results/phase4c_openvla_wrist_qlora_train_smoke_v1_static.yaml').is_file(),'wrapper_only_overrides_config':'trainer.CONFIG=' in ENTRY.read_text() and 'trainer.main()' in ENTRY.read_text()}
 result={'classification':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'config_sha256':digest(CONFIG),'fixture_sha256':digest(fixture),'entrypoint_sha256':digest(ENTRY),'base_trainer_sha256':digest(TRAINER),'cuda_calls':0,'training_steps':0,'model_predictions':0,'expected_peak_vram_from_phase3b_bytes':7128219648,'boundary':c['boundary']}
 if RESULT.exists(): raise FileExistsError(f'refusing overwrite: {RESULT}')
 RESULT.write_text(yaml.safe_dump(result,sort_keys=False)); print(json.dumps(result,indent=2))
 if result['classification']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
