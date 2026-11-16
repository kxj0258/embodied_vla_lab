#!/usr/bin/env python
"""Two-step BF16 GPU resource gate for deterministic ACT on the frozen LIBERO contract."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.dont_write_bytecode=True
import h5py, numpy as np, torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'libero/libero/datasets/libero_goal/turn_on_the_stove_demo.hdf5'
STATE_MEAN=np.array([0.14065630305038507,0.14608859918793637,0.3284340321192387,-2.654764316279583,0.012141825412267968,2.6063294360412943,1.0980553990782032,0.03135659827886598,-0.027591489386528608],np.float32)
STATE_STD=np.array([0.08660861826025365,0.22483161599744708,0.14652347014513997,0.16137885854998368,0.21233934303449734,0.33410315240152916,0.2398749348524432,0.01163933179420635,0.013850671871119304],np.float32)
ACTION_MEAN=np.array([-0.19369274312917523,0.21223271792615103,-0.34463935672280677,0.008190082809937212,0.03754140496859592,0.11458765051806254,-0.04284514141697004],np.float32)
ACTION_STD=np.array([0.23610090632509173,0.26188726968074433,0.279460241647471,0.03804072528474012,0.05093681370144726,0.17127109882587624,0.999081725314278],np.float32)

def main():
 assert torch.cuda.is_available(); torch.manual_seed(20260826); torch.cuda.manual_seed_all(20260826)
 device=torch.device('cuda:0'); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
 with h5py.File(DEMO,'r') as f:
  g=f['data/demo_1']; samples=[]
  for idx in [0,1]:
   state=np.concatenate([g['obs/joint_states'][idx],g['obs/gripper_states'][idx]]).astype(np.float32)
   action=np.zeros((20,7),np.float32); pad=np.ones(20,bool); n=min(20,len(g['actions'])-idx)
   action[:n]=(g['actions'][idx:idx+n]-ACTION_MEAN)/ACTION_STD; pad[:n]=False
   samples.append((np.moveaxis(g['obs/agentview_rgb'][idx],-1,0).astype(np.float32)/255,np.moveaxis(g['obs/eye_in_hand_rgb'][idx],-1,0).astype(np.float32)/255,(state-STATE_MEAN)/STATE_STD,action,pad))
 batch={
  'observation.images.agentview':torch.tensor(np.stack([x[0] for x in samples]),device=device),
  'observation.images.eye_in_hand':torch.tensor(np.stack([x[1] for x in samples]),device=device),
  'observation.state':torch.tensor(np.stack([x[2] for x in samples]),device=device),
  'action':torch.tensor(np.stack([x[3] for x in samples]),device=device),
  'action_is_pad':torch.tensor(np.stack([x[4] for x in samples]),device=device),
 }
 cfg=ACTConfig(input_features={'observation.images.agentview':PolicyFeature(FeatureType.VISUAL,(3,128,128)),'observation.images.eye_in_hand':PolicyFeature(FeatureType.VISUAL,(3,128,128)),'observation.state':PolicyFeature(FeatureType.STATE,(9,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))},chunk_size=20,n_action_steps=5,use_vae=False,pretrained_backbone_weights=None,device='cuda')
 cfg.validate_features(); policy=ACTPolicy(cfg).to(device).train(); opt=torch.optim.AdamW(policy.parameters(),lr=1e-5,weight_decay=1e-4)
 steps=int(os.environ.get('EVLA_OPTIMIZER_STEPS','2')); assert 1 <= steps <= 1000
 losses=[]; grads=[]; start=time.perf_counter()
 for _ in range(steps):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=policy(batch)
  assert torch.isfinite(loss); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.0)
  assert torch.isfinite(grad); opt.step(); losses.append(float(loss.detach())); grads.append(float(grad.detach()))
 torch.cuda.synchronize(); elapsed=time.perf_counter()-start
 overfit_pass = steps <= 2 or losses[-1] < losses[0] * 0.25
 print('classification:', 'PASS' if overfit_pass else 'FAIL'); print('parameters:',sum(p.numel() for p in policy.parameters()))
 print('loss_first_last_min:',losses[0],losses[-1],min(losses)); print('grad_norm_first_last_max:',grads[0],grads[-1],max(grads)); print('optimizer_steps:',steps)
 print('elapsed_s:',elapsed); print('steps_per_s:',steps/elapsed)
 print('peak_allocated_bytes:',torch.cuda.max_memory_allocated()); print('peak_reserved_bytes:',torch.cuda.max_memory_reserved())
 assert overfit_pass, (losses[0],losses[-1])
 print('FINAL: PASS_PHASE2B_ACT_LIBERO_GPU_RESOURCE_GATE')
if __name__=='__main__': main()
