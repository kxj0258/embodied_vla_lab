#!/usr/bin/env python
"""Strict-offline deterministic ACT training on the frozen single-task LIBERO HDF5."""
from __future__ import annotations
import dataclasses, json, os, random, sys, time
from pathlib import Path
import h5py, numpy as np, torch, yaml
from torch.utils.data import DataLoader, Dataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

class Frames(Dataset):
 def __init__(self,path,episodes,stats):
  self.path=path; self.episodes=episodes; self.stats=stats; self.items=[]; self.file=None
  with h5py.File(path,'r') as f:
   for e in episodes:
    for i in range(len(f[f'data/demo_{e}/actions'])): self.items.append((e,i))
 def __len__(self): return len(self.items)
 def __getitem__(self,k):
  if self.file is None: self.file=h5py.File(self.path,'r')
  e,i=self.items[k]; g=self.file[f'data/demo_{e}']; n=min(20,len(g['actions'])-i)
  a=np.zeros((20,7),np.float32); pad=np.ones(20,bool)
  a[:n]=(g['actions'][i:i+n]-self.stats['am'])/self.stats['as']; pad[:n]=False
  s=np.concatenate([g['obs/joint_states'][i],g['obs/gripper_states'][i]]).astype(np.float32)
  return {'observation.images.agentview':torch.from_numpy(np.moveaxis(g['obs/agentview_rgb'][i],-1,0).astype(np.float32)/255),'observation.images.eye_in_hand':torch.from_numpy(np.moveaxis(g['obs/eye_in_hand_rgb'][i],-1,0).astype(np.float32)/255),'observation.state':torch.from_numpy((s-self.stats['sm'])/self.stats['ss']),'action':torch.from_numpy(a),'action_is_pad':torch.from_numpy(pad)}

def main():
 assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('WANDB_MODE')=='disabled'
 config_path=Path(os.environ.get('EVLA_TRAIN_CONFIG',ROOT/'configs/train/act_libero_phase2b_single_task_10k.yaml'))
 cfg=yaml.safe_load(config_path.read_text()); dc=yaml.safe_load((ROOT/cfg['dataset_contract']).read_text()); sr=yaml.safe_load((ROOT/cfg['dataset_result']).read_text())
 steps=int(os.environ.get('EVLA_TRAIN_STEPS',cfg['training']['steps'])); out=Path(os.environ.get('EVLA_OUTPUT_DIR',ROOT/cfg['output_dir'])); resume_path=os.environ.get('EVLA_RESUME_CHECKPOINT')
 cfg['experiment_id']=os.environ.get('EVLA_EXPERIMENT_ID',cfg['experiment_id']); cfg['training']['steps']=steps
 cfg['training']['validation_every']=int(os.environ.get('EVLA_VALIDATION_EVERY',cfg['training']['validation_every']))
 cfg['training']['checkpoint_every']=int(os.environ.get('EVLA_CHECKPOINT_EVERY',cfg['training']['checkpoint_every']))
 cfg['output_dir']=str(out)
 if out.exists() and not resume_path: raise FileExistsError(out)
 if resume_path and not out.exists(): raise FileNotFoundError(f'resume output_dir missing: {out}')
 if not out.exists():
  out.mkdir(parents=True); (out/'checkpoints').mkdir(); (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 stats={k:np.array(v,np.float32) for k,v in {'sm':sr['state_mean'],'ss':sr['state_std'],'am':sr['action_mean'],'as':sr['action_std']}.items()}
 path=ROOT/dc['dataset']['path']; train=Frames(path,dc['split']['train_episodes'],stats); val=Frames(path,dc['split']['validation_episodes'],stats)
 gen=torch.Generator().manual_seed(seed); loader=DataLoader(train,batch_size=cfg['training']['batch_size'],shuffle=True,num_workers=0,generator=gen,drop_last=True)
 val_loader=DataLoader(val,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=0,drop_last=False)
 device=torch.device('cuda:0'); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
 pc=ACTConfig(input_features={'observation.images.agentview':PolicyFeature(FeatureType.VISUAL,(3,128,128)),'observation.images.eye_in_hand':PolicyFeature(FeatureType.VISUAL,(3,128,128)),'observation.state':PolicyFeature(FeatureType.STATE,(9,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))},chunk_size=20,n_action_steps=5,use_vae=False,pretrained_backbone_weights=None,device='cuda')
 policy=ACTPolicy(pc).to(device); opt=torch.optim.AdamW(policy.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']); log=out/'metrics.jsonl'; start_step=0
 if resume_path:
  ck=torch.load(resume_path,map_location=device,weights_only=False); start_step=int(ck['step']); policy.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
  if start_step%len(loader): raise ValueError(f'exact resume requires epoch-boundary checkpoint; step={start_step}, steps_per_epoch={len(loader)}')
  gen.set_state(ck['data_generator_state']); random.setstate(ck['python_rng_state']); np.random.set_state(ck['numpy_rng_state']); torch.set_rng_state(ck['torch_rng_state'].cpu()); torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
  print(json.dumps({'classification':'RESUME_LOADED','checkpoint':resume_path,'start_step':start_step,'steps_per_epoch':len(loader)}),flush=True)
 it=iter(loader); start=time.perf_counter()
 def move(b): return {k:v.to(device,non_blocking=True) for k,v in b.items()}
 def validate():
  policy.eval(); total=0.0; count=0
  with torch.inference_mode():
   for batch in val_loader:
    batch=move(batch)
    with torch.autocast('cuda',dtype=torch.bfloat16): val_loss,_=policy(batch)
    assert torch.isfinite(val_loss); n=len(batch['action']); total+=float(val_loss)*n; count+=n
  policy.train(); return total/count,count
 if start_step==0:
  val_loss,val_samples=validate(); row={'step':0,'validation_loss':val_loss,'validation_samples':val_samples,'elapsed_s':time.perf_counter()-start}; print(json.dumps(row),flush=True)
  with log.open('a') as f: f.write(json.dumps(row)+'\n')
 for step in range(start_step+1,steps+1):
  try: batch=next(it)
  except StopIteration: it=iter(loader); batch=next(it)
  batch=move(batch); policy.train(); opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=policy(batch)
  loss.backward(); grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg['training']['grad_clip_norm']); opt.step()
  assert torch.isfinite(loss) and torch.isfinite(grad)
  if step==1 or step%cfg['training']['log_every']==0 or step==steps:
   row={'step':step,'train_loss':float(loss.detach()),'grad_norm_preclip':float(grad.detach()),'elapsed_s':time.perf_counter()-start,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated()}; print(json.dumps(row),flush=True)
   with log.open('a') as f: f.write(json.dumps(row)+'\n')
  if step%cfg['training']['validation_every']==0 or step==steps:
   val_loss,val_samples=validate(); row={'step':step,'validation_loss':val_loss,'validation_samples':val_samples,'elapsed_s':time.perf_counter()-start}; print(json.dumps(row),flush=True)
   with log.open('a') as f: f.write(json.dumps(row)+'\n')
  if step%cfg['training']['checkpoint_every']==0 or step==steps:
   exact_resume_boundary=step%len(loader)==0
   checkpoint_path=out/'checkpoints'/f'model_{step:06d}.pt'; temporary_path=checkpoint_path.with_suffix('.pt.incomplete')
   torch.save({'step':step,'model':policy.state_dict(),'optimizer':opt.state_dict(),'act_config':dataclasses.asdict(pc),'stats':{k:v.tolist() for k,v in stats.items()},'data_generator_state':gen.get_state(),'python_rng_state':random.getstate(),'numpy_rng_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'exact_resume_boundary':exact_resume_boundary,'steps_per_epoch':len(loader)},temporary_path)
   os.replace(temporary_path,checkpoint_path)
 torch.cuda.synchronize(); print(json.dumps({'classification':'PASS','start_step':start_step,'steps':steps,'elapsed_s':time.perf_counter()-start,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved()}))
if __name__=='__main__': main()
