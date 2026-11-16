#!/usr/bin/env python3
"""Deterministic 20-step OpenVLA QLoRA checkpoint and validation-pipeline smoke."""
from __future__ import annotations
import gc, hashlib, json, os, random, sys, time, types
from pathlib import Path
import numpy as np, psutil, torch, yaml
from PIL import Image
import transformers.modeling_utils as modeling_utils
from accelerate.big_modeling import dispatch_model as accelerate_dispatch_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig,AutoImageProcessor,AutoModelForVision2Seq,AutoProcessor,BitsAndBytesConfig

ROOT=Path(__file__).resolve().parents[1]; OPENVLA=ROOT/'openvla'
SNAPSHOT=ROOT/'outputs/reference/phase3a_openvla_libero_goal_fa5ae1e/hf_cache/models--openvla--openvla-7b-finetuned-libero-goal/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf'
CONFIG=ROOT/'configs/train/openvla_phase3b_qlora_train_smoke_v3.yaml'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def namespaces():
 paths={'prismatic':OPENVLA/'prismatic','prismatic.extern':OPENVLA/'prismatic/extern','prismatic.extern.hf':OPENVLA/'prismatic/extern/hf',
  'prismatic.vla':OPENVLA/'prismatic/vla','prismatic.models':OPENVLA/'prismatic/models','prismatic.models.backbones':OPENVLA/'prismatic/models/backbones',
  'prismatic.models.backbones.llm':OPENVLA/'prismatic/models/backbones/llm','prismatic.models.backbones.llm.prompting':OPENVLA/'prismatic/models/backbones/llm/prompting'}
 for name,path in paths.items():
  module=types.ModuleType(name); module.__path__=[str(path)]; module.__package__=name; sys.modules[name]=module

def main():
 cfg=yaml.safe_load(CONFIG.read_text()); output_dir=ROOT/cfg['output_dir']
 if output_dir.exists(): raise FileExistsError(f'refusing overwrite: {output_dir}')
 if cfg['training_steps']!=20: raise ValueError('smoke contract mismatch')
 manifest_path=ROOT/cfg['fixture']; fixture=json.loads(manifest_path.read_text())
 if fixture['source_sha256']!='387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5': raise ValueError('dataset hash mismatch')
 image_path=ROOT/fixture['image_path']
 if sha(image_path)!=fixture['image_sha256']: raise ValueError('fixture image mismatch')
 seed=int(cfg['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)!=(12,0): raise RuntimeError('sm_120 CUDA required')
 namespaces()
 from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
 from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
 from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor,PrismaticProcessor
 from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
 from prismatic.vla.action_tokenizer import ActionTokenizer
 AutoConfig.register('openvla',OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig,PrismaticImageProcessor)
 AutoProcessor.register(OpenVLAConfig,PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig,OpenVLAForActionPrediction)
 original_dispatch=modeling_utils.dispatch_model
 def dispatch_hooks(model,*args,**kwargs): kwargs['force_hooks']=True; return accelerate_dispatch_model(model,*args,**kwargs)
 modeling_utils.dispatch_model=dispatch_hooks; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); process=psutil.Process(); started=time.perf_counter()
 processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False)
 model=AutoModelForVision2Seq.from_pretrained(SNAPSHOT,local_files_only=True,trust_remote_code=False,attn_implementation='eager',torch_dtype=torch.bfloat16,
  quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=False),low_cpu_mem_usage=True,device_map={'':0})
 modeling_utils.dispatch_model=original_dispatch; torch.cuda.synchronize(); load_s=time.perf_counter()-started
 adapter_started=time.perf_counter(); model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True); model.config.use_cache=False
 lc=cfg['lora']; model=get_peft_model(model,LoraConfig(r=lc['rank'],lora_alpha=lc['alpha'],lora_dropout=lc['dropout'],target_modules=lc['target_modules'],init_lora_weights=lc['initialization']))
 model.train(); adapter_prepare_s=time.perf_counter()-adapter_started; trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
 if trainable<=0: raise RuntimeError('no trainable LoRA parameters')
 passive_alloc=torch.cuda.memory_allocated(); passive_res=torch.cuda.memory_reserved()
 smoke_manifest_path=ROOT/cfg['smoke_fixture']
 if sha(smoke_manifest_path)!=cfg['smoke_fixture_sha256']: raise ValueError('smoke fixture manifest mismatch')
 smoke_manifest=json.loads(smoke_manifest_path.read_text())
 if smoke_manifest['source_sha256']!=cfg['dataset']['sha256']: raise ValueError('dataset provenance mismatch')
 tokenizer=ActionTokenizer(processor.tokenizer)
 chosen=[x for x in smoke_manifest['items'] if x['split']=='train']; validation_refs=[x for x in smoke_manifest['items'] if x['split']=='validation_smoke']
 if len(chosen)!=20 or len(validation_refs)!=10: raise ValueError('smoke fixture count mismatch')
 def sample_inputs(item):
  image_path=ROOT/item['image_path']
  if sha(image_path)!=item['image_sha256']: raise ValueError('smoke image hash mismatch')
  image=np.asarray(Image.open(image_path).convert('RGB')); action=np.asarray(item['openvla_training_action'],dtype=np.float64)
  builder=PurePromptBuilder('openvla'); builder.add_turn('human','What action should the robot take to turn on the stove?'); builder.add_turn('gpt',tokenizer(action))
  inputs=processor(builder.get_prompt(),Image.fromarray(image).convert('RGB'),return_tensors='pt').to('cuda:0',dtype=torch.bfloat16)
  labels=inputs['input_ids'].clone(); labels[:,:-8]=-100; labels[:,-1]=-100
  if int((labels!=-100).sum())!=7: raise ValueError('action supervision mismatch')
  inputs['labels']=labels; return inputs,labels
 def evaluate():
  model.eval(); losses=[]; correct=0; count=0
  with torch.no_grad():
   for item in validation_refs:
    inputs,labels=sample_inputs(item)
    with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**inputs)
    losses.append(float(out.loss.item()))
    wrapped=model.base_model.model; patches=wrapped.vision_backbone.featurizer.patch_embed.num_patches
    pred=out.logits[:,patches:-1].argmax(2); gt=labels[:,1:]; mask=gt>tokenizer.action_token_begin_idx
    correct+=int(((pred==gt)&mask).sum()); count+=int(mask.sum())
    del out,pred,gt,mask,inputs,labels
    torch.cuda.empty_cache()
  model.train(); return {'samples':len(validation_refs),'action_tokens':count,'mean_sample_loss':float(np.mean(losses)),'action_token_accuracy':correct/count}
 opt_cfg=cfg['optimizer']; optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=opt_cfg['lr'],betas=tuple(opt_cfg['betas']),eps=opt_cfg['eps'],weight_decay=opt_cfg['weight_decay'])
 train_metrics=[]; checkpoint_started=time.perf_counter()
 initial_validation=evaluate(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
 for step,item in enumerate(chosen,1):
   inputs,_=sample_inputs(item); optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**inputs)
   loss=out.loss
   if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
   loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),cfg['max_grad_norm']).item())
   if not np.isfinite(grad): raise RuntimeError('non-finite gradient')
   optimizer.step(); train_metrics.append({'step':step,'episode':item['episode'],'timestep':item['timestep'],'loss':float(loss.detach().item()),'grad_norm_before_clip':grad})
 final_validation=evaluate(); gc.collect(); torch.cuda.empty_cache()
 torch.cuda.synchronize(); train_validate_s=time.perf_counter()-checkpoint_started
 output_dir.mkdir(parents=True); adapter_dir=output_dir/'adapter'; model.save_pretrained(adapter_dir,safe_serialization=True)
 torch.save({'optimizer':optimizer.state_dict(),'step':cfg['training_steps'],'python_random_state':random.getstate(),'numpy_random_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state()},output_dir/'optimizer_and_rng.pt')
 (output_dir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 files=[]
 for path in sorted(output_dir.rglob('*')):
  if path.is_file(): files.append({'path':str(path.relative_to(output_dir)),'bytes':path.stat().st_size,'sha256':sha(path)})
 result={'classification':'PASS','experiment_id':cfg['experiment_id'],'checkpoint_revision':SNAPSHOT.name,'dataset_sha256':smoke_manifest['source_sha256'],'smoke_fixture_sha256':sha(smoke_manifest_path),
  'split':{'train_episodes':len(cfg['split']['train_episodes']),'train_pool_frames':smoke_manifest['train_pool_frames'],'train_smoke_frames':len(chosen),'validation_episodes':len(validation_refs),'validation_smoke_frames':len(validation_refs)},
  'lora':lc,'trainable_parameters':trainable,'training_steps':cfg['training_steps'],'optimizer':opt_cfg,'max_grad_norm':cfg['max_grad_norm'],
  'initial_validation_smoke':initial_validation,'final_validation_smoke':final_validation,'train_metrics':train_metrics,
  'train_loss_first':train_metrics[0]['loss'],'train_loss_last':train_metrics[-1]['loss'],'train_loss_mean':float(np.mean([x['loss'] for x in train_metrics])),
  'all_train_losses_finite':True,'all_gradients_finite':True,'model_load_s':load_s,'adapter_prepare_s':adapter_prepare_s,'train_validate_s':train_validate_s,
  'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_kib_at_result':process.memory_info().rss//1024,
  'optimizer_steps':cfg['training_steps'],'checkpoint_created':True,'checkpoint_files':files,'boundary':cfg['boundary']}
 (output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__':
 os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('WANDB_MODE','disabled'); main()
