#!/usr/bin/env python3
"""One real-sample rank-8 QLoRA optimizer-state and single-update resource gate."""
from __future__ import annotations
import hashlib, json, os, random, sys, time, types
from pathlib import Path
import numpy as np, psutil, torch, yaml
from PIL import Image
import transformers.modeling_utils as modeling_utils
from accelerate.big_modeling import dispatch_model as accelerate_dispatch_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig,AutoImageProcessor,AutoModelForVision2Seq,AutoProcessor,BitsAndBytesConfig

ROOT=Path(__file__).resolve().parents[1]; OPENVLA=ROOT/'openvla'
SNAPSHOT=ROOT/'outputs/reference/phase3a_openvla_libero_goal_fa5ae1e/hf_cache/models--openvla--openvla-7b-finetuned-libero-goal/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf'
CONFIG=ROOT/'configs/train/openvla_phase3b_qlora_optimizer_gate_v1.yaml'

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
 if cfg['forward_steps']!=1 or cfg['backward_steps']!=1 or cfg['optimizer_steps']!=1: raise ValueError('resource contract mismatch')
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
 action=np.asarray(fixture['openvla_training_action'],dtype=np.float64); tokenizer=ActionTokenizer(processor.tokenizer)
 builder=PurePromptBuilder('openvla'); builder.add_turn('human','What action should the robot take to turn on the stove?'); builder.add_turn('gpt',tokenizer(action)); prompt=builder.get_prompt()
 inputs=processor(prompt,Image.open(image_path).convert('RGB'),return_tensors='pt').to('cuda:0',dtype=torch.bfloat16)
 labels=inputs['input_ids'].clone(); labels[:,:-(7+1)]=-100; labels[:,-1]=-100
 supervised=int((labels!=-100).sum().item())
 if supervised!=7: raise ValueError(f'expected 7 supervised action tokens, got {supervised}')
 inputs['labels']=labels; model.zero_grad(set_to_none=True); torch.cuda.synchronize(); f0=time.perf_counter()
 with torch.autocast('cuda',dtype=torch.bfloat16): model_output=model(**inputs)
 torch.cuda.synchronize(); forward_s=time.perf_counter()-f0; loss=model_output.loss
 if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss: {loss}')
 wrapped=model.base_model.model; num_patches=wrapped.vision_backbone.featurizer.patch_embed.num_patches
 logits=model_output.logits[:,num_patches:-1]; gt=labels[:,1:]; mask=gt>tokenizer.action_token_begin_idx
 if logits.shape[:2]!=gt.shape or int(mask.sum())!=7: raise ValueError(f'action logit alignment mismatch {logits.shape} {gt.shape} {mask.sum()}')
 pred=logits.argmax(dim=2); accuracy=float(((pred==gt)&mask).sum().float().div(mask.sum()).item())
 pred_action=tokenizer.decode_token_ids_to_actions(pred[mask].detach().cpu().numpy()); gt_action=tokenizer.decode_token_ids_to_actions(gt[mask].detach().cpu().numpy())
 action_l1=float(np.mean(np.abs(pred_action-gt_action))); torch.cuda.synchronize(); b0=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward_s=time.perf_counter()-b0
 grads=[p.grad.detach().float() for p in model.parameters() if p.requires_grad and p.grad is not None]
 if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError('missing or non-finite LoRA gradients')
 grad_norm=float(torch.sqrt(sum(torch.sum(g*g) for g in grads)).item())
 first_param=next(p for p in model.parameters() if p.requires_grad)
 before=first_param.detach().clone()
 opt_cfg=cfg['optimizer']
 optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=opt_cfg['lr'],betas=tuple(opt_cfg['betas']),eps=opt_cfg['eps'],weight_decay=opt_cfg['weight_decay'])
 torch.cuda.synchronize(); o0=time.perf_counter()
 clipped_from=float(torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),cfg['max_grad_norm']).item())
 optimizer.step(); torch.cuda.synchronize(); optimizer_step_s=time.perf_counter()-o0
 delta=(first_param.detach()-before).float()
 if not torch.isfinite(delta).all() or float(delta.abs().max().item())<=0: raise RuntimeError('missing or non-finite parameter update')
 optimizer_state_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
 result={'classification':'PASS','experiment_id':cfg['experiment_id'],'checkpoint_revision':SNAPSHOT.name,'fixture_manifest_sha256':sha(manifest_path),
  'lora':lc,'trainable_parameters':trainable,'total_parameters_with_adapters':total,'trainable_percent':100*trainable/total,
  'input_token_count':int(inputs['input_ids'].shape[1]),'supervised_action_tokens':supervised,'loss':float(loss.detach().item()),'action_token_accuracy':accuracy,'continuous_action_l1':action_l1,
  'gradient_norm_before_clip':grad_norm,'clip_max_norm':cfg['max_grad_norm'],'clip_returned_norm':clipped_from,'all_gradients_finite':True,'model_load_s':load_s,'adapter_prepare_s':adapter_prepare_s,'forward_s':forward_s,'backward_s':backward_s,'optimizer_step_s':optimizer_step_s,
  'passive_vram_allocated_bytes':passive_alloc,'passive_vram_reserved_bytes':passive_res,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),
  'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_kib_at_result':process.memory_info().rss//1024,'optimizer':opt_cfg,'optimizer_state_bytes':optimizer_state_bytes,'optimizer_steps':1,'parameter_update_max_abs':float(delta.abs().max().item()),'parameter_update_l2':float(torch.linalg.vector_norm(delta).item()),'checkpoint_created':False,
  'boundary':cfg['boundary']}
 output_dir.mkdir(parents=True); (output_dir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False)); (output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))

if __name__=='__main__':
 os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('WANDB_MODE','disabled'); main()

