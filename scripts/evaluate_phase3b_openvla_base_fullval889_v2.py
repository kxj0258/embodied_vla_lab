#!/usr/bin/env python3
"""Read-only full 889-frame offline validation for canonical step-20 QLoRA adapter."""
from __future__ import annotations
import hashlib, json, os, random, sys, time, types
from pathlib import Path
import numpy as np, psutil, torch, yaml
from PIL import Image
import transformers.modeling_utils as modeling_utils
from accelerate.big_modeling import dispatch_model as accelerate_dispatch_model
from transformers import AutoConfig,AutoImageProcessor,AutoModelForVision2Seq,AutoProcessor,BitsAndBytesConfig

ROOT=Path(__file__).resolve().parents[1]; OPENVLA=ROOT/'openvla'
SNAPSHOT=ROOT/'outputs/reference/phase3a_openvla_libero_goal_fa5ae1e/hf_cache/models--openvla--openvla-7b-finetuned-libero-goal/snapshots/fa5ae1e7509348889295bba8e08621d8b55e9baf'
CONFIG=ROOT/'configs/eval/openvla_phase3b_base_fullval889_v2.yaml'

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
 if cfg['expected_validation_frames']!=889: raise ValueError('validation contract mismatch')
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
 model.eval(); adapter_load_s=0.0
 manifest_path=ROOT/cfg['fixture_dir']/'manifest.json'; manifest=json.loads(manifest_path.read_text())
 if sha(manifest_path)!=cfg['fixture_manifest_sha256'] or manifest['source_sha256']!=cfg['dataset']['sha256'] or manifest['frames']!=889: raise ValueError('fixture contract mismatch')
 from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
 from prismatic.vla.action_tokenizer import ActionTokenizer
 tokenizer=ActionTokenizer(processor.tokenizer); rows=[]; validation_started=time.perf_counter()
 with torch.no_grad():
  for index,item in enumerate(manifest['items']):
   image_path=ROOT/item['image_path']
   if sha(image_path)!=item['image_sha256']: raise ValueError('image hash mismatch')
   action=np.asarray(item['openvla_training_action'],dtype=np.float64)
   builder=PurePromptBuilder('openvla'); builder.add_turn('human','What action should the robot take to turn on the stove?'); builder.add_turn('gpt',tokenizer(action))
   inputs=processor(builder.get_prompt(),Image.open(image_path).convert('RGB'),return_tensors='pt').to('cuda:0',dtype=torch.bfloat16)
   labels=inputs['input_ids'].clone(); labels[:,:-8]=-100; labels[:,-1]=-100; inputs['labels']=labels
   torch.cuda.synchronize(); t0=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**inputs)
   torch.cuda.synchronize(); inference_s=time.perf_counter()-t0
   wrapped=model; patches=wrapped.vision_backbone.featurizer.patch_embed.num_patches
   logits=out.logits[:,patches:-1]; gt=labels[:,1:]; mask=gt>tokenizer.action_token_begin_idx; pred=logits.argmax(2)
   correct=int(((pred==gt)&mask).sum()); predicted=tokenizer.decode_token_ids_to_actions(pred[mask].cpu().numpy()); target=tokenizer.decode_token_ids_to_actions(gt[mask].cpu().numpy())
   rows.append({'episode':item['episode'],'timestep':item['timestep'],'loss':float(out.loss.item()),'correct_action_tokens':correct,'action_tokens':int(mask.sum()),'continuous_action_l1':float(np.mean(np.abs(predicted-target))),'inference_s':inference_s})
   del out,logits,gt,mask,pred,inputs,labels
   if (index+1)%50==0: print(json.dumps({'progress':index+1,'total':889}),flush=True)
 torch.cuda.synchronize(); validation_s=time.perf_counter()-validation_started
 episode_metrics=[]
 for episode in cfg['validation_episodes']:
  selected=[x for x in rows if x['episode']==episode]
  episode_metrics.append({'episode':episode,'frames':len(selected),'mean_loss':float(np.mean([x['loss'] for x in selected])),'action_token_accuracy':sum(x['correct_action_tokens'] for x in selected)/sum(x['action_tokens'] for x in selected),'mean_continuous_action_l1':float(np.mean([x['continuous_action_l1'] for x in selected]))})
 result={'classification':'PASS','experiment_id':cfg['experiment_id'],'base_revision':SNAPSHOT.name,'adapter_sha256':None,'fixture_manifest_sha256':sha(manifest_path),
  'validation_frames':len(rows),'validation_episodes':len(episode_metrics),'action_tokens':sum(x['action_tokens'] for x in rows),'mean_frame_loss':float(np.mean([x['loss'] for x in rows])),
  'action_token_accuracy':sum(x['correct_action_tokens'] for x in rows)/sum(x['action_tokens'] for x in rows),'mean_frame_continuous_action_l1':float(np.mean([x['continuous_action_l1'] for x in rows])),
  'episode_metrics':episode_metrics,'model_load_s':load_s,'adapter_load_s':adapter_load_s,'validation_s':validation_s,'frames_per_s':len(rows)/validation_s,
  'median_inference_s':float(np.median([x['inference_s'] for x in rows])),'p95_inference_s':float(np.percentile([x['inference_s'] for x in rows],95)),
  'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_kib_at_result':process.memory_info().rss//1024,
  'training_executed':False,'environment_executed':False,'checkpoint_selected':False,'rows':rows,'boundary':cfg['boundary']}
 output_dir.mkdir(parents=True); (output_dir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False)); (output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':
 os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('WANDB_MODE','disabled'); main()


