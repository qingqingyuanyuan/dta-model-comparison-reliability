"""Shared external-family trainer. Original model modules are not modified."""
import copy
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from functools import partial
pd_read_csv = partial(pd.read_csv, float_precision='round_trip')
import torch
from common_external import ROOT, MODELS, SEEDS, matrix, run_id, now, sha, read_json, write_json, verify_freeze
import config as original
from data.dataset import build_graph_cache, create_loader_from_frame
from baselines.dataset import create_sequence_loader_from_frame
from baselines.registry import build_baseline_model, get_baseline_config
from models.predictor import DTAPredictor

EXPECTED_PARAMS={'concat':592321,'product':592577,'attention':594305,'deepdta_style':422849,'graphdta_gcn_style':415809}

def seed_runtime(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Native scatter reductions may remain nondeterministic, as in the main study.

def make_model(name):
    if name in ['concat','product','attention']:
        cfg={k:copy.deepcopy(getattr(original,k)) for k in ['DRUG_ENCODER','PROTEIN_ENCODER','FUSION','PREDICTOR','TRAINING']}
        cfg['FUSION']['fusion_mode']=name
        model=DTAPredictor(cfg)
    else:
        cfg=get_baseline_config(name);model=build_baseline_model(name,cfg)
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n!=EXPECTED_PARAMS[name]: raise ValueError(f'Architecture drift {name}: {n} != {EXPECTED_PARAMS[name]}')
    return model,cfg

def make_loaders(name, frames, seed, graph_cache=None):
    if name!='deepdta_style' and graph_cache is None:
        graph_cache=build_graph_cache(pd.concat(list(frames.values()))['Canonical SMILES'],show_progress=False)
    loaders={}
    for part,f in frames.items():
        common=dict(batch_size=64,shuffle=part=='train',model_seed=seed,num_workers=0)
        if name=='deepdta_style':
            loaders[part]=create_sequence_loader_from_frame(f,max_smiles_len=200,max_protein_len=1500,**common)
        else:
            loaders[part]=create_loader_from_frame(f,split_name=part,graph_cache=graph_cache,max_len=1500,**common)
    return loaders

def atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,path)

def rng_state(loaders):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loaders={k:v.generator.get_state() for k,v in loaders.items()})

def restore_rng(state,loaders):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])
    for k,v in loaders.items(): v.generator.set_state(state['loaders'][k])

def trusted_load(path):
    return torch.load(path,map_location='cpu',weights_only=False)

def epoch_pass(model,loader,device,optimizer=None):
    model.train(optimizer is not None)
    total=0.;n=0;predictions=[]
    context=torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for a,b,y in loader:
            a=a.to(device);b=b.to(device);y=y.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            pred=model(a,b)
            loss=torch.nn.functional.mse_loss(pred,y)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all(): raise FloatingPointError('Non-finite loss or prediction')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
            else: predictions.extend(pred.detach().cpu().numpy().astype(float).tolist())
            total+=float(loss.detach())*len(y);n+=len(y)
    if not n: raise ValueError('Empty loader')
    return total/n,np.asarray(predictions)

def smoke():
    verify_freeze()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Stop; do not spend hours training on CPU.')
    # Train/validation only. Test files are not read by smoke or formal training.
    frame=pd_read_csv(ROOT/'prepared/cold_drug/train.csv').iloc[:128].copy()
    records=[]
    for name in MODELS:
        seed_runtime(42);model,cfg=make_model(name);model=model.cuda()
        loaders=make_loaders(name,{'train':frame,'val':frame.iloc[:64]},42)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
        loss,_=epoch_pass(model,loaders['train'],torch.device('cuda'),optimizer)
        val,_=epoch_pass(model,loaders['val'],torch.device('cuda'))
        path=ROOT/'results/smoke_roundtrip.pt';path.parent.mkdir(exist_ok=True)
        atomic_save(model.state_dict(),path)
        second,_=make_model(name);second.load_state_dict(trusted_load(path),strict=True)
        path.unlink()
        records.append({'model':name,'parameters':EXPECTED_PARAMS[name],'train_loss':loss,'val_loss':val,'passed':True})
        print('SMOKE PASS',name,'parameters',EXPECTED_PARAMS[name],flush=True)
        del model,second,loaders,optimizer;torch.cuda.empty_cache()
    write_json(ROOT/'results/SMOKE_PASSED.json',{'created_utc':now(),'freeze_hash':sha(ROOT/'FROZEN_BEFORE_TRAINING.json'),'models':records})

def train_one(setting,name,seed,device_override=None):
    freeze_hash=verify_freeze()
    rid=run_id(setting,name,seed);folder=ROOT/'results/runs'/rid;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'TRAINING_DONE.json'
    if marker.exists():
        done=read_json(marker)
        if done['freeze_hash']!=freeze_hash or done['checkpoint_sha256']!=sha(folder/'best.pt'):
            raise RuntimeError(f'Completed artifact changed: {rid}')
        print('SKIP completed',rid,flush=True);return
    seed_runtime(seed)
    frames={p:pd_read_csv(ROOT/'prepared'/setting/f'{p}.csv') for p in ['train','val']}
    model,cfg=make_model(name);device=torch.device(device_override or 'cuda');model=model.to(device)
    loaders=make_loaders(name,frames,seed)
    protocol=read_json(ROOT/'protocol.json')
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=10,min_lr=1e-6)
    best=float('inf');bad=0;start=0;best_epoch=0;elapsed=0.;best_checkpoint=None
    last=folder/'last.pt'
    if last.exists():
        ck=trusted_load(last)
        if ck['freeze_hash']!=freeze_hash or ck['run_id']!=rid: raise ValueError('Resume provenance mismatch')
        model.load_state_dict(ck['model_state_dict'],strict=True)
        optimizer.load_state_dict(ck['optimizer_state_dict']);scheduler.load_state_dict(ck['scheduler_state_dict'])
        # Torch optimizer restores tensor state onto parameter devices.
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.to(device) if k!='step' else v
        start=ck['epoch'];best=ck['best'];bad=ck['bad'];best_epoch=ck['best_epoch'];elapsed=ck['elapsed_seconds']
        best_checkpoint=ck['best_checkpoint']
        atomic_save(best_checkpoint,folder/'best.pt')
        restore_rng(ck['rng'],loaders)
        # Remove an uncommitted history line if the process died between CSV and checkpoint writes.
        hp=folder/'history.csv'
        if hp.exists():
            h=pd_read_csv(hp);h[h.epoch<=start].to_csv(hp,index=False)
        print('RESUME',rid,'next epoch',start+1,flush=True)
    write_json(folder/'run_config.json',{'run_id':rid,'created_utc':now(),'freeze_hash':freeze_hash,'model_config':cfg,'protocol':protocol,
                'parameter_count':EXPECTED_PARAMS[name],'train_rows':len(frames['train']),'validation_rows':len(frames['val'])})
    clock=time.monotonic()
    for epoch in range(start+1,protocol['max_epochs']+1):
        if bad>=protocol['early_stopping_patience']: break
        t=time.monotonic()
        train_loss,_=epoch_pass(model,loaders['train'],device,optimizer)
        val_loss,_=epoch_pass(model,loaders['val'],device)
        improved=val_loss<best
        if improved:
            best=val_loss;bad=0;best_epoch=epoch
            best_checkpoint={'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                             'model_config':cfg,'model':name,'seed':seed,'setting':setting,
                             'epoch':epoch,'validation_mse':best,'freeze_hash':freeze_hash}
            atomic_save(best_checkpoint,folder/'best.pt')
        else: bad+=1
        scheduler.step(val_loss)
        record={'epoch':epoch,'train_mse_norm':train_loss,'validation_mse_norm':val_loss,'lr':optimizer.param_groups[0]['lr'],
                'epoch_seconds':time.monotonic()-t,'best_epoch':best_epoch}
        hp=folder/'history.csv';pd.DataFrame([record]).to_csv(hp,index=False,mode='a',header=not hp.exists())
        atomic_save({'run_id':rid,'freeze_hash':freeze_hash,'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                     'epoch':epoch,'best':best,'best_epoch':best_epoch,'bad':bad,'best_checkpoint':best_checkpoint,
                     'elapsed_seconds':elapsed+time.monotonic()-clock,'rng':rng_state(loaders)},last)
        print(f'{rid} epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best_epoch={best_epoch} seconds={record["epoch_seconds"]:.1f}',flush=True)
    write_json(marker,{'run_id':rid,'completed_utc':now(),'freeze_hash':freeze_hash,'checkpoint_sha256':sha(folder/'best.pt'),
                       'best_epoch':best_epoch,'best_validation_mse':best,'training_seconds':elapsed+time.monotonic()-clock})
    print('TRAINING COMPLETE',rid,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','train'])
    parser.add_argument('--setting');parser.add_argument('--model');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    if args.action=='smoke':smoke()
    else:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        train_one(args.setting,args.model,args.seed)
