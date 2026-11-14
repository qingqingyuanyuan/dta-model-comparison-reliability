"""Open held-out tests only after all 30 runs have completed."""
import os
import numpy as np
import pandas as pd
from functools import partial
pd_read_csv = partial(pd.read_csv, float_precision='round_trip')
import torch
from common_external import ROOT, matrix, run_id, now, sha, read_json, write_json, verify_freeze
from engine_external import make_model, make_loaders, trusted_load, epoch_pass
from metrics_original import concordance_index

def metrics(y,p):
    return {'C_index':float(concordance_index(y,p)),'RMSE_pKi':float(np.sqrt(np.mean((np.asarray(y)-p)**2))),
            'MAE_pKi':float(np.mean(np.abs(np.asarray(y)-p)))}

def main(device_override=None):
    freeze_hash=verify_freeze()
    # This gate is checked before any test CSV is opened.
    for setting,name,seed in matrix():
        folder=ROOT/'results/runs'/run_id(setting,name,seed)
        if not (folder/'TRAINING_DONE.json').exists():raise RuntimeError('Not all 30 runs are trained; test stays sealed')
        done=read_json(folder/'TRAINING_DONE.json')
        if done['freeze_hash']!=freeze_hash or done['checkpoint_sha256']!=sha(folder/'best.pt'):raise RuntimeError('Training artifact mismatch')
    marker=ROOT/'results/TEST_OPENED.json'
    if not marker.exists():write_json(marker,{'opened_utc':now(),'freeze_hash':freeze_hash,'all_30_training_complete':True})
    rows=[]
    for setting,name,seed in matrix():
        rid=run_id(setting,name,seed);folder=ROOT/'results/runs'/rid
        out=folder/'EVALUATION_DONE.json'
        prediction_path=folder/'test_predictions.csv'
        if out.exists():
            m=read_json(out)
            if m['freeze_hash']!=freeze_hash or m['checkpoint_sha256']!=sha(folder/'best.pt') or m['prediction_sha256']!=sha(prediction_path):
                raise RuntimeError(f'Evaluation artifact mismatch: {rid}')
            rows.append(m['summary']);continue
        frame=pd_read_csv(ROOT/'prepared'/setting/'test.csv')
        device=torch.device(device_override or 'cuda')
        model,_=make_model(name);ck=trusted_load(folder/'best.pt');model.load_state_dict(ck['model_state_dict'],strict=True);model.to(device)
        loader=make_loaders(name,{'test':frame},seed)['test']
        _,pred_norm=epoch_pass(model,loader,device)
        scaler=read_json(ROOT/'prepared'/setting/'scaler.json')
        pred=pred_norm*(scaler['max']-scaler['min'])+scaler['min']
        frame['Prediction_pKi']=pred;frame.to_csv(prediction_path,index=False)
        row={'setting':setting,'model':name,'seed':seed,'records':len(frame),**metrics(frame.Label.to_numpy(),pred)}
        unique=frame.groupby(['Canonical SMILES','Target Sequence'],sort=False).agg(Label=('Label','median'),Prediction_pKi=('Prediction_pKi','mean')).reset_index()
        unique.to_csv(folder/'test_unique_input_predictions.csv',index=False)
        row.update({'unique_'+k:v for k,v in metrics(unique.Label.to_numpy(),unique.Prediction_pKi.to_numpy()).items()})
        write_json(out,{'created_utc':now(),'freeze_hash':freeze_hash,'checkpoint_sha256':sha(folder/'best.pt'),
                       'prediction_sha256':sha(prediction_path),'summary':row})
        rows.append(row);print('EVALUATED',rid,flush=True)
        del model,loader;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/ALL_30_METRICS.csv',index=False)

if __name__=='__main__':
    torch.set_num_threads(min(8,os.cpu_count() or 1));main()
