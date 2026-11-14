"""Small CPU QA only; uses synthetic labels and never writes campaign results."""
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import common_external as common
import engine_external as engine
from metrics_original import concordance_index,_concordance_index_reference
from statistics_external import weighted_ci
from prepare_external import assign_groups,audit_parts

def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(12)
    for _ in range(100):
        y=rng.integers(0,4,20);p=rng.integers(0,5,20);w=rng.integers(0,4,20)
        expected=_concordance_index_reference(np.repeat(y,w),np.repeat(p,w))
        actual=weighted_ci(y,p,w)
        assert abs(actual-expected)<1e-12
        assert abs(concordance_index(y,p)-_concordance_index_reference(y,p))<1e-12
    records=[{'Canonical SMILES':f'drug{d}','Target Sequence':f'protein{t}','Label':float((d+t)%13)} for d in range(200) for t in range(30)]
    frame=pd.DataFrame(records)
    for setting in common.SETTINGS:
        a=assign_groups(frame,setting);b=assign_groups(frame,setting)
        assert a.equals(b)
        # 200 synthetic drugs has <30 held out under 10%; use a larger identity-only case.
    frame=pd.DataFrame([{'Canonical SMILES':f'drug{d}','Target Sequence':f'protein{t}','Label':float((d+t)%13)} for d in range(400) for t in range(30)])
    for setting in common.SETTINGS:
        membership=assign_groups(frame,setting)
        audit_parts({s:frame[membership==s].copy() for s in ['train','val','test']},setting)
    saved_root=common.ROOT;engine_root=engine.ROOT
    with tempfile.TemporaryDirectory(prefix='dta_cpu_selftest_') as td:
        root=Path(td);common.ROOT=root;engine.ROOT=root
        # Tiny synthetic data keeps QA isolated from held-out scientific evaluation.
        smiles=['CCO','CCN','CCC','CC(=O)O','c1ccccc1','CCOC','CCS','CC(C)O']
        f=pd.DataFrame({'Canonical SMILES':smiles,'Target Sequence':['ACDEFGHIKLMNPQRSTVWY'*2]*8,
                        'Label_norm':np.linspace(.1,.9,8)})
        folder=root/'prepared/cold_drug';folder.mkdir(parents=True)
        for part in ['train','val']:f.to_csv(folder/f'{part}.csv',index=False)
        protocol=common.read_json(saved_root/'protocol.json');protocol['max_epochs']=2
        common.write_json(root/'protocol.json',protocol)
        common.write_json(root/'FROZEN_BEFORE_TRAINING.json',{'files':{'protocol.json':common.sha(root/'protocol.json')}})
        tested=[]
        for model in common.MODELS:
            engine.train_one('cold_drug',model,42,device_override='cpu')
            marker=root/'results/runs'/common.run_id('cold_drug',model,42)/'TRAINING_DONE.json'
            assert marker.exists();engine.train_one('cold_drug',model,42,device_override='cpu')
            tested.append({'model':model,'parameters':engine.EXPECTED_PARAMS[model],'two_epoch_train_checkpoint_and_completed_skip':'passed'})
        # Simulated interruption after a durable epoch checkpoint; resume only the unfinished epoch.
        save=engine.atomic_save
        def interrupt_after_checkpoint(obj,path):
            save(obj,path)
            if Path(path).name=='last.pt' and obj['epoch']==1:raise KeyboardInterrupt('simulated interruption')
        engine.atomic_save=interrupt_after_checkpoint
        try:engine.train_one('cold_drug','concat',123,device_override='cpu')
        except KeyboardInterrupt:pass
        finally:engine.atomic_save=save
        engine.train_one('cold_drug','concat',123,device_override='cpu')
        hp=root/'results/runs'/common.run_id('cold_drug','concat',123)/'history.csv'
        assert pd.read_csv(hp).epoch.tolist()==[1,2]
        common.ROOT=saved_root;engine.ROOT=engine_root
    result={'checked_utc':common.now(),'host':'local Windows CPU','torch':torch.__version__,
            'metric_randomized_tie_tests':100,'weighted_bootstrap_vs_explicit_duplication_tests':100,
            'split_disjointness':'passed','five_model_tests':tested,'epoch_resume_test':'passed',
            'gpu_status':'not run locally; server full-batch CUDA smoke is mandatory before formal training',
            'scientific_training_runs_started':0}
    common.write_json(saved_root/'LOCAL_VALIDATION.json',result)
    print('ALL LOCAL CPU SELFTESTS PASSED')

if __name__=='__main__':main()
