import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from common_external import ROOT, matrix, run_id, now, write_json, read_json, verify_freeze

def run(args,log=None):
    print('RUN:', ' '.join(map(str,args)),flush=True)
    if log:
        with open(log,'a',encoding='utf-8') as f:
            result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Command failed (exit {result.returncode}); see {log}')
    else:subprocess.run(args,cwd=ROOT,check=True)

def main():
    import fcntl
    lock=open(ROOT/'campaign.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A campaign is already running; duplicate launch blocked')
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Need at least 8 GiB free disk before starting')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: stop rather than training on CPU')
    if not torch.__version__.startswith('2.5.1'):raise RuntimeError('This package expects server PyTorch 2.5.1; existing PyTorch was not overwritten')
    env={'captured_utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
         'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
         'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    write_json(ROOT/'results/ENVIRONMENT.json',env)
    run([sys.executable,'-m','pip','freeze'],ROOT/'logs/environment_pip_freeze.txt')
    run([sys.executable,'prepare_external.py'])
    freeze_hash=verify_freeze()
    write_json(ROOT/'results/CAMPAIGN_STARTED.json',{'created_utc':now(),'freeze_hash':freeze_hash,'expected_runs':30})
    smoke=ROOT/'results/SMOKE_PASSED.json'
    if not smoke.exists() or read_json(smoke)['freeze_hash']!=freeze_hash:
        run([sys.executable,'engine_external.py','smoke'],ROOT/'logs/smoke.log')
    for i,(setting,name,seed) in enumerate(matrix(),1):
        rid=run_id(setting,name,seed)
        write_json(ROOT/'results/STATUS.json',{'updated_utc':now(),'stage':'training','run_number':i,'total':30,'run_id':rid})
        run([sys.executable,'engine_external.py','train','--setting',setting,'--model',name,'--seed',str(seed)],ROOT/'logs'/f'{rid}.log')
        print(f'PROGRESS {i}/30 training complete',flush=True)
    write_json(ROOT/'results/STATUS.json',{'updated_utc':now(),'stage':'held_out_evaluation','training_complete':30})
    run([sys.executable,'evaluate_external.py'],ROOT/'logs/evaluation.log')
    write_json(ROOT/'results/STATUS.json',{'updated_utc':now(),'stage':'statistics','training_complete':30})
    run([sys.executable,'statistics_external.py'],ROOT/'logs/statistics.log')
    write_json(ROOT/'results/STATUS.json',{'updated_utc':now(),'stage':'complete','training_complete':30})
    run([sys.executable,'pack_results.py'])
    print('SUCCESS: 30 runs, evaluation and statistics complete. Download DTA_NonKinase_30_RESULTS.zip from the parent folder.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(ROOT/'results/ERROR.json',{'created_utc':now(),'error':str(exc)})
        print('STOPPED:',exc,flush=True)
        raise
