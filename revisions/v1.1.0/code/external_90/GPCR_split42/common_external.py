from pathlib import Path
import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODELS = ['concat', 'product', 'attention', 'deepdta_style', 'graphdta_gcn_style']
SETTINGS = ['cold_drug', 'cold_target']
SEEDS = [42, 123, 2026]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def matrix():
    return [(s, m, seed) for s in SETTINGS for seed in SEEDS for m in MODELS]

def run_id(setting, model, seed):
    return f'{setting}__{model}__seed{seed}'

def freeze_files():
    paths = [ROOT/'protocol.json',ROOT/'requirements-server.txt'] + sorted(ROOT.glob('*.py')) + sorted(ROOT.glob('*.sh'))
    for folder in ['models', 'baselines', 'data']:
        paths += sorted((ROOT/folder).glob('*.py'))
    paths += sorted((ROOT/'prepared').rglob('*.csv'))
    paths += sorted((ROOT/'prepared').rglob('*.json'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def verify_freeze():
    freeze = read_json(ROOT/'FROZEN_BEFORE_TRAINING.json')
    for name, digest in freeze['files'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'Frozen file changed: {name}. Do not mix protocols in one campaign.')
    return sha(ROOT/'FROZEN_BEFORE_TRAINING.json')
