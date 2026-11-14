"""Download, clean and freeze a proposed human GPCR Ki external benchmark.

No model training or predictions are used in dataset eligibility decisions.
"""
import csv
import io
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests
from common_external import ROOT, now, sha, write_json, read_json, freeze_files, verify_freeze

def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    print('Downloading:', url, flush=True)
    with session.get(url, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        with open(str(path)+'.part', 'wb') as f:
            for b in r.iter_content(1024*1024):
                f.write(b)
    Path(str(path)+'.part').replace(path)

def column(headers, prefix, required=True):
    matches = [h for h in headers if h.strip().startswith(prefix)]
    if not matches and required:
        raise ValueError(f'BindingDB header missing: {prefix}; do not guess a replacement column')
    return matches[0] if matches else None

def assign_groups(frame, setting, seed=42):
    key = 'Canonical SMILES' if setting == 'cold_drug' else 'Target Sequence'
    entities = np.array(sorted(frame[key].unique()), dtype=object)
    np.random.RandomState(seed).shuffle(entities)
    if setting == 'cold_target':
        nt = max(5, math.ceil(len(entities)*.2))
        nv = max(3, math.ceil(len(entities)*.1))
        # Membership uses identifiers/counts only, never label or prediction values.
        mapping = {e: ('test' if i < nt else 'val' if i < nt+nv else 'train') for i,e in enumerate(entities)}
    else:
        counts = frame[key].value_counts().to_dict()
        boundaries = (.8*len(frame), .9*len(frame))
        total = 0
        mapping = {}
        for e in entities:
            mapping[e] = 'train' if total < boundaries[0] else 'val' if total < boundaries[1] else 'test'
            total += counts[e]
    return frame[key].map(mapping)

def audit_parts(parts, setting):
    details = {}
    for name, f in parts.items():
        if len(f) < 100 or f['Label'].nunique() < 2:
            raise ValueError(f'{setting}/{name}: too few rows or constant labels')
        details[name] = dict(records=len(f), drugs=f['Canonical SMILES'].nunique(), targets=f['Target Sequence'].nunique(),
                             unique_inputs=len(f.drop_duplicates(['Canonical SMILES','Target Sequence'])),
                             label_mean=float(f.Label.mean()), label_sd=float(f.Label.std()),
                             label_min=float(f.Label.min()), label_max=float(f.Label.max()))
    names = list(parts)
    key = 'Canonical SMILES' if setting == 'cold_drug' else 'Target Sequence'
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if set(parts[a][key]) & set(parts[b][key]):
                raise ValueError('Entity leakage')
            if set(zip(parts[a]['Canonical SMILES'],parts[a]['Target Sequence'])) & set(zip(parts[b]['Canonical SMILES'],parts[b]['Target Sequence'])):
                raise ValueError('Identical input leakage')
    if setting == 'cold_target' and (details['test']['targets'] < 5 or details['val']['targets'] < 3):
        raise ValueError('Not enough independent held-out proteins')
    if setting == 'cold_drug' and min(details[n]['drugs'] for n in ['test','val']) < 30:
        raise ValueError('Not enough independent held-out drugs')
    return details

def main():
    if (ROOT/'FROZEN_BEFORE_TRAINING.json').exists():
        verify_freeze()
        print('Existing frozen campaign verified; no resplitting.', flush=True)
        return
    from rdkit import Chem, rdBase
    p = read_json(ROOT/'protocol.json')
    raw = ROOT/'raw'; raw.mkdir(exist_ok=True)
    archive = raw/p['bindingdb_file']
    download(p['bindingdb_url'], archive)
    url = 'https://rest.uniprot.org/uniprotkb/stream?' + urlencode(dict(query=p['uniprot_query'],format='tsv',fields='accession,protein_name,sequence'))
    uniprot_file = raw/'human_reviewed_gpcr_uniprot.tsv'
    download(url, uniprot_file)
    up = pd.read_csv(uniprot_file, sep='\t', dtype=str).fillna('')
    if not {'Entry','Sequence','Protein names'} <= set(up.columns) or len(up) < 100:
        raise ValueError('UniProt GPCR query did not return the expected TSV; stopped')
    refs = {r['Entry']: r['Sequence'] for _,r in up.iterrows()}
    counts = Counter(); rows=[]; cache={}
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.lower().endswith('.tsv')]
        if len(members) != 1: raise ValueError(f'Unexpected archive members: {members}')
        with z.open(members[0]) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig'), delimiter='\t')
            h=reader.fieldnames
            smi=column(h,'Ligand SMILES'); seq=column(h,'BindingDB Target Chain Sequence')
            acc=column(h,'UniProt (SwissProt) Primary ID of Target Chain')
            chain=column(h,'Number of Protein Chains in Target'); ki=column(h,'Ki (nM)')
            name=column(h,'Target Name'); rsid=column(h,'BindingDB Reactant_set_id',False)
            if rsid is None: rsid=h[0]
            doi=column(h,'Article DOI',False)
            for r in reader:
                counts['source_rows']+=1
                accession=str(r.get(acc,'')).strip()
                if accession not in refs: counts['not_reviewed_human_gpcr']+=1; continue
                if str(r.get(chain,'')).strip() not in ['1','1.0']: counts['not_single_chain']+=1; continue
                sequence=str(r.get(seq,'')).strip().upper()
                if sequence != refs[accession]: counts['noncanonical_sequence']+=1; continue
                text=str(r.get(ki,'')).strip()
                if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',text):
                    counts['missing_or_censored_Ki']+=1; continue
                value=float(text)
                if not math.isfinite(value) or value <= 0: counts['invalid_Ki']+=1; continue
                smiles=str(r.get(smi,'')).strip()
                if smiles not in cache:
                    mol=Chem.MolFromSmiles(smiles)
                    cache[smiles]=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None
                canonical=cache[smiles]
                if canonical is None: counts['invalid_smiles']+=1; continue
                rows.append({'SourceRow':counts['source_rows'],'ReactionSetID':r.get(rsid,''),'Target ID':accession,
                             'Target Name':r.get(name,''),'Target Sequence':sequence,'SMILES':smiles,
                             'Canonical SMILES':canonical,'Ki_nM':value,'Label':9-math.log10(value),
                             'Article DOI':r.get(doi,'') if doi else ''})
    if not rows: raise ValueError('No eligible exact Ki rows; do not start training')
    df=pd.DataFrame(rows)
    before=len(df)
    df=df.drop_duplicates(['Canonical SMILES','Target Sequence','Label']).copy()
    counts['exact_duplicate_rows_removed']=before-len(df)
    coverage=df.groupby('Target Sequence')['Canonical SMILES'].nunique()
    selected=coverage[coverage >= p['minimum_unique_drugs_per_target']].index
    counts['low_coverage_rows_removed']=int((~df['Target Sequence'].isin(selected)).sum())
    df=df[df['Target Sequence'].isin(selected)].sort_values(['Target ID','Canonical SMILES','Label','SourceRow']).reset_index(drop=True)
    df.insert(0,'RowID',np.arange(len(df)))
    summary=dict(created_utc=now(), source_url=p['bindingdb_url'], source_sha256=sha(archive),
                 uniprot_url=url, uniprot_sha256=sha(uniprot_file), rdkit_version=rdBase.rdkitVersion,
                 records=len(df), drugs=df['Canonical SMILES'].nunique(), targets=df['Target Sequence'].nunique(),
                 filtering=dict(counts), eligibility='pending')
    write_json(ROOT/'DATA_FEASIBILITY.json',summary)
    print(json_summary(summary),flush=True)
    if summary['targets']<p['minimum_targets'] or summary['drugs']<p['minimum_drugs'] or len(df)<p['minimum_records']:
        raise ValueError('DATA FEASIBILITY FAILED: proposed PDSP GPCR subset is insufficient. No training started. Keep DATA_FEASIBILITY.json; do not lower thresholds after seeing results.')
    prepared=ROOT/'prepared';prepared.mkdir(exist_ok=True)
    df.to_csv(prepared/'cleaned.csv',index=False)
    df.groupby(['Target ID','Target Name','Target Sequence']).agg(records=('RowID','size'),drugs=('Canonical SMILES','nunique')).reset_index().to_csv(prepared/'target_coverage.csv',index=False)
    for setting in p['settings']:
        membership=assign_groups(df,setting,p['split_seed'])
        parts={s:df[membership==s].copy() for s in ['train','val','test']}
        details=audit_parts(parts,setting)
        lo=float(parts['train'].Label.min()); hi=float(parts['train'].Label.max())
        if hi<=lo: raise ValueError('Constant training labels')
        folder=prepared/setting;folder.mkdir(exist_ok=True)
        for s,f in parts.items():
            f['Label_norm']=(f.Label-lo)/(hi-lo)
            f.to_csv(folder/f'{s}.csv',index=False)
        write_json(folder/'scaler.json',{'min':lo,'max':hi,'method':'training_only_minmax_pKi'})
        write_json(folder/'split_audit.json',{'setting':setting,'seed':p['split_seed'],'partitions':details,'zero_entity_overlap':True,'zero_input_overlap':True})
        print(setting,details,flush=True)
    summary['eligibility']='passed';write_json(ROOT/'DATA_FEASIBILITY.json',summary)
    write_json(ROOT/'FROZEN_BEFORE_TRAINING.json',{'created_utc':now(),'protocol_id':p['protocol_id'],
               'statement':'Local timestamped freeze, not independent public preregistration',
               'files':freeze_files(), 'source_hashes':{'bindingdb':sha(archive),'uniprot':sha(uniprot_file)}})
    print('DATA READY: frozen split hashes saved. No models have been trained yet.',flush=True)

def json_summary(data):
    import json
    return json.dumps(data,ensure_ascii=False,indent=2)

if __name__=='__main__': main()
