import zipfile
from common_external import ROOT,sha,write_json,now

def main():
    paths=[]
    for folder in ['results','logs','prepared']:
        paths+=list((ROOT/folder).rglob('*'))
    paths += list(ROOT.glob('*.json')) + list(ROOT.glob('*.md')) + list(ROOT.glob('*.py'))
    for folder in ['models','baselines','data']:
        paths+=list((ROOT/folder).glob('*.py'))
    paths=[p for p in paths if p.is_file() and p.name not in ['last.pt','RESULTS_MANIFEST.json'] and not p.name.endswith('.tmp') and '__pycache__' not in p.parts]
    # last.pt is for server resumption. best.pt, histories and predictions are retained in the archive.
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    write_json(ROOT/'RESULTS_MANIFEST.json',{'created_utc':now(),'files':manifest})
    output=ROOT.parent/(ROOT.name+'_RESULTS.zip')
    with zipfile.ZipFile(str(output)+'.part','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in paths+[ROOT/'RESULTS_MANIFEST.json']:
            z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    from pathlib import Path
    Path(str(output)+'.part').replace(output)
    print('Results archive:',output,flush=True)

if __name__=='__main__':main()
