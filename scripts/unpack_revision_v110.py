"""Check and unpack the public revision. No training or archived code execution."""
from pathlib import Path
import argparse,hashlib,json,zipfile,shutil

def unpack(output):
 root=Path(__file__).resolve().parents[1]
 out=Path(output).expanduser().resolve()
 if out==root or root in out.parents:raise ValueError('Use a fresh output directory outside the repository')
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
 index=json.loads((root/'revisions/v1.1.0/payload_index.json').read_text())
 members=set()
 # Validate all parts and all names before writing anything.
 for info in index['parts']:
  p=root/info['path']
  if p.stat().st_size!=info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Part checksum mismatch: '+info['path'])
  with zipfile.ZipFile(p) as z:
   if z.testzip() is not None:raise ValueError('ZIP CRC mismatch: '+info['path'])
   for name in z.namelist():
    if name in members or Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or ':' in name:raise ValueError('Unsafe or duplicate member: '+name)
    members.add(name)
 if len(members)!=index['files']:raise ValueError('Member count mismatch')
 out.mkdir(parents=True,exist_ok=True)
 for info in index['parts']:
  with zipfile.ZipFile(root/info['path']) as z:z.extractall(out)
 mapping=json.loads((out/'PUBLIC_SOURCE_MAPPING.json').read_text(encoding='utf-8'))
 for item in mapping:
  p=out/item['path']
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Payload hash mismatch: '+item['path'])
 print(json.dumps({'version':index['version'],'parts_verified':len(index['parts']),'files_extracted':len(members),'source_mapped_payloads_verified':len(mapping),'output':str(out),'no_training_or_archived_code_executed':True},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);unpack(p.parse_args().output)
