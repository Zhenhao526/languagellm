"""Seal the completed formation replication, without changing prior batches."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/formation_001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    target=OUT/'completion_manifest.json';assert not target.exists(),'Do not overwrite sealed results'
    done=read(OUT/'training_complete.json');assert done['status']=='complete' and done['formal']
    assert read(OUT/'audit_execution.json')['passed'] and read(OUT/'raw_validation.json')['passed']
    assert read(OUT/'terminal_receipt.json')['exit_code']==0
    for path,h in {**done['source_hashes'],**done['input_hashes']}.items():assert sha(path)==h,path
    for rel,h in done['files'].items():assert sha(OUT/rel)==h,rel
    data=read(ROOT/'data/freeze.json')
    for path,h in data['input_hashes'].items():assert sha(path)==h,path
    assert sha(ROOT/'data/selection.json')==data['selection_sha256']
    for rel in ['README.md','paper_program/README.md','paper_program/证据与投稿门槛.md']:
        dest=OUT/'supporting_materials'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/rel,dest)
    report=OUT/'新材料上的私人经验与共同通信形成研究报告.md';links=[]
    for document in (report,ROOT/'README.md'):
        for url in re.findall(r'\]\(([^\)]+)\)',document.read_text()):
            if url.startswith('/'):
                assert Path(url).is_file(),url
                links.append(dict(document=str(document),target=url))
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target)
    artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    result=dict(status='complete_development_formation_replication',created_utc=datetime.now(timezone.utc).isoformat(),
        report_sha256=sha(report),artifacts=artifacts,artifact_count=len(artifacts),verified_local_links=links,
        private_fits=48,social_runs=36,independent_confirmation_complete=False,iclr_objective_complete=False,
        scope='All v28 code, data, development/production results, QA and report artifacts; prior sources bound by stored input hashes. Manifest excludes itself and Python bytecode.')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert all(sha(r['path'])==r['sha256'] for r in artifacts)
    print(json.dumps(dict(status=result['status'],artifacts=len(artifacts),links=len(links),manifest_sha256=sha(target),report_sha256=sha(report))))

if __name__=='__main__':main()
