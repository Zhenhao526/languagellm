"""Seal the completed batch and reviewable report, preserving living snapshots."""
import json,re,hashlib,shutil
from pathlib import Path
from urllib.parse import unquote
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/attention_001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    report=OUT/'私人预测动态读取与共同符号结构研究报告.md'
    receipt=read(OUT/'training_complete.json');audit=read(OUT/'audit_execution.json');analysis=read(OUT/'analysis.json')
    assert receipt['status']=='complete' and receipt['social_runs']==24 and receipt['pair_updates']==57600
    assert audit['passed'] and read(OUT/'comparison.json')['passed'] and read(OUT/'figure_qa.json')['passed']
    review=read(OUT/'audit_report_review.json');assert review['passed'] and review['report_sha256']==sha(report)
    structural=read(OUT/'structural_assay.json');assert structural['status']=='complete'
    for p,h in structural['source_fingerprints'].items():assert sha(Path(p))==h,p
    structure_audit=read(OUT/'structural_assay_audit.json');assert structure_audit.get('passed',structure_audit.get('status')=='passed')
    figure=read(OUT/'figure_qa.json');assert figure['analysis_sha256']==sha(OUT/'analysis.json')
    for p,h in figure['files'].items():assert sha(OUT/p)==h,p
    assert audit['analysis_sha256']==sha(OUT/'analysis.json') and audit['analysis_source_sha256']==sha(ROOT/'analyze_attention.py')
    assert audit['training_complete_sha256']==sha(OUT/'training_complete.json')
    for p,h in {**receipt['source_hashes'],**receipt['input_hashes']}.items():assert sha(Path(p))==h,p
    for p,h in receipt['files'].items():assert sha(OUT/p)==h,p
    snapshots=[]
    for rel in ('README.md','paper_program/README.md','paper_program/证据与投稿门槛.md'):
        src=PROJECT/rel;dst=OUT/'supporting_materials'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        snapshots.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst.relative_to(PROJECT)),snapshot_sha256=sha(dst)))
    links=[]
    for doc in [report,ROOT/'README.md',OUT/'结构重组评估.md']:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            target=target.strip('<>')
            if target.startswith(('https://','http://','#')):continue
            target=unquote(target.split('#')[0]);path=Path(target) if target.startswith('/') else doc.parent/target
            assert path.exists() or path.resolve()==(OUT/'completion_manifest.json').resolve(),(doc,target);links.append(dict(document=str(doc.relative_to(PROJECT)),target=target))
    write(OUT/'document_links_qa.json',dict(passed=True,checked=len(links),links=links))
    manifest=OUT/'completion_manifest.json';assert not manifest.exists()
    artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc') and p!=manifest}
    write(manifest,dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_seeds=4,social_runs=24,pair_updates=57600,
        messages=29491200,actions=58982400,new_dino_inferences=0,new_private_updates=0,raw_evaluation_files=288,
        main_report_sha256=sha(report),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(ROOT/'analyze_attention.py'),
        structural_assay_sha256=sha(OUT/'structural_assay.json'),source_hashes=receipt['source_hashes'],input_hashes=receipt['input_hashes'],
        audit_coverage=audit['coverage'],living_index_snapshots=snapshots,document_links_checked=len(links),artifacts=artifacts,
        primary_mean=analysis['primary_mean'],publication_readiness='not established; development sources, prior structured role predictions, same image pool, adapted existing architecture'))
    final=read(manifest)
    for p,h in final['artifacts'].items():assert sha(PROJECT/p)==h,p
    for link in links:
        target=link['target'];p=Path(target) if target.startswith('/') else (PROJECT/link['document']).parent/target;assert p.exists(),str(p)
    print(json.dumps(dict(status='complete',artifacts=len(artifacts),links=len(links),manifest_sha256=sha(manifest))))
if __name__=='__main__':main()
