"""Seal verified v25 evidence, report, and snapshots of the living indexes."""
import json,re,hashlib,shutil
from pathlib import Path
from urllib.parse import unquote
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/joint_001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    report=OUT/'资源角色分别读入与共同通信学习研究报告.md';manifest=OUT/'completion_manifest.json';assert not manifest.exists()
    receipt=read(OUT/'training_complete.json');audit=read(OUT/'audit_execution.json');analysis=read(OUT/'analysis.json')
    assert receipt['status']=='complete' and receipt['social_runs']==12 and receipt['pair_updates']==28800
    assert audit['passed'] and read(OUT/'comparison.json')['passed']
    assert audit['analysis_sha256']==sha(OUT/'analysis.json') and audit['analysis_source_sha256']==sha(ROOT/'analyze_joint.py')
    assert audit['training_complete_sha256']==sha(OUT/'training_complete.json')
    review=read(OUT/'audit_report_review.json');assert review['passed'] and review['report_sha256']==sha(report)
    structural=read(OUT/'structural_assay.json');assert structural['status']=='complete' and structural['wrapper_sha256']==sha(ROOT/'analyze_structure.py')
    for item in structural['baseline_bindings'].values():assert sha(Path(item['path']))==item['sha256'],item['path']
    sa=read(OUT/'structural_assay_audit.json');assert sa['status']=='passed' and sa['summary_sha256']==sha(OUT/'structural_assay.json')
    process=read(OUT/'process_summary.json');pa=read(OUT/'process_summary_qa.json');assert pa['passed'] and pa['summary_sha256']==sha(OUT/'process_summary.json')
    assert process['source_sha256']==sha(ROOT/'analyze_process.py')
    for p,h in process['input_hashes'].items():assert sha(Path(p))==h,p
    figure=read(OUT/'figure_qa.json');assert figure['passed'] and figure['analysis_sha256']==sha(OUT/'analysis.json')
    for p,h in figure['files'].items():assert sha(OUT/p)==h,p
    for p,h in {**receipt['source_hashes'],**receipt['input_hashes']}.items():assert sha(Path(p))==h,p
    for p,h in receipt['files'].items():assert sha(OUT/p)==h,p
    snapshots=[]
    for rel in ('README.md','paper_program/README.md','paper_program/证据与投稿门槛.md'):
        src=PROJECT/rel;dst=OUT/'supporting_materials'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        snapshots.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst.relative_to(PROJECT)),snapshot_sha256=sha(dst)))
    links=[]
    for doc in (report,ROOT/'README.md',OUT/'结构重组评估.md'):
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            target=target.strip('<>')
            if target.startswith(('https://','http://','#')):continue
            target=unquote(target.split('#')[0]);path=Path(target) if target.startswith('/') else doc.parent/target
            assert path.exists() or path.resolve()==manifest.resolve(),(doc,target)
            links.append(dict(document=str(doc.relative_to(PROJECT)),target=target))
    write(OUT/'document_links_qa.json',dict(passed=True,checked=len(links),links=links))
    artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc') and p!=manifest}
    write(manifest,dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_seeds=4,new_social_runs=12,reused_reference_runs=24,
        pair_updates=28800,messages=14745600,actions=29491200,new_dino_inferences=0,new_private_updates=0,new_cache_inferences=0,raw_new_evaluation_files=144,
        main_report_sha256=sha(report),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(ROOT/'analyze_joint.py'),
        structural_assay_sha256=sha(OUT/'structural_assay.json'),source_hashes=receipt['source_hashes'],input_hashes=receipt['input_hashes'],audit_coverage=audit['coverage'],
        living_index_snapshots=snapshots,document_links_checked=len(links),primary_old18_difference=analysis['primary_mean'],artifacts=artifacts,
        publication_readiness='not established; old-support learning improved, natural new12 transfer remains weak; independent material and central contribution incomplete'))
    for p,h in read(manifest)['artifacts'].items():assert sha(PROJECT/p)==h,p
    for link in links:
        target=link['target'];p=Path(target) if target.startswith('/') else (PROJECT/link['document']).parent/target;assert p.exists(),str(p)
    print(json.dumps(dict(status='complete',artifacts=len(artifacts),links=len(links),manifest_sha256=sha(manifest))))
if __name__=='__main__':main()
