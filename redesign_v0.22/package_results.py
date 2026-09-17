"""Verify and seal the completed v22 probe; does not complete the paper goal.

Run only after the report, report review, README and living indices are final.
This script checks stored evidence and bytes, not model formulas or forward
passes. It writes snapshots, delivery_qa.json and completion_manifest.json;
it never edits measured data, analysis, reviews or the original living indices.
"""
import hashlib
import itertools
import json
import re
import shutil
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=ROOT/'results/visibility_001'
REPORT=OUT/'布局保持与末帧可见对象的固定协议研究报告.md'
LIVING_INDICES=('README.md','paper_program/README.md','paper_program/证据与投稿门槛.md')
RECEIPTS=('audit_execution.json','comparison.json','figure_qa.json','audit_report_review.json')
EXPECTED_COUNTS=dict(private_heads=24,social_directions=48,private_world_forwards=23040,social_sender_world_forwards=46080)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def require(condition,label):
    if not condition:raise ValueError(label)
def bound(path,expected):require(sha(path)==expected,f'hash mismatch: {path}')


def local_links(document):
    """Check inline Markdown links/images; anchors and external URLs not fetched."""
    result=[]
    # These reports use ordinary inline Markdown destinations, optionally in
    # angle brackets and optionally followed by a quoted link title.
    for raw in re.findall(r'!?\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)',document.read_text()):
        target=raw.strip()
        if target.startswith('<'):target=target[1:target.index('>')]
        else:target=re.sub(r'\s+(?:"[^"]*"|\x27[^\x27]*\x27)\s*$','',target)
        if re.match(r'^(?:https?|mailto|data|app|codex):',target,re.I):continue
        decoded=unquote(target.split('#',1)[0])
        decoded=re.sub(r':\d+$','',decoded)  # Optional local code-line link.
        path=Path(decoded) if decoded else document
        if not path.is_absolute():path=document.parent/path
        path=path.resolve()
        require(path.exists(),f'broken local link in {document}: {target}')
        result.append(dict(source=str(document.relative_to(PROJECT)),target=target,resolved=str(path)))
    return result


def main():
    manifest=OUT/'completion_manifest.json'
    require(not manifest.exists(),'v22 already sealed; refusing to overwrite completion manifest')
    complete=read(OUT/'evaluation_complete.json');inv=read(OUT/'invocation.json')
    require(complete['status']=='complete' and complete['formal'] is True,'formal evaluation is incomplete')
    require(inv['seeds']==[32101,32102,32103,32104] and inv['partitions']==[1,2,3],'unexpected source scope')
    require(inv['masks']==['food_only','water_only'],'unexpected visibility masks')
    for key,value in EXPECTED_COUNTS.items():require(complete[key]==value,f'wrong completed count: {key}')
    require(complete['new_training']==complete['new_dino_inferences']==0,'v22 must contain no new training or DINO inference')
    require(inv['new_training']==inv['new_dino_inferences']==0,'invocation contradicts frozen probe')
    for key in ('source_hashes','input_hashes'):
        require(complete[key]==inv[key],f'invocation/completion {key} differ')
        for path,h in complete[key].items():bound(path,h)
    for path,h in complete['source_hashes'].items():bound(OUT/'frozen_sources'/Path(path).relative_to(PROJECT),h)
    for path,h in complete['files'].items():bound(OUT/path,h)

    expected_raw={f'private_s{s}_p{p}_d{d}.npz' for s,p,d in itertools.product(inv['seeds'],inv['partitions'],(0,1))}
    expected_raw|={f'social_{m}_s{s}_p{p}_d{d}.npz' for s,p,m,d in itertools.product(inv['seeds'],inv['partitions'],('immediate','delayed'),(0,1))}
    require({p.name for p in OUT.glob('*.npz')}==expected_raw|{'worlds.npz'},'expected exactly72 raw evaluations and one world table')

    evidence={name:read(OUT/name) for name in RECEIPTS}
    for name,item in evidence.items():require(item['passed'] is True,f'unsuccessful required receipt: {name}')
    analysis_sha=sha(OUT/'analysis.json');analysis_source_sha=sha(ROOT/'analyze_probe.py')
    audit=evidence['audit_execution.json'];comparison=evidence['comparison.json'];figures=evidence['figure_qa.json'];review=evidence['audit_report_review.json']
    require(audit['formal'] is True and audit['full_support_replay'] is True,'formal full-support replay required')
    require(audit['new_training']==0 and audit['failures']==[],'execution audit reports changes or failures')
    bound(OUT/'evaluation_complete.json',audit['evaluation_complete_sha256'])
    for item in (audit,comparison,figures):require(item['analysis_sha256']==analysis_sha,'receipt bound to different analysis')
    for item in (audit,figures):require(item['analysis_source_sha256']==analysis_source_sha,'receipt bound to different analyzer')
    bound(OUT/'analyze_probe_source.py',analysis_source_sha)
    for path,h in audit['replay_dependency_sha256'].items():bound(path,h)
    for key,value in EXPECTED_COUNTS.items():
        replay_key=key+'_replayed' if key.endswith('world_forwards') else key
        require(audit['counts'][replay_key]==value,f'wrong audit replay count: {replay_key}')
    bound(OUT/'summary.json',comparison['production_summary_sha256'])
    require(comparison['production_metrics_called'] is False,'comparison must retain independent formula scope')
    for path,h in figures['files'].items():bound(OUT/path,h)
    bound(REPORT,review['report_sha256'])
    require(review['analysis_sha256']==analysis_sha,'report review bound to different analysis')
    for path,h in review.get('bound_files',{}).items():bound(OUT/path,h)

    analysis=read(OUT/'analysis.json')
    require(analysis['status']=='complete' and analysis['formal'] is True,'incomplete analysis')
    require(analysis['seeds']==inv['seeds'] and analysis['partitions']==inv['partitions'],'analysis source scope differs')
    require(len(analysis['rows'])==72 and len(analysis['primary'])==4,'analysis does not retain72 cases/four source summaries')
    # Applicability is not a selection gate. Bind its outcome without requiring
    # that it passed; no individual or social direction may be omitted either way.
    applicability=read(OUT/'private_applicability.json')
    require(applicability['private_heads']==24 and applicability['support']=='old','wrong applicability scope')
    require(applicability==analysis['private_applicability'],'applicability result differs between files')

    gate=read(ROOT/'preflight_qa.json');prior=read(ROOT/'prior_integrity_qa.json')
    require(gate['passed'] is True and prior['passed'] is True,'preflight or prior integrity check failed')
    require(gate['source_hashes']==complete['source_hashes'],'production differs from successful preflight')
    require(gate['analysis_source_sha256']==analysis_source_sha,'analyzer differs from successful preflight')
    if inv.get('preflight'):bound(inv['preflight']['path'],inv['preflight']['sha256'])

    # Changes begin only here, after all required evidence has been checked.
    # A matching partial snapshot may be reused after an interrupted package;
    # conflicting historical bytes are never silently replaced.
    snapshot=OUT/'supporting_materials';snapshot.mkdir(exist_ok=True)
    index_snapshots=[]
    for name in LIVING_INDICES:
        source=PROJECT/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(source)
        if target.exists():bound(target,digest)
        else:shutil.copy2(source,target)
        bound(target,digest)
        index_snapshots.append(dict(source=str(source),source_sha256=digest,snapshot=str(target.relative_to(PROJECT)),snapshot_sha256=digest))
    links=local_links(REPORT)+local_links(ROOT/'README.md')
    bindings={str((OUT/name).relative_to(PROJECT)):sha(OUT/name) for name in RECEIPTS}
    bindings.update({str(p.relative_to(PROJECT)):sha(p) for p in (REPORT,ROOT/'README.md',OUT/'analysis.json',OUT/'summary.json',OUT/'evaluation_complete.json',ROOT/'preflight_qa.json',ROOT/'prior_integrity_qa.json')})
    write(OUT/'delivery_qa.json',dict(passed=True,local_links_checked=len(links),links=links,
        report_sha256=sha(REPORT),bound_evidence=bindings,living_index_snapshots=index_snapshots,
        scope='Stored hashes/counts and inline local-link existence only; no repeated model inference, numerical analysis, semantic-anchor or external-URL checking'))

    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p!=manifest]
    artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)}
    # Recheck the report/evidence after snapshot and link processing to avoid
    # sealing bytes that changed during packaging.
    for path,h in bindings.items():bound(PROJECT/path,h)
    write(manifest,dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_seeds=4,
        **EXPECTED_COUNTS,raw_evaluation_files=72,new_training=0,new_dino_inferences=0,
        main_report_sha256=sha(REPORT),source_hashes=complete['source_hashes'],input_hashes=complete['input_hashes'],
        evidence_hashes=bindings,living_index_snapshots=index_snapshots,artifacts=artifacts,
        scope='v22 frozen no-movement visibility-swap probe completed; broader ICLR-level research goal remains active',
        limitations=['No-movement event support is new; twelve non-old layouts also become initial observations',
            'Four inherited development source seeds; no new independent visual confirmation',
            'Private and social policy differences do not identify communication mediation',
            'Positive average visibility effect alone does not establish both resource-specific reversals',
            'Message changes do not by themselves demonstrate compositional language']))
    print(json.dumps(dict(status='complete',artifacts=len(artifacts),local_links=len(links),report_sha256=sha(REPORT))))


if __name__=='__main__':main()
