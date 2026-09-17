"""Once-only, no-forward response analysis of every frozen FI endpoint."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform, shutil, time
import numpy as np
from research_program.triadic_reciprocal_execution_study import environment
from . import cases

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
SOURCE=HERE.parent/'triadic_reciprocal_execution_study/results/rules_001'
ORIGINAL=HERE.parent/'triadic_action_dependency_study/results/context_001'
SEEDS=(57101,57102,57103,57104); RULES=('strict','reciprocal')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
ANCHORS={'plan.json':'59e9f8d3b2056538b339072867fde4c58b1478274d982e3158d1b4268edd0e53',
 'execution/results.json':'28dc1ee3ba9a6a460ad783b42f3d7faf6caef52d55be9753693122d894984a5e'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def input_manifest():
    for f,digest in ANCHORS.items():assert sha(SOURCE/f)==digest
    source=read(SOURCE/'execution/results.json');assert source['status']=='completed'
    manifest={str(SOURCE/f):d for f,d in ANCHORS.items()}
    prepared=ORIGINAL/'prepared.json'
    assert sha(prepared)=='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'
    manifest[str(prepared)]=sha(prepared)
    rows=[]
    for run in source['runs']:
        assert run['seed'] in SEEDS and run['rule'] in RULES
        for part in PARTS:
            rec=run['final'][part];path=Path(rec['path']).resolve()
            assert path.parent.parent==SOURCE/'execution'
            manifest[str(path)]=rec['data_sha256']
            rows.append(dict(seed=run['seed'],training_rule=run['rule'],part=part,path=str(path),sha256=rec['data_sha256']))
    assert len(rows)==32 and len({(r['seed'],r['training_rule'],r['part']) for r in rows})==32
    # Freeze the already completed independent execution audit as provenance.
    candidates=list(SOURCE.glob('audit*/verification.json'))
    matching=[p for p in candidates if sha(p)=='a0c375dc60833f1b4ce911125454bc2ea2d94a13aab14bfc4d9a4856c82ae638']
    assert len(matching)==1
    manifest[str(matching[0])]=sha(matching[0])
    return manifest,rows

def source_manifest():
    files=['__init__.py','runner.py','cases.py','test_cases.py','test_runner.py','plan.md','design_review.md','static_availability.json','cases_preflight_001.json','main_preflight.json']
    paths=[HERE/f for f in files]
    # Bind all imported frozen production task helpers, without loading networks.
    for p,d in read(SOURCE/'plan.json')['source_sha256'].items():
        assert sha(p)==d,('Frozen dependency changed',p)
        paths.append(Path(p))
    return {str(p):sha(p) for p in paths}

def prepare(out):
    out=Path(out).resolve();assert not out.exists()
    inputs,rows=input_manifest();sources=source_manifest()
    out.mkdir(parents=True)
    for source in sources:
        dest=out/'source_snapshot'/Path(source).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    plan=dict(schema='triadic_need_response_v1',status='prepared_without_endpoint_reads',at=now(),
        inputs_sha256=inputs,source_sha256=sources,endpoint_records=rows,
        primary='mean_four_reciprocal_trained_Q_minus_exact_shuffle_on_complete_double_holdout_reciprocal_settlement',
        runtime=dict(python=platform.python_version(),numpy=np.__version__),
        budget=dict(endpoint_files=32,settlement_evaluations=64,new_network_forwards=0,new_training_updates=0,new_model_calls=0))
    write(out/'plan.json',plan);write(out/'freeze.json',dict(plan_sha256=sha(out/'plan.json')))
    return dict(status=plan['status'],plan_sha256=sha(out/'plan.json'),budget=plan['budget'])

def verify(out):
    out=Path(out);plan=read(out/'plan.json')
    assert sha(out/'plan.json')==read(out/'freeze.json')['plan_sha256']
    assert plan['source_sha256']==source_manifest()
    assert plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__)
    for p,d in plan['source_sha256'].items():assert sha(out/'source_snapshot'/Path(p).relative_to(ROOT))==d
    return plan

def primary(records):
    assert len(records)==32
    indexed={(r['seed'],r['training_rule'],r['part']):r for r in records};assert len(indexed)==32
    rows=[]
    for seed in SEEDS:
        value=indexed[seed,'reciprocal','new_needs_and_layouts']['settlements']['reciprocal']
        assert all(np.isfinite(value[k]) for k in ('Q','Q_shuffle','Q_excess'))
        assert abs(value['Q']-value['Q_shuffle']-value['Q_excess'])<1e-12
        rows.append(dict(seed=seed,Q=value['Q'],Q_shuffle=value['Q_shuffle'],Q_excess=value['Q_excess']))
    return dict(name='reciprocal_Q_excess_complete_double_holdout',paired_units=4,by_seed=rows,
        mean_Q=float(np.mean([r['Q'] for r in rows])),mean_Q_shuffle=float(np.mean([r['Q_shuffle'] for r in rows])),
        mean_Q_excess=float(np.mean([r['Q_excess'] for r in rows])))

def execute(out):
    out=Path(out).resolve();plan=verify(out);assert not (out/'status.json').exists();start=time.perf_counter()
    write(out/'status.json',dict(status='running',at=now(),plan_sha256=sha(out/'plan.json')))
    try:
        for p,d in plan['inputs_sha256'].items():assert sha(p)==d,(p,'input checksum')
        specs=read(ORIGINAL/'prepared.json')['partitions'];cc={p:cases.build_cases(specs[p]) for p in PARTS}
        records=[]
        for meta in plan['endpoint_records']:
            with np.load(meta['path'],allow_pickle=False) as f:
                states=f['states'];ids=f['state_indices'];actions=f['action_indices']
            cases.validate_partition_arrays(cc[meta['part']],states,ids)
            stats={rule:cases.metrics(cc[meta['part']],environment.settle(states,actions,rule)['actual_pair_index']) for rule in RULES}
            rec=dict(meta,settlements=stats);records.append(rec)
            print(json.dumps(dict(stage='endpoint_analyzed',seed=meta['seed'],rule=meta['training_rule'],part=meta['part'])),flush=True)
        result=dict(status='completed',at=now(),plan_sha256=sha(out/'plan.json'),records=records,
            primary=primary(records),budget=plan['budget'],elapsed_seconds=time.perf_counter()-start)
        write(out/'result.json',result);verify(out)
        write(out/'status.json',dict(status='completed',at=now(),result_sha256=sha(out/'result.json')))
        return dict(status='completed',primary=result['primary'],elapsed_seconds=result['elapsed_seconds'])
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',at=now(),error=repr(exc),elapsed_seconds=time.perf_counter()-start));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','run']);p.add_argument('--out',required=True);a=p.parse_args()
    result=prepare(a.out) if a.command=='prepare' else execute(a.out) if a.command=='run' else dict(status='verified',plan_sha256=sha(Path(a.out)/'plan.json')) if verify(a.out) else None
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
