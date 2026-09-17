"""Separate raw traversal/aggregation using the frozen environment metrics.

The independently implemented metric formula and support checks are in
analyze_temporal.py. This file deliberately records its production-metric reuse.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import temporal_world as world

KEYS=('J','single','moved','stationary','Q','entropy','nll','exact_max_tie_rate','history_pair_J')
GROUPS=('old','added','sealed','common30');MODES=('immediate','delayed')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(v):return None if v[0] is None else float(np.mean(v))
def auc(t,v):return None if v[0] is None else float(np.trapz(v,t)/(t[-1]-t[0]))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');assert done['status']=='complete'
    checks=0;error=0.
    def compare(a,b):
        nonlocal checks,error
        if isinstance(b,dict):
            for k,v in b.items():compare(a[k],v)
        elif isinstance(b,(int,float)):
            checks+=1;err=abs(a-b);error=max(error,err);assert np.isclose(a,b,atol=2e-12,rtol=2e-12)
        else:assert a==b
    runs=[]
    for name in done['runs']:
        folder=out/name;cfg=read(folder/'config.json');curve=read(folder/'curve.json');points=[]
        for point in curve:
            scores={}
            for mode in MODES:
                p=folder/f"evaluation_{point['update']:04d}_{mode}.npz";assert sha(p)==done['files'][str(p.relative_to(out))]
                with np.load(p) as z:raw={k:z[k] for k in z.files}
                scores[mode]=world.metrics(raw.pop('logits'),raw,cfg['partition'])
                compare(scores[mode],point['scores'][mode])
            points.append(dict(update=point['update'],scores=scores))
        ep=folder/f"evaluation_{cfg['updates']:04d}_delayed_erase.npz";assert sha(ep)==done['files'][str(ep.relative_to(out))]
        with np.load(ep) as z:raw={k:z[k] for k in z.files}
        erase=world.metrics(raw.pop('logits'),raw,cfg['partition']);compare(erase,read(folder/'result.json')['erase_scores'])
        t=[x['update'] for x in points]
        runs.append(dict(**{k:cfg[k] for k in ('seed','partition','direction','arm')},curve=points,erase_scores=erase,
            auc={m:{g:{k:auc(t,[r['scores'][m][g][k] for r in points]) for k in KEYS} for g in GROUPS} for m in MODES}))
    times=[x['update'] for x in runs[0]['curve']]
    def average(rows,base):
        return dict(**base,curve=[dict(update=t,scores={m:{g:{k:mean([r['curve'][i]['scores'][m][g][k] for r in rows]) for k in KEYS} for g in GROUPS} for m in MODES}) for i,t in enumerate(times)],
            auc={m:{g:{k:mean([r['auc'][m][g][k] for r in rows]) for k in KEYS} for g in GROUPS} for m in MODES},
            erase_scores={g:{k:mean([r['erase_scores'][g][k] for r in rows]) for k in KEYS} for g in GROUPS})
    seed_rows=[average([r for r in runs if r['seed']==s and r['arm']==a],dict(seed=s,arm=a)) for s,a in itertools.product(inv['seeds'],inv['arms'])]
    aggregate=[average([r for r in seed_rows if r['arm']==a],dict(arm=a)) for a in inv['arms']]
    contrasts=[]
    for s in inv['seeds']:
        d={r['arm']:r for r in seed_rows if r['seed']==s};f,x=d['full'],d['detach'];F,X=f['curve'][-1]['scores'],x['curve'][-1]['scores']
        contrasts.append(dict(seed=s,primary_common30_delayed_J=F['delayed']['common30']['J']-X['delayed']['common30']['J'],
            old_delayed_J=F['delayed']['old']['J']-X['delayed']['old']['J'],old_immediate_J=F['immediate']['old']['J']-X['immediate']['old']['J'],
            common30_delayed_J_AUC=f['auc']['delayed']['common30']['J']-x['auc']['delayed']['common30']['J']))
    result=dict(passed=True,checks=checks,max_absolute_comparison_error=error,formal=inv['formal'],
        source_sha256=sha(__file__),training_receipt_sha256=sha(out/'training_complete.json'),metric_source_sha256=sha(world.__file__),
        metrics_reused_from_frozen_production=True,independent_formula_implementation='analyze_temporal.py',
        runs=runs,seed_rows=seed_rows,aggregate=aggregate,contrasts=contrasts,
        primary_mean=float(np.mean([r['primary_common30_delayed_J'] for r in contrasts])))
    (out/'independent_recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,checks=checks,max_error=error,primary_mean=result['primary_mean'])))

if __name__=='__main__':main()
