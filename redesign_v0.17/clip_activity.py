"""Prespecified v17 audit of logged clipping activity; no model calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parent;B=ROOT/'results/baseline_001'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(B/'training_complete.json')['status']=='complete'
inv=read(B/'invocation.json');args=inv['args']
arms=['matched','shuffled']
rows=[];hashes={}
for seed in args['seeds']:
    for partition in args['partitions']:
        for arm in arms:
            parent=Path(args['scaled_batch']) if arm=='matched' else B
            original='reset_scaled' if arm=='matched' else 'both_sender_baseline_shuffle'
            path=parent/f'social_s{seed}_p{partition}_{original}/training.jsonl'
            hashes[str(path)]=sha(path);data=[json.loads(s) for s in path.read_text().splitlines()]
            assert len(data)==2400 and [d['update'] for d in data]==list(range(1,2401))
            for who in (0,1):
                for role in ('sender','receiver'):
                    norms=np.array([d['agents'][who]['gradient_norm_by_role'][role] for d in data],dtype=np.float32)
                    # Exact scalar formula used by PyTorch's clip_grad_norm_ in the runner.
                    coefficients=np.minimum(np.float32(1),np.float32(2)/(norms+np.float32(1e-6)))
                    assert np.isfinite(norms).all() and (norms>=0).all()
                    for phase,sl in [('all',slice(0,2400)),('entropy_on',slice(0,2100)),('entropy_off',slice(2100,2400))]:
                        v=norms[sl];c=coefficients[sl]
                        rows.append(dict(seed=seed,partition=partition,arm=arm,person=who,role=role,phase=phase,
                            count=len(v),active_count=int(np.count_nonzero(c<1)),rate=float(np.mean(c<1)),
                            min_coefficient=float(c.min()),max_norm=float(v.max()),mean_norm=float(v.astype(np.float64).mean())))
summary={}
for arm in arms:
    summary[arm]={}
    for role in ('sender','receiver'):
        q=[r for r in rows if r['arm']==arm and r['role']==role and r['phase']=='all']
        summary[arm][role]=dict(person_updates=sum(r['count'] for r in q),active_count=sum(r['active_count'] for r in q),
            rate=sum(r['active_count'] for r in q)/sum(r['count'] for r in q),max_norm=max(r['max_norm'] for r in q),
            min_coefficient=min(r['min_coefficient'] for r in q),
            seed_rates={str(s):float(np.mean([r['rate'] for r in q if r['seed']==s])) for s in args['seeds']})
out=dict(status='complete',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
    scope='Prespecified for v17 before new training, following the exploratory v16 finding. Logged pre-clip norms only; no new training, inference, fitted predictor, or change to fixed primary/auxiliary outcomes.',
    formula='min(1, float32(2)/(float32(norm)+float32(1e-6))); active iff coefficient<1',
    source_sha256=sha(__file__),source_hashes=hashes,rows=rows,summary=summary,
    limitations=['Logged total norm cannot decompose value and policy contributions.','Frequency alone cannot establish clipping as a mediator; brief early changes can persist.','Float32 coefficient reconstruction is an audit of the logged scalar formula, not a replay of all gradients.'])
(B/'clip_activity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
