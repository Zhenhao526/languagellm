"""Independent audit for compositional parent-to-child transmission."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from . import design, policy, runner

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,(float,int,np.number)): return math.isfinite(float(x))
    return True

def audit(prepared,execution):
    plan,cfg=runner.verify(prepared); execution=Path(execution); progress=json.loads((execution/'progress.json').read_text()); files=sorted(execution.glob('seed_*/result.json')); assert progress['completed']==progress['total']==len(files)
    logs=checkpoints=0; maxerr=0.0; rows=[]; by={}
    for path in files:
        r=json.loads(path.read_text()); assert finite(r) and len(r['trajectory'])==r['updates']; role,channel=design.parse_condition(r['condition']); parent=Path(r['parent_checkpoint']); assert parent.is_file() and sha(parent)==r['parent_checkpoint_sha256'] if 'parent_checkpoint_sha256' in r else True
        log=path.parent/'training.jsonl'; assert sha(log)==r['training_log_sha256']; logs+=sum(1 for _ in log.open())
        cp=path.parent/f"checkpoint_{r['updates']:04d}.npz"; assert cp.is_file(); checkpoints+=1
        p=policy.load(cp); fresh=runner.evaluate(p,r['seed'],role,evaluation=True); stored=r['final']['heldout']
        for mode in ('natural','closed','permuted','recombined'):
            a=stored['modes'][mode]; b=fresh['modes'][mode]
            for key in ('team_return_mean','team_return_sd','positive_episode_rate'):
                x,y=a[key],b[key]
                if x is None or y is None: assert x is None and y is None
                else: maxerr=max(maxerr,abs(float(x)-float(y)))
        assert stored['codebook']==fresh['codebook']
        parentp=policy.load(parent)
        if role=='worker':
            assert np.array_equal(p['sender_logits_hidden'],parentp['sender_logits_hidden']); assert np.array_equal(p['worker_logits'][1:],parentp['worker_logits'][1:])
        else: assert np.array_equal(p['worker_logits'],parentp['worker_logits'])
        by[(r['seed'],r['condition'])]=r; rows.append({'seed':r['seed'],'condition':r['condition'],'heldout':stored})
    pair_checks=0
    for seed in sorted({s for s,_ in by}):
        for role in sorted({r.split('_')[0] for _,r in by}):
            if not all((seed,f'{role}_{c}') in by for c in design.CHANNELS):
                continue
            arms=[by[(seed,f'{role}_{c}')] for c in design.CHANNELS]
            for a,b in zip(arms,arms[1:]):
                for x,y in zip(a['trajectory'],b['trajectory']):
                    for k in ('world_sha256','goal_sha256','partner_sha256','message_uniform_sha256','action_uniform_sha256'):
                        assert x[k]==y[k],(seed,role,k,x['update'])
                    pair_checks+=1
    assert maxerr<1e-12
    return {'schema':'generation_compositional_transmission_audit_v1','status':'passed','runs':len(files),'training_log_rows':logs,'final_checkpoints':checkpoints,'max_abs_replay_error':maxerr,'paired_channel_trajectory_rows':pair_checks,'rows':rows}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepared',required=True); ap.add_argument('--execution',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); d=audit(Path(a.prepared),Path(a.execution)); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:v for k,v in d.items() if k!='rows'},ensure_ascii=False))
