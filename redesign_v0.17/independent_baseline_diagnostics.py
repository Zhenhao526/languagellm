"""NumPy-only permutation/trace audit and prespecified manipulation summaries.

All per-update permutations are regenerated independently. Baseline magnitudes
outside the two recorded raw traces remain logged summaries, not model replays.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent;B=ROOT/'results/baseline_001'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrsha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
FIELDS=('fixed_point_count','changed_value_count','unchanged_value_count','original_std','used_minus_original_mean_abs','used_minus_original_max_abs','used_minus_original_rms')
PHASES={'all':(0,2400),'entropy_on':(0,2100),'entropy_off':(2100,2400)}
assert read(B/'training_complete.json')['status']=='complete'
args=read(B/'invocation.json')['args'];rows=[];hashes={};trace_count=0;permutations=0;checks=0
for seed in args['seeds']:
 for p in args['partitions']:
  folder=B/f'social_s{seed}_p{p}_both_sender_baseline_shuffle'
  path=folder/'training.jsonl';hashes[str(path)]=sha(path)
  log=[json.loads(t) for t in path.read_text().splitlines()]
  assert len(log)==2400 and [x['update'] for x in log]==list(range(1,2401));checks+=1
  traces={step:np.load(folder/f'train_{step+1:04d}.npz') for step in (0,2100)}
  for step in traces:hashes[str(folder/f'train_{step+1:04d}.npz')]=sha(folder/f'train_{step+1:04d}.npz')
  for who in (0,1):
   series={key:[] for key in FIELDS}
   for step,line in enumerate(log):
    r=line['agents'][who]['sender_baseline'];entropy=[17017,seed,p,step,who]
    index=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).permutation(256).astype(np.int64)
    assert r['n']==256 and r['mode']=='shuffle' and r['namespace']==17017 and r['step_zero_based']==step and r['permutation_seed_entropy']==entropy;checks+=1
    assert r['permutation_sha256']==arrsha(index) and r['fixed_point_count']==np.count_nonzero(index==np.arange(256));checks+=1
    assert r['multiset_preserved'] and r['original_sorted_sha256']==r['used_sorted_sha256'];checks+=1
    assert 0<=r['changed_value_count']<=256 and 0<=r['unchanged_value_count']<=256;checks+=1
    assert all(np.isfinite(r[k]) and r[k]>=0 for k in FIELDS);checks+=1
    assert r['used_minus_original_max_abs']+1e-15>=r['used_minus_original_rms']>=r['used_minus_original_mean_abs']-1e-15;checks+=1
    permutations+=1
    for key in FIELDS:series[key].append(r[key])
    if step in traces:
     z=traces[step];original=z['baseline_original'][who];used=z['baseline_used'][who]
     assert original.dtype==used.dtype==np.float32 and original.shape==used.shape==(256,);checks+=1
     assert np.array_equal(z['baseline_permutation'][who],index);checks+=1
     assert np.array_equal(used.view(np.uint32),original[index].view(np.uint32));checks+=1
     assert Counter(original.view(np.uint32).tolist())==Counter(used.view(np.uint32).tolist());checks+=1
     assert arrsha(original)==r['original_sha256'] and arrsha(used)==r['used_sha256'];checks+=1
     assert arrsha(np.sort(original.view(np.uint32)))==r['original_sorted_sha256'];checks+=1
     delta=used.astype(np.float64)-original.astype(np.float64)
     expected={'fixed_point_count':int(np.count_nonzero(index==np.arange(256))),
       'changed_value_count':int(np.count_nonzero(delta)),
       'unchanged_value_count':int(np.count_nonzero(original.view(np.uint32)==used.view(np.uint32))),
       'original_std':float(np.std(original.astype(np.float64))),
       'used_minus_original_mean_abs':float(np.abs(delta).mean()),
       'used_minus_original_max_abs':float(np.abs(delta).max()),
       'used_minus_original_rms':float(np.sqrt(np.mean(delta**2)))}
     for key,value in expected.items():assert value==r[key];checks+=1
     reward=z['reward'][z['scout']==who]
     assert np.array_equal(z['baseline_target'][who],reward-np.float32(1));checks+=1
     trace_count+=1
   for phase,(start,end) in PHASES.items():
    stats={key:dict(mean=float(np.mean(values[start:end])),minimum=float(np.min(values[start:end])),maximum=float(np.max(values[start:end])),mean_square=float(np.mean(np.square(values[start:end])))) for key,values in series.items()}
    rows.append(dict(seed=seed,partition=p,person=who,phase=phase,updates=end-start,values=stats))
summary={}
for phase in PHASES:
 q=[r for r in rows if r['phase']==phase]
 summary[phase]=dict(person_updates=sum(r['updates'] for r in q),
  metrics={key:dict(mean=float(np.mean([r['values'][key]['mean'] for r in q])),minimum=min(r['values'][key]['minimum'] for r in q),maximum=max(r['values'][key]['maximum'] for r in q),
   seed_means={str(seed):float(np.mean([r['values'][key]['mean'] for r in q if r['seed']==seed])) for seed in args['seeds']}) for key in FIELDS})
for phase in PHASES:
 q=[r for r in rows if r['phase']==phase]
 summary[phase]['pooled_rms_change']=float(np.sqrt(np.mean([r['values']['used_minus_original_rms']['mean_square'] for r in q])))
 summary[phase]['seed_pooled_rms_change']={str(seed):float(np.sqrt(np.mean([r['values']['used_minus_original_rms']['mean_square'] for r in q if r['seed']==seed]))) for seed in args['seeds']}
out=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),check_count=checks,person_permutations_checked=permutations,raw_person_traces_checked=trace_count,
 scope='Prespecified manipulation diagnostic. All deterministic permutations checked; two recorded raw value traces per person verified without model calls. Other magnitudes summarize recorded values, not all-trajectory model replays.',
 source_sha256=sha(__file__),source_hashes=hashes,rows=rows,summary=summary)
(B/'independent_baseline_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','source_hashes','summary')},indent=2))
print(json.dumps(summary['all'],indent=2))
