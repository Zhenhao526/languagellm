from pathlib import Path
import json,numpy as np,hashlib
root=Path('/Users/xia/Documents/ChatGPT/语言');batch=root/'redesign_v0.14/results/reset_001';rec=json.loads((batch/'independent_recount.json').read_text());assert rec['passed'];out={'arms':{},'primary_by_seed':rec['primary_by_seed'],'contrast':'retained minus reset'};hashes={str(batch/'independent_recount.json'):hashlib.sha256((batch/'independent_recount.json').read_bytes()).hexdigest()}
for a in ['retained','reset']:
 rows=[r for r in rec['rows'] if r['arm']==a];means={}
 for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
  means[mode]={g:float(np.mean([r['modes'][mode][g]['J'] for r in rows])) for g in ('old','added','sealed','common30')}
 proto=[d for r in rows for d in r['protocol']['2400']]
 means['protocol']={mode:{k:float(np.mean([d[mode]['common30'][k] for d in proto])) for k in proto[0][mode]['common30']} for mode in ('native','greedy')}
 aucs={g:[] for g in ('old','added','sealed')}
 for r in rows:
  base=root/'redesign_v0.13/results/spatial_001' if a=='retained' else batch;arm='control' if a=='retained' else 'reset';p=base/f"social_s{r['seed']}_p{r['partition']}_{arm}/curve.json";c=json.loads(p.read_text());hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();t=np.array([v['update'] for v in c])
  for g in aucs:
   y=np.array([v['scores']['normal']['map_groups'][g]['both_accuracy'] for v in c]);aucs[g].append(float(np.sum((t[1:]-t[:-1])*(y[1:]+y[:-1])/2)/(t[-1]-t[0])))
 means['auc']={g:float(np.mean(v)) for g,v in aucs.items()};out['arms'][a]=means
out['input_hashes']=hashes;(batch/'report_numbers.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='input_hashes'},indent=2))
