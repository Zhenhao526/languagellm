"""Compact aggregation for compositional transmission."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def ci(x):
 x=np.asarray(x,dtype=float); n=len(x)
 if n<2:return [float(x.mean()),float(x.mean())] if n else [None,None]
 h=2.13145*float(x.std(ddof=1))/math.sqrt(n); return [float(x.mean()-h),float(x.mean()+h)]
def load(p): return {(int(r['seed']),r['condition']):r for r in json.loads(Path(p).read_text())['results']}
def mean(r,mode='natural'): return float(r['final']['heldout']['modes'][mode]['team_return_mean'])
def aggregate(path):
 by=load(path); rows=[]; effects=[]
 for role in ('worker','sender'):
  for channel in ('live','silent','permuted'):
   vals=[mean(by[(s,f'{role}_{channel}')]) for s in sorted({x for x,_ in by})]; rows.append({'role':role,'channel':channel,'n':len(vals),'natural_mean':float(np.mean(vals)),'natural_ci95_t':ci(vals)})
  seeds=sorted({x for x,_ in by});
  for label,control in [('live-minus-silent','silent'),('live-minus-permuted','permuted')]:
   vals=[mean(by[(s,f'{role}_live')])-mean(by[(s,f'{role}_{control}')]) for s in seeds]; effects.append({'role':role,'effect':label,'values':vals,'mean':float(np.mean(vals)),'ci95_t':ci(vals)})
  vals=[mean(by[(s,f'{role}_live')], 'recombined')-mean(by[(s,f'{role}_live')]) for s in seeds]; effects.append({'role':role,'effect':'recombined-minus-natural','values':vals,'mean':float(np.mean(vals)),'ci95_t':ci(vals)})
 return {'schema':'generation_compositional_transmission_aggregate_v1','runs':len(by),'rows':rows,'effects':effects}
def md(path,d):
 L=['# Generation compositional transmission compact aggregation','','| role | live | silent | permuted | live−silent | live−permuted | recombined−natural |','|---|---:|---:|---:|---:|---:|---:|']
 for role in ('worker','sender'):
  x={r['channel']:r['natural_mean'] for r in d['rows'] if r['role']==role}; e={r['effect']:r['mean'] for r in d['effects'] if r['role']==role}; L.append(f"| `{role}` | {x['live']:.3f} | {x['silent']:.3f} | {x['permuted']:.3f} | {e['live-minus-silent']:.3f} | {e['live-minus-permuted']:.3f} | {e['recombined-minus-natural']:.3f} |")
 L+=['','','Intervals are paired across the same parent seed. `recombined−natural` tests whether the new component preserves the parent slot decomposition.']; Path(path).write_text('\n'.join(L)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--markdown',required=True); a=ap.parse_args(); d=aggregate(a.results); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); md(a.markdown,d); print(json.dumps({'status':'written','runs':d['runs']},ensure_ascii=False))
