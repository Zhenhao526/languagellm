"""Stratified analysis of compositional protocol recovery."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

RETURN_MIN=.60; GAP_MAX=.02

def ci(x):
 x=np.asarray(x,dtype=float); n=len(x)
 if n<2:return [float(x.mean()),float(x.mean())] if n else [None,None]
 h=2.0395*float(x.std(ddof=1))/math.sqrt(n); return [float(x.mean()-h),float(x.mean()+h)]
def wilson(k,n,z=1.96):
 if n==0:return [None,None]
 p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return [float(c-h),float(c+h)]
def load_result(path): return json.loads(Path(path).read_text())
def parent_status(root,seed):
 d=load_result(Path(root)/f'seed_{seed}_dual2_rotating_hidden_live_factorized_staged/result.json'); w=d['final']['heldout']['workers']; n=float(np.mean([x['natural']['team_return_mean'] for x in w.values()])); r=float(np.mean([x['recombined']['team_return_mean'] for x in w.values()])); return {'seed':seed,'parent_natural':n,'parent_recombined_minus_natural':r-n,'parent_composable':bool(n>=RETURN_MIN and abs(r-n)<=GAP_MAX),'parent_sender_sequences':d['final']['heldout']['codebook']['sender_sequence_by_goal_worker']}
def child_row(root,seed,role,channel,parent):
 d=load_result(Path(root)/f'seed_{seed}_{role}_{channel}/result.json'); m=d['final']['heldout']['modes']; cb=d['final']['heldout']['codebook']; seq=cb['sender_sequences']; parentseq=[x[0] for x in parent['parent_sender_sequences']]
 return {'seed':seed,'role':role,'channel':channel,'parent_composable':parent['parent_composable'],'natural':float(m['natural']['team_return_mean']),'silent_or_mode':float(m['natural']['team_return_mean']),'recombined_minus_natural':float(m['recombined']['team_return_mean']-m['natural']['team_return_mean']),'semantic_success':float(cb['semantic_success_mean']),'sender_sequence_exact':bool(seq==parentseq),'child_composable':bool(m['natural']['team_return_mean']>=RETURN_MIN and abs(m['recombined']['team_return_mean']-m['natural']['team_return_mean'])<=GAP_MAX)}
def analyze(parent_root,child_root):
 rows=[]
 for seed in range(76101,76133):
  p=parent_status(parent_root,seed)
  for role in ('worker','sender'):
   for channel in ('live','silent','permuted'):
    r=child_row(child_root,seed,role,channel,p); r['natural']=float(load_result(Path(child_root)/f'seed_{seed}_{role}_{channel}/result.json')['final']['heldout']['modes']['natural']['team_return_mean']); rows.append(r)
 out=[]
 for role in ('worker','sender'):
  for group,label in [(lambda x:True,'all'),(lambda x:x['parent_composable'],'parent_composable'),(lambda x:not x['parent_composable'],'parent_noncomposable')]:
   rr=[x for x in rows if x['role']==role and group(x)]; live=[x for x in rr if x['channel']=='live']; n=len(live); succ=sum(x['child_composable'] for x in live)
   def vals(ch,key): return [x[key] for x in rr if x['channel']==ch]
   effects=[]
   for control in ('silent','permuted'):
    pairs=[]
    for s in sorted({x['seed'] for x in rr}):
     a=next(x for x in live if x['seed']==s); b=next(x for x in rr if x['seed']==s and x['channel']==control); pairs.append(a['natural']-b['natural'])
    effects.append({'control':control,'mean':float(np.mean(pairs)),'ci95_t':ci(pairs)})
   out.append({'role':role,'group':label,'n':n,'live_natural_mean':float(np.mean([x['natural'] for x in live])),'live_natural_ci95_t':ci([x['natural'] for x in live]),'live_recombined_minus_natural_mean':float(np.mean([x['recombined_minus_natural'] for x in live])),'live_recombined_minus_natural_ci95_t':ci([x['recombined_minus_natural'] for x in live]),'child_composable_count':succ,'child_composable_rate':succ/len(live),'child_composable_wilson95':wilson(succ,len(live)),'sender_sequence_exact_count':sum(x['sender_sequence_exact'] for x in live),'semantic_success_mean':float(np.mean([x['semantic_success'] for x in live])),'effects':effects})
 return {'schema':'generation_compositional_transmission_analysis_v1','rule':{'natural_min':RETURN_MIN,'absolute_recombined_gap_max':GAP_MAX},'rows':rows,'groups':out,'parent_composable_count':sum(parent_status(parent_root,s)['parent_composable'] for s in range(76101,76133)),'parent_count':32}
def write_md(path,d):
 L=['# Stratified compositional transmission analysis','',f"- parent composable: {d['parent_composable_count']}/{d['parent_count']}",f"- child rule: natural ≥ {d['rule']['natural_min']:.2f} and |recombined−natural| ≤ {d['rule']['absolute_recombined_gap_max']:.2f}",'','| role | parent group | n | child live composable | live natural | live recombined−natural | sender sequence exact |','|---|---|---:|---:|---:|---:|---:|']
 for x in d['groups']: L.append(f"| `{x['role']}` | `{x['group']}` | {x['n']} | {x['child_composable_count']}/{x['n']} | {x['live_natural_mean']:.3f} | {x['live_recombined_minus_natural_mean']:.3f} | {x['sender_sequence_exact_count']}/{x['n']} |")
 L += ['','','| role | parent group | live−silent | live−permuted |','|---|---|---:|---:|']
 for x in d['groups']:
  es={e['control']:e for e in x['effects']}; L.append(f"| `{x['role']}` | `{x['group']}` | {es['silent']['mean']:.3f} [{es['silent']['ci95_t'][0]:.3f},{es['silent']['ci95_t'][1]:.3f}] | {es['permuted']['mean']:.3f} [{es['permuted']['ci95_t'][0]:.3f},{es['permuted']['ci95_t'][1]:.3f}] |")
 L += ['','','The parent stratum is fixed by the parent endpoint before child training. The analysis therefore separates recovery of a compositional parent protocol from recovery of a degenerate whole-code protocol.']
 Path(path).write_text('\n'.join(L)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--parent-root',required=True); ap.add_argument('--child-root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--markdown',required=True); a=ap.parse_args(); d=analyze(a.parent_root,a.child_root); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); write_md(a.markdown,d); print(json.dumps({'status':'written','rows':len(d['rows'])},ensure_ascii=False))
