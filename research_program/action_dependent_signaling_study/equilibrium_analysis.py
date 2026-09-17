"""Pre-registered seed-level equilibrium analysis for action-dependent signaling."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

THRESHOLD_RETURN = 0.60
THRESHOLD_GAP = 0.02
T95 = {7: 2.365, 15: 2.131, 31: 2.040}


def ci_mean(x):
    x=np.asarray(x,dtype=float); n=len(x)
    if n == 0: return [None,None]
    if n == 1: return [float(x[0]),float(x[0])]
    t=T95.get(n-1,1.96); h=t*float(x.std(ddof=1))/math.sqrt(n)
    return [float(x.mean()-h),float(x.mean()+h)]


def wilson(k,n,z=1.96):
    if n == 0: return [None,None]
    p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(cen-half),float(cen+half)]


def mcnemar_exact(b,c):
    n=b+c
    if n == 0: return 1.0
    tail=sum(math.comb(n,k) for k in range(0,min(b,c)+1))/2**n
    return float(min(1.0,2*tail))


def load(path):
    payload=json.loads(Path(path).read_text()); return {(int(r['seed']),r['condition']):r for r in payload['results']}


def heldout(r):
    w=r['final']['heldout']['workers']; nat=float(np.mean([x['natural']['team_return_mean'] for x in w.values()])); rec=float(np.mean([x['recombined']['team_return_mean'] for x in w.values()])); return nat,rec


def analyze(path):
    by=load(path); seeds=sorted({s for s,_ in by}); rows=[]; contrasts=[]
    for task in ('factorized','entangled'):
        condition_rows={}
        for protocol in ('simultaneous','staged'):
            cond=f'dual2_rotating_hidden_live_{task}_{protocol}'
            vals=[]
            for seed in seeds:
                n,r=heldout(by[(seed,cond)]); vals.append({'seed':seed,'task':task,'protocol':protocol,'natural':n,'recombined':r,'recombined_minus_natural':r-n,'composable_equilibrium':bool(n>=THRESHOLD_RETURN and abs(r-n)<=THRESHOLD_GAP)})
            condition_rows[protocol]=vals; rows.extend(vals)
        sim=condition_rows['simultaneous']; sta=condition_rows['staged']; sm={x['seed']:x for x in sim}; st={x['seed']:x for x in sta}
        nd=[st[s]['natural']-sm[s]['natural'] for s in seeds]
        gd=[st[s]['recombined_minus_natural']-sm[s]['recombined_minus_natural'] for s in seeds]
        b=sum(st[s]['composable_equilibrium'] and not sm[s]['composable_equilibrium'] for s in seeds)
        c=sum(sm[s]['composable_equilibrium'] and not st[s]['composable_equilibrium'] for s in seeds)
        for protocol, vals in condition_rows.items():
            succ=sum(x['composable_equilibrium'] for x in vals); n=len(vals)
            contrasts.append({'task':task,'protocol':protocol,'seeds':n,'composable_equilibrium_count':succ,'composable_equilibrium_rate':succ/n,'composable_equilibrium_wilson95':wilson(succ,n),'natural_mean':float(np.mean([x['natural'] for x in vals])),'natural_ci95_t':ci_mean([x['natural'] for x in vals]),'recombined_minus_natural_mean':float(np.mean([x['recombined_minus_natural'] for x in vals])),'recombined_minus_natural_ci95_t':ci_mean([x['recombined_minus_natural'] for x in vals])})
        contrasts.append({'task':task,'protocol':'staged_minus_simultaneous','seeds':len(seeds),'natural_difference_mean':float(np.mean(nd)),'natural_difference_ci95_t':ci_mean(nd),'recombined_gap_difference_mean':float(np.mean(gd)),'recombined_gap_difference_ci95_t':ci_mean(gd),'mcnemar_staged_only':b,'mcnemar_simultaneous_only':c,'mcnemar_exact_p':mcnemar_exact(b,c)})
    return {'schema':'action_dependent_equilibrium_analysis_v1','thresholds':{'natural_min':THRESHOLD_RETURN,'absolute_recombined_gap_max':THRESHOLD_GAP},'runs':len(by),'seeds':seeds,'rows':rows,'contrasts':contrasts}


def write_md(path,data):
    lines=['# Action-dependent equilibrium analysis','',f"- runs: {data['runs']}",f"- composable-equilibrium rule: natural ≥ {data['thresholds']['natural_min']:.2f} and |recombined−natural| ≤ {data['thresholds']['absolute_recombined_gap_max']:.2f}",'','| task | protocol | composable seeds | rate | Wilson 95% | natural | recombined−natural |','|---|---|---:|---:|---|---:|---:|']
    for x in data['contrasts']:
        if x['protocol']=='staged_minus_simultaneous': continue
        lines.append(f"| `{x['task']}` | `{x['protocol']}` | {x['composable_equilibrium_count']}/{x['seeds']} | {x['composable_equilibrium_rate']:.3f} | [{x['composable_equilibrium_wilson95'][0]:.3f}, {x['composable_equilibrium_wilson95'][1]:.3f}] | {x['natural_mean']:.3f} | {x['recombined_minus_natural_mean']:.3f} |")
    lines += ['','','| task | staged−sim natural | staged−sim recombination gap | McNemar staged-only / simultaneous-only | exact p |','|---|---:|---:|---:|---:|']
    for x in data['contrasts']:
        if x['protocol']!='staged_minus_simultaneous': continue
        lines.append(f"| `{x['task']}` | {x['natural_difference_mean']:.3f} [{x['natural_difference_ci95_t'][0]:.3f},{x['natural_difference_ci95_t'][1]:.3f}] | {x['recombined_gap_difference_mean']:.3f} | {x['mcnemar_staged_only']} / {x['mcnemar_simultaneous_only']} | {x['mcnemar_exact_p']:.4f} |")
    lines += ['','','The equilibrium rule was fixed before the additional 16 seeds. These counts describe convergence into a high-performing, recombination-preserving basin; they are not a claim that every staged run develops a language.']
    Path(path).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--markdown',required=True); a=ap.parse_args(); d=analyze(a.results); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); write_md(a.markdown,d); print(json.dumps({'status':'written','runs':d['runs']},ensure_ascii=False))
