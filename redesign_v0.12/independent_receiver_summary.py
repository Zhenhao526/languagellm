"""Independent direct-world recount of all 96 frozen receiver endpoints."""
from pathlib import Path
from itertools import product
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/receiver_001'
SEEDS=[29101,29102,29103,29104]
ARMS=['warm_rl24','warm_ce24','fresh_ce24','fresh_ce30']


def logprob(x):
    x=np.asarray(x,dtype=np.float64)
    z=x-x.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))


def load_context(path):
    with np.load(path) as z:a={k:z[k] for k in z.files if k!='h'}
    first=np.exp(logprob(a['first_logits']))
    second=np.exp(logprob(a['second_logits']))
    a['p']=(first[:,:,None]*second).reshape(-1,49)
    return a


def recount(context,logits,pool):
    take=np.isin(context['map_ids'],pool)
    pos=context['positions'][take];p=context['p'][take]
    code=context['greedy_message'][take]@np.array([7,1])
    lp=logprob(logits);rp=np.exp(lp);actions=logits.argmax(-1)
    food=rp[:,0,pos[:,0]].T;water=rp[:,1,pos[:,1]].T
    gf=(actions[:,0][None,:]==pos[:,0,None]);gw=(actions[:,1][None,:]==pos[:,1,None])
    nll=-np.mean(np.sum(p*(lp[:,0,pos[:,0]].T+lp[:,1,pos[:,1]].T),1))
    c=np.zeros((49,6,6))
    for f,w in product(range(6),repeat=2):
        c[:,f,w]=p[(pos[:,0]==f)&(pos[:,1]==w)].sum(0)/len(pos)
    mass=c.sum((1,2));cf=c.sum(2);cw=c.sum(1)
    hf=0.;hw=0.;hj=0.
    for m in range(49):
        if mass[m]==0:continue
        for f in range(6):
            if cf[m,f]>0:hf-=cf[m,f]*np.log(cf[m,f]/mass[m])
        for w in range(6):
            if cw[m,w]>0:hw-=cw[m,w]*np.log(cw[m,w]/mass[m])
        for f,w in product(range(6),repeat=2):
            if c[m,f,w]>0:hj-=c[m,f,w]*np.log(c[m,f,w]/mass[m])
    joint=c.reshape(49,36).max(1).sum()
    utilities=np.stack([.25*(cf[:,f]+cw[:,w])+.5*c[:,f,w] for f,w in product(range(6),repeat=2)],1)
    natural=(actions[code]==pos)
    return dict(worlds=len(pos),Q=float(np.mean(np.sum(p*food*water,1))),
        G=float(np.mean(np.sum(p*gf*gw,1))),N=float(natural.all(1).mean()),
        Q_reward=float(np.mean(np.sum(p*(.25*(food+water)+.5*food*water),1))),
        G_reward=float(np.mean(np.sum(p*(.25*(gf.astype(float)+gw)+.5*gf*gw),1))),
        N_reward=float(np.mean(.25*natural.sum(1)+.5*natural.all(1))),
        branch_nll=float(nll),branch_entropy=float(hf+hw),joint_entropy=float(hj),
        CMI=float(hf+hw-hj),nll_residual=float(nll-hf-hw),
        joint_oracle=float(joint),mixed_oracle=float(utilities.max(1).sum()))


def main():
    names=[f's{s}_p{p}_d{d}_{a}' for s,p,d,a in product(SEEDS,(1,2,3),(0,1),ARMS)]
    assert set(names)=={p.parent.name for p in BATCH.glob('s*_p*_d*_*/result.json')}
    assert json.loads((BATCH/'training_complete.json').read_text())['complete']
    rows=[]
    for seed,part,direction in product(SEEDS,(1,2,3),(0,1)):
        src=BATCH/f'source_s{seed}_p{part}_d{direction}'
        ev=load_context(src/'context_evaluation.npz');fit=load_context(src/'context_fit.npz')
        for arm in ARMS:
            folder=BATCH/f's{seed}_p{part}_d{direction}_{arm}'
            cfg=json.loads((folder/'config.json').read_text())
            logits=np.load(folder/'receiver_2400.npz')['logits']
            groups=dict(cfg['map_groups'],all=list(range(30)),train24=sorted(cfg['map_groups']['old']+cfg['map_groups']['added']))
            row=dict(seed=seed,partition=part,direction=direction,arm=arm,
                evaluation={g:recount(ev,logits,pool) for g,pool in groups.items()},
                fit=recount(fit,logits,cfg['training_pool']))
            rows.append(row)
    metrics=[k for k in rows[0]['fit'] if k!='worlds']
    cells=[]
    for seed,arm in product(SEEDS,ARMS):
        r=[x for x in rows if x['seed']==seed and x['arm']==arm];assert len(r)==6
        cells.append(dict(seed=seed,arm=arm,
            evaluation={g:{k:float(np.mean([x['evaluation'][g][k] for x in r])) for k in metrics} for g in rows[0]['evaluation']},
            fit={k:float(np.mean([x['fit'][k] for x in r])) for k in metrics}))
    means={a:{'evaluation':{g:{k:float(np.mean([r['evaluation'][g][k] for r in cells if r['arm']==a])) for k in metrics}
        for g in rows[0]['evaluation']},'fit':{k:float(np.mean([r['fit'][k] for r in cells if r['arm']==a])) for k in metrics}} for a in ARMS}
    differences={}
    for left,right in [('warm_ce24','warm_rl24'),('fresh_ce24','warm_ce24')]:
        differences[left+' - '+right]={g:{k:[next(r for r in cells if r['seed']==s and r['arm']==left)['evaluation'][g][k]-
             next(r for r in cells if r['seed']==s and r['arm']==right)['evaluation'][g][k] for s in SEEDS]
             for k in metrics} for g in rows[0]['evaluation']}
    output=dict(scope='Independent direct-world NumPy endpoint recount; no import of runner or production metrics; 4 seeds, average 3 partitions and 2 directions within seed.',
        rows=rows,seed_cells=cells,means=means,differences=differences,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (BATCH/'independent_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(means,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
