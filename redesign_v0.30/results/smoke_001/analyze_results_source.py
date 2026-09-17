"""Independent NumPy-only v30 protocol statistics; no production imports.

All saved checkpoints are rescored. This file does not establish forward or
training provenance; audit_results.py performs the separate bounded replay.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, shutil, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
MAPS=np.array(list(itertools.permutations(range(6),2)),np.int64)
ARMS=('connected_cycle12','three_products')
MASKS=('pooled','food_only','water_only')
PERMS=((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4))
QS=((2,3,4,5,0,1),(2,5,4,1,0,3))
WORLD_KEYS=('map_id','photo_ids','positions','shown')

def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def arrays_sha(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

class Checks:
    def __init__(self): self.count=0;self.comparisons=0;self.max_error=0.;self.coverage=Counter()
    def require(self,ok,label):
        self.count+=1
        if not bool(ok): raise AssertionError(label)
    def exact(self,a,b,label):
        if isinstance(b,dict):
            self.require(set(a)==set(b),label+' keys')
            for k in b:self.exact(a[k],b[k],label+'/'+k)
        elif isinstance(b,(tuple,list)):
            self.require(len(a)==len(b),label+' length')
            for i,(v,w) in enumerate(zip(a,b)):self.exact(v,w,label+'/'+str(i))
        else:self.require(np.array_equal(a,b),label)
    def close(self,a,b,label):
        if isinstance(b,dict):
            self.require(set(a)==set(b),label+' keys')
            for k in b:self.close(a[k],b[k],label+'/'+k)
        elif isinstance(b,(list,tuple)):
            self.require(len(a)==len(b),label+' length')
            for i,(v,w) in enumerate(zip(a,b)):self.close(v,w,label+'/'+str(i))
        elif isinstance(b,(bool,str)) or b is None:self.exact(a,b,label)
        else:
            x,y=np.asarray(a),np.asarray(b);self.require(x.shape==y.shape,label+' shape')
            self.require(np.isfinite(x).all() and np.isfinite(y).all(),label+' finite')
            self.comparisons+=int(x.size);err=float(np.max(np.abs(x-y))) if x.size else 0.
            self.max_error=max(self.max_error,err);self.require(np.allclose(x,y,rtol=1e-9,atol=1e-10),label+' numerical')

def groups(panel,arm):
    if panel not in (1,2,3) or arm not in ARMS:raise ValueError('invalid support')
    perm=PERMS[panel-1];index={tuple(x):i for i,x in enumerate(MAPS)}
    sets=[]
    for q in QS:
        sets.append({index[perm[i],perm[j]] for i in range(6) for j in range(6) if i!=j and (j-i)%6 not in (1,5) and j!=q[i]})
    target={index[perm[i],perm[(i+offset)%6]] for i in range(6) for offset in (-1,1)}
    train=sets[ARMS.index(arm)];shared=sets[0]&sets[1];common=set(range(30))-sets[0]-sets[1]
    return {k:np.array(sorted(v),np.int64) for k,v in dict(train12=train,target12=target,
        shared_train9=shared,held18=set(range(30))-train,other_held6=set(range(30))-train-target,
        common_unseen15=common,extra_common_unseen3=common-target,common30=set(range(30))).items()}

def softmax(x):
    x=np.asarray(x,np.float64);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def donor_codes(raw,p,arm):
    """Each target matches donor photograph pair and mask; exactly2 per role."""
    g=groups(p,arm);train=set(g['train12']);target=np.flatnonzero(np.isin(raw['map_id'],g['target12']))
    code=raw['tokens'][:,0]*7+raw['tokens'][:,1];food=[];water=[]
    for i in target:
        allowed=[j for j in range(len(code)) if int(raw['map_id'][j]) in train and raw['shown'][j]==raw['shown'][i]
                 and np.array_equal(raw['photo_ids'][i],raw['photo_ids'][j])]
        fs=[j for j in allowed if raw['positions'][j,0]==raw['positions'][i,0]]
        ws=[j for j in allowed if raw['positions'][j,1]==raw['positions'][i,1]]
        if len(fs)!=2 or len(ws)!=2:raise ValueError('nonunique or incomplete donor support')
        food.append(code[fs]);water.append(code[ws])
    return target,np.asarray(food),np.asarray(water)

def recombination(raw,p,arm,permutation=None):
    ix,fc,wc=donor_codes(raw,p,arm);decoder=np.argmax(raw['receiver_logits'],axis=-1)
    if permutation is not None:
        perm=np.asarray(permutation);inverse=np.argsort(perm)
        if not np.array_equal(np.sort(perm),np.arange(49)):raise ValueError('not a complete49-code bijection')
        fc,wc=perm[fc],perm[wc];decoder=decoder[inverse]
    codes={'recombine_FW_J':(fc[:,:,None]//7)*7+(wc[:,None,:]%7),
           'recombine_WF_J':(wc[:,:,None]//7)*7+(fc[:,None,:]%7)}
    correct={k:np.all(decoder[v]==raw['positions'][ix,None,None,:],axis=-1) for k,v in codes.items()}
    return {mask:{k:float(v[(np.ones(len(ix),bool) if mask=='pooled' else raw['shown'][ix]==MASKS.index(mask)-1)].mean())
                  for k,v in correct.items()} for mask in MASKS}

def target_information(raw,p):
    """Complete-code information/use; never exposes labels to trained models."""
    ix=np.flatnonzero(np.isin(raw['map_id'],groups(p,ARMS[0])['target12']))
    pos=raw['positions'][ix];photo=raw['photo_ids'][ix];shown=raw['shown'][ix]
    code=raw['tokens'][ix]@np.array([7,1]);decoder=raw['receiver_logits'].argmax(-1)
    sp=softmax(raw['sender_log_probs'][ix]);natural=np.all(decoder[code]==pos,axis=1)
    ent=lambda x:-np.sum(np.where(x>0,x*np.log(np.maximum(x,np.finfo(float).tiny)),0.),axis=-1)
    fields={}
    for known in (0,1):
        partners=[]
        for i in range(len(ix)):
            match=np.flatnonzero((pos[:,known]==pos[i,known])&(pos[:,1-known]!=pos[i,1-known])&
                (photo==photo[i]).all(1)&(shown==shown[i]))
            if len(match)!=1:raise ValueError('target partner must be unique at same photo/mask')
            partners.append(int(match[0]))
        mate=np.array(partners)
        if not np.array_equal(mate[mate],np.arange(len(ix))):raise ValueError('partners must be involutions')
        label='food' if known==0 else 'water'
        swapped=decoder[code[mate]]==pos
        fields['swap_same_'+label+'_J']=swapped.all(1).astype(float)
        fields['swap_same_'+label+'_food']=swapped[:,0].astype(float)
        fields['swap_same_'+label+'_water']=swapped[:,1].astype(float)
        fields['I_greedy_'+('water_given_food' if known==0 else 'food_given_water')+'_bits']=(code!=code[mate]).astype(float)
        fields['pair_same_'+label+'_J']=(natural&natural[mate]).astype(float)
        # Entropy of mixture minus mean conditional entropy; equivalent mean KL,
        # computed independently rather than production rowwise log-ratio formula.
        information=ent((sp+sp[mate])/2)-(ent(sp)+ent(sp[mate]))/2
        fields['I_'+('water_given_food' if known==0 else 'food_given_water')+'_bits']=information/np.log(2.)
    result={}
    for mask in MASKS:
        use=np.ones(len(ix),bool) if mask=='pooled' else shown==MASKS.index(mask)-1
        scores={k:float(v[use].mean()) for k,v in fields.items()};table=np.zeros((30,49));greedy=np.zeros((30,49))
        # Labels enter this independent oracle table only; no fitted model changes.
        np.add.at(table,raw['map_id'][ix][use],sp[use]/int(use.sum()))
        np.add.at(greedy,(raw['map_id'][ix][use],code[use]),1/int(use.sum()))
        scores.update(bayes_message_Q=float(table.max(axis=0).sum()),bayes_greedy_J=float(greedy.max(axis=0).sum()))
        result[mask]=scores
    return result

def score(raw,p,arm):
    pos=raw['positions'];code=raw['tokens'][:,0]*7+raw['tokens'][:,1]
    decoder=np.argmax(raw['receiver_logits'],axis=-1);action=decoder[code];correct=action==pos
    send,recv=softmax(raw['sender_log_probs']),softmax(raw['receiver_logits'])
    success=(decoder[None,:,:]==pos[:,None,:]).all(-1)
    q=np.sum(send*recv[:,0,pos[:,0]].T*recv[:,1,pos[:,1]].T,axis=1)
    global_freq=np.bincount(code,minlength=49)/len(code);global_shuf=success@global_freq
    rec=recombination(raw,p,arm);info=target_information(raw,p);result={}
    for name,maps in groups(p,arm).items():
        result[name]={};base=np.isin(raw['map_id'],maps)
        for mask in MASKS:
            use=base if mask=='pooled' else base&(raw['shown']==MASKS.index(mask)-1)
            n=int(use.sum());good=correct[use]
            if n==0:raise ValueError('empty support '+name+'/'+mask)
            freq=np.bincount(code[use],minlength=49)/n
            values=dict(n=n,J=float(good.all(-1).mean()),Q=float(q[use].mean()),food=float(good[:,0].mean()),water=float(good[:,1].mean()),
                blank_J=float(success[use,0].mean()),shuffle_J=float(global_shuf[use].mean()),within_shuffle_J=float((success[use]@freq).mean()))
            if name=='target12':values.update(rec[mask]);values.update(info[mask])
            result[name][mask]=values
    return result

def mean_tree(rows):
    if isinstance(rows[0],dict):return {k:mean_tree([x[k] for x in rows]) for k in rows[0]}
    return float(np.mean(rows))
def auc(curve):
    times=np.array([x['update'] for x in curve]);span=times[-1]-times[0]
    def integrate(vals):
        if isinstance(vals[0],dict):return {k:integrate([x[k] for x in vals]) for k in vals[0] if k!='n'}
        return float(np.sum(np.diff(times)*(np.asarray(vals[1:])+np.asarray(vals[:-1]))/2)/span)
    return integrate([x['scores'] for x in curve])

def null_raw(raw,p,arm,perms):
    """Reconstruct every199x72 score independently from all four donor pairs."""
    ix,fc,wc=donor_codes(raw,p,arm);decoder=raw['receiver_logits'].argmax(-1)
    inverse=np.argsort(perms,axis=1);f=perms[:,fc];w=perms[:,wc]
    codes={'FW':7*(f[:,:,:,None]//7)+w[:,:,None,:]%7,
           'WF':7*(w[:,:,:,None]//7)+f[:,:,None,:]%7}
    result={'target_rows':ix}
    for name,code in codes.items():
        original=np.take_along_axis(inverse,code.reshape(len(perms),-1),axis=1).reshape(code.shape)
        correct=np.all(decoder[original]==raw['positions'][None,ix,None,None,:],axis=-1)
        result[name]=correct.mean(axis=(-1,-2))
    return result

def summarize_null(raw,p,arm,table):
    observed=recombination(raw,p,arm);ix=table['target_rows']
    return {mask:{'recombine_'+k+'_J':rank_summary(observed[mask]['recombine_'+k+'_J'],v[:,
        np.ones(len(ix),bool) if mask=='pooled' else raw['shown'][ix]==MASKS.index(mask)-1].mean(1))
        for k,v in table.items() if k!='target_rows'} for mask in MASKS}

def mean_null(rows):
    # Apply the same199 relabelings throughout; average first, then calculate rank.
    return {mask:{key:rank_summary(np.mean([r[mask][key]['observed'] for r in rows]),
        np.mean([r[mask][key]['null'] for r in rows],axis=0)) for key in rows[0][mask]} for mask in MASKS}
def rank_summary(observed,null):
    values=np.asarray(null);equal=np.isclose(values,observed,atol=1e-12,rtol=0)
    return dict(observed=float(observed),null_mean=float(values.mean()),excess=float(observed-values.mean()),
        null=values.tolist(),n_less=int(((values<observed)&~equal).sum()),n_equal=int(equal.sum()),
        n_greater=int(((values>observed)&~equal).sum()),upper_tail_fraction=float((1+((values>observed)|equal).sum())/(len(values)+1)))

def check_raw(raw,w,c):
    c.exact({k:raw[k] for k in WORLD_KEYS},w,'complete common world')
    n=len(w['map_id']);c.require(raw['tokens'].shape==(n,2) and raw['tokens'].dtype.kind in 'iu','two integer tokens')
    c.require(((raw['tokens']>=0)&(raw['tokens']<7)).all(),'token domain')
    c.require(raw['sender_log_probs'].shape==(n,49) and np.isfinite(raw['sender_log_probs']).all(),'sender49 finite')
    c.require(raw['receiver_logits'].shape==(49,2,6) and np.isfinite(raw['receiver_logits']).all(),'receiver finite')

def analyze(out,c):
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');c.exact(done['status'],'complete','terminal')
    seeds,panels=inv['seeds'],inv['partitions'];c.exact(seeds,[34101,34102,34103,34104] if inv['formal'] else [99528],'fixed sources')
    c.exact(panels,[1,2,3] if inv['formal'] else [1],'fixed panels')
    c.exact(inv['updates'],2400 if inv['formal'] else 40,'fixed updates')
    perms=np.load(ROOT/'code_relabelings.npy');rng=np.random.default_rng(np.random.SeedSequence([30030,49,199]))
    c.exact(perms,np.stack([rng.permutation(49) for _ in range(199)]),'fixed complete-code relabelings')
    c.exact(sha(ROOT/'code_relabelings.npy'),inv['source_hashes'][str(ROOT/'code_relabelings.npy')],'relabeling source binding')
    w=npz(out/'test_worlds.npz');c.exact(len(w['map_id']),180,'test worlds180')
    c.exact(w['positions'],MAPS[w['map_id']],'map positions');rows=[];raw_hashes={}
    for seed,p,arm in itertools.product(seeds,panels,ARMS):
        folder=out/'social'/f's{seed}_p{p}_{arm}';cfg=read(folder/'config.json');res=read(folder/'result.json');saved=read(folder/'curve.json')
        c.exact(res['status'],'complete','run complete');c.exact(cfg['checkpoints'],[x['update'] for x in saved],'checkpoint list')
        directional=[[],[]];nulls={}
        for item in saved:
            t=item['update']
            for d in (0,1):
                path=folder/f'protocol_{t:04d}_d{d}.npz';digest=sha(path);rel=str(path.relative_to(out))
                c.exact(done['files'][rel],digest,'completion bound raw');raw=npz(path);check_raw(raw,w,c)
                metrics=score(raw,p,arm);c.close(metrics,item['scores'][d],'all scalar scores '+rel)
                directional[d].append(dict(update=t,scores=metrics));raw_hashes[rel]=digest;c.coverage['protocol_tables']+=1;c.coverage['protocol_worlds']+=len(w['map_id'])
                if t==inv['updates']:
                    c.close(metrics,res['scores'][d],'final score')
                    nr=null_raw(raw,p,arm,perms);nf=folder/f'recombination_null_d{d}.npz'
                    c.close(nr,npz(nf),'all199 relabelings and target rows')
                    c.exact(sha(nf),done['files'][str(nf.relative_to(out))],'null raw completion binding')
                    raw_hashes[str(nf.relative_to(out))]=sha(nf)
                    nulls[d]=summarize_null(raw,p,arm,nr);c.coverage['null_target_scores']+=2*199*len(nr['target_rows'])
        for d,curve in enumerate(directional):rows.append(dict(seed=seed,partition=p,direction=d,condition=arm,curve=curve,scores=curve[-1]['scores'],auc=auc(curve),recombination_null=nulls[d]))
    c.exact(len(list((out/'social').glob('*/result.json'))),len(seeds)*len(panels)*2,'run inventory')
    seed_rows=[];aggregate={}
    for seed,arm in itertools.product(seeds,ARMS):
        selected=[x for x in rows if x['seed']==seed and x['condition']==arm];c.exact(len(selected),len(panels)*2,'directions per source')
        curve=[dict(update=t['update'],scores=mean_tree([x['curve'][i]['scores'] for x in selected])) for i,t in enumerate(selected[0]['curve'])]
        seed_rows.append(dict(seed=seed,condition=arm,curve=curve,scores=curve[-1]['scores'],auc=auc(curve),recombination_null=mean_null([x['recombination_null'] for x in selected])))
    for arm in ARMS:
        selected=[x for x in seed_rows if x['condition']==arm]
        curve=[dict(update=t['update'],scores=mean_tree([x['curve'][i]['scores'] for x in selected])) for i,t in enumerate(selected[0]['curve'])]
        aggregate[arm]=dict(independent_sources=len(seeds),curve=curve,scores=curve[-1]['scores'],auc=auc(curve),recombination_null=mean_null([x['recombination_null'] for x in selected]))
    primary=[]
    for seed in seeds:
        a,b=[next(x for x in seed_rows if x['seed']==seed and x['condition']==arm) for arm in ARMS]
        extract=lambda x:x['scores']['target12']['pooled']['J']
        primary.append(dict(seed=seed,connected=extract(a),products=extract(b),difference=extract(a)-extract(b),
            auc_difference=a['auc']['target12']['pooled']['J']-b['auc']['target12']['pooled']['J']))
    result=dict(status='complete',formal=inv['formal'],seeds=seeds,partitions=panels,rows=rows,seed_rows=seed_rows,aggregate=aggregate,
        primary=primary,primary_mean=float(np.mean([x['difference'] for x in primary])),primary_auc_mean=float(np.mean([x['auc_difference'] for x in primary])),
        analysis_source_sha256=sha(__file__),training_complete_sha256=sha(out/'training_complete.json'),
        definitions=dict(J='Saved sequential-greedy message, greedy receiver for both goals.',Q='Normalized float64 sum over49 sender codes and independent receiver goal actions.',
            shuffle_J='One complete greedy code drawn from all30 maps, both masks and all photos.',within_shuffle_J='One complete greedy code drawn from the evaluated scope AND mask.',
            recombination='All2x2 same-photo/same-mask train donors; fixed FW/WF assignments, no maximization.',
            conditional='Target partners share photo/mask and one resource position. Swap uses unique different other-resource partner; pair requires both natural joint successes. I_* is complete-code native probability information in bits; I_greedy_* uses deterministic greedy codes. Neither measures token structure.',
            bayes='Target-label fitted arbitrary49-code joint decoder, separately per reported mask or pooled; no photo/mask input within the fitted table. Upper bound, not actual receiver/learning performance.',
            null='199 fixed complete-code bijections, inverse receiver relabeling; ranks calculated after averaging per source. Equality atol1e-12; upper-tail fraction is a coordinate-assay reference, not a source-level significance test.',
            unit='Source initialization; panels/directions/masks nested, not independent observations.',AUC='Saved-checkpoint trapezoid, normalized by update interval; auxiliary.'),
        limits=['No production metrics imported.','No model/optimizer/DINO replay by this analysis.','Degree2 target removes deterministic fixed-factor inference; high J still does not establish token-factor composition.', 'Selecting which resource to report using the entire layout can itself encode both factors; the50% bound assumes a fixed/exogenous known factor.'])
    return result,raw_hashes

def self_test():
    c=Checks();ids=np.tile(np.repeat(np.arange(30),3),2)
    photo=np.tile(np.array([[0,3],[1,3],[2,3]]),(60,1));shown=np.repeat(np.arange(2),90)
    raw=dict(map_id=ids,positions=MAPS[ids],photo_ids=photo,shown=shown,tokens=MAPS[ids].copy(),
             sender_log_probs=np.zeros((180,49)),receiver_logits=np.zeros((49,2,6)))
    uniform=score(raw,1,ARMS[0]);c.close(uniform['common30']['pooled']['Q'],1/36,'uniform joint')
    c.exact(uniform['common30']['pooled']['J'],0.,'uniform argmax same-site cannot solve any map')
    raw['sender_log_probs'].fill(-1000);raw['receiver_logits'].fill(-1000)
    code=raw['tokens']@np.array([7,1]);raw['sender_log_probs'][np.arange(180),code]=0
    for m in range(49):
        raw['receiver_logits'][m,0,(m//7)%6]=0;raw['receiver_logits'][m,1,(m%7)%6]=0
    rng=np.random.default_rng(901);perms=np.stack([np.arange(49),*[rng.permutation(49) for _ in range(3)]])
    for panel,arm in itertools.product((1,2,3),ARMS):
        s=score(raw,panel,arm);c.exact(s['common30']['pooled']['J'],1.,'compositional perfect natural')
        c.exact(s['target12']['pooled']['recombine_FW_J'],1.,'compositional FW')
        c.exact(s['target12']['pooled']['recombine_WF_J'],0.,'switched resource donors')
        for mask,values in target_information(raw,panel).items():
            for name,v in values.items():
                expected=0. if name.startswith('swap_') and name not in ('swap_same_food_food','swap_same_water_water') else 1.
                c.close(v,expected,'perfect complete-code information/pair/oracle '+name)
        nr=null_raw(raw,panel,arm,perms)
        for i,perm in enumerate(perms):
            slow=recombination(raw,panel,arm,perm)
            for k in ('FW','WF'):c.close(nr[k][i].mean(),slow['pooled']['recombine_'+k+'_J'],'vectorized vs individual relabeling')
            decoder=raw['receiver_logits'].argmax(-1)
            c.exact(decoder[np.argsort(perm)][perm[code]],raw['positions'],'whole-code invariance')
    factor={k:v.copy() for k,v in raw.items()};factor['tokens'][:,1]=0
    factor['sender_log_probs'].fill(-1000);factor['sender_log_probs'][np.arange(180),factor['tokens']@np.array([7,1])]=0
    for panel in (1,2,3):
        for mask,values in target_information(factor,panel).items():
            c.close(values['I_water_given_food_bits'],0.,'F-only no conditional water information')
            c.close(values['I_food_given_water_bits'],1.,'F-only conditional F information')
            c.close(values['I_greedy_water_given_food_bits'],0.,'F-only greedy conditional water information')
            c.close(values['I_greedy_food_given_water_bits'],1.,'F-only greedy conditional F information')
            c.close(values['bayes_message_Q'],.5,'F-only native whole-code oracle bound')
            c.close(values['bayes_greedy_J'],.5,'F-only greedy whole-code oracle bound')
    return dict(passed=True,checks=c.count,scope='Uniform/perfect resource code, fixed-factor50% oracle, target conditional MI/swap/pair, fixed donor assignments, relabeling inverse and vectorized independent references; no model calls.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if args.out is None:ap.error('--out is required unless --self-test')
    out=args.out.resolve();c=Checks();start=time.monotonic()
    try:
        result,hashes=analyze(out,c);write(out/'analysis.json',result)
        qa=dict(passed=True,checks=c.count,scalar_comparisons=c.comparisons,maximum_metric_absolute_difference=c.max_error,
            coverage=dict(c.coverage),analysis_source_sha256=sha(__file__),analysis_sha256=sha(out/'analysis.json'),checked_raw_sha256=hashes,
            production_modules_imported=False,model_calls=0,seconds=time.monotonic()-start)
        write(out/'raw_validation.json',qa);shutil.copy2(__file__,out/'analyze_results_source.py')
        print(json.dumps({k:qa[k] for k in ('passed','checks','scalar_comparisons','maximum_metric_absolute_difference','coverage','seconds')}))
    except Exception as e:
        stamp=time.time_ns();write(out/f'analysis_failure_{stamp}.json',dict(passed=False,error=repr(e),traceback=traceback.format_exc(),source_sha256=sha(__file__)))
        shutil.copy2(__file__,out/f'analyze_results_failure_{stamp}.py')
        raise
if __name__=='__main__':main()
