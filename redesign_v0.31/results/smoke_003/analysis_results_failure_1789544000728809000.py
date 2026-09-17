"""Independent NumPy-only v0.31 population statistics; no production imports."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,shutil,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
MAPS=np.array(list(itertools.permutations(range(6),2)),np.int64);CONDITIONS=('fixed_partners','rotating_partners');MASKS=('pooled','food_only','water_only');PRIVATE_TYPES=(0,1,0,1);AGENTS=4
PERMS=((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4))
TARGET=((0,1),(0,3),(1,2),(1,4),(2,0),(2,5),(3,2),(3,4),(4,0),(4,5),(5,1),(5,3))
TRAIN=((0,2),(0,5),(1,0),(1,3),(2,1),(2,4),(3,1),(3,5),(4,2),(4,3),(5,0),(5,4))
WORLD_KEYS=('map_id','photo_ids','positions','shown');PAIR_KEYS=tuple((i,j) for i in range(AGENTS) for j in range(AGENTS) if i!=j and PRIVATE_TYPES[i]!=PRIVATE_TYPES[j])
FIXED_TRAINED={(0,1),(1,0),(2,3),(3,2)}

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def arrays_sha(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

class Checks:
    def __init__(self):self.count=0;self.comparisons=0;self.max_error=0.;self.coverage=Counter()
    def require(self,ok,label):self.count+=1;assert bool(ok),label
    def exact(self,a,b,label):
        if isinstance(b,dict):self.require(set(a)==set(b),label+' keys');[self.exact(a[k],b[k],label+'/'+k) for k in b]
        elif isinstance(b,(tuple,list)):self.require(len(a)==len(b),label+' length');[self.exact(v,w,label+'/'+str(i)) for i,(v,w) in enumerate(zip(a,b))]
        else:self.require(np.array_equal(a,b),label)
    def close(self,a,b,label):
        if isinstance(b,dict):self.require(set(a)==set(b),label+' keys');[self.close(a[k],b[k],label+'/'+k) for k in b]
        elif isinstance(b,(tuple,list)):self.require(len(a)==len(b),label+' length');[self.close(v,w,label+'/'+str(i)) for i,(v,w) in enumerate(zip(a,b))]
        elif isinstance(b,(bool,str)) or b is None:self.exact(a,b,label)
        else:
            x,y=np.asarray(a),np.asarray(b);self.require(x.shape==y.shape,label+' shape');self.require(np.isfinite(x).all() and np.isfinite(y).all(),label+' finite');self.comparisons+=int(x.size);err=float(np.max(np.abs(x-y))) if x.size else 0.;self.max_error=max(self.max_error,err);self.require(np.allclose(x,y,rtol=1e-9,atol=1e-10),label+' numerical')

def groups(panel,condition):
    if panel not in (1,2,3) or condition not in CONDITIONS:raise ValueError('invalid panel or condition')
    q=PERMS[panel-1];index={tuple(x):i for i,x in enumerate(MAPS)};target={index[q[i],q[j]] for i,j in TARGET};train={index[q[i],q[j]] for i,j in TRAIN}
    return {k:np.array(sorted(v),np.int64) for k,v in dict(train12=train,target12=target,held18=set(range(30))-train,common30=set(range(30))).items()}
def softmax(x):
    x=np.asarray(x,np.float64);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def donor_codes(raw,p,condition):
    g=groups(p,condition);train=set(g['train12']);target=np.flatnonzero(np.isin(raw['map_id'],g['target12']));code=raw['tokens'][:,0]*7+raw['tokens'][:,1];food=[];water=[]
    for i in target:
        allowed=[j for j in range(len(code)) if int(raw['map_id'][j]) in train and raw['shown'][j]==raw['shown'][i] and np.array_equal(raw['photo_ids'][i],raw['photo_ids'][j])];fs=[j for j in allowed if raw['positions'][j,0]==raw['positions'][i,0]];ws=[j for j in allowed if raw['positions'][j,1]==raw['positions'][i,1]]
        if len(fs)!=2 or len(ws)!=2:raise ValueError('nonunique donor support')
        food.append(code[fs]);water.append(code[ws])
    return target,np.asarray(food),np.asarray(water)
def recombination(raw,p,condition,permutation=None):
    ix,fc,wc=donor_codes(raw,p,condition);decoder=np.argmax(raw['receiver_logits'],axis=-1)
    if permutation is not None:
        perm=np.asarray(permutation);inverse=np.argsort(perm)
        if not np.array_equal(np.sort(perm),np.arange(49)):raise ValueError('not a 49-code bijection')
        fc,wc=perm[fc],perm[wc];decoder=decoder[inverse]
    codes={'recombine_FW_J':(fc[:,:,None]//7)*7+(wc[:,None,:]%7),'recombine_WF_J':(wc[:,:,None]//7)*7+(fc[:,None,:]%7)};correct={k:np.all(decoder[v]==raw['positions'][ix,None,None,:],axis=-1) for k,v in codes.items()}
    return {mask:{k:float(v[(np.ones(len(ix),bool) if mask=='pooled' else raw['shown'][ix]==MASKS.index(mask)-1)].mean()) for k,v in correct.items()} for mask in MASKS}
def partner_pairs(raw,p,condition):
    target=np.flatnonzero(np.isin(raw['map_id'],groups(p,condition)['target12']));code=raw['tokens'][:,0]*7+raw['tokens'][:,1];action=np.argmax(raw['receiver_logits'],axis=-1)[code];correct=np.all(action==raw['positions'],axis=-1);out={}
    for kind,name in ((0,'food_partner_pair_J'),(1,'water_partner_pair_J')):
        vals=[]
        for loc in range(6):
            rows0=target[raw['positions'][target,kind]==loc]
            for photo in np.unique(raw['photo_ids'][rows0],axis=0):
                for shown in (0,1):
                    rows=rows0[(raw['shown'][rows0]==shown)&(raw['photo_ids'][rows0]==photo).all(-1)]
                    if len(rows)!=2:raise ValueError('partner group is not two')
                    vals.append(correct[rows].all())
        out[name]=float(np.mean(vals))
    return out
def score(raw,p,condition):
    pos=raw['positions'];code=raw['tokens'][:,0]*7+raw['tokens'][:,1];decoder=np.argmax(raw['receiver_logits'],axis=-1);action=decoder[code];correct=action==pos;send,recv=softmax(raw['sender_log_probs']),softmax(raw['receiver_logits']);success=(decoder[None,:,:]==pos[:,None,:]).all(-1);q=np.sum(send*recv[:,0,pos[:,0]].T*recv[:,1,pos[:,1]].T,axis=1);freq=np.bincount(code,minlength=49)/len(code);global_shuf=success@freq;rec=recombination(raw,p,condition);pairs=partner_pairs(raw,p,condition);result={}
    for name,maps in groups(p,condition).items():
        base=np.isin(raw['map_id'],maps);result[name]={}
        for mask in MASKS:
            use=base if mask=='pooled' else base&(raw['shown']==MASKS.index(mask)-1);good=correct[use];local=np.bincount(code[use],minlength=49)/int(use.sum());values=dict(n=int(use.sum()),J=float(good.all(-1).mean()),Q=float(q[use].mean()),food=float(good[:,0].mean()),water=float(good[:,1].mean()),blank_J=float(success[use,0].mean()),shuffle_J=float(global_shuf[use].mean()),within_shuffle_J=float((success[use]@local).mean()))
            if name=='target12':values.update(rec[mask])
            if name=='target12' and mask=='pooled':values.update(pairs)
            result[name][mask]=values
    return result
def mean_tree(rows):
    if isinstance(rows[0],dict):return {k:mean_tree([x[k] for x in rows]) for k in rows[0]}
    return float(np.mean(rows))
def auc(curve):
    times=np.asarray([x['update'] for x in curve]);span=times[-1]-times[0]
    def integrate(vals):
        if isinstance(vals[0],dict):return {k:integrate([x[k] for x in vals]) for k in vals[0] if k!='n'}
        return float(np.sum(np.diff(times)*(np.asarray(vals[1:])+np.asarray(vals[:-1]))/2)/span)
    return integrate([x['scores'] for x in curve])
def null_raw(raw,p,condition,perms):
    ix,fc,wc=donor_codes(raw,p,condition);decoder=raw['receiver_logits'].argmax(-1);inverse=np.argsort(perms,axis=1);f=perms[:,fc];w=perms[:,wc];codes={'FW':7*(f[:,:,:,None]//7)+w[:,:,None,:]%7,'WF':7*(w[:,:,:,None]//7)+f[:,:,None,:]%7};result={'target_rows':ix}
    for name,code in codes.items():
        original=np.take_along_axis(inverse,code.reshape(len(perms),-1),axis=1).reshape(code.shape);correct=np.all(decoder[original]==raw['positions'][None,ix,None,None,:],axis=-1);result[name]=correct.mean(axis=(-1,-2))
    return result
def rank_summary(observed,null):
    values=np.asarray(null);equal=np.isclose(values,observed,atol=1e-12,rtol=0);return dict(observed=float(observed),null_mean=float(values.mean()),excess=float(observed-values.mean()),null=values.tolist(),n_less=int(((values<observed)&~equal).sum()),n_equal=int(equal.sum()),n_greater=int(((values>observed)&~equal).sum()),upper_tail_fraction=float((1+((values>observed)|equal).sum())/(len(values)+1)))
def summarize_null(raw,p,condition,table):
    observed=recombination(raw,p,condition);ix=table['target_rows'];return {mask:{'recombine_'+k+'_J':rank_summary(observed[mask]['recombine_'+k+'_J'],v[:,np.ones(len(ix),bool) if mask=='pooled' else raw['shown'][ix]==MASKS.index(mask)-1].mean(1)) for k,v in table.items() if k!='target_rows'} for mask in MASKS}
def mean_null(rows):return {mask:{key:rank_summary(np.mean([r[mask][key]['observed'] for r in rows]),np.mean([r[mask][key]['null'] for r in rows],axis=0)) for key in rows[0][mask]} for mask in MASKS}
def check_raw(raw,w,c):
    c.exact({k:raw[k] for k in WORLD_KEYS},w,'complete world');n=len(w['map_id']);c.require(raw['tokens'].shape==(n,2) and raw['tokens'].dtype.kind in 'iu','two integer tokens');c.require(((raw['tokens']>=0)&(raw['tokens']<7)).all(),'token domain');c.require(raw['sender_log_probs'].shape==(n,49) and np.isfinite(raw['sender_log_probs']).all(),'sender table');c.require(raw['receiver_logits'].shape==(49,2,6) and np.isfinite(raw['receiver_logits']).all(),'receiver table')

def agreement(folder,t,source_pairs):
    # Sender type copies receive the same frozen h.  Compare complete greedy
    # messages for agents0/2 and agents1/3; receiver choice is irrelevant.
    values=[]
    for i,j in ((0,2),(1,3)):
        a=npz(folder/f'protocol_{t:04d}_i{i}_j{source_pairs[i]}.npz')['tokens'];b=npz(folder/f'protocol_{t:04d}_i{j}_j{source_pairs[j]}.npz')['tokens'];values.append((a==b).all(-1).mean());values.append((a[:,0]==b[:,0]).mean());values.append((a[:,1]==b[:,1]).mean())
    return dict(full_message_agreement=float(np.mean(values[0::3])),token0_agreement=float(np.mean(values[1::3])),token1_agreement=float(np.mean(values[2::3])))

def analyze(out,c):
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');c.exact(done['status'],'complete','terminal');seeds,panels=inv['seeds'],inv['partitions'];c.exact(seeds,[34101,34102,34103,34104] if inv['formal'] else [99528],'sources');c.exact(panels,[1,2,3] if inv['formal'] else [1],'panels');c.exact(inv['updates'],2400 if inv['formal'] else 40,'updates');perms=np.load(ROOT/'code_relabelings.npy');rng=np.random.default_rng(np.random.SeedSequence([31031,49,199]));c.exact(perms,np.stack([rng.permutation(49) for _ in range(199)]),'relabelings');c.exact(sha(ROOT/'code_relabelings.npy'),inv['source_hashes'][str(ROOT/'code_relabelings.npy')],'relabeling binding');w=npz(out/'test_worlds.npz');c.exact(len(w['map_id']),180,'test worlds');c.exact(w['positions'],MAPS[w['map_id']],'positions')
    rows=[];raw_hashes={};times=[0,100,600,1200,2100,2400] if inv['formal'] else [0,40]
    for seed,p,condition in itertools.product(seeds,panels,CONDITIONS):
        folder=out/'social'/f's{seed}_p{p}_{condition}';cfg=read(folder/'config.json');res=read(folder/'result.json');saved=read(folder/'curve.json');c.exact(res['status'],'complete','run');c.exact(cfg['checkpoints'],times,'checkpoints');c.exact(set(saved[0]['scores']),{f'i{i}_j{j}' for i,j in PAIR_KEYS},'pair inventory');nulls={};pair_curves={k:[] for k in saved[0]['scores']}
        for item in saved:
            t=item['update']
            for i,j in PAIR_KEYS:
                key=f'i{i}_j{j}';path=folder/f'protocol_{t:04d}_{key}.npz';rel=str(path.relative_to(out));digest=sha(path);c.exact(done['files'][rel],digest,'raw bound');raw=npz(path);check_raw(raw,w,c);m=score(raw,p,condition);c.close(m,item['scores'][key],'production score recheck');pair_curves[key].append(dict(update=t,scores=m));raw_hashes[rel]=digest;c.coverage['protocol_tables']+=1;c.coverage['protocol_worlds']+=len(w['map_id'])
                if t==inv['updates']:
                    c.close(m,res['scores'][key],'terminal score');nr=null_raw(raw,p,condition,perms);nf=folder/f'recombination_null_{key}.npz';c.close(nr,npz(nf),'199 relabel rows');c.exact(sha(nf),done['files'][str(nf.relative_to(out))],'null bound');raw_hashes[str(nf.relative_to(out))]=sha(nf);nulls[key]=summarize_null(raw,p,condition,nr);c.coverage['null_target_scores']+=2*199*len(nr['target_rows'])
        source_pairs={0:1,2:1,1:0,3:0};agreements=[dict(update=item['update'],agreement=agreement(folder,item['update'],source_pairs)) for item in saved]
        for key,curve in pair_curves.items():i,j=map(int,key[1:].split('_j'));rows.append(dict(seed=seed,partition=p,condition=condition,sender=i,receiver=j,pair=key,curve=curve,scores=curve[-1]['scores'],auc=auc(curve),recombination_null=nulls[key]))
        # Attach panel-level agreement to each row for aggregation below.
        for r in rows[-len(PAIR_KEYS):]:r['agreement_curve']=agreements
    c.exact(len(list((out/'social').glob('*/result.json'))),len(seeds)*len(panels)*len(CONDITIONS),'run inventory')
    seed_rows=[];aggregate={}
    for seed,condition in itertools.product(seeds,CONDITIONS):
        selected=[r for r in rows if r['seed']==seed and r['condition']==condition];c.exact(len(selected),len(panels)*len(PAIR_KEYS),'source pair rows');curve=[dict(update=t['update'],scores=mean_tree([r['curve'][k]['scores'] for r in selected])) for k,t in enumerate(selected[0]['curve'])];agree=[dict(update=t['update'],agreement=mean_tree([r['agreement_curve'][k]['agreement'] for r in selected])) for k,t in enumerate(selected[0]['agreement_curve'])];pair_end={r['pair']:r['scores']['target12']['pooled']['J'] for r in selected};pair_auc={r['pair']:r['auc']['target12']['pooled']['J'] for r in selected};seed_rows.append(dict(seed=seed,condition=condition,curve=curve,scores=curve[-1]['scores'],auc=auc(curve),agreement=agree,pair_endpoint_J=pair_end,pair_auc_J=pair_auc,recombination_null=mean_null([r['recombination_null'] for r in selected])))
    for condition in CONDITIONS:
        selected=[r for r in seed_rows if r['condition']==condition];curve=[dict(update=t['update'],scores=mean_tree([r['curve'][k]['scores'] for r in selected])) for k,t in enumerate(selected[0]['curve'])];agree=[dict(update=t['update'],agreement=mean_tree([r['agreement'][k]['agreement'] for r in selected])) for k,t in enumerate(selected[0]['agreement'])];aggregate[condition]=dict(independent_sources=len(seeds),curve=curve,scores=curve[-1]['scores'],auc=auc(curve),agreement=agree,pair_endpoint_J={k:float(np.mean([r['pair_endpoint_J'][k] for r in selected])) for k in sorted(selected[0]['pair_endpoint_J'])},pair_auc_J={k:float(np.mean([r['pair_auc_J'][k] for r in selected])) for k in sorted(selected[0]['pair_auc_J'])},recombination_null=mean_null([r['recombination_null'] for r in selected]))
    primary=[]
    for seed in seeds:
        f=next(r for r in seed_rows if r['seed']==seed and r['condition']=='fixed_partners');rot=next(r for r in seed_rows if r['seed']==seed and r['condition']=='rotating_partners');primary.append(dict(seed=seed,fixed=f['scores']['target12']['pooled']['J'],rotating=rot['scores']['target12']['pooled']['J'],difference=rot['scores']['target12']['pooled']['J']-f['scores']['target12']['pooled']['J'],auc_difference=rot['auc']['target12']['pooled']['J']-f['auc']['target12']['pooled']['J'],full_message_agreement_fixed=f['agreement'][-1]['agreement']['full_message_agreement'],full_message_agreement_rotating=rot['agreement'][-1]['agreement']['full_message_agreement']))
    return dict(status='complete',formal=inv['formal'],seeds=seeds,partitions=panels,rows=rows,seed_rows=seed_rows,aggregate=aggregate,primary=primary,primary_mean=float(np.mean([r['difference'] for r in primary])),primary_auc_mean=float(np.mean([r['auc_difference'] for r in primary])),analysis_source_sha256=sha(__file__),training_complete_sha256=sha(out/'training_complete.json'),definitions=dict(J='Saved sequential-greedy message and greedy receiver, averaged over eight ordered cross-type pairs.',Q='Normalized float64 sum over49 sender codes and two receiver goals.',partner_pair='For each fixed resource/location/photo/mask, both target partner edges must be correct.',agreement='Exact greedy-message equality between independent agents of the same private type, evaluated on the same frozen h.',recombination='All2x2 same-photo/same-mask training donors; fixed FW/WF assignments.',null='199 fixed complete-code bijections; descriptive reference only.',unit='Initialization source; panels, agent pairs and directions are nested.',AUC='Fixed-checkpoint trapezoid, auxiliary.'),limits=['No production metrics imported.','No model/optimizer/DINO replay by this analysis.','All four population members reuse two private encoder types; partner schedule is the experimental manipulation.']),raw_hashes

def self_test():
    c=Checks();ids=np.tile(np.repeat(np.arange(30),3),2);photo=np.tile(np.array([[0,3],[1,3],[2,3]]),(60,1));shown=np.repeat(np.arange(2),90);raw=dict(map_id=ids,positions=MAPS[ids],photo_ids=photo,shown=shown,tokens=MAPS[ids].copy(),sender_log_probs=np.zeros((180,49)),receiver_logits=np.zeros((49,2,6)));u=score(raw,1,CONDITIONS[0]);c.close(u['common30']['pooled']['Q'],1/36,'uniform joint');c.exact(u['common30']['pooled']['J'],0.,'uniform argmax');raw['sender_log_probs'].fill(-1000);raw['receiver_logits'].fill(-1000);code=raw['tokens']@np.array([7,1]);raw['sender_log_probs'][np.arange(180),code]=0
    for m in range(49):raw['receiver_logits'][m,0,(m//7)%6]=0;raw['receiver_logits'][m,1,(m%7)%6]=0
    rng=np.random.default_rng(901);perms=np.stack([np.arange(49),*[rng.permutation(49) for _ in range(3)]])
    for p,cond in itertools.product((1,2,3),CONDITIONS):
        s=score(raw,p,cond);c.exact(s['common30']['pooled']['J'],1.,'perfect natural');c.exact(s['target12']['pooled']['recombine_FW_J'],1.,'perfect FW');nr=null_raw(raw,p,cond,perms)
        for i,perm in enumerate(perms):
            slow=recombination(raw,p,cond,perm)
            for k in ('FW','WF'):c.close(nr[k][i].mean(),slow['pooled']['recombine_'+k+'_J'],'vectorized relabel');decoder=raw['receiver_logits'].argmax(-1);c.exact(decoder[np.argsort(perm)][perm[code]],raw['positions'],'whole-code invariance')
    return dict(passed=True,checks=c.count,scope='Uniform/perfect protocol, four-donor recombination, relabel inverse; no model calls.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if args.out is None:ap.error('--out required unless --self-test')
    out=args.out.resolve();c=Checks();start=time.monotonic()
    try:
        result,hashes=analyze(out,c);write(out/'analysis.json',result);qa=dict(passed=True,checks=c.count,scalar_comparisons=c.comparisons,maximum_metric_absolute_difference=c.max_error,coverage=dict(c.coverage),analysis_source_sha256=sha(__file__),analysis_sha256=sha(out/'analysis.json'),checked_raw_sha256=hashes,production_modules_imported=False,model_calls=0,seconds=time.monotonic()-start);write(out/'raw_validation.json',qa);shutil.copy2(__file__,out/'analyze_results_source.py');print(json.dumps({k:qa[k] for k in ('passed','checks','scalar_comparisons','maximum_metric_absolute_difference','coverage','seconds')}))
    except Exception as e:
        stamp=time.time_ns();write(out/f'analysis_failure_{stamp}.json',dict(passed=False,error=repr(e),traceback=traceback.format_exc(),source_sha256=sha(__file__)));shutil.copy2(__file__,out/f'analysis_results_failure_{stamp}.py');raise
if __name__=='__main__':main()
