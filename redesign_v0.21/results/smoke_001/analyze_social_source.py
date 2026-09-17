"""Independent finite-support social analysis from saved protocol arrays.

No social_metrics, model forwards, fitted probes, or training functions are
called. Conditional information uses joint contingency counts; pair metrics
use count algebra rather than enumerating all pairwise row comparisons.
"""
from pathlib import Path
from itertools import permutations,product
import argparse,hashlib,json,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays(p):
    with np.load(p,allow_pickle=False) as data:return {k:data[k] for k in data.files}
def groups(p):
    def matching(k):
        pairs={z for a,b in MATCHINGS[k] for z in ((a,b),(b,a))}
        return np.asarray([i for i,t in enumerate(MAPS) if tuple(t) in pairs])
    added,sealed=matching(p-1),matching(p%3)
    return dict(old=np.setdiff1d(np.arange(30),np.r_[added,sealed]),added=added,sealed=sealed,common30=np.arange(30))
def bucket(keys):
    rows={}
    for i,key in enumerate(keys):rows.setdefault(tuple(key),[]).append(i)
    return [np.asarray(v,np.int64) for v in rows.values()]
def softmax(x):
    x=np.asarray(x,np.float64);z=np.exp(x-x.max(-1,keepdims=True));return z/z.sum(-1,keepdims=True)
def entropy(count):
    count=np.asarray(count,np.float64);q=count[count>0]/count.sum();return float(-q@np.log2(q))
def code_counts(code):return np.bincount(code,minlength=49)

def statistics(sender_lp,receiver_logits,tokens,w,p):
    ps=softmax(sender_lp);pr=softmax(receiver_logits);decode=np.asarray(receiver_logits).argmax(-1)
    tokens=np.asarray(tokens);assert tokens.shape==(len(ps),2) and ((tokens>=0)&(tokens<7)).all()
    code=7*tokens[:,0]+tokens[:,1];target=w['positions'];n=len(code);m=w['mover'];ii=np.arange(n)
    assert ps.shape==(n,49) and pr.shape==(49,2,6) and np.array_equal(target,MAPS[w['target_map']])
    action=decode[code];correct=action==target;correct_codes=decode[None,:,:]==target[:,None,:];joint=correct_codes.all(-1)
    q=sum(ps[:,c]*pr[c,0,target[:,0]]*pr[c,1,target[:,1]] for c in range(49))
    global_shuffle=joint@ (code_counts(code)/n);local=np.zeros(n);local_static=np.zeros(n)
    keys=np.column_stack((m,target[ii,m],w['photo_ids']))
    for rr in bucket(keys):
        donor=code_counts(code[rr])/len(rr);local[rr]=joint[rr]@donor
        local_static[rr]=correct_codes[rr,:,1-int(m[rr[0]])]@donor
    message_entropy=-(ps*np.log2(np.where(ps>0,ps,1.))).sum(-1);result={}
    for name,support in groups(p).items():
        rows=np.flatnonzero(np.isin(w['target_map'],support));ok=correct[rows];r=dict(n=len(rows),J=float(ok.all(-1).mean()),single=float(ok.mean()),Q=float(q[rows].mean()),
            moved=float(correct[ii,m][rows].mean()),stationary=float(correct[ii,1-m][rows].mean()),receiver_coverage=float(joint[rows].any(-1).mean()),
            native_sender_entropy_bits=float(message_entropy[rows].mean()),unique_greedy_codes=int(len(np.unique(code[rows]))),
            global_shuffle_J=float(global_shuffle[rows].mean()),conditional_shuffle_J=float(local[rows].mean()),conditional_shuffle_stationary=float(local_static[rows].mean()),constant00_J=float(joint[rows,0].mean()))
        if name in ('old','common30'):
            pairs=pairgood=pairboth=pairdifferent=0;information=0.
            for localrows in bucket(keys[rows]):
                rr=rows[localrows];station=target[rr,1-int(m[rr[0]])];cont=np.zeros((6,49),np.int64)
                np.add.at(cont,(station,code[rr]),1);tot=cont.sum(1);good=np.bincount(station,weights=correct[rr,1-int(m[rr[0]])],minlength=6);both=np.bincount(station,weights=correct[rr].all(-1),minlength=6)
                denominator=(int(tot.sum())**2-int(tot@tot))//2;pairs+=denominator
                pairgood+=int(round((good.sum()**2-good@good)/2));pairboth+=int(round((both.sum()**2-both@both)/2))
                samecross=(int(cont.sum(0)@cont.sum(0))-int((cont*cont).sum()))//2;pairdifferent+=denominator-samecross
                a,c=np.nonzero(cont);mass=cont[a,c]/len(rr);information+=len(rr)*float(np.sum(mass*np.log2(cont[a,c]*len(rr)/(tot[a]*cont.sum(0)[c]))))
            r.update(history_pair_J=pairgood/pairs,history_pairs=pairs,history_pair_correct=pairgood,static_message_cmi_bits=information/len(rows),history_message_separation=pairdifferent/pairs,history_pair_both_joint_J=pairboth/pairs)
            assert r['history_pair_J']<=r['history_message_separation']+1e-12
        else:r.update(history_pair_J=None,history_pairs=0,history_pair_correct=0,static_message_cmi_bits=None,history_message_separation=None,history_pair_both_joint_J=None)
        agreement=[];route_entropy=[];same_correct=[]
        for localrows in bucket(np.column_stack((w['target_map'][rows],w['photo_ids'][rows]))):
            rr=rows[localrows];routes={}
            for row in rr:routes.setdefault((int(w['source_map'][row]),int(m[row])),[]).append(row)
            assert all(len(set(code[ix]))==1 for ix in routes.values()),'repeat of one identical legal route changed greedy code'
            rr=np.asarray([ix[0] for ix in routes.values()]);freq=code_counts(code[rr]);cnt=len(rr);den=cnt*(cnt-1)
            hit=np.bincount(code[rr],weights=correct[rr].all(-1),minlength=49)
            agreement.append(float(np.sum(freq*(freq-1))/den));route_entropy.append(entropy(freq));same_correct.append(float(np.sum(hit*(hit-1))/den))
        r.update(route_code_agreement=float(np.mean(agreement)),route_code_entropy_bits=float(np.mean(route_entropy)),route_samecode_bothcorrect=float(np.mean(same_correct)))
        assert r['J']<=r['receiver_coverage']+1e-12
        result[name]=r
    return result

def expected_world(p,manifest):
    old=set(groups(p)['old']);events=[]
    for a,t,m in product(sorted(old),range(30),(0,1)):
        if MAPS[a,1-m]==MAPS[t,1-m] and MAPS[t,m] not in MAPS[a]:events.extend([(a,t,m)]*(3 if t in old else 2))
    assert len(events)==360
    pools=[[i for i,x in enumerate(manifest['images']) if x['split']=='test' and x['category']==kind][:4] for kind in ('food','water')]
    photos=np.asarray(list(product(*pools)),np.int64);e=np.repeat(np.asarray(events,np.int64),16,axis=0)
    return dict(source_map=e[:,0],target_map=e[:,1],mover=e[:,2],positions=MAPS[e[:,1]],photo_ids=np.tile(photos,(360,1)))

COUNT_KEYS={'n','history_pairs','history_pair_correct'}
def mean_scores(scores):
    return {g:{k:None if scores[0][g][k] is None else float(np.mean([s[g][k] for s in scores])) for k in scores[0][g] if k not in COUNT_KEYS} for g in scores[0]}
def auc(curve):
    t=np.asarray([c['update'] for c in curve],float);assert t[-1]>0
    return {g:{k:None if curve[0]['scores'][g][k] is None else float(np.trapz([c['scores'][g][k] for c in curve],t)/t[-1]) for k in curve[0]['scores'][g]} for g in curve[0]['scores']}

def analyze(out):
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');formal=inv['formal'];checks=0;worst=0.;files=0;rawrows=0
    assert done['status']=='complete' and done['formal']==formal and inv['source_hashes']==done['source_hashes'] and inv['input_hashes']==done['input_hashes']
    assert all(sha(p)==h for p,h in {**inv['source_hashes'],**inv['input_hashes']}.items())
    assert inv['seeds']==([32101,32102,32103,32104] if formal else [99520]) and inv['partitions']==([1,2,3] if formal else [1]) and inv['modes']==['immediate','delayed']
    if formal:assert inv['updates']==2400 and read(inv['preflight']['path'])['passed'] and sha(inv['preflight']['path'])==inv['preflight']['sha256']
    manifest=read(ROOT.parent/'redesign_v0.4/data/manifest.json');runs=[];worlds={p:expected_world(p,manifest) for p in inv['partitions']}
    def compare(a,b,path):
        nonlocal checks,worst
        if isinstance(a,dict):
            assert a.keys()==b.keys(),(path,a.keys(),b.keys())
            for k in a:compare(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif a is None:assert b is None,path
        else:
            error=abs(float(a)-float(b));worst=max(worst,error);checks+=1
            assert np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,atol=2e-12,rtol=2e-12),(path,a,b,error)
    for seed,p,mode in product(inv['seeds'],inv['partitions'],inv['modes']):
        name=f's{seed}_p{p}_{mode}';folder=out/name;cfg=read(folder/'config.json');oldcurve=read(folder/'curve.json');end=read(folder/'result.json');curve=[];extra={}
        assert end['status']=='complete' and cfg['checkpoints']==[x['update'] for x in oldcurve] and cfg['checkpoints'][0]==0 and cfg['checkpoints'][-1]==inv['updates']
        expected_files=[]
        for t,label in [(x['update'],'normal') for x in oldcurve]+[(inv['updates'],x) for x in ('cross_view','erase')]:
            directional=[]
            for d in (0,1):
                f=folder/f'protocol_{t:04d}_{label}_d{d}.npz';relative=str(f.relative_to(out));expected_files.append(f.name)
                assert sha(f)==done['files'][relative],relative
                data=arrays(f);w={k:data[k] for k in worlds[p]};assert all(np.array_equal(v,worlds[p][k]) for k,v in w.items()),relative
                scores=statistics(data['sender_log_probs'],data['receiver_logits'],data['tokens'],w,p);directional.append(scores);files+=1;rawrows+=len(w['mover'])
                # The LP table is rounded float32. Check token-wise greedy
                # compatibility with its marginals, not complete-code argmax.
                ps=softmax(data['sender_log_probs']).reshape(-1,7,7);tokens=data['tokens'];ii=np.arange(len(tokens));marg=ps.sum(-1)
                assert np.max(marg.max(-1)-marg[ii,tokens[:,0]])<2e-6,relative
                conditional=ps[ii,tokens[:,0]];assert np.max(conditional.max(-1)-conditional[ii,tokens[:,1]])<2e-6,relative
            ref=next(x['scores'] for x in oldcurve if x['update']==t) if label=='normal' else end['extra'][label]
            compare(directional,ref,name+f'/{t}/{label}')
            if label=='normal':curve.append(dict(update=t,directions=directional,scores=mean_scores(directional)))
            else:extra[label]=dict(directions=directional,scores=mean_scores(directional))
        assert sorted(expected_files)==sorted(f.name for f in folder.glob('protocol_*.npz')),name
        compare(curve[-1]['directions'],end['scores'],name+'/final')
        runs.append(dict(seed=seed,partition=p,mode=mode,curve=curve,auc=auc(curve),extra=extra));print(json.dumps(dict(analyzed=name,protocol_files=files)),flush=True)
    assert sorted(done['runs'])==sorted(f's{x["seed"]}_p{x["partition"]}_{x["mode"]}' for x in runs) and len(runs)==done['social_runs']
    seed_rows=[];aggregate=[]
    for seed,mode in product(inv['seeds'],inv['modes']):
        source=[r for r in runs if r['seed']==seed and r['mode']==mode]
        curve=[dict(update=t,scores=mean_scores([r['curve'][i]['scores'] for r in source])) for i,t in enumerate(cfg['checkpoints'])]
        seed_rows.append(dict(seed=seed,mode=mode,curve=curve,auc=auc(curve),extra={label:mean_scores([r['extra'][label]['scores'] for r in source]) for label in extra}))
    for mode in inv['modes']:
        source=[r for r in seed_rows if r['mode']==mode];curve=[dict(update=t,scores=mean_scores([r['curve'][i]['scores'] for r in source])) for i,t in enumerate(cfg['checkpoints'])]
        aggregate.append(dict(mode=mode,curve=curve,auc=auc(curve),extra={label:mean_scores([r['extra'][label] for r in source]) for label in extra}))
    contrasts=[]
    for seed in inv['seeds']:
        a=next(x for x in seed_rows if x['seed']==seed and x['mode']=='immediate');b=next(x for x in seed_rows if x['seed']==seed and x['mode']=='delayed')
        delta=lambda x,y:{g:{k:None if x[g][k] is None else x[g][k]-y[g][k] for k in x[g]} for g in x}
        contrasts.append(dict(seed=seed,endpoint=delta(b['curve'][-1]['scores'],a['curve'][-1]['scores']),auc=delta(b['auc'],a['auc'])))
    return dict(status='complete',formal=formal,analysis_source_sha256=sha(__file__),training_receipt_sha256=sha(out/'training_complete.json'),seeds=inv['seeds'],partitions=inv['partitions'],times=cfg['checkpoints'],runs=runs,seed_rows=seed_rows,aggregate=aggregate,contrasts=contrasts,
        primary_mean=float(np.mean([r['endpoint']['common30']['J'] for r in contrasts])),primary='common30 normal greedy same-message J at final2400: delayed minus immediate',
        interpretation='Observation-condition total effect with all inherited v20 full interfaces; v20 failed full-detach capability gate remains failed. Four source units, not48 directions or weighted rows.',
        metric_definition='Q integrates normalized exp of saved float32 autoregressive logprobs and two independent receiver softmax actions; J uses saved token-wise greedy messages and greedy decoder. CMI/routes use greedy codes in bits. Common-support shuffle marginals are set before subgroup scoring.',
        protocol_files=files,protocol_world_rows=rawrows),dict(passed=True,comparisons=checks,max_absolute_error=worst,production_metrics_called=False,model_forwards=0,new_training=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();out=args.out.resolve()
    try:analysis,comparison=analyze(out)
    except Exception:
        digest=sha(__file__);write(out/f'analysis_failure_{digest[:12]}.json',dict(passed=False,analysis_source_sha256=digest,exception=traceback.format_exc()));(out/f'analysis_failure_source_{digest[:12]}.py').write_bytes(Path(__file__).read_bytes());raise
    write(out/'analysis.json',analysis);comparison.update(analysis_sha256=sha(out/'analysis.json'),analysis_source_sha256=sha(__file__),training_receipt_sha256=sha(out/'training_complete.json'))
    write(out/'comparison.json',comparison);print(json.dumps(comparison),flush=True)
if __name__=='__main__':main()
