"""Read-only independent protocol/material audit, with bounded forward replay.

Reuses v26's independently implemented world/partition/check utilities. Does
not import run_transfer, production metrics, or social_model policy helpers.
"""
import argparse, itertools, json, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.26'))
import audit_probe as prior_audit
from audit_probe import QA, MAPS, groups, table, sha, read, write, load, now
from camp import remake_agents

PRIOR=PROJECT/'redesign_v0.23/results/experience_001'
V26=PROJECT/'redesign_v0.26';DATA=V26/'results/material_001'
SEEDS=(33101,33102,33103,33104);ARMS=('old_old','all_old','all_all');CELLS=('old_old','new_old','old_new','new_new')
WORLD=('map_id','positions','photo_ids','shown')
FRONT=('project.','memory.','slot_phi.','input_transform')
SENDER=('send_context.','send_embedding.','send_recur.','send_out.')
RECEIVER=('receive_embedding.','actor.')
MEASURES=('J','Q','food','water','visible','historical','blank_J','shuffle_J')
CHANGES=('message_TV','action_TV','cross_message_agreement','cross_action_agreement','old_within_message_agreement','new_within_message_agreement','old_within_action_agreement','new_within_action_agreement')
MASKS=('pooled','food_only','water_only');GROUPS=('old','new12','common30')
COUNTS=dict(old_old=960,new_old=2160,old_new=480,new_new=1080)

def softmax(x):
    x=np.asarray(x,np.float64);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def indicators(raw):
    codes=raw['tokens']@np.asarray([7,1],np.int64)
    actions=np.argmax(raw['receiver_logits'],axis=2)
    correct=actions[codes]==raw['positions'];prob=softmax(raw['receiver_logits']);sp=softmax(raw['sender_log_probs'])
    f=prob[:,0,raw['positions'][:,0]].T;w=prob[:,1,raw['positions'][:,1]].T
    q=np.sum(sp*f*w,axis=1)
    h=np.bincount(codes,minlength=49).astype(np.float64)/len(codes)
    successful=(actions[None,:,:]==raw['positions'][:,None,:]).all(-1)
    blank=successful[:,0].astype(np.float64);shuffle=successful@h
    return codes,actions,correct,q,blank,shuffle

def scores(raw,p):
    codes,actions,correct,q,blank,shuffle=indicators(raw);r=np.arange(len(codes));shown=raw['shown'];out={}
    for g,ids in groups(p).items():
        selected=np.isin(raw['map_id'],ids);out[g]={}
        for label,mask in [('pooled',selected),('food_only',selected&(shown==0)),('water_only',selected&(shown==1))]:
            c=correct[mask]
            out[g][label]=dict(n=int(mask.sum()),J=float((c[:,0]&c[:,1]).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask].mean()),blank_J=float(blank[mask].mean()),shuffle_J=float(shuffle[mask].mean()),visible=float(correct[r,shown][mask].mean()),historical=float(correct[r,1-shown][mask].mean()))
    return out

def histograms(raw):
    codes=raw['tokens']@np.asarray([7,1],np.int64);acts=raw['receiver_logits'].argmax(-1)[codes];decoded=acts@np.asarray([6,1],np.int64)
    out={}
    for m,k in itertools.product(range(30),(0,1)):
        ix=(raw['map_id']==m)&(raw['shown']==k);n=int(ix.sum());assert n>0
        # Decode actual row messages, independently of the production pushforward.
        out[m,k]=(np.bincount(codes[ix],minlength=49)/n,np.bincount(decoded[ix],minlength=36)/n)
    return out

def changes(reference,current,p):
    a=histograms(reference);b=histograms(current);rows=[]
    for m,k in itertools.product(range(30),(0,1)):
        x,y=a[m,k];u,v=b[m,k]
        rows.append(dict(map_id=m,shown=k,message_TV=float(np.abs(x-u).sum()/2),action_TV=float(np.abs(y-v).sum()/2),cross_message_agreement=float(np.sum(x*u)),cross_action_agreement=float(np.sum(y*v)),old_within_message_agreement=float(np.sum(x*x)),new_within_message_agreement=float(np.sum(u*u)),old_within_action_agreement=float(np.sum(y*y)),new_within_action_agreement=float(np.sum(v*v))))
    out={}
    for g,ids in groups(p).items():
        out[g]={}
        for label,k in [('pooled',None),('food_only',0),('water_only',1)]:
            rr=[r for r in rows if r['map_id'] in ids and (k is None or r['shown']==k)]
            out[g][label]={key:float(np.mean([r[key] for r in rr])) for key in CHANGES}
    return dict(rows=rows,scores=out)

def aggregate(rows):
    def mean(rr,field,keys):
        return {g:{m:{k:float(np.mean([r[field][g][m][k] for r in rr])) for k in keys} for m in MASKS} for g in GROUPS}
    sources=[];overall=[]
    for s,arm,cell in itertools.product(SEEDS,ARMS,CELLS):
        rr=[r for r in rows if (r['seed'],r['arm'],r['cell'])==(s,arm,cell)];assert len(rr)==6
        sources.append(dict(seed=s,arm=arm,cell=cell,scores=mean(rr,'scores',MEASURES),changes=mean(rr,'changes',CHANGES)))
    for arm,cell in itertools.product(ARMS,CELLS):
        rr=[r for r in sources if (r['arm'],r['cell'])==(arm,cell)];assert len(rr)==4
        overall.append(dict(arm=arm,cell=cell,scores=mean(rr,'scores',MEASURES),changes=mean(rr,'changes',CHANGES)))
    contrasts=[]
    for s in SEEDS:
        lookup={(r['arm'],r['cell']):r['scores']['new12']['pooled']['J'] for r in sources if r['seed']==s}
        bo=lookup['all_old','old_old'];bn=lookup['all_old','new_new'];ao=lookup['old_old','old_old'];an=lookup['old_old','new_new']
        contrasts.append(dict(seed=s,B_original_J=bo,B_new_J=bn,primary_B_material_delta=bn-bo,aux_new_B_minus_A=bn-an,aux_original_B_minus_A=bo-ao,aux_experience_material_interaction=(bn-an)-(bo-ao),C_new_J=lookup['all_all','new_new']))
    return dict(primary='B new12 natural J: new_new minus old_old',sources=sources,overall=overall,contrasts=contrasts,mean_contrasts={k:float(np.mean([r[k] for r in contrasts])) for k in contrasts[0] if k!='seed'})

def sender(a,h):
    state=a.send_context(torch.cat((h,torch.zeros(len(h),4,dtype=torch.float32)),dim=1));first_logits=a.send_out(state)
    first=F.log_softmax(first_logits,dim=1);conditional=[]
    for code in range(7):
        token=torch.full((len(h),),code,dtype=torch.int64)
        hidden=a.send_recur(a.send_embedding(token),state)
        conditional.append(F.log_softmax(a.send_out(hidden),dim=1))
    joint=(first[:,:,None]+torch.stack(conditional,dim=1)).reshape(len(h),49)
    t0=first_logits.argmax(dim=1)
    t1=a.send_out(a.send_recur(a.send_embedding(t0),state)).argmax(dim=1)
    return joint.numpy(),torch.stack((t0,t1),dim=1).numpy()

def receiver(a):
    tokens=torch.tensor(list(itertools.product(range(7),repeat=2)),dtype=torch.int64)
    embedded=a.receive_embedding(tokens).flatten(1)
    return a.actor(torch.cat((embedded,torch.zeros(49,20,dtype=torch.float32)),dim=1)).reshape(49,2,6).numpy()

@torch.no_grad()
def replay(out,qa):
    s,p,d=33101,1,0;prepared=torch.load(PRIOR/f'prepared_{s}.pt',weights_only=True);rows=[]
    for arm in ARMS:
        agents=remake_agents(s,prepared,7,2,'identity');state=torch.load(PRIOR/'social'/f's{s}_p{p}_{arm}'/'final.pt',weights_only=True)
        for who,a in enumerate(agents):
            a.load_state_dict(state[who],strict=True);a.eval()
            for name,value in a.named_parameters():value.requires_grad_(name.startswith(SENDER+RECEIVER))
        rc=receiver(agents[1-d])
        for cell in CELLS:
            raw=load(out/'tables'/f's{s}_p{p}_d{d}_{arm}_{cell}.npz')
            path=DATA/'tables'/f's{s}_p{p}_d{d}_{arm.split("_")[0]}_{cell}.npz';h=torch.from_numpy(load(path)['h']);lp=[];tokens=[]
            for lo in range(0,len(h),256):
                prob,token=sender(agents[d],h[lo:lo+256]);lp.append(prob);tokens.append(token)
            lp=np.concatenate(lp);tokens=np.concatenate(tokens)
            qa.close(lp,raw['sender_log_probs'],arm+'/'+cell+' independently composed sender',atol=0)
            qa.check(np.array_equal(tokens,raw['tokens']),arm+'/'+cell+' sequential greedy')
            qa.close(rc,raw['receiver_logits'],arm+'/'+cell+' independently composed receiver',atol=0)
            rows.append(dict(seed=s,partition=p,direction=d,arm=arm,cell=cell,worlds=len(h),sender_log_probs_exact=True,tokens_exact=True,receiver_logits_exact=True))
        for who,a in enumerate(agents):
            qa.check(all(torch.equal(value,state[who][name]) for name,value in a.state_dict().items()),arm+' state unchanged')
            qa.check(all(v.grad is None for v in a.parameters()),arm+' no gradients')
    return rows

def selftest():
    qa=QA();w=table([1,2],[3]);n=len(w['map_id'])
    raw=dict(w,tokens=np.zeros((n,2),np.int64),receiver_logits=np.zeros((49,2,6),np.float32),sender_log_probs=np.full((n,49),-np.log(49),np.float32))
    z=scores(raw,1)['common30']['pooled'];qa.close(z['Q'],1/36,'uniform Q');qa.check(z['J']==0 and z['blank_J']==0,'sameplace tie action')
    rc=np.full((49,2,6),-100.,np.float32)
    for c,(f,wloc) in enumerate(MAPS):rc[c,0,f]=rc[c,1,wloc]=100.
    full=dict(raw,receiver_logits=rc,tokens=np.column_stack((raw['map_id']//7,raw['map_id']%7)))
    qa.check(scores(full,1)['common30']['pooled']['J']==1,'perfect native messages')
    original=changes(raw,raw,1);qa.check(all(r['message_TV']==r['action_TV']==0 for r in original['rows']),'same distribution')
    switched=dict(raw,tokens=np.tile([0,1],(n,1)))
    contrast=changes(raw,switched,1);qa.check(all(r['message_TV']==1 and r['action_TV']==0 for r in contrast['rows']),'decoder merges two codes')
    return dict(passed=True,checks=qa.checks,model_inference=False)

def audit(out):
    torch.set_num_threads(1);start=time.monotonic();qa=QA();completion=read(out/'completion.json')
    for k,v in dict(status='complete',pairs=36,protocol_directions=72,tables=288,sender_worlds=336960,receiver_tables=72,training_updates=0,DINO_inference=0,new_pixel_reads=0,confirmation_pixels=0,network_requests=0,inputs_unchanged=True).items():qa.check(completion[k]==v,'complete/'+k)
    frozen=read(out/'freeze.json')['source_hashes']
    for path,digest in frozen.items():qa.check(sha(path)==digest,'source SHA/'+path)
    seal=read(DATA/'completion_manifest.json');qa.check(seal['artifact_count']==len(seal['artifacts'])==228,'v26 seal228')
    for path,digest in seal['artifacts'].items():qa.check(sha(V26/path)==digest,'v26 seal/'+path)
    pools=read(DATA/'world_manifest.json')['pools'];tables={c:table(pools[c.split('_')[0]][0],pools[c.split('_')[1]][1]) for c in CELLS}
    source_rows=read(out/'raw_summaries.json');lookup={(r['seed'],r['partition'],r['direction'],r['arm'],r['cell']):r for r in source_rows}
    keys=list(itertools.product(SEEDS,(1,2,3),(0,1),ARMS,CELLS));qa.check(len(lookup)==len(source_rows)==288 and set(lookup)==set(keys),'full288 inventory')
    results=[];legacy=[];inheritance=[];outputs={};total=0
    for s,p,arm in itertools.product(SEEDS,(1,2,3),ARMS):
        folder=PRIOR/'social'/f's{s}_p{p}_{arm}';states=torch.load(folder/'final.pt',weights_only=True)
        for d in (0,1):
            scope=arm.split('_')[0];private=torch.load(PRIOR/'private'/f's{s}_p{p}_d{d}_{scope}'/'final.pt',weights_only=True)['agent']
            front=[k for k in private if k.startswith(FRONT)]
            qa.check(len(front)==9 and all(torch.equal(private[k],states[d][k]) for k in front),'frozen private frontend')
            oldh=load(DATA/'tables'/f's{s}_p{p}_d{d}_{scope}_old_old.npz')['h'];oldcache=np.load(PRIOR/'cache'/f's{s}_p{p}_{scope}'/f'test_d{d}.npy')
            qa.close(oldh,oldcache,'old frontend finite precision',atol=1e-6,rtol=1e-5)
            inheritance.append(dict(seed=s,partition=p,arm=arm,direction=d,frontend_tensors=len(front),frontend_equal=True,old_h_cache_max_error=float(np.max(np.abs(oldh-oldcache)))))
            first=None
            for cell in CELLS:
                record=lookup[s,p,d,arm,cell];path=out/'tables'/f's{s}_p{p}_d{d}_{arm}_{cell}.npz';digest=sha(path);outputs[str(path)]=digest
                qa.check(str(path)==record['file'] and digest==record['sha256'],'raw path/SHA')
                raw=load(path);source=DATA/'tables'/f's{s}_p{p}_d{d}_{scope}_{cell}.npz'
                qa.check(record['v26_source']==str(source),'correct private h source')
                n=COUNTS[cell];total+=n;w=tables[cell];src=load(source)
                qa.check(set(raw)==set(WORLD)|{'tokens','sender_log_probs','receiver_logits'},'raw schema')
                for k in WORLD:qa.check(raw[k].dtype==np.int64 and np.array_equal(raw[k],w[k]) and np.array_equal(raw[k],src[k]),'rectangular world/'+k)
                qa.check(raw['tokens'].shape==(n,2) and raw['tokens'].dtype==np.int64 and ((raw['tokens']>=0)&(raw['tokens']<7)).all(),'token shape/range')
                qa.check(raw['sender_log_probs'].shape==(n,49) and raw['receiver_logits'].shape==(49,2,6),'policy table shapes')
                qa.check(all(raw[k].dtype==np.float32 and np.isfinite(raw[k]).all() for k in ('sender_log_probs','receiver_logits')),'finite float32 policy')
                qa.close(np.exp(raw['sender_log_probs'].astype(np.float64)).sum(-1),np.ones(n),'stored logprob normalization',atol=2e-6)
                if cell=='old_old':
                    first=raw;ref=load(folder/f'protocol_2400_d{d}.npz')
                    for k in WORLD+('tokens',):qa.check(np.array_equal(raw[k],ref[k]),'old protocol exact/'+k)
                    errors={k:float(np.max(np.abs(raw[k]-ref[k]))) for k in ('sender_log_probs','receiver_logits')}
                    for k in errors:qa.close(raw[k],ref[k],'old protocol/'+k,atol=1e-6,rtol=1e-5)
                    probability=softmax(raw['sender_log_probs']);old_probability=softmax(ref['sender_log_probs'])
                    qa.close(probability,old_probability,'old normalized49 probabilities',atol=1e-6,rtol=1e-5)
                    legacy.append(dict(seed=s,partition=p,arm=arm,direction=d,same_tokens=True,errors=errors,probability_max_error=float(np.max(np.abs(probability-old_probability))),passed=True))
                else:qa.check(np.array_equal(raw['receiver_logits'],first['receiver_logits']),'same decoder across materials')
                calculated=scores(raw,p);delta=changes(first,raw,p)
                qa.tree(calculated,record['scores'],'metrics/'+path.name);qa.tree(delta['scores'],record['changes'],'change groups/'+path.name);qa.tree(delta['rows'],record['change_rows'],'change rows/'+path.name)
                qa.check(all(-1e-12<=r['action_TV']<=r['message_TV']+1e-12<=1+2e-12 for r in delta['rows']),'fixed decoder TV contraction')
                results.append(dict(seed=s,partition=p,direction=d,arm=arm,cell=cell,file=str(path),sha256=digest,v26_source=str(source),scores=calculated,changes=delta['scores'],change_rows=delta['rows']))
    qa.check(total==336960,'sender world total');qa.tree(legacy,read(out/'legacy_control.json'),'old72 control')
    qa.tree(inheritance,read(out/'frontend_inheritance.json'),'all72 frontend source inheritance')
    summary=aggregate(results);qa.tree(summary,read(out/'summary.json'),'4source aggregate and contrasts')
    measured_comparisons=qa.comparisons;measured_max_error=qa.max_abs_error
    replays=replay(out,qa)
    for path,digest in frozen.items():qa.check(sha(path)==digest,'unchanged source/'+path)
    for name in ('completion.json','freeze.json','raw_summaries.json','summary.json','legacy_control.json','frontend_inheritance.json','synthetic_qa.json'):outputs[str(out/name)]=sha(out/name)
    dependency_hashes={str(Path(prior_audit.__file__).resolve()):sha(prior_audit.__file__),str(PROJECT/'redesign_v0.8/camp.py'):sha(PROJECT/'redesign_v0.8/camp.py')}
    analysis=dict(status='complete',source_sha256=sha(__file__),dependency_hashes=dependency_hashes,rows=results,**summary)
    write(out/'analysis.json',analysis)
    comparison=dict(passed=True,comparisons=qa.comparisons,max_abs_error=qa.max_abs_error,raw_and_summary_comparisons=measured_comparisons,raw_and_summary_max_abs_error=measured_max_error,analysis_sha256=sha(out/'analysis.json'),production_summary_sha256=sha(out/'summary.json'),notes='Maximum includes fixed old-h/old-logit numerical controls and float32 probability normalization QA; scalar metrics tolerance1e-12. Forward replay is exact.')
    write(out/'comparison.json',comparison)
    result=dict(passed=True,status='passed',created_utc=now(),seconds=time.monotonic()-start,checks=qa.checks,failures=[],comparisons=qa.comparisons,max_abs_error=qa.max_abs_error,source_sha256=sha(__file__),dependency_hashes=dependency_hashes,source_hashes=frozen,output_hashes=outputs,selftest=selftest(),replays=replays,
        coverage=dict(protocol_tables=288,protocol_directions=72,sender_worlds=336960,per_map_mask_histogram_comparisons=288*60,old_protocol_controls=72,frontend_source_bindings=72,replayed_sender_directions=3,replayed_protocol_tables=12,replayed_sender_worlds=sum(r['worlds'] for r in replays),replayed_receiver_tables=3,private_frontend_forward=0,DINO_forward=0,training_updates=0,pixel_reads=0),
        files={name:sha(out/name) for name in ('analysis.json','comparison.json')},
        limits=['All288 raw tables and72 old protocol controls checked; sender forward replay limited before outcomes to33101/p1/d0 A/B/C across4 material cells.', 'Sender/receiver policy composed directly from original atomic modules; no production protocol or metric helper used. Partitions/world utilities reused from sealed independent v26 audit.', 'No reexecution of visual frontend/DINO/optimization; h and frontend source tensors are read and bound.', '49-logprob and receiver-logit legacy comparisons use fixed atol1e-6/rtol1e-5 plus identical sequential greedy tokens; do not describe these legacy arrays as all bitwise equal.', 'Four inherited initialization sources; pooled worlds are finite-image reuses, not independent population samples.'])
    write(out/'audit_qa.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest()));sys.exit(0)
    if args.out is None:parser.error('--out or --selftest required')
    out=args.out.resolve()
    try:
        result=audit(out);print(json.dumps({k:result[k] for k in ('passed','checks','comparisons','max_abs_error','coverage','source_sha256')},ensure_ascii=False))
    except Exception as error:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        if out.exists():write(out/f'audit_failure_{stamp}.json',dict(status='failed',source_sha256=sha(__file__),error=repr(error),traceback=traceback.format_exc()))
        raise
