"""Frozen A/B/C protocols on four fixed image-material cells. No optimization."""
import argparse,hashlib,itertools,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import run_experience as exp
import metrics,world
from camp import remake_agents
import social_model as social
PRIOR=PROJECT/'redesign_v0.23/results/experience_001'
V26=PROJECT/'redesign_v0.26';DATA=V26/'results/material_001'
SEEDS=(33101,33102,33103,33104);ARMS=('old_old','all_old','all_all');CELLS=('old_old','new_old','old_new','new_new')
FRONT=('project.','slot_phi.','memory.','input_transform')
SCORES=('J','Q','food','water','visible','historical','blank_J','shuffle_J')
GROUPS=('old','new12','common30');MASKS=('pooled','food_only','water_only')
CHANGES=('message_TV','action_TV','cross_message_agreement','cross_action_agreement','old_within_message_agreement','new_within_message_agreement','old_within_action_agreement','new_within_action_agreement')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def now():return datetime.now(timezone.utc).isoformat()
def v26_path(s,p,d,scope,cell):return DATA/'tables'/f's{s}_p{p}_d{d}_{scope}_{cell}.npz'

def freeze(out):
    assert not out.exists(), 'Choose a new output directory'
    sealed=read(DATA/'completion_manifest.json')
    assert sealed['artifact_count']==228 and all(sha(V26/f)==h for f,h in sealed['artifacts'].items())
    files=[Path(__file__),ROOT/'固定执行方案.md',ROOT/'前置科学审查.md',ROOT/'前置数据核查.json',DATA/'completion_manifest.json',DATA/'summary.json',DATA/'audit_qa.json',DATA/'selection.json',DATA/'world_manifest.json']
    files += [v26_path(s,p,d,scope,cell) for s,p,d,scope,cell in itertools.product(SEEDS,(1,2,3),(0,1),('old','all'),CELLS)]
    files += [PRIOR/f'prepared_{s}.pt' for s in SEEDS]
    files += [PRIOR/'private'/f's{s}_p{p}_d{d}_{scope}'/'final.pt' for s,p,d,scope in itertools.product(SEEDS,(1,2,3),(0,1),('old','all'))]
    files += [PRIOR/'cache'/f's{s}_p{p}_{scope}'/f'test_d{d}.npy' for s,p,d,scope in itertools.product(SEEDS,(1,2,3),(0,1),('old','all'))]
    files += [PRIOR/'social'/f's{s}_p{p}_{arm}'/name for s,p,arm in itertools.product(SEEDS,(1,2,3),ARMS) for name in ('final.pt','protocol_2400_d0.npz','protocol_2400_d1.npz')]
    files += [PROJECT/name for name in ('redesign_v0.4/agents.py','redesign_v0.4/run_pilot.py','redesign_v0.4/resource_env.py','redesign_v0.8/camp.py','redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py','redesign_v0.23/world.py','redesign_v0.23/metrics.py','redesign_v0.23/run_experience.py')]
    hashes={str(p):sha(p) for p in sorted(set(files))};out.mkdir(parents=True);(out/'tables').mkdir()
    write(out/'freeze.json',dict(created_utc=now(),before_new_protocol_inference=True,source_hashes=hashes,previous_turn_classification='progress: completed v26 material probe and228 sealed artifacts',training_updates=0,new_DINO_inference=0))
    return hashes

def protocol_scores(raw,p):
    scores=metrics.social(raw,p);code=raw['tokens'][:,0]*7+raw['tokens'][:,1]
    correct=raw['receiver_logits'].argmax(-1)[code]==raw['positions'];ix=np.arange(len(code));shown=raw['shown']
    for g,selected in metrics.masks(raw,p).items():
        for label,mask in [('pooled',selected),('food_only',selected&(shown==0)),('water_only',selected&(shown==1))]:
            scores[g][label].update(visible=float(correct[ix,shown][mask].mean()),historical=float(correct[ix,1-shown][mask].mean()))
    return scores

def histogram(raw,m,k):
    selected=(raw['map_id']==m)&(raw['shown']==k);codes=raw['tokens'][selected,0]*7+raw['tokens'][selected,1]
    message=np.bincount(codes,minlength=49)/len(codes)
    actions=raw['receiver_logits'].argmax(-1);action_codes=actions[:,0]*6+actions[:,1]
    action=np.bincount(action_codes,weights=message,minlength=36)
    return message,action

def change_scores(reference,current,p):
    assert np.array_equal(reference['receiver_logits'],current['receiver_logits'])
    rows=[]
    for m,k in itertools.product(range(30),(0,1)):
        a,b=histogram(reference,m,k);c,d=histogram(current,m,k)
        item=dict(map_id=m,shown=k,message_TV=float(.5*np.abs(a-c).sum()),action_TV=float(.5*np.abs(b-d).sum()),cross_message_agreement=float(a@c),cross_action_agreement=float(b@d),old_within_message_agreement=float(a@a),new_within_message_agreement=float(c@c),old_within_action_agreement=float(b@b),new_within_action_agreement=float(d@d))
        assert item['action_TV']<=item['message_TV']+1e-12
        rows.append(item)
    dummy={key:np.asarray([r[key] for r in rows]) for key in ('map_id','shown')};scores={}
    for g,selected in metrics.masks(dummy,p).items():
        scores[g]={}
        for label,mask in [('pooled',selected),('food_only',selected&(dummy['shown']==0)),('water_only',selected&(dummy['shown']==1))]:
            scores[g][label]={key:float(np.mean([r[key] for r,keep in zip(rows,mask) if keep])) for key in CHANGES}
    return dict(rows=rows,scores=scores)

def aggregate(records):
    def mean(rr,field,keys):return {g:{m:{key:float(np.mean([r[field][g][m][key] for r in rr])) for key in keys} for m in MASKS} for g in GROUPS}
    sources=[];overall=[]
    for seed,arm,cell in itertools.product(SEEDS,ARMS,CELLS):
        rr=[r for r in records if (r['seed'],r['arm'],r['cell'])==(seed,arm,cell)];assert len(rr)==6
        sources.append(dict(seed=seed,arm=arm,cell=cell,scores=mean(rr,'scores',SCORES),changes=mean(rr,'changes',CHANGES)))
    for arm,cell in itertools.product(ARMS,CELLS):
        rr=[r for r in sources if (r['arm'],r['cell'])==(arm,cell)];assert len(rr)==4
        overall.append(dict(arm=arm,cell=cell,scores=mean(rr,'scores',SCORES),changes=mean(rr,'changes',CHANGES)))
    idx={(r['seed'],r['arm'],r['cell']):r['scores']['new12']['pooled'] for r in sources}
    contrasts=[]
    for seed in SEEDS:
        value=lambda arm,cell:idx[seed,arm,cell]['J']
        old=value('all_old','old_old');new=value('all_old','new_new')
        d0=old-value('old_old','old_old');d1=new-value('old_old','new_new')
        contrasts.append(dict(seed=seed,B_original_J=old,B_new_J=new,primary_B_material_delta=new-old,aux_new_B_minus_A=d1,aux_original_B_minus_A=d0,aux_experience_material_interaction=d1-d0,C_new_J=value('all_all','new_new')))
    keys=[k for k in contrasts[0] if k!='seed']
    return dict(primary='B new12 natural J: new_new minus old_old',sources=sources,overall=overall,contrasts=contrasts,mean_contrasts={k:float(np.mean([r[k] for r in contrasts])) for k in keys})

def synthetic_test():
    # Every world sends code0; receiver0 maps to correct positions0,1. Uniform
    # sender/receiver policies give Q=1/36 independently of the greedy success.
    raw=dict(map_id=np.tile(np.arange(30),2),shown=np.repeat((0,1),30),positions=world.MAPS[np.tile(np.arange(30),2)],tokens=np.zeros((60,2),np.int64),sender_log_probs=np.full((60,49),-np.log(49)),receiver_logits=np.zeros((49,2,6)))
    score=protocol_scores(raw,1);assert abs(score['common30']['pooled']['Q']-1/36)<1e-12
    change=change_scores(raw,raw,1);assert all(r['message_TV']==r['action_TV']==0 for r in change['rows'])
    other={k:v.copy() for k,v in raw.items()};other['tokens'][:,1]=1
    changed=change_scores(raw,other,1);assert all(r['message_TV']==1 and r['action_TV']==0 for r in changed['rows'])
    return dict(passed=True,model_inference=False,checks=['uniform49 codes/2 uniform6 actions gives Q1/36','identical materials have TV0','different codes merged by one decoder have messageTV1/actionTV0'])

@torch.no_grad()
def run(out):
    torch.set_num_threads(1);synthetic=synthetic_test();hashes=freeze(out);write(out/'synthetic_qa.json',synthetic)
    records=[];legacy=[];inheritance=[];start=time.monotonic()
    for seed,p,arm in itertools.product(SEEDS,(1,2,3),ARMS):
        scope=arm.split('_')[0];prepared=torch.load(PRIOR/f'prepared_{seed}.pt',weights_only=True)
        agents=remake_agents(seed,prepared,7,2,'identity');path=PRIOR/'social'/f's{seed}_p{p}_{arm}'
        states=torch.load(path/'final.pt',weights_only=True)
        before=[]
        for d,a in enumerate(agents):
            a.load_state_dict(states[d],strict=True);a.eval()
            for name,value in a.named_parameters():value.requires_grad_(name.startswith(social.SENDER_MODULES+social.RECEIVER_MODULES))
            private=torch.load(PRIOR/'private'/f's{seed}_p{p}_d{d}_{scope}'/'final.pt',weights_only=True)['agent']
            selected=[k for k in private if k.startswith(FRONT)]
            assert selected and all(torch.equal(private[k],states[d][k]) for k in selected)
            old_h=load(v26_path(seed,p,d,scope,'old_old'))['h'];cache=np.load(PRIOR/'cache'/f's{seed}_p{p}_{scope}'/f'test_d{d}.npy')
            assert np.allclose(old_h,cache,atol=1e-6,rtol=1e-5)
            inheritance.append(dict(seed=seed,partition=p,arm=arm,direction=d,frontend_tensors=len(selected),frontend_equal=True,old_h_cache_max_error=float(np.max(np.abs(old_h-cache)))))
            before.append(exp.state_sha(a.state_dict()))
        for d in (0,1):
            receiver=social.receiver_logits(agents[1-d]).numpy();raws={}
            for cell in CELLS:
                cached=load(v26_path(seed,p,d,scope,cell));h=torch.from_numpy(cached['h']);lp=[];tokens=[]
                for lo in range(0,len(h),256):
                    lp.append(social.message_log_probs(agents[d],h[lo:lo+256]).numpy());tokens.append(social.greedy_messages(agents[d],h[lo:lo+256]).numpy())
                raw={k:cached[k] for k in ('map_id','photo_ids','positions','shown')};raw.update(sender_log_probs=np.concatenate(lp),tokens=np.concatenate(tokens),receiver_logits=receiver)
                if cell=='old_old':
                    ref=load(path/f'protocol_2400_d{d}.npz')
                    assert all(np.array_equal(raw[k],ref[k]) for k in ('map_id','photo_ids','positions','shown','tokens'))
                    errors={k:float(np.max(np.abs(raw[k]-ref[k]))) for k in ('sender_log_probs','receiver_logits')}
                    assert all(np.allclose(raw[k],ref[k],atol=1e-6,rtol=1e-5) for k in errors)
                    prob=metrics.probabilities(raw['sender_log_probs']);old_prob=metrics.probabilities(ref['sender_log_probs'])
                    assert np.allclose(prob,old_prob,atol=1e-6,rtol=1e-5)
                    legacy.append(dict(seed=seed,partition=p,arm=arm,direction=d,same_tokens=True,errors=errors,probability_max_error=float(np.max(np.abs(prob-old_prob))),passed=True))
                file=out/'tables'/f's{seed}_p{p}_d{d}_{arm}_{cell}.npz';np.savez_compressed(file,**raw);raws[cell]=raw
                changes=change_scores(raws['old_old'],raw,p)
                records.append(dict(seed=seed,partition=p,direction=d,arm=arm,cell=cell,file=str(file),sha256=sha(file),v26_source=str(v26_path(seed,p,d,scope,cell)),scores=protocol_scores(raw,p),changes=changes['scores'],change_rows=changes['rows']))
        assert before==[exp.state_sha(a.state_dict()) for a in agents]
        assert all(v.grad is None for a in agents for v in a.parameters())
        print(json.dumps(dict(pairs_completed=len(legacy)//2,total_pairs=36)),flush=True)
    assert len(records)==288 and len(legacy)==len(inheritance)==72
    write(out/'raw_summaries.json',records);write(out/'summary.json',aggregate(records));write(out/'legacy_control.json',legacy);write(out/'frontend_inheritance.json',inheritance)
    assert all(sha(f)==h for f,h in hashes.items())
    write(out/'completion.json',dict(status='complete',created_utc=now(),seconds=time.monotonic()-start,pairs=36,protocol_directions=72,tables=288,sender_worlds=336960,receiver_tables=72,training_updates=0,DINO_inference=0,new_pixel_reads=0,confirmation_pixels=0,network_requests=0,inputs_unchanged=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);run(parser.parse_args().out.resolve())
