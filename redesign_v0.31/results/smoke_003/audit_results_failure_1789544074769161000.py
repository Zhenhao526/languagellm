"""Bounded independent audit for the v0.31 four-agent population run."""
from __future__ import annotations
import argparse,itertools,json,shutil,sys,time,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT));import analyze_results as stats
sys.path.insert(0,str(PROJECT/'redesign_v0.21'));from audit_execution import enumerate_sender,enumerate_receiver
sys.path.insert(0,str(PROJECT/'redesign_v0.8'));from camp import remake_agents
SEND=('send_context','send_embedding','send_recur','send_out');RECV=('receive_embedding','actor');RESET=SEND+RECV+('send_value','receive_value');PRIVATE_TYPES=(0,1,0,1);AGENTS=4;PAIR_KEYS=tuple((i,j) for i in range(AGENTS) for j in range(AGENTS) if i!=j and PRIVATE_TYPES[i]!=PRIVATE_TYPES[j])
read,write,sha,npz=stats.read,stats.write,stats.sha,stats.npz
load=lambda p:torch.load(p,weights_only=True,map_location='cpu')
def seed_value(seed,p,agent):return int(np.random.SeedSequence([31031,seed,p,agent,1]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))
def reset(a,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name in RESET:
            for module in getattr(a,name).modules():
                if hasattr(module,'reset_parameters'):module.reset_parameters()
    a.requires_grad_(False)
    for name,value in a.named_parameters():value.requires_grad_(name.split('.')[0] in SEND+RECV)
    return a
def frozen(state,reset_stage=False):
    excluded=RESET if reset_stage else SEND+RECV
    return {k:v for k,v in state.items() if k.split('.')[0] not in excluded}
def fixture(seed,p,d,step,condition):
    groups=stats.groups(p,condition);rng=lambda stream:np.random.default_rng(np.random.SeedSequence([31031,seed,p,d,step,stream]));maps=np.tile(groups['train12'],10)[rng(0).permutation(120)];photos=rng(1).random((120,2));base=maps*12+(photos[:,0]*6).astype(np.int64)*2+(photos[:,1]*2).astype(np.int64);return dict(indices=np.r_[base,base+360],uniforms=rng(2).random((240,4)).astype(np.float32))
def check_fixture(seed,p,step,condition,w,raw,log,cache,c):
    traces={k.removeprefix('trace__'):v for k,v in raw.items() if k.startswith('trace__')};expected={}
    for i,d in enumerate(PRIVATE_TYPES):
        f=fixture(seed,p,d,step,condition);expected.update({f'a{i}__{k}':v for k,v in f.items()});idx=f['indices'];tr={k.removeprefix(f'a{i}__'):v for k,v in traces.items() if k.startswith(f'a{i}__')};c.exact(tr['h'],cache['train',d][idx],f'a{i} frozen h');c.exact(tr['uniforms'],f['uniforms'],f'a{i} uniforms');c.exact(tr['positions'],w['positions'][idx],f'a{i} truth')
        for kind in ('messages','actions'):
            probs=tr['token_probabilities' if kind=='messages' else 'action_probabilities'];us=f['uniforms'][:,:2] if kind=='messages' else f['uniforms'][:,2:];sampled=np.minimum((np.cumsum(probs,axis=-1)<us[:,:,None]).sum(-1),probs.shape[-1]-1);c.exact(sampled,tr[kind],f'a{i} native {kind}')
        correct=(tr['actions']==w['positions'][idx]).astype(np.float32);reward=.25*correct.sum(1)+.5*correct.prod(1);c.exact(correct,tr['success'],f'a{i} success');c.exact(reward,tr['reward'],f'a{i} reward');c.exact(reward-np.float32(.5),tr['advantage'],f'a{i} baseline');c.exact(tr['entropy_weight'],.02 if step<2100 else 0.,f'a{i} entropy');c.coverage['sampled_direction_fixtures']+=1;c.coverage['sampled_message_trajectories']+=240
    c.exact(expected,{k.removeprefix('world__'):v for k,v in raw.items() if k.startswith('world__')},'full population fixture');c.exact(stats.arrays_sha(expected),log['world_sha256'],'world hash');c.exact(stats.arrays_sha(traces),log['trace_sha256'],'trace hash')
def audit(out,c):
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');source=Path(inv['source']);c.exact(done['status'],'complete','terminal');c.exact(inv['threads'],1,'threads');c.exact(inv['source_hashes'],done['source_hashes'],'source map');c.exact(inv['input_hashes'],done['input_hashes'],'input map')
    for p,h in inv['source_hashes'].items():c.exact(sha(p),h,'current source');c.exact(sha(out/'frozen_sources'/Path(p).relative_to(PROJECT)),h,'frozen source')
    source_done=read(source/'training_complete.json');c.exact(source_done['status'],'complete','inherited source');source_qa=read(source/'audit_execution.json');c.exact(source_qa['passed'],True,'inherited audit');c.exact(source_qa['training_complete_sha256'],sha(source/'training_complete.json'),'inherited terminal binding')
    for p,h in inv['input_hashes'].items():c.exact(sha(p),h,'input unchanged');c.coverage['consumed_inputs_hashed']+=1
    if inv['formal']:
        seal=read(source/'completion_manifest.json');artifacts={x['path']:x['sha256'] for x in seal['artifacts']}
        for p,h in inv['input_hashes'].items():c.exact(artifacts[p],h,'input in source seal')
    for rel,h in done['files'].items():c.exact(sha(out/rel),h,'completion file');c.coverage['completion_bound_files']+=1
    tables={split:npz(out/f'{split}_worlds.npz') for split in ('train','test')}
    for split,w in tables.items():
        c.exact(w,npz(source/f'{split}_worlds.npz'),f'{split} world table');c.exact(w['positions'],stats.MAPS[w['map_id']],f'{split} coordinates');half=len(w['map_id'])//2;photos=np.unique(w['photo_ids'][:half],axis=0);c.exact(w['shown'],np.repeat(np.arange(2),half),f'{split} masks');c.exact(w['map_id'],np.tile(np.repeat(np.arange(30),len(photos)),2),f'{split} layouts')
    c.require(not set(tables['train']['photo_ids'].flat)&set(tables['test']['photo_ids'].flat),'disjoint photos');seeds,panels=inv['seeds'],inv['partitions'];times=[0,100,600,1200,2100,2400] if inv['formal'] else [0,40];c.exact(seeds,[34101,34102,34103,34104] if inv['formal'] else [99528],'seeds');c.exact(panels,[1,2,3] if inv['formal'] else [1],'panels');c.exact(inv['updates'],times[-1],'budget');c.exact(inv['social_conditions'],list(stats.CONDITIONS),'conditions');c.exact(inv['population_agents'],4,'population size');c.exact(inv['private_types'],list(PRIVATE_TYPES),'private types')
    for seed,p in itertools.product(seeds,panels):
        prepared=load(source/f'prepared_{seed}.pt');templates=remake_agents(seed,prepared,7,2,'identity');agents=[]
        for i,d in enumerate(PRIVATE_TYPES):
            a=__import__('copy').deepcopy(templates[d]);private=load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt')['agent'];a.load_state_dict(private);reset(a,seed_value(seed,p,i));agents.append(a);c.exact(frozen(a.state_dict(),True),frozen(private,True),'private frozen tensors');c.exact([n for n,v in a.named_parameters() if v.requires_grad],[n for n,v in a.named_parameters() if n.split('.')[0] in SEND+RECV],'trainable communication only')
        cachefolder=source/'cache'/f's{seed}_p{p}_all';manifest=read(cachefolder/'manifest.json');c.exact(manifest['private_head_used'],False,'private head excluded');cache={(split,d):np.load(cachefolder/f'{split}_d{d}.npy') for split,d in itertools.product(('train','test'),(0,1))}
        for condition in stats.CONDITIONS:
            folder=out/'social'/f's{seed}_p{p}_{condition}';cfg=read(folder/'config.json');res=read(folder/'result.json');initial=load(folder/'initial.pt');c.exact(initial,[a.state_dict() for a in agents],'paired independent initial states');c.exact(cfg['population_agents'],4,'config population');c.exact(cfg['private_types'],list(PRIVATE_TYPES),'config private types');c.exact(cfg['groups'],{k:v.tolist() for k,v in stats.groups(p,condition).items()},'config groups');c.exact((cfg['seed'],cfg['partition'],cfg['condition'],cfg['updates']),(seed,p,condition,times[-1]),'run identity');c.exact(cfg['checkpoints'],times,'checkpoints');c.exact((cfg['learning_rate'],cfg['clip']),(.0007,2.),'optimizer');opts=load(folder/'initial_optimizer.pt')
            for opt in opts:c.exact(opt['state'],{},'fresh Adam');c.exact(opt['param_groups'][0]['lr'],.0007,'Adam lr')
            for t in times:
                states=load(folder/f'checkpoint_{t:04d}.pt');
                for d in range(AGENTS):c.exact(frozen(states[d]),frozen(initial[d]),'frozen checkpoint')
                if t==0:c.exact(states,initial,'checkpoint zero')
                c.coverage['pair_checkpoints_verified']+=1
            final=load(folder/'final.pt');c.exact(final,states,'final checkpoint')
            for i,j in PAIR_KEYS:
                a=agents[i];b=agents[j];a.load_state_dict(final[i]);b.load_state_dict(final[j]);raw=npz(folder/f'protocol_{times[-1]:04d}_i{i}_j{j}.npz');lp,tok=enumerate_sender(a,torch.from_numpy(cache['test',PRIVATE_TYPES[i]]));c.exact(lp.numpy(),raw['sender_log_probs'],f'i{i}j{j} sender');c.exact(tok.numpy(),raw['tokens'],f'i{i}j{j} greedy');c.exact(enumerate_receiver(b).numpy(),raw['receiver_logits'],f'i{i}j{j} receiver');c.coverage['endpoint_sender_worlds_replayed']+=len(lp);c.coverage['endpoint_receiver_tables_replayed']+=1
            logs=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()];c.exact([x['update'] for x in logs],list(range(1,times[-1]+1)),'log rows');c.coverage['training_log_rows_counted']+=len(logs)
            for step in sorted({0,times[-1]-1}|({2100} if times[-1]>2100 else set())):check_fixture(seed,p,step,condition,tables['train'],npz(folder/f'train_{step+1:04d}.npz'),logs[step],cache,c)
            c.exact((res['status'],res['updates'],res['messages'],res['actions'],res['frozen_verified']),('complete',times[-1],times[-1]*960,times[-1]*1920,True),'run terminal');c.coverage['social_runs']+=1
    n=len(seeds)*len(panels)*len(stats.CONDITIONS);c.exact(len(list((out/'social').glob('*/result.json'))),n,'run inventory');
    for k,v in dict(social_runs=n,pair_updates=n*times[-1],messages=n*times[-1]*960,actions=n*times[-1]*1920,new_private_fits=0,new_dino_inferences=0).items():c.exact(done[k],v,'complete '+k)
    c.exact(c.coverage['social_runs'],n,'all population runs');return inv,dict(source_audit_sha256=sha(source/'audit_execution.json'),source_training_complete_sha256=sha(source/'training_complete.json'),source_completion_manifest_sha256=sha(source/'completion_manifest.json') if inv['formal'] else None)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve();torch.set_num_threads(1);c=stats.Checks();start=time.monotonic()
    try:
        inv,prov=audit(out,c);deps={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(PROJECT)) and Path(m.__file__).is_file()};result=dict(passed=True,status='passed_bounded_population_audit_and_endpoint_replay',formal=inv['formal'],checks=c.count,coverage=dict(c.coverage),source_hashes=inv['source_hashes'],input_hashes=inv['input_hashes'],source_provenance=prov,audit_source_sha256=sha(__file__),audit_dependencies=deps,training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-start,exclusions=['No complete gradient/Adam replay; checkpoint frozen tensors and selected trace fixtures checked.','Inherited visual/private preparation and DINO encoding bound by input/source hashes.','All cross-type endpoint sender worlds and49-code receiver tables replayed independently.']);write(out/'audit_execution.json',result);shutil.copy2(__file__,out/'audit_results_source.py');print(json.dumps({k:result[k] for k in ('passed','checks','coverage','seconds')}))
    except Exception as e:
        stamp=time.time_ns();write(out/f'audit_failure_{stamp}.json',dict(passed=False,error=repr(e),traceback=traceback.format_exc(),audit_source_sha256=sha(__file__)));shutil.copy2(__file__,out/f'audit_results_failure_{stamp}.py');raise
if __name__=='__main__':main()
