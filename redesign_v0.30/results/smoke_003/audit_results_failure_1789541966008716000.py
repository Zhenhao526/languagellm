"""Bounded independent v30 execution audit; no production support/update imports.

Reuse v21 independent sender/receiver atomic replay and the unchanged CampAgent
constructor, not its production communication helpers. No gradient/Adam replay,
private training, frontend cache reconstruction, pixel access or DINO inference.
"""
from __future__ import annotations
import argparse,itertools,json,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT))
import analyze_results as stats
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
from audit_execution import enumerate_sender,enumerate_receiver
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import remake_agents
SEND=('send_context','send_embedding','send_recur','send_out');RECV=('receive_embedding','actor')
RESET=SEND+RECV+('send_value','receive_value')
read,write,sha,npz=stats.read,stats.write,stats.sha,stats.npz
load=lambda p:torch.load(p,weights_only=True,map_location='cpu')

def seed_value(seed,p,d):
    return int(np.random.SeedSequence([30030,seed,p,d,1]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

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
    return {k:v for k,v in state.items() if k.split('.')[0] not in (RESET if reset_stage else SEND+RECV)}

def fixture(seed,p,d,step,arm):
    groups=stats.groups(p,arm);rng=lambda stream:np.random.default_rng(np.random.SeedSequence([30030,seed,p,d,step,stream]))
    maps=np.tile(groups['train12'],10)[rng(0).permutation(120)]
    photos=rng(1).random((120,2));base=maps*12+(photos[:,0]*6).astype(np.int64)*2+(photos[:,1]*2).astype(np.int64)
    return dict(indices=np.r_[base,base+360],uniforms=rng(2).random((240,4)).astype(np.float32))

def check_fixture(seed,p,step,arm,w,raw,log,cache,c):
    expected={};traces={k.removeprefix('trace__'):v for k,v in raw.items() if k.startswith('trace__')}
    for d in (0,1):
        f=fixture(seed,p,d,step,arm);expected.update({f'd{d}__{k}':v for k,v in f.items()});idx=f['indices']
        other=fixture(seed,p,d,step,stats.ARMS[1-stats.ARMS.index(arm)])
        c.exact(f['uniforms'],other['uniforms'],'arms same policy uniforms')
        for k in ('photo_ids','shown'):c.exact(w[k][idx],w[k][other['indices']],'arms same '+k)
        other_arm=stats.ARMS[1-stats.ARMS.index(arm)]
        # The two arms use the same random photo/mask stream but their map
        # permutations differ.  Match shared worlds by their full identity,
        # rather than comparing rows at the same batch position.
        shared_maps=np.intersect1d(stats.groups(p,arm)['train12'],stats.groups(p,other_arm)['train12'])
        left=np.column_stack((w['map_id'][idx],w['photo_ids'][idx],w['shown'][idx]))
        right=np.column_stack((w['map_id'][other['indices']],w['photo_ids'][other['indices']],w['shown'][other['indices']]))
        lk={tuple(row):i for i,row in enumerate(left)};rk={tuple(row):i for i,row in enumerate(right)}
        keys=[key for key in lk if key[0] in set(shared_maps)]
        c.exact(len(keys),140,'shared actual worlds140/240')
        c.exact(w['positions'][idx[[lk[key] for key in keys]]],w['positions'][other['indices'][[rk[key] for key in keys]]],'shared positions paired')
        counts=np.bincount(w['map_id'][idx],minlength=30);wanted=np.zeros(30,int);wanted[stats.groups(p,arm)['train12']]=20
        c.exact(counts,wanted,'exact training support and equal map exposures')
        for resource in (0,1):c.exact(np.bincount(w['positions'][idx,resource],minlength=6),np.full(6,40),'exact position margins')
        c.exact(w['shown'][idx],np.repeat(np.arange(2),120),'paired masks')
        c.exact(w['map_id'][idx[:120]],w['map_id'][idx[120:]],'same maps two masks')
        tr={k.removeprefix(f'd{d}__'):v for k,v in traces.items() if k.startswith(f'd{d}__')}
        c.exact(tr['h'],cache['train',d][idx],'frozen cached h used')
        c.exact(tr['uniforms'],f['uniforms'],'trace uniforms');c.exact(tr['positions'],w['positions'][idx],'feedback truth')
        for k in ('messages','actions'):
            probs=tr['token_probabilities' if k=='messages' else 'action_probabilities']
            us=f['uniforms'][:,:2] if k=='messages' else f['uniforms'][:,2:]
            sampled=np.minimum((np.cumsum(probs,axis=-1)<us[:,:,None]).sum(-1),probs.shape[-1]-1)
            c.exact(sampled,tr[k],'native categorical draws '+k)
        correct=(tr['actions']==w['positions'][idx]).astype(np.float32)
        reward=.25*correct.sum(1)+.5*correct.prod(1)
        c.exact(correct,tr['success'],'success from actual chosen action');c.exact(reward,tr['reward'],'team reward')
        c.exact(reward-np.float32(.5),tr['advantage'],'constant baseline')
        c.exact(tr['entropy_weight'],.02 if step<2100 else 0.,'fixed entropy transition')
        c.coverage['sampled_direction_fixtures']+=1;c.coverage['sampled_message_trajectories']+=240
    c.exact(expected,{k.removeprefix('world__'):v for k,v in raw.items() if k.startswith('world__')},'full saved fixture')
    c.exact(stats.arrays_sha(expected),log['world_sha256'],'sampled log world hash')
    c.exact(stats.arrays_sha(traces),log['trace_sha256'],'sampled log trace hash')

def audit(out,c):
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');source=Path(inv['source'])
    c.exact(done['status'],'complete','terminal');c.exact(inv['threads'],1,'CPU threads')
    c.exact(inv['source_hashes'],done['source_hashes'],'frozen source maps')
    c.exact(inv['input_hashes'],done['input_hashes'],'input maps')
    for p,h in inv['source_hashes'].items():
        c.exact(sha(p),h,'current production source');c.exact(sha(out/'frozen_sources'/Path(p).relative_to(PROJECT)),h,'archived production source')
    source_done=read(source/'training_complete.json');c.exact(source_done['status'],'complete','source training completed')
    for p,h in inv['input_hashes'].items():
        c.exact(sha(p),h,'consumed input unchanged')
        if Path(p).is_relative_to(source) and Path(p).name!='training_complete.json':
            c.exact(source_done['files'][str(Path(p).relative_to(source))],h,'input bound by old completion')
        c.coverage['consumed_inputs_hashed']+=1
    source_qa=read(source/'audit_execution.json');c.exact(source_qa['passed'],True,'inherited source execution QA')
    c.exact(source_qa['training_complete_sha256'],sha(source/'training_complete.json'),'old audit terminal binding')
    provenance=dict(source_audit_sha256=sha(source/'audit_execution.json'),source_training_complete_sha256=sha(source/'training_complete.json'))
    if inv['formal']:
        seal=read(source/'completion_manifest.json');artifacts={x['path']:x['sha256'] for x in seal['artifacts']}
        for p,h in inv['input_hashes'].items():c.exact(artifacts[p],h,'consumed inputs in original seal')
        c.exact(artifacts[str(source/'audit_execution.json')],sha(source/'audit_execution.json'),'source audit sealed')
        provenance['source_completion_manifest_sha256']=sha(source/'completion_manifest.json')
    for rel,h in done['files'].items():c.exact(sha(out/rel),h,'all new completion-bound artifacts')
    c.coverage['completion_bound_files']=len(done['files'])
    tables={split:npz(out/f'{split}_worlds.npz') for split in ('train','test')}
    for split,w in tables.items():
        c.exact(w,npz(source/f'{split}_worlds.npz'),'unmodified source world table')
        c.exact(w['positions'],stats.MAPS[w['map_id']],'physical coordinates')
        half=len(w['map_id'])//2;photos=np.unique(w['photo_ids'][:half],axis=0)
        c.exact(len(photos),12 if split=='train' else 3,'fixed non-square photo support')
        c.exact(w['shown'],np.repeat(np.arange(2),half),'complete ordered masks')
        c.exact(w['map_id'],np.tile(np.repeat(np.arange(30),len(photos)),2),'all layouts equally represented')
        c.exact(w['photo_ids'],np.tile(photos,(60,1)),'layout-independent photo order')
    c.require(not set(tables['train']['photo_ids'].flat)&set(tables['test']['photo_ids'].flat),'train/test photos disjoint')
    seeds=inv['seeds'];panels=inv['partitions'];steps=inv['updates'];times=[0,100,600,1200,2100,2400] if inv['formal'] else [0,40]
    c.exact(seeds,[34101,34102,34103,34104] if inv['formal'] else [99528],'predefined source seeds')
    c.exact(panels,[1,2,3] if inv['formal'] else [1],'coordinate panels');c.exact(steps,times[-1],'fixed budget')
    c.exact(inv['social_arms'],list(stats.ARMS),'all arms');c.exact(inv['batch_per_direction'],240,'direction batch')
    for seed,p in itertools.product(seeds,panels):
        prepared=load(source/f'prepared_{seed}.pt');agents=remake_agents(seed,prepared,7,2,'identity');fresh=[]
        for d,a in enumerate(agents):
            private=load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt')['agent'];a.load_state_dict(private)
            reset(a,seed_value(seed,p,d));fresh.append({k:v.detach().clone() for k,v in a.state_dict().items()})
            c.exact(frozen(fresh[d],True),frozen(private,True),'private interface inherited exactly')
            c.exact([n for n,v in a.named_parameters() if v.requires_grad],[n for n,v in a.named_parameters() if n.split('.')[0] in SEND+RECV],'only communication parameters trainable')
        cachefolder=source/'cache'/f's{seed}_p{p}_all';manifest=read(cachefolder/'manifest.json')
        c.exact(manifest['private_head_used'],False,'no private head in cache');c.exact(manifest['scope'],'all','all-private frozen source')
        cache={(split,d):np.load(cachefolder/f'{split}_d{d}.npy') for split,d in itertools.product(('train','test'),(0,1))}
        for (split,d),h in cache.items():
            c.require(h.shape==(len(tables[split]['map_id']),96) and h.dtype==np.float32 and np.isfinite(h).all(),'frozen cache shape')
            c.exact(manifest['files'][f'{split}_d{d}.npy'],sha(cachefolder/f'{split}_d{d}.npy'),'cache manifest binding')
        for arm in stats.ARMS:
            folder=out/'social'/f's{seed}_p{p}_{arm}';cfg=read(folder/'config.json');res=read(folder/'result.json');initial=load(folder/'initial.pt')
            c.exact(initial,fresh,'all initial tensors independent reset and paired arms')
            c.exact(cfg['groups'],{k:v.tolist() for k,v in stats.groups(p,arm).items()},'independently derived graph groups')
            c.exact((cfg['seed'],cfg['partition'],cfg['arm'],cfg['updates']),(seed,p,arm,steps),'run identity')
            c.exact(cfg['checkpoints'],times,'prespecified checkpoints');c.exact((cfg['learning_rate'],cfg['clip']),(.0007,2.),'optimizer settings')
            opts=load(folder/'initial_optimizer.pt')
            for d,opt in enumerate(opts):
                c.exact(opt['state'],{},'fresh empty Adam');c.exact(opt['param_groups'][0]['lr'],.0007,'Adam learning rate')
                c.exact(cfg['reset'][d]['seed'],seed_value(seed,p,d),'independent comm reset identity')
            for t in times:
                states=load(folder/f'checkpoint_{t:04d}.pt')
                for d in (0,1):c.exact(frozen(states[d]),frozen(initial[d]),'frozen tensors at every saved checkpoint')
                if t==0:c.exact(states,initial,'checkpoint zero')
                c.coverage['pair_checkpoints_verified']+=1
            final=load(folder/'final.pt');c.exact(final,states,'final equals terminal checkpoint')
            for d,a in enumerate(agents):a.load_state_dict(final[d])
            with torch.no_grad():
                for d in (0,1):
                    raw=npz(folder/f'protocol_{steps:04d}_d{d}.npz');lp,tokens=enumerate_sender(agents[d],torch.from_numpy(cache['test',d]))
                    c.exact(lp.numpy(),raw['sender_log_probs'],'all endpoint49 sender log probabilities')
                    c.exact(tokens.numpy(),raw['tokens'],'sequential greedy not joint argmax')
                    c.exact(enumerate_receiver(agents[1-d]).numpy(),raw['receiver_logits'],'complete49 integer-only receiver')
                    c.coverage['endpoint_sender_worlds_replayed']+=len(lp);c.coverage['endpoint_receiver_tables_replayed']+=1
            logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
            c.exact([x['update'] for x in logs],list(range(1,steps+1)),'all logged update identities')
            c.coverage['training_log_rows_counted']+=len(logs)
            for step in sorted({0,steps-1}|({2100} if steps>2100 else set())):
                check_fixture(seed,p,step,arm,tables['train'],npz(folder/f'train_{step+1:04d}.npz'),logs[step],cache,c)
                after=load(folder/f'after_{step+1:04d}.pt')
                for d in (0,1):c.exact(frozen(after[d]),frozen(initial[d]),'frozen sampled post-update tensors')
            c.exact((res['status'],res['updates'],res['messages'],res['actions'],res['frozen_verified']),('complete',steps,steps*480,steps*960,True),'run terminal budgets')
            c.coverage['social_runs']+=1
    n=len(seeds)*len(panels)*2
    c.exact(len(list((out/'social').glob('*/result.json'))),n,'exact run inventory')
    for k,v in dict(social_runs=n,pair_updates=n*steps,messages=n*steps*480,actions=n*steps*960,new_private_fits=0,new_dino_inferences=0).items():c.exact(done[k],v,'complete '+k)
    c.exact(c.coverage['social_runs'],n,'all new runs audited')
    return inv,provenance

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
    torch.set_num_threads(1);c=stats.Checks();started=time.monotonic()
    try:
        inv,provenance=audit(out,c)
        dependencies={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(PROJECT)) and Path(m.__file__).is_file() and Path(m.__file__).suffix=='.py'}
        result=dict(passed=True,status='passed_bounded_execution_and_complete_endpoint_replay',formal=inv['formal'],checks=c.count,
            coverage=dict(c.coverage),source_hashes=inv['source_hashes'],input_hashes=inv['input_hashes'],source_provenance=provenance,
            audit_source_sha256=sha(__file__),audit_dependencies=dependencies,training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-started,
            exclusions=['No gradient or Adam update replay. Initial Adam state checked empty; all checkpoint frozen tensors checked.',
                'Training fixture and stored native categorical actions sampled at updates1,2101,last; all log row counts only.',
                'Frontend cache/private preparation/DINO inherited by consumed-file hashes and source audit/seal, not rerun.',
                'Endpoint complete sender probabilities, sequential greedy,49-code receiver independently replayed using prior audit atoms.'])
        write(out/'audit_execution.json',result);shutil.copy2(__file__,out/'audit_results_source.py')
        print(json.dumps({k:result[k] for k in ('passed','checks','coverage','seconds')}))
    except Exception as exc:
        stamp=time.time_ns();write(out/f'audit_failure_{stamp}.json',dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),audit_source_sha256=sha(__file__)))
        shutil.copy2(__file__,out/f'audit_results_failure_{stamp}.py');raise
if __name__=='__main__':main()
