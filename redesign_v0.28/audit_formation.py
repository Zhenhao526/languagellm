"""Bounded v28 execution replay; no training or DINO inference.

Independent rectangular enumeration, normalization and sampled fixture identity.
Reuses sealed v23 independent atomic-render/communication helpers, never v28
world, metrics or runner. All endpoints are replayed, not all optimizer updates.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import analyze_experience as prior

read=prior.read; write=prior.write; sha=prior.sha; load=prior.load; npz=prior.npz


def bank_and_tables(out,check):
    entries=read(ROOT/'data/selection.json')['images']; ids=[e['id'] for e in entries]
    cache=load(ROOT/'data/feature_cache.pt'); z=cache['features'].numpy()
    check.exact(ids,sorted(set(ids)),'unique sorted image identifiers')
    check.exact(cache['image_ids'],ids,'feature row identity')
    check.check(z.shape==(12,1024) and z.dtype==np.float32 and np.isfinite(z).all(),'raw feature matrix')
    for category in ('apple','banana','orange','water'):
        for split,n in (('train',2),('test',1)):
            check.exact(sum(e['stratum']==category and e['split']==split for e in entries),n,'stratified split')
    ix=np.asarray([i for i,e in enumerate(entries) if e['split']=='train'])
    center=z[ix].mean(0); scale=float(np.sqrt(((z[ix]-center)**2).mean()))
    norm=npz(out/'normalization.npz');check.exact(center,norm['center'],'train-only center')
    check.exact(np.asarray(scale),norm['scale'],'train-only overall RMS')
    pools={(split,k):np.asarray([i for i,e in enumerate(entries) if e['split']==split and e['category']==kind],np.int64)
           for split in ('train','test') for k,kind in enumerate(('food','water'))}
    check.exact({f'{s}_{k}':v.tolist() for (s,k),v in pools.items()},read(out/'image_pools.json'),'actual pools')
    check.exact({key:len(v) for key,v in pools.items()},{('train',0):6,('train',1):2,('test',0):3,('test',1):1},'rectangular support')
    tables={}
    for split in ('train','test'):
        rows=[(m,f,w,shown) for shown in (0,1) for m in range(30) for f in pools[split,0] for w in pools[split,1]]
        r=np.asarray(rows,np.int64)
        tables[split]=dict(map_id=r[:,0],positions=prior.MAPS[r[:,0]],photo_ids=r[:,1:3],shown=r[:,3])
        check.exact(tables[split],npz(out/f'{split}_worlds.npz'),'complete rectangular world enumeration')
        check.add(split+'_world_rows',len(rows))
    return SimpleNamespace(features=torch.from_numpy((z-center)/max(scale,1e-6)),pools=pools),tables


def fixture(seed,p,d,step,scope,phase,w):
    ph=0 if phase=='private' else 1;n=256 if phase=='private' else 128
    rng=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,ph,step,1]))
    u=rng.random((n,3));maps=prior.group_maps(p)['old'] if scope=='old' else np.arange(30)
    m=maps[(u[:,0]*len(maps)).astype(np.int64)]
    food=np.unique(w['photo_ids'][:,0]);water=np.unique(w['photo_ids'][:,1]);count=len(food)*len(water)
    base=m*count+(u[:,1]*len(food)).astype(np.int64)*len(water)+(u[:,2]*len(water)).astype(np.int64)
    draws=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,ph,step,2])).random((2*n,1 if ph==0 else 4)).astype(np.float32)
    return dict(indices=np.concatenate((base,base+len(w['map_id'])//2)),uniforms=draws,goals=np.tile(rng.integers(2,size=n),2))


def logs(folder,cfg,phase,scope,table,check):
    rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    check.exact([r['update'] for r in rows],list(range(1,cfg['updates']+1)),'complete ordered log count')
    for step in sorted({0,cfg['updates']-1,*([2100] if cfg['updates']>2100 else [])}):
        if phase=='private': f=fixture(cfg['seed'],cfg['partition'],cfg['direction'],step,scope,phase,table)
        else: f={f'd{d}__{k}':v for d in (0,1) for k,v in fixture(cfg['seed'],cfg['partition'],d,step,scope,phase,table).items()}
        check.exact(prior.arrays_sha(f),rows[step]['world_sha256'],'prespecified external fixture hash')
        check.add(phase+'_sampled_fixture_updates')
    check.add(phase+'_log_rows_counted',len(rows))
    return [r['world_sha256'] for r in rows]


def audit(out,check):
    started=time.monotonic();inv=read(out/'invocation.json');done=read(out/'training_complete.json')
    check.exact(done['status'],'complete','complete batch required')
    check.exact(inv['formal'],done['formal'],'formal identity')
    check.exact(inv['seeds'],[34101,34102,34103,34104] if inv['formal'] else [99528],'fixed source list')
    check.exact(inv['partitions'],[1,2,3] if inv['formal'] else [1],'fixed partition list')
    check.exact(inv['updates'],2400 if inv['formal'] else 40,'fixed budget')
    check.exact(inv['new_dino_inferences'],0,'no DINO in training')
    for key in ('source_hashes','input_hashes'):
        check.exact(inv[key],done[key],'consistent source/input boundary')
        for path,h in inv[key].items():
            check.exact(sha(path),h,'current source/input bytes')
            if key=='source_hashes':check.exact(sha(out/'frozen_sources'/Path(path).relative_to(PROJECT)),h,'frozen source copy')
    for name,h in done['files'].items():check.exact(sha(out/name),h,'all completion-bound artifact bytes')
    check.add('completion_bound_files',len(done['files']))
    bank,tables=bank_and_tables(out,check);seeds=inv['seeds'];parts=inv['partitions'];updates=inv['updates']
    check.exact(done['private_fits'],len(seeds)*len(parts)*4,'private count')
    check.exact(done['social_runs'],len(seeds)*len(parts)*3,'social count')
    check.exact(len(list((out/'private').glob('*/result.json'))),done['private_fits'],'private actual inventory')
    check.exact(len(list((out/'social').glob('*/result.json'))),done['social_runs'],'social actual inventory')
    for seed in seeds:
        prepared=load(out/f'prepared_{seed}.pt')
        for p in parts:
            private_states={};private_initials={};cache_by_scope={};social_initials={};streams={}
            for d,scope in itertools.product((0,1),('old','all')):
                folder=out/'private'/f's{seed}_p{p}_d{d}_{scope}';cfg=read(folder/'config.json');res=read(folder/'result.json')
                check.exact((cfg['seed'],cfg['partition'],cfg['direction'],cfg['scope'],cfg['updates']),(seed,p,d,scope,updates),'private identity')
                check.exact(res['status'],'complete','private terminal')
                check.exact(cfg['prepared_sha256'],sha(out/f'prepared_{seed}.pt'),'private prepared bytes')
                a,head=prior.make_private(seed,p,d,prepared);initial=load(folder/'initial.pt')
                check.exact(dict(agent=a.state_dict(),head=head.state_dict()),initial,'independent private initialization')
                check.exact(load(folder/'initial_optimizer.pt')['state'],{},'fresh private Adam')
                for t in cfg['checkpoints']:
                    state=load(folder/f'checkpoint_{t:04d}.pt')
                    check.exact(prior.nonprivate(state['agent']),prior.nonprivate(initial['agent']),'all private frozen checkpoints')
                final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{updates:04d}.pt'),'private final checkpoint')
                a.load_state_dict(final['agent']);head.load_state_dict(final['head'])
                raw=npz(folder/f'evaluation_{updates:04d}.npz');check.exact({k:raw[k] for k in tables['test']},tables['test'],'private endpoint world')
                with torch.no_grad(): projected=a.project(bank.features).detach()
                h=prior.replay_h(a,projected,tables['test']);check.exact(h,raw['h'],'full private endpoint h')
                with torch.no_grad(): logits=head(torch.from_numpy(h)).reshape(-1,2,6).numpy()
                check.exact(logits,raw['logits'],'full private endpoint logits')
                check.close(prior.metrics(raw,p,True),res['scores'],'independent private endpoint scalar metrics')
                logs(folder,cfg,'private',scope,tables['train'],check)
                private_states[scope,d]=final;private_initials[scope,d]=initial
                check.add('private_endpoint_worlds_replayed',len(h));check.add('private_fits')
            for d in (0,1):check.exact(private_initials['old',d],private_initials['all',d],'paired private initialization')
            for scope in ('old','all'):
                cache_by_scope[scope],_=prior.social_cache(seed,p,scope,prepared,private_states,bank,tables,out,check)
            for arm in ('old_old','all_old','all_all'):
                scope,train_scope=arm.split('_');folder=out/'social'/f's{seed}_p{p}_{arm}'
                cfg=read(folder/'config.json');res=read(folder/'result.json');check.exact(res['status'],'complete','social terminal')
                check.exact((cfg['seed'],cfg['partition'],cfg['arm'],cfg['updates']),(seed,p,arm,updates),'social identity')
                for path,h in cfg['source_private'].items():check.exact(sha(path),h,'social private source bytes')
                agents=prior.remake_agents(seed,prepared,7,2,'identity');initial=load(folder/'initial.pt')
                for d,a in enumerate(agents):
                    a.load_state_dict(private_states[scope,d]['agent']);prior.reset_social(a,seed,p,d)
                    check.exact(a.state_dict(),initial[d],'fresh social initialization and frozen private interface')
                for state in load(folder/'initial_optimizer.pt'):check.exact(state['state'],{},'fresh social Adam')
                for t in cfg['checkpoints']:
                    states=load(folder/f'checkpoint_{t:04d}.pt')
                    for d in (0,1):check.exact(prior.nonsocial(states[d]),prior.nonsocial(initial[d]),'all social frozen checkpoints')
                final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{updates:04d}.pt'),'social final checkpoint')
                for a,state in zip(agents,final):a.load_state_dict(state)
                for d in (0,1):
                    raw=npz(folder/f'protocol_{updates:04d}_d{d}.npz');check.exact({k:raw[k] for k in tables['test']},tables['test'],'social endpoint world')
                    with torch.no_grad():
                        lp,tok=prior.enumerate_sender(agents[d],cache_by_scope[scope]['test',d])
                        rcv=prior.enumerate_receiver(agents[1-d])
                    check.exact(lp.numpy(),raw['sender_log_probs'],'independent full endpoint 49 log probabilities')
                    check.exact(tok.numpy(),raw['tokens'],'independent sequential greedy messages')
                    check.exact(rcv.numpy(),raw['receiver_logits'],'independent endpoint49 receiver actions')
                    check.close(prior.metrics(raw,p,False),res['scores'][d],'independent social endpoint scalar metrics')
                    check.add('social_endpoint_sender_worlds_replayed',len(lp));check.add('endpoint_receiver_tables')
                streams[arm]=logs(folder,cfg,'social',train_scope,tables['train'],check);social_initials[arm]=initial;check.add('social_pairs')
            check.exact(social_initials['all_old'],social_initials['all_all'],'B and C all initial tensors paired')
            for d in (0,1):
                select=lambda st:{k:v for k,v in st.items() if k.split('.')[0] in prior.SEND+prior.RECV}
                check.exact(select(social_initials['old_old'][d]),select(social_initials['all_old'][d]),'A B communication initial tensors paired')
            check.exact(streams['old_old'],streams['all_old'],'all A B social logged world identities paired')
    dependencies={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(PROJECT/'redesign_')) and Path(m.__file__).suffix=='.py'}
    return dict(passed=True,status='passed_bounded_execution_and_endpoint_replay',formal=inv['formal'],checks=check.count,
        scalar_comparisons=check.comparisons,scalar_max_error=check.max_error,coverage=check.scope_counts,
        source_hashes=inv['source_hashes'],input_hashes=inv['input_hashes'],audit_dependencies=dependencies,
        audit_source_sha256=sha(__file__),training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-started,
        exclusions=['No DINO or image inference repeated','No independent preparation or optimizer-update replay',
                    'Training fixture reconstruction limited to updates 1/2101/final; all log identities counted and A/B paired',
                    'Only endpoint scalar metrics here; independent complete-curve aggregation is a separate analysis'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();torch.set_num_threads(1)
    out=args.out.resolve();check=prior.Checks()
    try:
        result=audit(out,check);write(out/'audit_execution.json',result)
        print(json.dumps({k:result[k] for k in ('passed','checks','scalar_comparisons','scalar_max_error','coverage','seconds')},ensure_ascii=False))
    except Exception as exc:
        failure=out/f'audit_failure_{time.time_ns()}.json'
        write(failure,dict(passed=False,audit_source_sha256=sha(__file__),checks=check.count,error=repr(exc),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
