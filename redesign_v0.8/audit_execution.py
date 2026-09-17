"""Independent read-only audit of all 60 complementarity runs and 4 personal controls.

Reconstructs training worlds with NumPy, checks exact initialization provenance,
and re-scores saved evaluation traces. Does not train or edit experiment files.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SEEDS=(27101,27102,27103,27104)
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}
REPRESENTATIONS=('identity',)
REWARDS={'additive':0.,'mixed':.5,'joint':1.}
CONDITIONS={f'split{s}_{kind}':dict(split=s,representation='identity',schedule='direct',vocab=7,
    length=2,known=False,blocked=False,reward_kind=kind,complementarity=lam)
    for s in MATCHINGS for kind,lam in REWARDS.items()}
for prefix,blocked in (('full',False),('blocked',True)):
    CONDITIONS.update({f'{prefix}_{kind}':dict(split=0,representation='identity',schedule='direct',vocab=7,
        length=2,known=False,blocked=blocked,reward_kind=kind,complementarity=lam)
        for kind,lam in REWARDS.items()})

MODES=('normal','shuffle','blank','stochastic','erase_memory')
CHECKPOINTS=[0,100,300,600,1200,1800,2100,2400]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return torch.load(path,weights_only=True,map_location='cpu')


def state_hash(states,prefix=None):
    h=hashlib.sha256()
    for who,state in enumerate(states):
        for k,v in sorted(state.items()):
            if prefix is None or k.startswith(prefix):
                h.update(f'{who}/{k}'.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def equal_states(a,b,prefix=None):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        keys={k for k in x if prefix is None or k.startswith(prefix)}
        if keys!={k for k in y if prefix is None or k.startswith(prefix)}:return False
        if any(not torch.equal(x[k],y[k]) for k in keys):return False
    return True


class Audit:
    def __init__(self):self.counts=Counter();self.failures=[]
    def check(self,value,name,context=None):
        self.counts[name]+=1
        if not bool(value):self.failures.append(dict(check=name,context=context))


@lru_cache(None)
def maps_for(split):
    held=set()
    if split:
        held={p for a,b in MATCHINGS[split] for p in ((a,b),(b,a))}
    return (tuple(i for i,p in enumerate(MAPS) if tuple(p) not in held),
            tuple(i for i,p in enumerate(MAPS) if tuple(p) in held))


@lru_cache(None)
def access_options(split,level):
    train,_=maps_for(split)
    return tuple((sites,tuple(i for i in train if set(MAPS[i])<=set(sites)))
                 for sites in combinations(range(6),level)
                 if any(set(MAPS[i])<=set(sites) for i in train))


def access(seed,identity,level,split):
    options=access_options(split,level)
    weights=np.asarray([len(pool) for _,pool in options],float)
    # Independent inverse-CDF implementation of the recorded weighted choice.
    u=np.random.default_rng(seed+47120000+identity).random()
    index=int(np.searchsorted(np.cumsum(weights/weights.sum()),u,side='right'))
    return options[min(index,len(options)-1)]


def schedule(seed,kind):
    if kind!='direct':raise ValueError(kind)
    return np.full(2400,6,dtype=np.int64),np.arange(2400)


def world_hash(rows):
    h=hashlib.sha256()
    for row in rows:
        for k in ('positions','photo_ids','goals','menu'):
            h.update(np.ascontiguousarray(row[k]).tobytes())
    return h.hexdigest()


def training_world(seed,identity,pool,photo_pools):
    rng=np.random.default_rng(seed*100000+60000000+identity+1);rows=[]
    for scout in (0,1):
        positions=MAPS[rng.choice(np.asarray(pool,np.int64),256)].copy()
        first=rng.integers(2,size=256)
        ids=np.column_stack([rng.choice(photo_pools[k],256) for k in (0,1)])
        menu=np.argsort(rng.random((256,2,6)),axis=2)
        rows.append(dict(positions=positions,goals=np.column_stack((first,1-first)),photo_ids=ids,menu=menu))
    return world_hash(rows)


def evaluation_world(seed,photo_pools):
    rng=np.random.default_rng(seed+69200000);rows=[]
    for scout in (0,1):
        ix=np.tile(np.asarray(list(product(range(30),range(2)))),(80,1))
        ix=ix[rng.permutation(4800)]
        ids=np.column_stack([rng.choice(photo_pools[k],4800) for k in (0,1)])
        menu=np.argsort(rng.random((4800,2,6)),axis=2)
        rows.append(dict(positions=MAPS[ix[:,0]].copy(),goals=np.column_stack((ix[:,1],1-ix[:,1])),
                         photo_ids=ids,menu=menu,scout=np.full(4800,scout),episode=np.arange(4800)))
    arrays={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return world_hash(rows),arrays


def metrics_from_counts(counts,lam):
    c00,c01,c10,c11=(int(counts[k]) for k in ('00','01','10','11'))
    n=c00+c01+c10+c11;partial=c01+c10;single=partial+2*c11
    total=(1-lam)*single/2+lam*c11
    second=((1-lam)/2)**2*partial+c11
    return dict(n=n,decisions=2*n,reward_sum=total,mean_reward=total/n if n else None,
        reward_variance=second/n-(total/n)**2 if n else None,
        positive_rewards=(c11+(partial if lam<1 else 0)),
        single_correct=single,single_accuracy=single/(2*n) if n else None,
        both_correct=c11,both_accuracy=c11/n if n else None,
        food_correct=c10+c11,water_correct=c01+c11,
        outcome_counts=dict(zip(('00','01','10','11'),(c00,c01,c10,c11))))


def exact_metric_match(actual,expected):
    # Mean/variance may use a different but equivalent float64 reduction order.
    for key,value in expected.items():
        if value is not None and isinstance(value,float):
            if not np.isclose(actual.get(key,np.nan),value,atol=1e-12,rtol=0):return False
        elif actual.get(key)!=value:return False
    return True


def expected_transform(seed,who,representation):
    rng=np.random.default_rng(np.random.SeedSequence([seed,who,4707]))
    if representation=='identity':matrix=np.eye(6)
    elif representation=='permute':
        cycle=rng.permutation(6);indices=np.empty(6,dtype=int)
        for j in range(6):indices[cycle[j]]=cycle[(j-1)%6]
        matrix=np.eye(6)[indices]
    else:
        q,r=np.linalg.qr(rng.normal(size=(6,6)))
        matrix=q*np.where(np.diag(r)<0,-1.,1.)
    return torch.from_numpy(matrix.astype(np.float32))


def trainable_hash(states,agents):
    return state_hash([{k:v for k,v in state.items()
        if k in {n for n,p in a.named_parameters() if p.requires_grad}}
        for state,a in zip(states,agents)])


def matrix_audit(audit,initial,final,cfg,seed,representation,label):
    for who,(before,after) in enumerate(zip(initial,final)):
        expected=expected_transform(seed,who,representation)
        matrix=before['input_transform']
        audit.check(torch.equal(matrix,expected) and torch.equal(matrix,after['input_transform']) and
                    np.array_equal(matrix.numpy(),np.asarray(cfg['input_transforms'][who],np.float32)),
                    'fixed_private_transform_exact',[label,who])
        audit.check(torch.allclose(matrix.T@matrix,torch.eye(6),atol=3e-7,rtol=0),
                    'invertible_orthonormal_transform',[label,who])


def personal_world(seed,who,update,pools,training=True):
    n=512 if training else 9600
    ws=seed*100000+50000000+who*10000+update if training else seed+49400000+who*1000
    rng=np.random.default_rng(ws)
    if training:maps=rng.integers(30,size=n);goals=rng.integers(2,size=n)
    else:
        ix=np.tile(np.asarray(list(product(range(30),range(2)))),(160,1))[rng.permutation(n)]
        maps,goals=ix.T
    ids=np.column_stack([rng.choice(pools[k],n) for k in (0,1)])
    world=dict(positions=MAPS[maps].copy(),goals=goals,photo_ids=ids,
               menu=np.argsort(rng.random((n,6)),axis=1))
    h=hashlib.sha256()
    for k in ('positions','goals','photo_ids','menu'):h.update(np.ascontiguousarray(world[k]).tobytes())
    return h.hexdigest(),world


def controls_audit(audit,root,ns,pools,allow_incomplete):
    completed=[];pending=[];rows_by_run={};initials={};results=[];gates=[]
    world_cache={};eval_cache={};trace_rows=0
    source_names=('camp.py','run_experiment.py','run_controls.py','固定执行方案.md')
    source_hashes={n:sha(root/'control_sources'/n) for n in source_names}
    for name,digest in source_hashes.items():
        audit.check(sha(ROOT/name)==digest,'control_archived_source_matches_frozen',name)
    invocation=read(root/'control_invocation.json')
    for key,value in dict(seeds=list(SEEDS),representations=['identity'],updates=2400,
        batch_per_person=512,eval_n_per_person=9600,source_hashes=source_hashes).items():
        audit.check(invocation.get(key)==value,'fixed_control_invocation',key)
    for seed in SEEDS:
      for representation in REPRESENTATIONS:
        label=f's{seed}_{representation}';folder=root/'individual_controls'/label
        if not(folder/'result.json').exists():pending.append(label);continue
        completed.append(label);cfg=read(folder/'config.json');result=read(folder/'result.json');results.append(result)
        for k,v in dict(seed=seed,representation=representation,updates=2400,batch=512,
            eval_n_per_agent=9600,gate=.9,fitted_agent_prefixes=['memory.','slot_phi.'],
            optimizer='Adam',learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=2100,
            source_hashes=source_hashes).items():audit.check(cfg.get(k)==v,'fixed_control_config',[label,k])
        for k,v in dict(complete=True,seed=seed,representation=representation,
            frozen_unused_parameters_verified=True).items():audit.check(result.get(k)==v,'control_result_identity',[label,k])
        source=cfg['prepared_source'];path=root/f'prepared_{seed}.pt'
        audit.check(Path(source['path']).resolve()==path and source['sha256']==sha(path),
                    'control_prepared_source_digest',label)
        prepared=load(path);agents=ns['remake_agents'](seed,prepared,7,2,representation)
        initial=load(folder/'initial.pt');final=load(folder/'final.pt');initials[label]=initial
        audit.check(equal_states(initial['agents'],[a.state_dict() for a in agents]),
                    'control_initial_agents_exact_fresh',label)
        audit.check(state_hash(initial['agents'])==cfg['initial_sha256'] and
                    trainable_hash(initial['agents'],agents)==cfg['common_initial_sha256'],
                    'control_initial_hashes_exact',label)
        matrix_audit(audit,initial['agents'],final['agents'],cfg,seed,representation,label)
        for who in (0,1):
            torch.manual_seed(seed*1000+881+who)
            head=torch.nn.Sequential(torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,12))
            audit.check(equal_states([head.state_dict()],[initial['heads'][who]]),
                        'control_head_initial_exact_fresh',[label,who])
            untouched={k:v for k,v in initial['agents'][who].items() if not k.startswith(('memory.','slot_phi.'))}
            audit.check(all(torch.equal(v,final['agents'][who][k]) for k,v in untouched.items()),
                        'control_all_unused_weights_frozen',[label,who])
            opt=load(folder/'final_optimizer.pt')[who]
            nparams=len(list(head.parameters()))+sum(k.startswith(('memory.','slot_phi.'))
                                                     for k,_ in agents[who].named_parameters())
            audit.check(len(opt['state'])==nparams and all(float(s['step'])==2400 for s in opt['state'].values()) and
                        all(g['lr']==.0007 for g in opt['param_groups']),
                        'control_fresh_adam_update_budget',[label,who])
        rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        rows_by_run[label]=rows
        audit.check(len(rows)==2400,'control_exact_training_updates',label)
        for update,row in enumerate(rows,1):
            audit.check(row['update']==update and [r['agent'] for r in row['agents']]==[0,1],
                        'control_personal_batch_structure',[label,update])
            for record in row['agents']:
                who=record['agent'];key=(seed,who,update)
                if key not in world_cache:
                    world_cache[key]=personal_world(seed,who,update,{k:pools['train',k] for k in (0,1)})[0]
                audit.check(record['world_sha256']==world_cache[key],
                            'independent_control_training_world_digest',[label,who,update])
                audit.check(all(np.isfinite(record[k]) for k in ('reward','loss','gradient_norm')) and
                            0<=record['reward']<=1,'finite_control_training_values',[label,who,update])
        curve=read(folder/'curve.json')
        audit.check([r['update'] for r in curve]==[0,600,1200,1800,2400],
                    'control_fixed_curve_checkpoints',label)
        for point in curve:
            audit.check(set(point['scores'])=={'normal','erase_memory'},'control_checkpoint_modes',[label,point['update']])
            for mode,stats in point['scores'].items():
                counts=stats['per_agent']
                audit.check(len(counts)==2 and all(x['n']==1200 and x['accuracy']==x['correct']/1200 for x in counts)
                    and stats['mean_accuracy']==sum(x['accuracy'] for x in counts)/2,
                    'control_checkpoint_counts',[label,point['update'],mode])
        audit.check(set(result['scores'])=={'normal','erase_memory'},'control_final_modes',label)
        for mode in ('normal','erase_memory'):
            stats=result['scores'][mode];counts=[]
            with np.load(folder/f'final_{mode}.npz') as z:
                trace_rows+=len(z['reward'])
                audit.check(len(z['reward'])==19200 and np.array_equal(z['agent'],np.repeat([0,1],9600)),
                            'control_final_personal_eval_budget',[label,mode])
                for who in (0,1):
                    ix=z['agent']==who;key=(seed,who)
                    if key not in eval_cache:
                        eval_cache[key]=personal_world(seed,who,0,{k:pools['test',k] for k in (0,1)},False)[1]
                    world=eval_cache[key]
                    audit.check(all(np.array_equal(z[k][ix],v) for k,v in world.items()),
                                'independent_control_final_world_arrays',[label,who,mode])
                    places=z['menu'][ix][np.arange(9600),z['action'][ix]]
                    truth=(places==z['positions'][ix][np.arange(9600),z['goals'][ix]])
                    correct=int(truth.sum());counts.append(dict(n=9600,correct=correct,accuracy=correct/9600))
                    audit.check(np.array_equal(z['place'][ix],places) and np.array_equal(z['reward'][ix],truth),
                                'independent_control_final_reward_recount',[label,who,mode])
                    mapid=z['positions'][ix,0]*5+z['positions'][ix,1]-(z['positions'][ix,1]>z['positions'][ix,0])
                    audit.check(Counter(zip(mapid.tolist(),z['goals'][ix].tolist()))==
                                Counter({(m,g):160 for m in range(30) for g in (0,1)}),
                                'control_each_map_goal_has_160_cases',[label,who,mode])
                audit.check(stats==dict(per_agent=counts,mean_accuracy=sum(x['accuracy'] for x in counts)/2),
                            'control_report_matches_exact_counts',[label,mode])
            if mode=='normal':
                passed=all(x['correct']*10>=x['n']*9 for x in counts)
                gates.append(dict(seed=seed,representation=representation,passed=passed,counts=counts))
                audit.check(result['passed']==passed,'control_integer_gate_exact',label)
                audit.check(passed,'control_positive_gate_passed',label)
    for seed in SEEDS:
        names=[f's{seed}_{r}' for r in REPRESENTATIONS]
        if not all(n in initials for n in names):continue
        base=initials[names[0]]
        for name in names[1:]:
            other=initials[name]
            keep=lambda states:[{k:v for k,v in s.items() if k!='input_transform'} for s in states]
            audit.check(equal_states(keep(base['agents']),keep(other['agents'])) and
                        equal_states(base['heads'],other['heads']),'controls_same_initial_parameters',[seed,name])
            for a,b in zip(rows_by_run[names[0]],rows_by_run[name]):
                audit.check(a['update']==b['update'] and all(x['world_sha256']==y['world_sha256']
                    for x,y in zip(a['agents'],b['agents'])),'control_representation_matched_worlds',[seed,name,a['update']])
    summary_path=root/'individual_controls/summary.json'
    if summary_path.exists():
        summary=read(summary_path)
        audit.check(summary==dict(complete=True,passed=all(r['passed'] for r in results),runs=len(results),results=results),
                    'control_summary_matches_individual_results')
    gate_path=root/'social_launch_gate.json'
    if gate_path.exists():
        gate=read(gate_path)
        audit.check(gate['controls_complete']==4 and gate['all_passed'] and gate['social_runs_planned']==60 and
                    not gate['diagnostic_weights_transferred'] and gate['control_summary_sha256']==sha(summary_path),
                    'social_launch_gate_matches_completed_control_summary')
    if not allow_incomplete:audit.check(len(completed)==4 and not pending,'all_four_personal_controls_complete',pending)
    return dict(completed=len(completed),pending=pending,gates=gates,
        training_person_batches_verified=audit.counts['independent_control_training_world_digest'],
        unique_person_batches_reconstructed=len(world_cache),final_trace_rows_recounted=trace_rows,
        scope='Each person has 2400 x 512 personal choices; social runs have 2400 x 512 pair worlds with two choices per world. Control weights are independent copies.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'results/complementarity_001')
    p.add_argument('--out',type=Path)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--development-triplet',nargs=2,type=int,metavar=('SEED','SPLIT'),
                   help='With --allow-incomplete, audit controls and one complete main reward triplet.')
    args=p.parse_args();root=args.root.resolve();out=args.out or root/'audit_execution.json'
    if args.development_triplet and not args.allow_incomplete:p.error('Development subset requires --allow-incomplete')
    if out.exists():raise FileExistsError(f'Preserve prior audit: {out}')
    torch.set_num_threads(1);audit=Audit()
    invocations=sorted(root.glob('invocation_*.json'))
    if not invocations:raise RuntimeError('No frozen invocation found')
    hashes=read(invocations[0])['hashes']
    for path in invocations:
        record=read(path)
        audit.check(record['hashes']==hashes,'invocation_sources_unchanged',path.name)
        audit.check(record['seeds']==list(SEEDS),'fixed_seed_set',path.name)
        audit.check(len(record['conditions'])==len(set(record['conditions'])) and
                    set(record['conditions'])<=set(CONDITIONS),'invocation_condition_scope',path.name)
        for key,value in dict(updates=2400,batch=512,eval_n=9600).items():
            audit.check(record[key]==value,'invocation_budget',[path.name,key])
    for path,digest in hashes.items():
        audit.check(Path(path).exists() and sha(path)==digest,'frozen_original_asset_hash',path)
    camp_hash=hashes[str(ROOT/'camp.py')];runner_hash=hashes[str(ROOT/'run_experiment.py')]
    snapshot=root/f'sources_{camp_hash[:8]}_{runner_hash[:8]}'
    for path,digest in hashes.items():
        if Path(path).suffix in ('.py','.md'):
            audit.check(sha(snapshot/Path(path).name)==digest,'frozen_snapshot_hash',Path(path).name)
    ns={'__file__':str(ROOT/'camp.py'),'__name__':'complementarity_audit_frozen_camp'}
    exec(compile((snapshot/'camp.py').read_text(),str(snapshot/'camp.py'),'exec'),ns)
    entries=read(ROOT.parent/'redesign_v0.4/data/manifest.json')['images']
    pools={(split,k):np.asarray([i for i,e in enumerate(entries) if e['split']==split and e['category']==name],np.int64)
        for split in ('train','test') for k,name in enumerate(('food','water'))}
    audit.check([len(pools[s,k]) for s in ('train','test') for k in (0,1)]==[22,22,8,8],'photo_pool_sizes')
    train_shas={e['sha256'] for e in entries if e['split']=='train'}
    test_shas={e['sha256'] for e in entries if e['split']=='test'}
    audit.check(not(train_shas&test_shas),'photo_split_disjoint')
    controls=controls_audit(audit,root,ns,pools,args.allow_incomplete)

    prepared={};people=[];personal_hashes=[];project_hashes=[]
    for seed in SEEDS:
        path=root/f'prepared_{seed}.pt'
        if not path.exists():continue
        states=load(path);prepared[seed]=states
        random=[a.state_dict() for a in ns['make_agents'](seed)]
        report=read(root/f'preparation_{seed}.json')
        audit.check(len(states)==2 and report['seed']==seed,'preparation_seed_and_count',seed)
        audit.check(all(x['updates']==200 and x['choices']==12800 and x['heldout_need_sensitive_choice']>=.9
                        for x in report['two_sites']) and min(report['six_sites'])>=.9,'preparation_gate_and_budget',seed)
        for who,(state,initial) in enumerate(zip(states,random)):
            audit.check(equal_states([state],[initial],('sender.','value.')),'new_seed_untouched_fingerprint',[seed,who])
            audit.check(not equal_states([state],[initial],('project.',)) and
                        not equal_states([state],[initial],('actor.',)),'resource_preparation_changed_trainable_branches',[seed,who])
            h=state_hash([state]);z=state_hash([state],'project.')
            personal_hashes.append(h);project_hashes.append(z)
            people.append(dict(seed=seed,agent=who,personal_torch_seed=seed*1000+who*137,state_sha256=h,project_sha256=z))
    audit.check(len(personal_hashes)==len(set(personal_hashes)) and len(project_hashes)==len(set(project_hashes)),
                'new_personal_preparations_unique')
    overlaps=[];history_count=0
    for version in ('redesign_v0.4','redesign_v0.5','redesign_v0.6','redesign_v0.7','redesign_v0.8'):
        for path in sorted((ROOT.parent/version/'results').glob('**/prepared*.pt')):
            if path.resolve().parent==root:continue
            states=load(path)
            if isinstance(states,dict):states=[states]
            for who,state in enumerate(states):
                if not isinstance(state,dict) or 'project.0.weight' not in state:continue
                history_count+=1
                if state_hash([state]) in personal_hashes or state_hash([state],'project.') in project_hashes:
                    overlaps.append(dict(path=str(path),agent=who))
    audit.check(not overlaps,'no_historical_preparation_or_projection_overlap',overlaps)

    completed=[];pending=[];rows_by_run={};world_cache={};evaluations={};trace_rows=0
    expected_names={f's{s}_{name}' for s in SEEDS for name in CONDITIONS}
    actual_names={p.name for p in root.glob('s*') if p.is_dir() and p.name[1:2].isdigit()}
    audit.check(not(actual_names-expected_names),'no_extra_study_conditions',sorted(actual_names-expected_names))
    for seed in SEEDS:
      for name,plan in CONDITIONS.items():
        label=f's{seed}_{name}';folder=root/label
        if args.development_triplet and (seed!=args.development_triplet[0] or
                plan['split']!=args.development_triplet[1] or plan['blocked']):
            pending.append(label);continue
        if not(folder/'result.json').exists():pending.append(label);continue
        completed.append(label);cfg=read(folder/'config.json');result=read(folder/'result.json')
        train_ids,held_ids=maps_for(plan['split']);levels,identities=schedule(seed,plan['schedule'])
        expected=dict(seed=seed,condition=name,plan=plan,updates=2400,batch=512,eval_n=9600,
            checkpoint_eval_n=1200,checkpoints=CHECKPOINTS,sites=6,history_dim=18,
            train_map_ids=list(train_ids),heldout_map_ids=list(held_ids),map_table=MAPS.tolist(),
            learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=2100,gamma=1,
            training_seed_offset=60000000,choices_per_world=2,
            reward_formula='(1-lambda)*mean(successes)+lambda*product(successes)',
            receiver_policy_score='sum of two action log probabilities',trace_schema='paired_world_v1',source_hashes={Path(p).name:h for p,h in hashes.items() if Path(p).parent==ROOT})
        for k,v in expected.items():audit.check(cfg.get(k)==v,'fixed_run_config',[label,k])
        for k,v in dict(seed=seed,condition=name,plan=plan,updates=2400,batch=512,frozen_projection_verified=True).items():
            audit.check(result.get(k)==v,'result_identity',[label,k])
        initial=load(folder/'initial.pt');final=load(folder/'final.pt')
        audit.check(state_hash(initial)==cfg['initial_sha256']==result['initial_sha256'],'initial_digest',label)
        audit.check(state_hash(final)==result['final_sha256'],'final_digest',label)
        audit.check(equal_states(initial,final,('project.',)) and equal_states(initial,prepared[seed],('project.',)),
                    'frozen_prepared_projection',label)
        agents=ns['remake_agents'](seed,prepared[seed],plan['vocab'],plan['length'],plan['representation'])
        audit.check(equal_states(initial,[a.state_dict() for a in agents]),'new_random_social_interface_exact',label)
        audit.check(trainable_hash(initial,agents)==cfg['trainable_initial_sha256'],
                    'trainable_initial_digest',label)
        audit.check(state_hash(initial,('receive_embedding.','actor.','receive_value.'))==cfg['receiver_initial_sha256'],
                    'receiver_initial_digest',label)
        matrix_audit(audit,initial,final,cfg,seed,plan['representation'],label)
        control_initial=load(root/'individual_controls'/f"s{seed}_{plan['representation']}"/'initial.pt')['agents']
        audit.check(equal_states(initial,control_initial),'social_starts_from_control_initial_not_final',label)
        audit.check(cfg['trainable_parameters']==[sum(p.numel() for p in a.parameters() if p.requires_grad) for a in agents],
                    'trainable_parameter_count',label)
        source=cfg['prepared_source'];path=root/f'prepared_{seed}.pt'
        audit.check(Path(source['path']).resolve()==path.resolve() and source['sha256']==sha(path),'prepared_source_digest',label)
        for point in CHECKPOINTS:
            checkpoint=folder/f'checkpoint_{point:04d}.pt'
            audit.check(checkpoint.exists(),'checkpoint_present',[label,point])
            if checkpoint.exists():
                audit.check(equal_states(initial,load(checkpoint),('project.','input_transform')),
                            'projection_and_transform_frozen_at_all_checkpoints',[label,point])
        audit.check(equal_states(initial,load(folder/'checkpoint_0000.pt')) and
                    equal_states(final,load(folder/'checkpoint_2400.pt')),'checkpoint_endpoints_match',label)
        for who,opt in enumerate(load(folder/'final_optimizer.pt')):
            audit.check(bool(opt['state']) and all(g['lr']==.0007 for g in opt['param_groups']) and
                        all(float(s['step'])==2400 for s in opt['state'].values()),'fresh_adam_update_budget',[label,who])
            audit.check(len(opt['state'])==sum(p.requires_grad for p in agents[who].parameters()),
                        'optimizer_state_has_only_all_trainable_parameters',[label,who])
        sched=read(folder/'training_schedule.json')
        audit.check(sched['levels']==levels.tolist() and sched['batch_identities']==identities.tolist(),'exact_training_schedule',label)
        rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        rows_by_run[label]=rows;audit.check(len(rows)==2400,'exact_training_updates',label)
        for step,row in enumerate(rows):
            if step>=2400:break
            identity=int(identities[step]);level=int(levels[step]);sites,pool=access(seed,identity,level,plan['split'])
            audit.check(row['update']==step+1 and row['batch_identity']==identity and row['active_sites']==level and
                        row['allowed_sites']==list(sites) and row['map_pool']==list(pool),
                        'independent_weighted_access_and_batch',[label,step+1])
            audit.check(row['entropy_weight']==(.02 if step<2100 else 0.),'exact_exploration_weight',[label,step+1])
            audit.check(set(pool)<=set(train_ids) and not(set(pool)&set(held_ids)),'training_holdout_support_excluded',[label,step+1])
            audit.check(np.isfinite(row['reward']) and all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm'])
                        for x in row['agents']),'finite_training_values',[label,step+1])
            outcomes=row['outcome_counts'];metrics=metrics_from_counts(outcomes,plan['complementarity'])
            audit.check(set(outcomes)=={'00','01','10','11'} and
                        all(isinstance(x,int) and x>=0 for x in outcomes.values()) and metrics['n']==512,
                        'training_four_outcomes_count_worlds',[label,step+1])
            audit.check(all(np.isclose(row[key],metrics[target],atol=1e-12,rtol=0) for key,target in
                [('reward','mean_reward'),('single_accuracy','single_accuracy'),('both_accuracy','both_accuracy'),
                 ('reward_variance','reward_variance'),('positive_rewards','positive_rewards')]),
                'training_native_utility_and_common_scores_from_outcomes',[label,step+1])
            key=(seed,identity,tuple(pool))
            if key not in world_cache:
                world_cache[key]=training_world(seed,identity,pool,{k:pools['train',k] for k in (0,1)})
            audit.check(row['world_sha256']==world_cache[key],'independent_training_world_digest',[label,step+1])
        curve=read(folder/'curve.json');lam=plan['complementarity']
        audit.check([r['update'] for r in curve]==CHECKPOINTS,'fixed_evaluation_checkpoints',label)
        for point in curve:
            audit.check(set(point['scores'])==set(MODES),'checkpoint_all_modes',[label,point['update']])
            for mode,value in point['scores'].items():
                groups=value['map_groups'];seen=1200*len(train_ids)//30;unseen=1200*len(held_ids)//30
                audit.check(value['episodes']==1200 and value['horizon']==1 and value['n']==1200 and
                    value['decisions']==2400 and groups['seen']['n']==seen and groups['unseen']['n']==unseen,
                    'balanced_checkpoint_budget',[label,point['update'],mode])
                for group in [value,*value['direction_metrics'],*value['map_groups'].values()]:
                    audit.check(exact_metric_match(group,metrics_from_counts(group['outcome_counts'],lam)),
                                'checkpoint_metrics_from_four_outcomes',[label,point['update'],mode])
        if seed not in evaluations:evaluations[seed]=evaluation_world(seed,{k:pools['test',k] for k in (0,1)})
        expected_hash,expected_world=evaluations[seed]
        audit.check(set(result['scores'])==set(MODES),'all_final_modes',label)
        for mode in MODES:
            stats=result['scores'][mode]
            audit.check(stats['episodes']==9600 and stats['horizon']==1 and stats['world_sha256']==expected_hash,
                        'balanced_final_world_digest',[label,mode])
            with np.load(folder/f'final_{mode}.npz') as z:
                trace_rows+=len(z['reward'])
                audit.check(all(np.array_equal(z[k],v) for k,v in expected_world.items()),
                            'independent_final_world_arrays',[label,mode])
                audit.check(z['history'].shape==(9600,18) and (z['history']==0).all() and
                            z['inventory'].shape==(9600,2) and (z['inventory']==0).all(),
                            'no_inventory_or_history_leak',[label,mode])
                audit.check(z['sent'].shape==(9600,2) and z['sent'].dtype==np.int64 and
                            ((z['sent']>=0)&(z['sent']<7)).all(),
                            'finite_integer_message_channel',[label,mode])
                delivered=z['sent'].copy()
                if plan['blocked'] or mode=='blank':delivered[:]=0
                elif mode=='shuffle':
                    for scout in (0,1):
                        rng=np.random.default_rng(seed+69200000+10001+scout)
                        for first in (0,1):
                            ix=np.flatnonzero((z['scout']==scout)&(z['goals'][:,0]==first))
                            delivered[ix]=z['sent'][rng.permutation(ix)]
                audit.check(np.array_equal(z['delivered'],delivered),'exact_delivered_message_intervention',[label,mode])
                audit.check(z['action'].shape==(9600,2) and z['action'].dtype==np.int64 and
                            ((z['action']>=0)&(z['action']<6)).all(), 'two_legal_actions_per_world',[label,mode])
                places=np.take_along_axis(z['menu'],z['action'][:,:,None],axis=2)[:,:,0]
                target=np.take_along_axis(z['positions'],z['goals'],axis=1)
                truth=(places==target).astype(np.int64)
                reward=(1-lam)*(truth[:,0]+truth[:,1])/2+lam*(truth[:,0]*truth[:,1])
                by_resource=np.empty_like(truth)
                for query in (0,1):by_resource[np.arange(9600),z['goals'][:,query]]=truth[:,query]
                outcomes=[str(f)+str(w) for f,w in by_resource]
                counts=dict(Counter(outcomes));counts={k:counts.get(k,0) for k in ('00','01','10','11')}
                audit.check(np.array_equal(z['place'],places) and np.array_equal(z['successes'],truth) and
                            np.array_equal(z['reward'],reward), 'independent_final_actions_and_joint_reward_recount',[label,mode])
                audit.check(exact_metric_match(stats,metrics_from_counts(counts,lam)),
                            'final_metrics_match_integer_outcomes',[label,mode])
                mapid=z['positions'][:,0]*5+z['positions'][:,1]-(z['positions'][:,1]>z['positions'][:,0])
                for group,pool in [('seen',train_ids),('unseen',held_ids)]:
                    ix=np.isin(mapid,pool)
                    c=Counter(np.asarray(outcomes)[ix]);c={k:c.get(k,0) for k in ('00','01','10','11')}
                    audit.check(exact_metric_match(stats['map_groups'][group],metrics_from_counts(c,lam)),
                                'exact_seen_unseen_outcomes',[label,mode,group])
                for scout in (0,1):
                    ix=z['scout']==scout
                    cells=Counter(zip(mapid[ix].tolist(),z['goals'][ix,0].tolist()))
                    audit.check(cells==Counter({(m,g):80 for m in range(30) for g in (0,1)}),
                                'each_map_first_goal_direction_has_80_worlds',[label,mode,scout])
                    c=Counter(np.asarray(outcomes)[ix]);c={k:c.get(k,0) for k in ('00','01','10','11')}
                    expected=metrics_from_counts(c,lam)
                    audit.check(exact_metric_match(stats['direction_metrics'][scout],expected) and
                                abs(stats['direction_means'][scout]-expected['mean_reward'])<=1e-12,
                                'direction_metrics_match_outcomes',[label,mode,scout])
    matched=[]
    for seed in SEEDS:
      for group in ('split1','split2','split3','full','blocked'):
        names=[f's{seed}_{group}_{kind}' for kind in REWARDS]
        if not all(name in rows_by_run for name in names):continue
        base=rows_by_run[names[0]]
        for name in names[1:]:
            other=rows_by_run[name]
            for a,b in zip(base,other):
                audit.check(all(a[k]==b[k] for k in ('update','batch_identity','active_sites','allowed_sites',
                    'map_pool','world_sha256','entropy_weight')),'reward_conditions_matched_worlds',[seed,group,a['update'],name])
        initial=[load(root/name/'initial.pt') for name in names]
        audit.check(all(equal_states(initial[0],x) for x in initial[1:]),
                    'reward_conditions_same_initial_parameters',[seed,group])
        matched.append(dict(seed=seed,group=group))
      for kind in REWARDS:
        full=f's{seed}_full_{kind}';blocked=f's{seed}_blocked_{kind}'
        if full in rows_by_run and blocked in rows_by_run:
            audit.check(equal_states(load(root/full/'initial.pt'),load(root/blocked/'initial.pt')),
                        'full_blocked_same_initial_state',[seed,kind])
            audit.check(all(all(a[k]==b[k] for k in ('world_sha256','batch_identity','entropy_weight','map_pool'))
                for a,b in zip(rows_by_run[full],rows_by_run[blocked])), 'full_blocked_same_world_stream',[seed,kind])
    if not args.allow_incomplete:
        audit.check(not pending and len(completed)==60,'all_60_runs_complete',pending)
        audit.check(len(people)==8,'all_eight_new_people_present')
        audit.check(len(matched)==20,'all_twenty_reward_triplets_complete')
    report=dict(status='failed' if audit.failures else 'incomplete' if pending or controls['pending'] else 'passed',planned_runs=60,
        completed_runs=len(completed),pending=pending,checks=dict(audit.counts),failures=audit.failures,
        train_updates_verified=audit.counts['independent_training_world_digest'],
        unique_training_batches_reconstructed=len(world_cache),final_trace_rows_recounted=trace_rows,final_primitive_decisions_recounted=2*trace_rows,
        matched_reward_triplets=matched,preparations=people,historical_preparations_compared=history_count,
        historical_overlap=overlaps,personal_controls=controls,frozen_source_hashes=hashes,audit_script_sha256=sha(__file__),
        boundaries=['No training or full optimizer replay. New prepared seed provenance is established by untouched sender/value fingerprints and source/state checks.',
          'All world reconstructions use independent six-site matching supports; each reward condition receives identical raw worlds at a given seed/split.',
          'Social reward means and variances allow 1e-12 reduction-order tolerance; exact integer counts govern success metrics and personal-control gates.',
          'Four independent subject pairs; splits, directions, episodes and reward conditions are repeated measurements, not additional independent pairs.',
          'The same old 44/16 photos remain exploratory development stimuli.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],completed=len(completed),pending=len(pending),
        failures=len(audit.failures),training_updates=report['train_updates_verified'],trace_rows=trace_rows,out=str(out))))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
