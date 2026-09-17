"""Independent saved-state/world/outcome audit; never trains the social agents."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'redesign_v0.9/audit_adaptation.py'
spec=importlib.util.spec_from_file_location('v9_independent_audit',HELPER)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
CONTROL_HELPER=ROOT.parent/'redesign_v0.8/audit_execution.py'
spec8=importlib.util.spec_from_file_location('v8_independent_audit',CONTROL_HELPER)
audit8=importlib.util.module_from_spec(spec8);spec8.loader.exec_module(audit8)
Audit,read,sha,load,same,state_hash,metrics,metric_match=(getattr(old,k) for k in
    ('Audit','read','sha','load','same','state_hash','metrics','metric_match'))
MAPS,MATCHINGS,SENDER,RECEIVER,MODES=old.MAPS,old.MATCHINGS,old.SENDER,old.RECEIVER,old.MODES
SEEDS=(29101,29102,29103,29104)
ARMS=('expand_both','stay_old_both','expand_sender','expand_receiver')
ALL_ARMS=('base',)+ARMS
PLAN=dict(split=0,representation='identity',schedule='direct',vocab=7,length=2,
          known=False,blocked=False,reward_kind='mixed',complementarity=.5)


def partitions(p):
    def matching(k):
        pairs={z for a,b in MATCHINGS[k] for z in ((a,b),(b,a))}
        return tuple(i for i,z in enumerate(MAPS) if tuple(z) in pairs)
    added,sealed=matching(p),matching(p%3+1)
    return dict(old=tuple(i for i in range(30) if i not in added+sealed),added=added,sealed=sealed)


def audit_partitions(audit):
    for p in (1,2,3):
        pools=partitions(p)
        audit.check([len(x) for x in pools.values()]==[18,6,6] and
                    set().union(*map(set,pools.values()))==set(range(30)) and
                    sum(len(x) for x in pools.values())==30,'partition_disjoint_complete',p)
        for name,ids in {**pools,'expanded':pools['old']+pools['added']}.items():
            for resource in (0,1):
                audit.check(np.array_equal(np.bincount(MAPS[list(ids),resource],minlength=6),
                            np.full(6,len(ids)//6)),'every_resource_marginal_balanced',[p,name,resource])
    audit.check(Counter(x for p in (1,2,3) for x in partitions(p)['added'])==
                Counter(x for p in (1,2,3) for x in partitions(p)['sealed']),
                'added_sealed_matching_roles_counterbalanced')


def seed_for(seed,p,purpose,step=0):
    return int(np.random.SeedSequence([10010,seed,p,purpose,step]).generate_state(1,dtype=np.uint64)[0])//2


def plasticity(arm):
    return {'base':'both','expand_both':'both','stay_old_both':'both',
            'expand_sender':'sender_only','expand_receiver':'receiver_only'}[arm]


def active_key(key,arm):return old.active_key(key,plasticity(arm))


def world(seed,n,photos,pool=None):
    rng=np.random.default_rng(seed);rows=[];h=hashlib.sha256()
    for who in (0,1):
        size=n//2
        if pool is not None:
            ids=rng.choice(np.asarray(pool),size);first=rng.integers(2,size=size)
        else:
            cells=np.tile(np.array(list(product(range(30),range(2)))),(size//60,1))[rng.permutation(size)]
            ids,first=cells.T
        row=dict(positions=MAPS[ids].copy(),photo_ids=np.column_stack([rng.choice(photos[k],size) for k in (0,1)]),
            goals=np.column_stack((first,1-first)),menu=np.argsort(rng.random((size,2,6)),axis=2),
            scout=np.full(size,who),episode=np.arange(size))
        for key in ('positions','photo_ids','goals','menu'):h.update(np.ascontiguousarray(row[key]).tobytes())
        rows.append(row)
    return h.hexdigest(),{k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def mapids(positions):return positions[:,0]*5+positions[:,1]-(positions[:,1]>positions[:,0])


def trace_audit(audit,path,stats,seed,p,mode,n,photos):
    label=[path.parent.name,mode];key=(seed,n)
    if key not in audit.eval_cache:audit.eval_cache[key]=world(seed,n,{k:photos['test',k] for k in (0,1)})
    digest,external=audit.eval_cache[key]
    audit.check(stats['world_sha256']==digest and stats['episodes']==n and stats['horizon']==1,
                'evaluation_world_hash_budget',label)
    with np.load(path) as z:
        audit.trace_rows+=len(z['reward'])
        audit.check(all(np.array_equal(z[k],v) for k,v in external.items()),'independent_evaluation_world_arrays',label)
        audit.check(z['inventory'].shape==(n,2) and not z['inventory'].any() and
                    z['history'].shape==(n,18) and not z['history'].any(),'zero_history_inventory',label)
        audit.check(z['sent'].shape==(n,2) and z['sent'].dtype==np.int64 and
                    ((z['sent']>=0)&(z['sent']<7)).all(),'two_legal_symbols',label)
        delivered=z['sent'].copy()
        if mode=='blank':delivered[:]=0
        elif mode=='shuffle':
            for who in (0,1):
                rng=np.random.default_rng(seed+10001+who)
                for first in (0,1):
                    ix=np.flatnonzero((z['scout']==who)&(z['goals'][:,0]==first))
                    delivered[ix]=z['sent'][rng.permutation(ix)]
        audit.check(np.array_equal(z['delivered'],delivered),'message_intervention_exact',label)
        audit.check(z['action'].shape==(n,2) and z['action'].dtype==np.int64 and
                    ((z['action']>=0)&(z['action']<6)).all(),'two_legal_actions',label)
        place=np.take_along_axis(z['menu'],z['action'][:,:,None],2)[:,:,0]
        success=(place==np.take_along_axis(z['positions'],z['goals'],1)).astype(np.int64)
        reward=(success[:,0]+success[:,1])/4+success.prod(1)/2
        audit.check(np.array_equal(z['place'],place) and np.array_equal(z['successes'],success) and
                    np.array_equal(z['reward'],reward),'actions_success_mixed_reward_recount',label)
        by_resource=np.take_along_axis(success,np.argsort(z['goals'],axis=1),1)
        code=2*by_resource[:,0]+by_resource[:,1];mids=mapids(z['positions'])
        def counts(mask):return {k:int((code[mask]==i).sum()) for i,k in enumerate(('00','01','10','11'))}
        audit.check(metric_match(stats,metrics(counts(np.ones(n,bool)))),'overall_metrics_recount',label)
        audit.check(stats['sealed_maps_never_in_this_social_training'] is True,'sealed_scope_label',label)
        for name,ids in partitions(p).items():
            ix=np.isin(mids,ids)
            audit.check(metric_match(stats['map_groups'][name],metrics(counts(ix))),
                        'old_added_sealed_metrics_recount',[*label,name])
            for who in (0,1):
                audit.check(metric_match(stats['direction_groups'][who][name],metrics(counts(ix&(z['scout']==who)))),
                            'direction_group_metrics_recount',[*label,who,name])
        for who in (0,1):
            ix=z['scout']==who;expected=metrics(counts(ix))
            audit.check(metric_match(stats['direction_metrics'][who],expected) and
                        stats['direction_means'][who]==expected['mean_reward'],'direction_metrics_recount',[*label,who])
            audit.check(Counter(zip(mids[ix].tolist(),z['goals'][ix,0].tolist()))==
                        Counter({(m,g):n//120 for m in range(30) for g in (0,1)}),
                        'balanced_120_evaluation_cells',[*label,who])


def audit_sources(audit,root,formal,pre_updates,updates,batch,eval_n):
    paths=sorted(root.glob('invocation_*.json'));audit.check(bool(paths),'invocation_present')
    hashes=read(paths[0])['source_hashes']
    for path in paths:
        inv=read(path)
        for key,value in dict(formal=formal,arms=list(ARMS),pre_updates=pre_updates,updates=updates,
                              batch=batch,eval_n=eval_n,source_hashes=hashes,device='cpu').items():
            audit.check(inv.get(key)==value,'invocation_fixed_fields',[path.name,key])
        if formal:audit.check(inv['seeds']==list(SEEDS) and inv['partitions']==[1,2,3],'fixed_four_seeds_three_partitions')
        values=[seed_for(s,p,k,u) for s in inv['seeds'] for p in inv['partitions']
                for k,n in ((1,pre_updates),(2,updates),(3,1)) for u in range(n)]
        audit.check(len(values)==len(set(values))==inv['unique_rng_count'],'all_derived_rng_unique',path.name)
    snapshot=root/f"sources_{hashes[str(ROOT/'run_generalization.py')][:12]}"
    for i,(name,digest) in enumerate(hashes.items()):
        audit.check(Path(name).is_file() and sha(name)==digest,'current_source_asset_hash',name)
        if Path(name).suffix in ('.py','.md'):
            audit.check(sha(snapshot/f'{i:02d}_{Path(name).name}')==digest,'executed_source_snapshot_hash',name)
    return hashes


def audit_one(audit,folder,seed,p,arm,updates,batch,eval_n,hashes,photos):
    label=folder.name;cfg=read(folder/'config.json');result=read(folder/'result.json');groups=partitions(p)
    pool=groups['old'] if arm in ('base','stay_old_both') else tuple(sorted(groups['old']+groups['added']))
    boundary=updates-min(300 if arm=='base' else 100,max(1,updates//(8 if arm=='base' else 6)))
    proposed=(0,100,300,600,1200,1800,2100,2400) if arm=='base' else (0,25,50,100,200,400,500,600)
    checkpoints=sorted({0,updates}|{x for x in proposed if x<updates});purpose=1 if arm=='base' else 2
    expected=dict(seed=seed,partition=p,arm=arm,plasticity=plasticity(arm),plan=PLAN,
        map_groups={k:list(v) for k,v in groups.items()},training_pool=list(pool),
        training_plan=dict(PLAN,map_pool=list(pool),allowed_sites=list(range(6))),updates=updates,batch=batch,
        eval_n=eval_n,checkpoints=checkpoints,learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=boundary,
        optimizer='fresh Adam',gradient_clip='norm 2 separately per active sender/receiver module',
        role_loss_reduction='mean of both roles, including frozen constants',rng_namespace=10010,
        training_purpose=purpose,evaluation_seed=seed_for(seed,p,3),source_hashes=hashes,
        sealed_never_trained=True,development_photos=True)
    for key,value in expected.items():audit.check(cfg.get(key)==value,'fixed_run_configuration',[label,key])
    initial,final=load(folder/'initial.pt'),load(folder/'final.pt')
    source=folder.parent/f's{seed}_p{p}_base/final.pt'
    expected_source=None if arm=='base' else dict(path=str(source),sha256=sha(source))
    source_matches=(cfg['source_checkpoint'] is None) if expected_source is None else (
        Path(cfg['source_checkpoint']['path']).resolve()==source.resolve() and cfg['source_checkpoint']['sha256']==expected_source['sha256'])
    audit.check(source_matches and cfg['source_checkpoint']==result['source_checkpoint'],'source_checkpoint_provenance',label)
    if arm!='base':audit.check(same(initial,load(source)),'initial_exact_common_base_endpoint',label)
    audit.check(state_hash(initial)==cfg['initial_sha256']==result['initial_sha256'],'initial_digest',label)
    audit.check(state_hash(final)==result['final_sha256'],'final_digest',label)
    prepared_path=folder.parent/f'prepared_{seed}.pt';prepared=load(prepared_path)
    audit.check(Path(cfg['prepared_source']['path']).resolve()==prepared_path.resolve() and
                cfg['prepared_source']['sha256']==sha(prepared_path),'prepared_source_digest',label)
    audit.check(all(torch.equal(s[k],prepared[who][k]) for who,s in enumerate(initial) for k in s if k.startswith('project.')),
                'project_exact_private_prepared_weights',label)
    for key,value in dict(seed=seed,partition=p,arm=arm,updates=updates,batch=batch,frozen_modules_verified=True,sealed_never_trained=True).items():
        audit.check(result.get(key)==value,'result_configuration',[label,key])
    initial_opt,final_opt=load(folder/'initial_optimizer.pt'),load(folder/'final_optimizer.pt')
    for who,state in enumerate(initial):
        active=[k for k in state if active_key(k,arm)];frozen=[k for k in state if k!='input_transform' and not active_key(k,arm)]
        partition=dict(active=active,frozen=frozen,trainable_count=sum(state[k].numel() for k in active))
        audit.check(cfg['partition_parameters'][who]==partition,'exact_plasticity_partition',[label,who])
        audit.check(partition['trainable_count']=={'both':209975,'sender_only':198234,'receiver_only':11741}[plasticity(arm)],
                    'expected_parameter_budget',[label,who])
        audit.check(torch.equal(state['input_transform'],torch.eye(6)),'identity_private_transform',[label,who])
        audit.check(all(torch.equal(state[k],final[who][k]) for k in (*frozen,'input_transform')),'all_frozen_endpoint_weights',[label,who])
        start,end=initial_opt[who],final_opt[who]
        audit.check(not start['state'],'fresh_adam_empty',[label,who])
        audit.check(len(end['state'])==len(active) and all(float(s['step'])==updates for s in end['state'].values()),
                    'adam_active_count_exact_update_steps',[label,who])
        for opt in (start,end):
            audit.check(sum(len(g['params']) for g in opt['param_groups'])==len(active) and
                all(g['lr']==.0007 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in opt['param_groups']),
                'adam_fixed_hyperparameters',[label,who])
    for point in checkpoints:
        states=load(folder/f'checkpoint_{point:04d}.pt')
        audit.check(all(torch.equal(v,states[who][k]) for who,s in enumerate(initial) for k,v in s.items() if not active_key(k,arm)),
                    'checkpoint_all_frozen_weights',[label,point])
        if point in (0,updates):audit.check(same(states,initial if point==0 else final),'checkpoint_endpoint_exact',[label,point])
    rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
    audit.check(len(rows)==updates,'training_budget',label);exposures=Counter()
    for step,row in enumerate(rows):
        rng=seed_for(seed,p,purpose,step);key=(rng,batch,pool)
        if key not in audit.world_cache:
            digest,external=world(rng,batch,{k:photos['train',k] for k in (0,1)},pool)
            ids=mapids(external['positions'])
            count={k:int(np.isin(ids,v).sum()) for k,v in groups.items()}
            audit.world_cache[key]=(digest,count)
        digest,count=audit.world_cache[key];exposures.update(count)
        audit.check(row['update']==step+1 and row['rng_seed']==rng and row['world_sha256']==digest,
                    'independent_training_world_hash',[label,step+1])
        audit.check(row['exposure']==count and count['sealed']==0 and sum(count.values())==batch and
                    (arm not in ('base','stay_old_both') or count['added']==0),'zero_sealed_exact_group_exposure',[label,step+1])
        weight=.02 if step<boundary else 0
        audit.check(row['entropy_weight']==weight,'fixed_exploration_weight',[label,step+1])
        audit.check(0<=row['both_accuracy']<=row['single_accuracy']<=1 and
                    np.isclose(row['mean_reward'],.5*(row['single_accuracy']+row['both_accuracy']),rtol=0,atol=1e-12),
                    'training_mixed_reward_consistency',[label,step+1])
        for who,info in enumerate(row['agents']):
            roles=('sender','receiver') if who==0 else ('receiver','sender');loss=0.
            for role,c in zip(roles,info['components']):
                loss+=(c['policy_loss']+c['value_loss']-weight*c['entropy'])/2
                audit.check(c['trainable']==(plasticity(arm)=='both' or plasticity(arm)==role+'_only'),
                            'role_gradient_partition',[label,step+1,who,role])
            audit.check(np.isfinite(info['loss']) and np.isclose(info['loss'],loss,atol=2e-6,rtol=1e-6),
                        'two_role_mean_loss',[label,step+1,who])
            roles={'sender','receiver'} if plasticity(arm)=='both' else {plasticity(arm).removesuffix('_only')}
            audit.check(set(info['gradient_norm_by_role'])==roles and all(np.isfinite(v) and v>=0 for v in info['gradient_norm_by_role'].values()),
                        'separate_active_module_norms',[label,step+1,who])
    curve=read(folder/'curve.json');audit.check([x['update'] for x in curve]==checkpoints,'fixed_curve_schedule',label)
    for point in curve:
        stats=point['scores']['normal']
        audit.check(point['frozen_modules_verified'] and set(point['scores'])=={'normal'},'curve_mode_and_freeze',[label,point['update']])
        audit.check(stats['n']==eval_n and stats['decisions']==2*eval_n and all(stats['map_groups'][g]['n']==eval_n*len(ids)//30 for g,ids in groups.items()),
                    'curve_balanced_group_budgets',[label,point['update']])
        for item in [stats,*stats['map_groups'].values(),*stats['direction_metrics'],*[x for d in stats['direction_groups'] for x in d.values()]]:
            audit.check(metric_match(item,metrics(item['outcome_counts'])),'curve_metrics_from_counts',[label,point['update']])
    audit.check(set(result['scores'])==set(MODES) and result['scores']['normal']==curve[-1]['scores']['normal'],
                'five_modes_last_curve_identical',label)
    for mode in MODES:trace_audit(audit,folder/f'final_{mode}.npz',result['scores'][mode],seed_for(seed,p,3),p,mode,eval_n,photos)
    return dict(seed=seed,partition=p,arm=arm,rows=rows,initial=initial,final=final,curve=curve,training_exposure=dict(exposures))


def paired_audit(audit,groups,first_step=False):
    for key,runs in groups.items():
        base=runs['base'];both=runs['expand_both']
        for arm in ARMS:
            r=runs[arm]
            audit.check(same(r['initial'],base['final']) and r['curve'][0]['scores']==base['curve'][-1]['scores'],
                        'all_four_arms_same_source_and_zero_step',[*key,arm])
            audit.check(all(a['rng_seed']==b['rng_seed'] and a['entropy_weight']==b['entropy_weight'] for a,b in zip(r['rows'],both['rows'])),
                        'four_arms_rng_identity_and_entropy_matched',[*key,arm])
            if arm!='stay_old_both':
                audit.check(all(a['world_sha256']==b['world_sha256'] and a['exposure']==b['exposure'] for a,b in zip(r['rows'],both['rows'])),
                            'three_expansion_arms_exact_worlds',[*key,arm])
        if first_step:
            for arm,prefix in (('expand_sender',SENDER),('expand_receiver',RECEIVER)):
                r=runs[arm]
                for who in (0,1):
                    keys=[k for k in r['initial'][who] if k.startswith(prefix)]
                    audit.check(all(torch.equal(r['final'][who][k],both['final'][who][k]) for k in keys),
                                'one_step_single_module_equals_both_exact',[*key,arm,who])
                    audit.check(any(not torch.equal(r['final'][who][k],r['initial'][who][k]) for k in keys),
                                'one_step_active_module_really_changes',[*key,arm,who])
                audit.check(all(r['rows'][0][k]==both['rows'][0][k] for k in
                            ('rng_seed','world_sha256','mean_reward','both_accuracy','single_accuracy','entropy_weight')),
                            'first_step_expansion_actions_returns_matched',[*key,arm])


def model_namespace(hashes):
    # Initialization only. No model inference or optimization is executed here.
    path=ROOT.parent/'redesign_v0.8/camp.py'
    assert sha(path)==hashes[str(path)]
    ns={'__file__':str(path),'__name__':'v10_audit_frozen_model_initializer'}
    exec(compile(path.read_text(),str(path),'exec'),ns)
    return ns


def preparation_audit(audit,root,ns,seeds):
    people=[];fresh={};personal_hashes=[];project_hashes=[]
    for seed in seeds:
        states=load(root/f'prepared_{seed}.pt');report=read(root/f'preparation_{seed}.json')
        random=[a.state_dict() for a in ns['make_agents'](seed)]
        audit.check(len(states)==2 and report['seed']==seed,'preparation_seed_count',seed)
        audit.check(all(x['updates']==200 and x['choices']==12800 and x['heldout_need_sensitive_choice']>=.9 for x in report['two_sites'])
                    and min(report['six_sites'])>=.9,'preparation_recorded_budget_and_gate',seed)
        for who,(state,before) in enumerate(zip(states,random)):
            audit.check(audit8.equal_states([state],[before],('sender.','value.')),
                        'prepared_untouched_parameters_match_new_seed',[seed,who])
            audit.check(not audit8.equal_states([state],[before],('project.',)) and
                        not audit8.equal_states([state],[before],('actor.',)),
                        'individual_practice_changed_project_actor',[seed,who])
            ph=state_hash([state]);zh=audit8.state_hash([state],'project.')
            personal_hashes.append(ph);project_hashes.append(zh)
            people.append(dict(seed=seed,agent=who,personal_torch_seed=seed*1000+who*137,
                               state_sha256=ph,project_sha256=zh))
        fresh[seed]=[a.state_dict() for a in ns['remake_agents'](seed,states,7,2,'identity')]
    audit.check(len(set(personal_hashes))==len(personal_hashes) and len(set(project_hashes))==len(project_hashes),
                'preparations_and_projections_within_batch_unique')
    overlaps=[];history_count=0
    search_roots=[ROOT.parent/f'redesign_v0.{v}'/'results' for v in range(4,11)]
    search_roots.append(ROOT.parent/'research_program/policy_gain_study/results')
    paths=sorted({x.resolve() for r in search_roots for x in r.glob('**/prepared*.pt') if root not in x.resolve().parents})
    for path in paths:
        states=load(path)
        if isinstance(states,dict):states=[states]
        for who,state in enumerate(states):
            if not isinstance(state,dict) or 'project.0.weight' not in state:continue
            history_count+=1
            if state_hash([state]) in personal_hashes or audit8.state_hash([state],'project.') in project_hashes:
                overlaps.append(dict(path=str(path),agent=who))
    audit.check(not overlaps,'no_historical_preparation_projection_overlap',overlaps)
    return fresh,dict(people=people,historical_people_compared=history_count,historical_overlap=overlaps,
        scope='Recorded 200x64 preparation budget, exact untouched random fingerprints and distinct fitted projections; preparation optimization was not replayed.')


def controls_audit(audit,root,ns,photos,hashes,fresh,seeds):
    results=[];gates=[];trace_rows=0
    names=('camp.py','run_experiment.py','run_controls.py','固定执行方案.md')
    expected_hashes={name:hashes[str(ROOT.parent/'redesign_v0.8'/name)] for name in names}
    for seed in seeds:
        folder=root/f'individual_controls/s{seed}_identity';cfg=read(folder/'config.json');result=read(folder/'result.json')
        results.append(result);initial=load(folder/'initial.pt');final=load(folder/'final.pt')
        label=folder.name
        expected=dict(seed=seed,representation='identity',updates=2400,batch=512,eval_n_per_agent=9600,
            gate=.9,fitted_agent_prefixes=['memory.','slot_phi.'],optimizer='Adam',learning_rate=.0007,
            entropy_coefficient=.02,entropy_off_after=2100,source_hashes=expected_hashes)
        for key,value in expected.items():audit.check(cfg.get(key)==value,'fixed_personal_control_config',[label,key])
        audit.check(result['complete'] and result['seed']==seed and result['representation']=='identity' and
                    result['frozen_unused_parameters_verified'],'personal_control_complete_identity',label)
        audit.check(same(initial['agents'],fresh[seed]) and state_hash(initial['agents'])==cfg['initial_sha256'],
                    'personal_control_starts_exact_fresh_social_initial',label)
        agents=ns['remake_agents'](seed,load(root/f'prepared_{seed}.pt'),7,2,'identity')
        audit.check(audit8.trainable_hash(initial['agents'],agents)==cfg['common_initial_sha256'],
                    'personal_common_trainable_hash',label)
        audit.check(Path(cfg['prepared_source']['path']).resolve()==(root/f'prepared_{seed}.pt').resolve() and
                    cfg['prepared_source']['sha256']==sha(root/f'prepared_{seed}.pt'),'personal_prepared_source',label)
        audit.check(cfg['input_transforms']==[torch.eye(6).tolist()]*2,'personal_identity_transform_config',label)
        opts=load(folder/'final_optimizer.pt')
        for who in (0,1):
            torch.manual_seed(seed*1000+881+who)
            head=torch.nn.Sequential(torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,12))
            audit.check(same([head.state_dict()],[initial['heads'][who]]),'personal_head_exact_initial_seed',[label,who])
            unchanged=[k for k in initial['agents'][who] if not k.startswith(('memory.','slot_phi.'))]
            audit.check(all(torch.equal(initial['agents'][who][k],final['agents'][who][k]) for k in unchanged),
                        'personal_all_unused_weights_frozen',[label,who])
            audit.check(any(not torch.equal(initial['agents'][who][k],final['agents'][who][k]) for k in initial['agents'][who]
                            if k.startswith(('memory.','slot_phi.'))),'personal_diagnostic_path_did_learn',[label,who])
            count=len(list(head.parameters()))+sum(k.startswith(('memory.','slot_phi.')) for k,_ in agents[who].named_parameters())
            opt=opts[who]
            audit.check(len(opt['state'])==count and all(float(s['step'])==2400 for s in opt['state'].values()) and
                        all(g['lr']==.0007 for g in opt['param_groups']),'personal_optimizer_budget',[label,who])
        rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        audit.check(len(rows)==2400,'personal_training_updates',label)
        for update,row in enumerate(rows,1):
            audit.check(row['update']==update and [x['agent'] for x in row['agents']]==[0,1],'personal_training_row_identity',[label,update])
            for who,info in enumerate(row['agents']):
                digest,_=audit8.personal_world(seed,who,update,{k:photos['train',k] for k in (0,1)})
                audit.check(info['world_sha256']==digest,'independent_personal_training_world_hash',[label,who,update])
                audit.check(all(np.isfinite(info[k]) for k in ('reward','loss','gradient_norm')) and 0<=info['reward']<=1,
                            'personal_training_finite_values',[label,who,update])
        curve=read(folder/'curve.json');audit.check([x['update'] for x in curve]==[0,600,1200,1800,2400],'personal_curve_schedule',label)
        for point in curve:
            audit.check(set(point['scores'])=={'normal','erase_memory'},'personal_curve_modes',[label,point['update']])
            for mode,score in point['scores'].items():
                counts=score['per_agent']
                audit.check(len(counts)==2 and all(x['n']==1200 and x['accuracy']==x['correct']/1200 for x in counts) and
                            score['mean_accuracy']==sum(x['accuracy'] for x in counts)/2,'personal_curve_counts',[label,point['update'],mode])
        for mode in ('normal','erase_memory'):
            counts=[]
            with np.load(folder/f'final_{mode}.npz') as z:
                trace_rows+=len(z['reward'])
                audit.check(len(z['reward'])==19200 and np.array_equal(z['agent'],np.repeat([0,1],9600)),
                            'personal_final_budget',[label,mode])
                for who in (0,1):
                    ix=z['agent']==who
                    _,external=audit8.personal_world(seed,who,0,{k:photos['test',k] for k in (0,1)},False)
                    audit.check(all(np.array_equal(z[k][ix],v) for k,v in external.items()),'personal_final_worlds_exact',[label,who,mode])
                    action=z['action'][ix]
                    audit.check(action.dtype==np.int64 and ((action>=0)&(action<6)).all(),'personal_actions_legal',[label,who,mode])
                    place=z['menu'][ix][np.arange(9600),action]
                    truth=place==z['positions'][ix][np.arange(9600),z['goals'][ix]];correct=int(truth.sum())
                    counts.append(dict(n=9600,correct=correct,accuracy=correct/9600))
                    audit.check(np.array_equal(z['place'][ix],place) and np.array_equal(z['reward'][ix],truth),
                                'personal_final_reward_recount',[label,who,mode])
                audit.check(result['scores'][mode]==dict(per_agent=counts,mean_accuracy=sum(x['accuracy'] for x in counts)/2),
                            'personal_final_metrics_exact',[label,mode])
            if mode=='normal':
                passed=all(x['correct']*10>=x['n']*9 for x in counts)
                audit.check(result['passed']==passed and passed,'personal_integer_gate_passed',label)
                gates.append(dict(seed=seed,counts=counts,passed=passed))
    summary=read(root/'individual_controls/summary.json')
    audit.check(summary==dict(complete=True,passed=True,runs=len(seeds),results=results,diagnostic_weights_never_transferred=True),
                'personal_summary_all_four_results')
    return dict(completed=len(results),gates=gates,training_person_batches_verified=audit.counts['independent_personal_training_world_hash'],
                final_world_rows_recounted=trace_rows,source_hashes=expected_hashes,
                scope='Independent full30 personal diagnostic copies; all source/base comparisons use their untrained initial states, never their trained endpoint.')


def write_report(audit,out,**extra):
    report=dict(passed=not audit.failures,checks=dict(audit.counts),total_checks=sum(audit.counts.values()),
        failures=audit.failures,training_world_hashes_verified=audit.counts['independent_training_world_hash'],
        unique_training_batches=len(audit.world_cache),evaluation_world_rows=audit.trace_rows,
        evaluation_primitive_decisions=2*audit.trace_rows,audit_script_sha256=sha(__file__),
        shared_independent_helper_sha256=sha(HELPER),control_helper_sha256=sha(CONTROL_HELPER),
        completed_utc=datetime.now(timezone.utc).isoformat(),**extra)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=report['total_checks'],failures=len(audit.failures),out=str(out)),ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/generalization_001');p.add_argument('--out',type=Path)
    args=p.parse_args();root=args.root.resolve();out=args.out or root/'audit_execution.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);audit=Audit();audit_partitions(audit);photos=old.get_pools(audit)
    hashes=audit_sources(audit,root,True,2400,600,512,9600);gate=read(ROOT/'preflight_qa.json')
    audit.check(gate['passed'] and gate['runner_sha256']==hashes[str(ROOT/'run_generalization.py')],'preflight_gate_current_source')
    ns=model_namespace(hashes);fresh,preparations=preparation_audit(audit,root,ns,SEEDS)
    controls=controls_audit(audit,root,ns,photos,hashes,fresh,SEEDS)
    expected={f's{s}_p{p}_{a}' for s in SEEDS for p in (1,2,3) for a in ALL_ARMS}
    actual={x.name for x in root.glob('s*') if x.is_dir() and x.name[1:2].isdigit()}
    audit.check(actual==expected,'exact_sixty_social_directories')
    groups={};summaries=[]
    for seed in SEEDS:
      for p in (1,2,3):
        runs={a:audit_one(audit,root/f's{seed}_p{p}_{a}',seed,p,a,2400 if a=='base' else 600,512,9600,hashes,photos) for a in ALL_ARMS}
        groups[seed,p]=runs
        summaries.extend({k:v for k,v in r.items() if k not in ('rows','initial','final','curve')} for r in runs.values())
    paired_audit(audit,groups)
    for seed in SEEDS:
        states=[groups[seed,p]['base']['initial'] for p in (1,2,3)]
        audit.check(all(same(states[0],s) for s in states[1:]),'three_base_partitions_identical_initialization',seed)
        audit.check(same(states[0],fresh[seed]),'base_exact_new_random_social_interface',seed)
        control_initial=load(root/f'individual_controls/s{seed}_identity/initial.pt')['agents']
        control_final=load(root/f'individual_controls/s{seed}_identity/final.pt')['agents']
        audit.check(same(states[0],control_initial) and not same(states[0],control_final),
                    'social_initial_does_not_inherit_trained_personal_diagnostic',seed)
    report=write_report(audit,out,completed_social_runs=len(summaries),sources_and_exposures=summaries,source_hashes=hashes,
        preparations=preparations,personal_controls=controls,
        scope='All 60 social runs and 4 independent personal controls. Saved weights, RNG worlds, role losses and outcomes audited; no optimizer replay or new model inference. Four independent pairs with three repeated partitions; old photos are development data.')
    if not report['passed']:sys.exit(1)


if __name__=='__main__':main()
