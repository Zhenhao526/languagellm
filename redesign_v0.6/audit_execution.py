"""Independent read-only audit of all 56 six-site runs and the oracle gate.

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
SEEDS=(25101,25102,25103,25104)
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}
CONDITIONS={f'split{s}_{kind}':dict(split=s,schedule='direct' if kind=='atomic_direct' else kind,
    vocab=49 if kind=='atomic_direct' else 7,length=1 if kind=='atomic_direct' else 2,
    known=False,blocked=False) for s in MATCHINGS for kind in ('course','mixed','direct','atomic_direct')}
CONDITIONS.update(full_direct=dict(split=0,schedule='direct',vocab=7,length=2,known=False,blocked=False),
                  full_blocked=dict(split=0,schedule='direct',vocab=7,length=2,known=False,blocked=True))
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
    u=np.random.default_rng(seed+37120000+identity).random()
    index=int(np.searchsorted(np.cumsum(weights/weights.sum()),u,side='right'))
    return options[min(index,len(options)-1)]


def schedule(seed,kind):
    levels=np.asarray([2]*600+[4]*600+[6]*1200)
    order=np.arange(2400)
    if kind=='mixed':order[:2100]=np.random.default_rng(seed+37110000).permutation(2100)
    elif kind!='course':levels[:]=6
    return levels[order],order


def world_hash(rows):
    h=hashlib.sha256()
    for row in rows:
        for k in ('positions','photo_ids','goals','menu','refill_uniform'):
            h.update(np.ascontiguousarray(row[k]).tobytes())
    return h.hexdigest()


def training_world(seed,identity,pool,photo_pools):
    rng=np.random.default_rng(seed*100000+30000000+identity+1)
    rows=[]
    for scout in (0,1):
        positions=MAPS[rng.choice(np.asarray(pool,np.int64),256)].copy()
        goals=rng.integers(2,size=256)
        ids=np.column_stack([rng.choice(photo_pools[k],256) for k in (0,1)])
        menu=np.argsort(rng.random((256,6)),axis=1)
        rows.append(dict(positions=positions,goals=goals,photo_ids=ids,menu=menu,refill_uniform=rng.random(256)))
    return world_hash(rows)


def evaluation_world(seed,photo_pools):
    rng=np.random.default_rng(seed+39200000);rows=[]
    for scout in (0,1):
        ix=np.tile(np.asarray(list(product(range(30),range(2)))),(80,1))
        ix=ix[rng.permutation(4800)]
        ids=np.column_stack([rng.choice(photo_pools[k],4800) for k in (0,1)])
        menu=np.argsort(rng.random((4800,6)),axis=1)
        rows.append(dict(positions=MAPS[ix[:,0]].copy(),goals=ix[:,1],photo_ids=ids,menu=menu,
                         refill_uniform=rng.random(4800),scout=np.full(4800,scout)))
    arrays={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return world_hash(rows),arrays


def oracle_gate_audit(audit):
    failures_before=len(audit.failures)
    root=ROOT/'results/receiver_control_99102'
    cfg=read(root/'config.json');original=read(root/'result_float32_gate.json');corrected=read(root/'result.json')
    audit.check(cfg['seed']==99102 and cfg['updates']==2400 and cfg['batch']==512 and cfg['gate']==.9,
                'oracle_fixed_design')
    audit.check(sha(root/'diagnostic_receiver_executed.py')==cfg['source_hashes']['diagnostic_receiver.py'],
                'oracle_executed_source_preserved')
    expected_map=np.random.default_rng(399102).permutation(49)[:30]
    audit.check(cfg['mapping']==expected_map.tolist(),'oracle_fixed_random_codebook')
    audit.check(original['complete'] and not original['passed'] and corrected['complete'] and corrected['passed'],
                'oracle_original_failed_rounding_record_preserved')
    counts=[]
    for name in ('sequence','atomic'):
        folder=root/name
        logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        audit.check(len(logs)==2400 and [r['update'] for r in logs]==list(range(1,2401)),
                    'oracle_no_extra_recorded_updates',name)
        a,b=load(folder/'initial.pt'),load(folder/'final.pt')
        prefixes=('receive_embedding.','actor.','receive_value.')
        audit.check(all(torch.equal(v,b[who][k]) for who,s in enumerate(a)
                         for k,v in s.items() if not k.startswith(prefixes)),
                    'oracle_only_receiver_parameters_changed',name)
        with np.load(folder/'final_normal.npz') as z:
            codes=np.column_stack((expected_map//7,expected_map%7)) if name=='sequence' else expected_map[:,None]
            audit.check(np.array_equal(z['positions'],MAPS[z['maps']]) and
                        np.array_equal(z['delivered'],codes[z['maps']]),'oracle_saved_code_route',name)
            truth=(z['place']==MAPS[z['maps'],z['goals']]).astype(np.int64)
            audit.check(np.array_equal(truth,z['reward']),'oracle_reward_independent_recount',name)
            for who in (0,1):
                mask=z['agent']==who;n=int(mask.sum());correct=int(truth[mask].sum())
                ratio=correct/n
                audit.check(n==9600 and 10*correct>=9*n,'oracle_exact_integer_90pct_gate',[name,who])
                audit.check(corrected['conditions'][name]['counts'][who]==dict(correct=correct,n=n) and
                            corrected['conditions'][name]['per_agent'][who]==ratio,
                            'oracle_corrected_counts_match_trace',[name,who])
                old=float(z['reward'][mask].mean(dtype=np.float32))
                audit.check(original['conditions'][name]['per_agent'][who]==old,
                            'oracle_original_float32_reproduced',[name,who])
                cells=Counter(zip(z['maps'][mask].tolist(),z['goals'][mask].tolist()))
                audit.check(cells==Counter({(m,g):160 for m in range(30) for g in (0,1)}),
                            'oracle_balanced_evaluation',[name,who])
                counts.append(dict(condition=name,agent=who,correct=correct,n=n,exact_rate=ratio,
                                   original_float32=old,passed=10*correct>=9*n))
    return dict(correction_verified=len(audit.failures)==failures_before,counts=counts,
        interpretation='Recount of preserved trajectories at unchanged 90% gate; no training is performed by this audit. No extra updates appear in recorded logs.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'results/recombination_001')
    p.add_argument('--out',type=Path)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args();root=args.root.resolve();out=args.out or root/'audit_execution.json'
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
    ns={'__file__':str(ROOT/'camp.py'),'__name__':'six_site_audit_frozen_camp'}
    exec(compile((snapshot/'camp.py').read_text(),str(snapshot/'camp.py'),'exec'),ns)
    entries=read(ROOT.parent/'redesign_v0.4/data/manifest.json')['images']
    pools={(split,k):np.asarray([i for i,e in enumerate(entries) if e['split']==split and e['category']==name],np.int64)
        for split in ('train','test') for k,name in enumerate(('food','water'))}
    audit.check([len(pools[s,k]) for s in ('train','test') for k in (0,1)]==[22,22,8,8],'photo_pool_sizes')
    train_shas={e['sha256'] for e in entries if e['split']=='train'}
    test_shas={e['sha256'] for e in entries if e['split']=='test'}
    audit.check(not(train_shas&test_shas),'photo_split_disjoint')
    oracle=oracle_gate_audit(audit)

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
    for version in ('redesign_v0.4','redesign_v0.5','redesign_v0.6'):
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
        if not(folder/'result.json').exists():pending.append(label);continue
        completed.append(label);cfg=read(folder/'config.json');result=read(folder/'result.json')
        train_ids,held_ids=maps_for(plan['split']);levels,identities=schedule(seed,plan['schedule'])
        expected=dict(seed=seed,condition=name,plan=plan,updates=2400,batch=512,eval_n=9600,
            checkpoint_eval_n=1200,checkpoints=CHECKPOINTS,sites=6,history_dim=18,
            train_map_ids=list(train_ids),heldout_map_ids=list(held_ids),map_table=MAPS.tolist(),
            learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=2100,gamma=1,
            training_seed_offset=30000000,source_hashes={Path(p).name:h for p,h in hashes.items() if Path(p).parent==ROOT})
        for k,v in expected.items():audit.check(cfg.get(k)==v,'fixed_run_config',[label,k])
        for k,v in dict(seed=seed,condition=name,plan=plan,updates=2400,batch=512,frozen_projection_verified=True).items():
            audit.check(result.get(k)==v,'result_identity',[label,k])
        initial=load(folder/'initial.pt');final=load(folder/'final.pt')
        audit.check(state_hash(initial)==cfg['initial_sha256']==result['initial_sha256'],'initial_digest',label)
        audit.check(state_hash(final)==result['final_sha256'],'final_digest',label)
        audit.check(equal_states(initial,final,('project.',)) and equal_states(initial,prepared[seed],('project.',)),
                    'frozen_prepared_projection',label)
        agents=ns['remake_agents'](seed,prepared[seed],plan['vocab'],plan['length'])
        audit.check(equal_states(initial,[a.state_dict() for a in agents]),'new_random_social_interface_exact',label)
        audit.check(cfg['trainable_parameters']==[sum(p.numel() for p in a.parameters() if p.requires_grad) for a in agents],
                    'trainable_parameter_count',label)
        source=cfg['prepared_source'];path=root/f'prepared_{seed}.pt'
        audit.check(Path(source['path']).resolve()==path.resolve() and source['sha256']==sha(path),'prepared_source_digest',label)
        for point in CHECKPOINTS:audit.check((folder/f'checkpoint_{point:04d}.pt').exists(),'checkpoint_present',[label,point])
        audit.check(equal_states(initial,load(folder/'checkpoint_0000.pt')) and
                    equal_states(final,load(folder/'checkpoint_2400.pt')),'checkpoint_endpoints_match',label)
        for who,opt in enumerate(load(folder/'final_optimizer.pt')):
            audit.check(bool(opt['state']) and all(g['lr']==.0007 for g in opt['param_groups']) and
                        all(float(s['step'])==2400 for s in opt['state'].values()),'fresh_adam_update_budget',[label,who])
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
            key=(seed,identity,tuple(pool))
            if key not in world_cache:
                world_cache[key]=training_world(seed,identity,pool,{k:pools['train',k] for k in (0,1)})
            audit.check(row['world_sha256']==world_cache[key],'independent_training_world_digest',[label,step+1])
        curve=read(folder/'curve.json')
        audit.check([r['update'] for r in curve]==CHECKPOINTS,'fixed_evaluation_checkpoints',label)
        for point in curve:
            audit.check(set(point['scores'])==set(MODES),'checkpoint_all_modes',[label,point['update']])
            for mode,value in point['scores'].items():
                groups=value['map_groups'];seen=1200*len(train_ids)//30;unseen=1200*len(held_ids)//30
                audit.check(value['episodes']==1200 and value['horizon']==1 and groups['seen']['n']==seen and
                            groups['unseen']['n']==unseen,'balanced_checkpoint_budget',[label,point['update'],mode])
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
                            (z['inventory']==0).all() and (z['step']==0).all(),
                            'no_inventory_or_history_leak',[label,mode])
                audit.check(z['sent'].shape==(9600,plan['length']) and
                            z['sent'].dtype==np.int64 and ((z['sent']>=0)&(z['sent']<plan['vocab'])).all(),
                            'finite_integer_message_channel',[label,mode])
                delivered=z['sent'].copy()
                if plan['blocked'] or mode=='blank':delivered[:]=0
                elif mode=='shuffle':
                    for scout in (0,1):
                        rng=np.random.default_rng(seed+39200000+10001+scout)
                        for goal in (0,1):
                            ix=np.flatnonzero((z['scout']==scout)&(z['goals']==goal))
                            delivered[ix]=z['sent'][rng.permutation(ix)]
                audit.check(np.array_equal(z['delivered'],delivered),'exact_delivered_message_intervention',[label,mode])
                places=z['menu'][np.arange(9600),z['action']]
                picked=(z['positions']==places[:,None]).astype(np.int64)
                truth=picked[np.arange(9600),z['goals']]
                after=picked.copy();after[np.arange(9600),z['goals']]-=truth
                audit.check(np.array_equal(z['place'],places) and np.array_equal(z['reward'],truth) and
                            np.array_equal(z['gathered'],picked) and np.array_equal(z['next_inventory'],after) and
                            np.array_equal(z['next_positions'],z['positions']) and (z['overflow']==0).all(),
                            'independent_final_action_and_resource_recount',[label,mode])
                audit.check(abs(stats['mean_reward']-float(truth.mean()))<=1e-7,
                            'final_report_mean_matches_integer_counts',[label,mode])
                mapid=z['positions'][:,0]*5+z['positions'][:,1]-(z['positions'][:,1]>z['positions'][:,0])
                for group,pool in [('seen',train_ids),('unseen',held_ids)]:
                    ix=np.isin(mapid,pool);n=int(ix.sum());correct=int(truth[ix].sum())
                    audit.check(stats['map_groups'][group]==dict(n=n,correct=correct,mean_reward=correct/n if n else None),
                                'exact_seen_unseen_counts',[label,mode,group])
                for scout in (0,1):
                    ix=z['scout']==scout
                    cells=Counter(zip(mapid[ix].tolist(),z['goals'][ix].tolist()))
                    audit.check(cells==Counter({(m,g):80 for m in range(30) for g in (0,1)}),
                                'each_map_goal_direction_has_80_cases',[label,mode,scout])
                    audit.check(abs(stats['direction_means'][scout]-float(truth[ix].mean()))<=1e-7,
                                'direction_means_match_counts',[label,mode,scout])
    matched=[]
    for seed in SEEDS:
      for split in (1,2,3):
        akey=f's{seed}_split{split}_course';bkey=f's{seed}_split{split}_mixed'
        if akey not in rows_by_run or bkey not in rows_by_run:continue
        a=rows_by_run[akey];b=rows_by_run[bkey];index={r['batch_identity']:r for r in b}
        for row in a:
            other=index[row['batch_identity']]
            audit.check(all(row[k]==other[k] for k in ('batch_identity','active_sites','allowed_sites','map_pool','world_sha256','entropy_weight')),
                        'same_curriculum_batch_world_and_entropy',[seed,split,row['batch_identity']])
        audit.check(all(a[i]['batch_identity']==b[i]['batch_identity'] for i in range(2100,2400)),
                    'same_final_300_order',[seed,split])
        matched.append(dict(seed=seed,split=split))
    if not args.allow_incomplete:
        audit.check(not pending and len(completed)==56,'all_56_runs_complete',pending)
        audit.check(len(people)==8,'all_eight_new_people_present')
        audit.check(len(matched)==12,'all_twelve_course_mixed_pairs_complete')
    report=dict(status='failed' if audit.failures else 'incomplete' if pending else 'passed',planned_runs=56,
        completed_runs=len(completed),pending=pending,checks=dict(audit.counts),failures=audit.failures,
        train_updates_verified=audit.counts['independent_training_world_digest'],
        unique_training_batches_reconstructed=len(world_cache),final_trace_rows_recounted=trace_rows,
        matched_course_mixed=matched,preparations=people,historical_preparations_compared=history_count,
        historical_overlap=overlaps,oracle_gate=oracle,frozen_source_hashes=hashes,audit_script_sha256=sha(__file__),
        boundaries=['No training or full optimizer replay. New prepared seed provenance is established by untouched sender/value fingerprints and source/state checks.',
          'All world reconstructions use independently implemented six-site matching supports and weighted access sampling.',
          'Float32 reported aggregate means allow 1e-7 representation tolerance; exact correct/n counts govern the 90% diagnostic gate.',
          'Four independent subject pairs; splits, directions, episodes and random recoding references are repeated measurements, not additional independent pairs.',
          'The same old 44/16 photos remain exploratory development stimuli.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],completed=len(completed),pending=len(pending),
        failures=len(audit.failures),training_updates=report['train_updates_verified'],trace_rows=trace_rows,out=str(out))))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
