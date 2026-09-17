"""Saved-state, RNG, loss-weight and outcome audit. No training/model inference."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'redesign_v0.10/audit_execution.py'
spec=importlib.util.spec_from_file_location('v10_saved_audit',HELPER)
v10=importlib.util.module_from_spec(spec);spec.loader.exec_module(v10)
Audit,read,sha,load,same,state_hash,metrics,metric_match=(getattr(v10,k) for k in
    ('Audit','read','sha','load','same','state_hash','metrics','metric_match'))
MAPS=v10.MAPS;SEEDS=(29101,29102,29103,29104);ARMS=('current','anchor','release')
VARIANTS=('current','current_to_old','old_to_current');TAGS=('old_s','old_r','new')
STREAMS={'old_s':11,'old_r':21,'new':31};MODES=('normal','shuffle','blank','stochastic','erase_memory')
TIMES=(0,25,50,100,200,300,325,350,400,500,600)


def seed_for(seed,p,purpose,step=0,who=0):
    return int(np.random.SeedSequence([11011,seed,p,purpose,step,who]).generate_state(1,dtype=np.uint64)[0])//2


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def world(seed,p,step,purpose,n,photos,pool=None):
    rows=[];h=hashlib.sha256()
    for who in (0,1):
        rng=np.random.default_rng(seed_for(seed,p,purpose,step,who));size=n//2
        if pool is None:
            cells=np.tile(np.array(list(product(range(30),range(2)))),(size//60,1))[rng.permutation(size)]
            ids,first=cells.T
        else:ids=rng.choice(np.array(pool),size);first=rng.integers(2,size=size)
        row=dict(positions=MAPS[ids].copy(),photo_ids=np.column_stack([rng.choice(photos[k],size) for k in (0,1)]),
            goals=np.column_stack((first,1-first)),menu=np.argsort(rng.random((size,2,6)),axis=2),
            scout=np.full(size,who),episode=np.arange(size))
        for key in ('positions','photo_ids','goals','menu'):h.update(np.ascontiguousarray(row[key]).tobytes())
        rows.append(row)
    return h.hexdigest(),{k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def check_stats(audit,stats,p,n,label):
    groups=v10.partitions(p)
    audit.check(stats['n']==n and stats['decisions']==2*n,'stats_communication_action_budget',label)
    audit.check(stats['sealed_maps_never_in_this_social_training'] is True,'stats_sealed_scope',label)
    all_stats=[stats,*stats['map_groups'].values(),*[x for d in stats['direction_groups'] for x in d.values()]]
    for item in all_stats:audit.check(metric_match(item,metrics(item['outcome_counts'])),'metrics_from_four_integer_outcomes',label)
    audit.check(sum(stats['map_groups'][g]['n'] for g in groups)==n and
                all(sum(d[g]['n'] for g in groups)==n//2 for d in stats['direction_groups']),
                'groups_partition_communications',label)
    for name,pool in groups.items():
        summed={k:sum(stats['direction_groups'][who][name]['outcome_counts'][k] for who in (0,1)) for k in ('00','01','10','11')}
        audit.check(summed==stats['map_groups'][name]['outcome_counts'],'group_directions_add_up',[label,name])
    total={k:sum(stats['map_groups'][g]['outcome_counts'][k] for g in groups) for k in ('00','01','10','11')}
    audit.check(total==stats['outcome_counts'],'three_groups_add_to_total',label)


def trace_audit(audit,path,stats,external,digest,seed,p,step,stream,mode,label,evaluation):
    with np.load(path) as z:
        n=len(external['positions']);audit.trace_rows+=n
        audit.check(stats['world_sha256']==digest and all(np.array_equal(z[k],v) for k,v in external.items()),
                    'saved_trace_exact_independent_world',label)
        audit.check(z['inventory'].shape==(n,2) and not z['inventory'].any() and
                    z['history'].shape==(n,18) and not z['history'].any(),'zero_private_history_inventory',label)
        audit.check(z['sent'].shape==(n,2) and z['sent'].dtype==np.int64 and ((z['sent']>=0)&(z['sent']<7)).all(),
                    'finite_two_symbol_channel',label)
        delivered=z['sent'].copy()
        if mode=='blank':delivered[:]=0
        elif mode=='shuffle':
            for who in (0,1):
                rng=np.random.default_rng(seed_for(seed,p,stream+2,step,who))
                for first in (0,1):
                    ix=np.flatnonzero((z['scout']==who)&(z['goals'][:,0]==first))
                    delivered[ix]=z['sent'][rng.permutation(ix)]
        audit.check(np.array_equal(z['delivered'],delivered),'actual_message_intervention',label)
        audit.check(z['action'].shape==(n,2) and z['action'].dtype==np.int64 and ((z['action']>=0)&(z['action']<6)).all(),
                    'two_legal_resource_actions',label)
        place=np.take_along_axis(z['menu'],z['action'][:,:,None],2)[:,:,0]
        correct=(place==np.take_along_axis(z['positions'],z['goals'],1)).astype(np.int64)
        reward=(correct[:,0]+correct[:,1])/4+correct.prod(1)/2
        audit.check(np.array_equal(z['place'],place) and np.array_equal(z['successes'],correct) and np.array_equal(z['reward'],reward),
                    'trace_actions_and_mixed_reward_recount',label)
        behavioral=hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in
            ('sent','delivered','action','place','successes','reward'))).hexdigest()
        audit.check(stats['trace_sha256']==behavioral,'recorded_behavior_hash',label)
        by_resource=np.take_along_axis(correct,np.argsort(z['goals'],axis=1),1)
        codes=2*by_resource[:,0]+by_resource[:,1];mids=v10.mapids(z['positions'])
        def counts(mask):return {k:int((codes[mask]==i).sum()) for i,k in enumerate(('00','01','10','11'))}
        audit.check(metric_match(stats,metrics(counts(np.ones(n,bool)))),'trace_complete_metrics_recount',label)
        for group,pool in v10.partitions(p).items():
            ix=np.isin(mids,pool)
            audit.check(metric_match(stats['map_groups'][group],metrics(counts(ix))),'trace_group_metrics_recount',[label,group])
            for who in (0,1):
                audit.check(metric_match(stats['direction_groups'][who][group],metrics(counts(ix&(z['scout']==who)))),
                            'trace_direction_group_metrics_recount',[label,who,group])
        if evaluation:
            for who in (0,1):
                ix=z['scout']==who
                audit.check(Counter(zip(mids[ix].tolist(),z['goals'][ix,0].tolist()))==
                            Counter({(m,g):n//120 for m in range(30) for g in (0,1)}),'balanced_final_cells',[label,who])


def sources_audit(audit,root,formal):
    invocation=read(root/'invocation.json');args=invocation['args'];hashes=invocation['source_hashes']
    expected=dict(seeds=list(SEEDS),partitions=[1,2,3],updates=600,release_after=300,entropy_off_after=500,contexts=512,eval_n=9600) if formal else dict(
        seeds=[99510],partitions=[1],updates=4,release_after=2,entropy_off_after=3,contexts=512,eval_n=1200)
    audit.check(invocation['formal']==formal and args['smoke']==(not formal) and invocation['arms']==list(ARMS) and
                invocation['device']=='cpu','invocation_mode_arms_device')
    for key,value in expected.items():audit.check(args[key]==value,'fixed_invocation_budget',key)
    audit.check(Path(args['out']).resolve()==root,'invocation_output_directory')
    source=Path(args['source_root']).resolve();expected_source=ROOT.parent/'redesign_v0.10/results'/('generalization_001' if formal else 'smoke_001')
    audit.check(source==expected_source.resolve(),'fixed_source_directory')
    ids=[seed_for(s,p,k,u,d) for s,p,k,u,d in product(args['seeds'],args['partitions'],(1,2,11,12,21,22,31,32),range(args['updates']),(0,1))]
    ids += [seed_for(s,p,k,0,d) for s,p,k,d in product(args['seeds'],args['partitions'],(90,91,92,93),(0,1))]
    audit.check(len(ids)==len(set(ids))==invocation['rng_identity_count'],'all_world_policy_rng_identities_unique')
    for i,(name,digest) in enumerate(hashes.items()):
        audit.check(sha(name)==digest,'current_frozen_source_hash',name)
        if Path(name).suffix in ('.py','.md'):
            audit.check(sha(root/f'frozen_sources/{i:02d}_{Path(name).name}')==digest,'archived_execution_source_hash',name)
    if formal:
        previous=read(source/'audit_execution.json')
        audit.check(previous['passed'] and previous['completed_social_runs']==60 and previous['personal_controls']['completed']==4,
                    'inherited_v10_audit_and_personal_controls')
        for name,digest in previous['source_hashes'].items():audit.check(sha(name)==digest,'inherited_source_still_exact',name)
        gate=read(root/'frozen_sources/preflight_qa.json')
        audit.check(gate['passed'] and gate['runner_sha256']==hashes[str(ROOT/'run_anchoring.py')],'archived_preflight_matches_runner')
        audit.check(sha(ROOT/'audit_anchoring_preflight_source.py')==gate['audit_script_sha256'] and
                    sha(HELPER)==gate['helper_sha256'],'preflight_auditor_and_helper_source_preserved')
    return args,source,hashes


def optimizer_audit(audit,opt,states,step,label):
    for who,row in enumerate(opt):
        names=[k for k in states[who] if v10.active_key(k,'base')]
        audit.check(sum(len(g['params']) for g in row['param_groups'])==len(names) and all(g['lr']==.0007 and
            g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in row['param_groups']),
            'optimizer_parameter_set_hyperparameters',[label,who,step])
        audit.check((not row['state']) if step==0 else len(row['state'])==len(names) and
                    all(float(s['step'])==step for s in row['state'].values()),'adam_exact_update_count',[label,who,step])


def audit_one(audit,root,source,seed,p,arm,args,hashes,photos):
    folder=root/f's{seed}_p{p}_{arm}';label=folder.name
    cfg=read(folder/'config.json');result=read(folder/'result.json');updates=args['updates'];contexts=args['contexts'];n=args['eval_n']
    rel=args['release_after'];off=args['entropy_off_after'];groups=v10.partitions(p)
    points=sorted({0,updates,rel,off}|{x for x in TIMES if x<updates});points=[x for x in points if x<=updates]
    for key,value in dict(seed=seed,partition=p,arm=arm,updates=updates,contexts_per_update=contexts,
        communications_per_update=3*contexts//2,actions_per_update=3*contexts,eval_n=n,release_after=rel,entropy_off_after=off,
        entropy_coefficient=.02,learning_rate=.0007,optimizer='fresh Adam',checkpoints=points,plan=v10.PLAN,
        role_loss_weights={'old_s_sender':.25,'old_r_receiver':.25,'new_sender':.25,'new_receiver':.25},
        map_groups={k:list(v) for k,v in groups.items()},source_hashes=hashes,rng_namespace=11011,
        inherited_v10_sources=True,old_photos_development=True,sealed_never_trained=True).items():
        audit.check(cfg.get(key)==value,'fixed_run_configuration',[label,key])
    for key,expected in [('source_checkpoint',source/f's{seed}_p{p}_base/final.pt'),
                         ('source_config',source/f's{seed}_p{p}_base/config.json'),('prepared_source',source/f'prepared_{seed}.pt')]:
        audit.check(Path(cfg[key]['path']).resolve()==expected.resolve() and cfg[key]['sha256']==sha(expected),
                    'exact_inherited_source_path_hash',[label,key])
    initial=load(folder/'initial.pt');reference=load(folder/'reference.pt');original=load(source/f's{seed}_p{p}_base/final.pt');final=load(folder/'final.pt')
    audit.check(same(initial,reference) and same(initial,original),'current_and_old_initial_exact_source',label)
    audit.check(state_hash(initial)==cfg['initial_sha256']==cfg['reference_state_sha256']==result['initial_sha256']==result['reference_state_sha256'],
                'initial_reference_digest',label)
    audit.check(state_hash(final)==result['final_sha256'],'final_current_digest',label)
    for key,value in dict(seed=seed,partition=p,arm=arm,updates=updates,frozen_modules_verified=True,sealed_never_trained=True).items():
        audit.check(result.get(key)==value,'result_identity',label)
    for who,state in enumerate(initial):
        active=[k for k in state if v10.active_key(k,'base')];frozen=[k for k in state if k!='input_transform' and not v10.active_key(k,'base')]
        expected=dict(active=active,frozen=frozen,trainable_count=sum(state[k].numel() for k in active))
        audit.check(cfg['parameter_partition'][who]==expected and expected['trainable_count']==209975,'matched_current_parameter_partition',[label,who])
        audit.check(all(torch.equal(state[k],final[who][k]) for k in (*frozen,'input_transform')),'all_current_frozen_endpoint_weights',[label,who])
    opt0=load(folder/'initial_optimizer.pt');optend=load(folder/'final_optimizer.pt')
    optimizer_audit(audit,opt0,initial,0,label);optimizer_audit(audit,optend,initial,updates,label)
    checkpoints={};optimizers={}
    for point in points:
        states=load(folder/f'checkpoint_{point:04d}.pt');opt=load(folder/f'optimizer_{point:04d}.pt')
        checkpoints[point]=states;optimizers[point]=opt
        audit.check(all(torch.equal(v,states[who][k]) for who,s in enumerate(initial) for k,v in s.items() if not v10.active_key(k,'base')),
                    'every_checkpoint_frozen_project_transform',[label,point])
        optimizer_audit(audit,opt,initial,point,label)
        if point in (0,updates):
            audit.check(same(states,initial if point==0 else final) and tree_equal(opt,opt0 if point==0 else optend),
                        'checkpoint_endpoints_model_and_adam',[label,point])
    first=load(folder/'after_first_update.pt');first_opt=load(folder/'after_first_optimizer.pt')
    optimizer_audit(audit,first_opt,initial,1,label)
    audit.check(not same(first,initial),'first_update_changes_current_policy',label)
    audit.check(all(torch.equal(v,first[who][k]) for who,s in enumerate(initial) for k,v in s.items() if not v10.active_key(k,'base')),
                'first_update_current_frozen_weights',label)
    rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
    audit.check(len(rows)==updates,'exact_training_updates',label)
    exposure=Counter(dict(old=0,added=0,sealed=0));training_trace_rows=0
    for step,row in enumerate(rows):
        anchored=arm=='anchor' or (arm=='release' and step<rel);weight=.02 if step<off else 0
        audit.check(row['update']==step+1 and row['anchored']==anchored and row['entropy_weight']==weight,
                    'exact_release_and_entropy_boundaries',[label,step+1])
        query_stats=row['queries'];audit.check(set(query_stats)==set(TAGS),'exact_three_communication_queries',[label,step+1])
        for tag in TAGS:
            group='added' if tag=='new' else 'old';purpose=2 if tag=='new' else 1;pool=groups[group]
            key=(seed,p,step,purpose,contexts)
            if key not in audit.world_cache:
                digest,external=world(seed,p,step,purpose,contexts//2,{k:photos['train',k] for k in (0,1)},pool)
                audit.world_cache[key]=digest
            else:digest=audit.world_cache[key]
            stats=query_stats[tag];check_stats(audit,stats,p,contexts//2,[label,step+1,tag])
            audit.check(stats['world_sha256']==digest,'independent_training_query_world_hash',[label,step+1,tag])
            audit.check(stats['map_groups'][group]['n']==contexts//2 and stats['map_groups']['sealed']['n']==0 and
                        all(stats['map_groups'][g]['n']==0 for g in groups if g!=group),'training_support_sealed_zero',[label,step+1,tag])
            exposure[group]+=contexts//2
            if step in (0,rel,off):
                digest,external=world(seed,p,step,purpose,contexts//2,{k:photos['train',k] for k in (0,1)},pool)
                trace_audit(audit,folder/f'train_{step+1:04d}_{tag}.npz',stats,external,digest,seed,p,step,STREAMS[tag],
                            'stochastic',[label,step+1,tag],False);training_trace_rows+=contexts//2
        audit.check(query_stats['old_s']['world_sha256']==query_stats['old_r']['world_sha256'],
                    'old_queries_same_contexts_independent_policy_streams',[label,step+1])
        for who,info in enumerate(row['agents']):
            expected_pairs=[('old_s','sender'),('old_r','receiver'),('new','sender'),('new','receiver')]
            audit.check([(x['query'],x['role']) for x in info['parts']]==expected_pairs,'explicit_four_loss_routes',[label,step+1,who])
            totals=[]
            for part,(tag,role) in zip(info['parts'],expected_pairs):
                scout=who if role=='sender' else 1-who;group='added' if tag=='new' else 'old'
                target=query_stats[tag]['direction_groups'][scout][group]['mean_reward']-1
                audit.check(part['n']==contexts//4 and np.isclose(part['target_mean'],target,rtol=0,atol=1e-12),
                            'each_role_matched_samples_actual_direction_reward',[label,step+1,who,tag,role])
                total=part['policy_loss']+part['value_loss']-weight*part['entropy'];totals.append(part['total'])
                audit.check(np.isclose(part['total'],total,atol=2e-6,rtol=1e-6),'each_role_loss_components',[label,step+1,who,tag,role])
            audit.check(np.isfinite(info['loss']) and np.isclose(info['loss'],sum(totals)/4,atol=2e-6,rtol=1e-6),
                        'four_parts_equal_quarter_weight',[label,step+1,who])
            audit.check(set(info['gradient_norm_by_role'])=={'sender','receiver'} and all(np.isfinite(x) and x>=0 for x in info['gradient_norm_by_role'].values()),
                        'current_sender_receiver_separate_clip_norms',[label,step+1,who])
        raw=sum(x['reward_sum'] for x in query_stats.values())/(3*contexts//2)
        weighted=.25*query_stats['old_s']['mean_reward']+.25*query_stats['old_r']['mean_reward']+.5*query_stats['new']['mean_reward']
        audit.check(row['raw_communication_mean_reward']==raw and row['role_weighted_mean_reward']==weighted,
                    'raw_communication_vs_role_weighted_return',[label,step+1])
    curve=read(folder/'curve.json');audit.check([x['update'] for x in curve]==points,'fixed_curve_schedule',label)
    for point in curve:
        at=point['update'];audit.check(point['current_sha256']==state_hash(checkpoints[at]) and point['reference_state_sha256']==state_hash(reference),
                                      'curve_saved_current_and_reference_digests',[label,at])
        audit.check(set(point['scores'])==set(VARIANTS),'three_partner_variants_each_curve',[label,at])
        for variant in VARIANTS:
            audit.check(set(point['scores'][variant])=={'normal'},'normal_only_curve_variant',[label,at,variant])
            stats=point['scores'][variant]['normal'];check_stats(audit,stats,p,n,[label,at,variant])
            audit.check(all(stats['map_groups'][g]['n']==n*len(ids)//30 for g,ids in groups.items()),'balanced_curve_group_budget',[label,at,variant])
    audit.check(all(curve[0]['scores'][v]==curve[0]['scores']['current'] for v in VARIANTS),'zero_step_three_partner_behaviors_identical',label)
    key=(seed,p,n)
    if key not in audit.eval_cache:audit.eval_cache[key]=world(seed,p,0,90,n,{k:photos['test',k] for k in (0,1)})
    digest,external=audit.eval_cache[key]
    final_trace_rows=0
    for variant in VARIANTS:
        modes=MODES if variant=='current' else ('normal',)
        audit.check(set(result['scores'][variant])==set(modes) and
                    result['scores'][variant]['normal']==curve[-1]['scores'][variant]['normal'],'final_variants_modes_match_curve',[label,variant])
        for mode in modes:
            trace_audit(audit,folder/f'final_{variant}_{mode}.npz',result['scores'][variant][mode],external,digest,
                        seed,p,0,91,mode,[label,variant,mode],True);final_trace_rows+=n
    return dict(folder=folder,seed=seed,partition=p,arm=arm,initial=initial,reference=reference,final=final,
        first=first,first_optimizer=first_opt,checkpoints=checkpoints,optimizers=optimizers,rows=rows,curve=curve,
        exposure=dict(exposure),training_trace_rows=training_trace_rows,final_trace_rows=final_trace_rows)


def paired_audit(audit,runs,args):
    for key,group in runs.items():
        current,anchor,release=(group[a] for a in ARMS)
        for arm,other in group.items():
            audit.check(same(current['initial'],other['initial']) and same(current['reference'],other['reference']),
                        'all_arms_exact_current_reference_initial',[*key,arm])
            audit.check(same(current['first'],other['first']) and tree_equal(current['first_optimizer'],other['first_optimizer']),
                        'all_arms_first_update_and_adam_exact',[*key,arm])
            audit.check(all(current['rows'][0][field]==other['rows'][0][field] for field in
                        ('queries','agents','raw_communication_mean_reward','role_weighted_mean_reward')),
                        'all_arms_first_query_rewards_and_losses_exact',[*key,arm])
            audit.check(all(all(x['queries'][tag]['world_sha256']==y['queries'][tag]['world_sha256'] for tag in TAGS) and
                            x['entropy_weight']==y['entropy_weight'] for x,y in zip(current['rows'],other['rows'])),
                        'all_arms_external_worlds_exploration_paired',[*key,arm])
        rel=args['release_after']
        audit.check(anchor['rows'][:rel]==release['rows'][:rel],'anchor_release_every_pre_release_log_exact',key)
        for point in anchor['checkpoints']:
            if point<=rel:
                audit.check(same(anchor['checkpoints'][point],release['checkpoints'][point]) and
                            tree_equal(anchor['optimizers'][point],release['optimizers'][point]),
                            'anchor_release_pre_release_model_adam_exact',[*key,point])
        audit.check([x for x in anchor['curve'] if x['update']<=rel]==[x for x in release['curve'] if x['update']<=rel],
                    'anchor_release_pre_release_curves_exact',key)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();root=(args.root or ROOT/'results'/('smoke_002' if args.preflight else 'anchoring_001')).resolve()
    out=args.out or (ROOT/'preflight_qa.json' if args.preflight else root/'audit_execution.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);audit=Audit();v10.audit_partitions(audit);photos=v10.old.get_pools(audit)
    config,source,hashes=sources_audit(audit,root,not args.preflight)
    expected={f's{s}_p{k}_{a}' for s,k,a in product(config['seeds'],config['partitions'],ARMS)}
    actual={x.name for x in root.glob('s*') if x.is_dir() and x.name[1:2].isdigit()}
    audit.check(actual==expected,'exact_run_directories')
    groups={};summaries=[]
    for seed,pnum in product(config['seeds'],config['partitions']):
        group={arm:audit_one(audit,root,source,seed,pnum,arm,config,hashes,photos) for arm in ARMS}
        groups[seed,pnum]=group
        summaries.extend({k:r[k] for k in ('seed','partition','arm','exposure','training_trace_rows','final_trace_rows')} for r in group.values())
    paired_audit(audit,groups,config)
    update_count=sum(len(r['rows']) for g in groups.values() for r in g.values())
    report=dict(passed=not audit.failures,completed_runs=len(summaries),preflight=args.preflight,
        runner_sha256=hashes[str(ROOT/'run_anchoring.py')],checks=dict(audit.counts),total_checks=sum(audit.counts.values()),failures=audit.failures,
        training_updates=update_count,training_contexts=update_count*config['contexts'],
        training_communications=update_count*config['contexts']*3//2,
        training_primitive_actions=update_count*config['contexts']*3,
        training_communication_exposure={name:sum(x['exposure'][name] for x in summaries) for name in ('old','added','sealed')},
        independent_training_query_world_hashes=audit.counts['independent_training_query_world_hash'],unique_training_context_batches=len(audit.world_cache),
        final_world_rows_recounted=sum(x['final_trace_rows'] for x in summaries),saved_training_communication_rows_recounted=sum(x['training_trace_rows'] for x in summaries),
        all_recounted_primitive_actions=2*audit.trace_rows,source_root=str(source),source_hashes=hashes,run_counts=summaries,
        audit_script_sha256=sha(__file__),helper_sha256=sha(HELPER),completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only saved models/Adam/logs and independent NumPy worlds/rewards. Model-based routing replay is an independent root-thread audit; not claimed here. Four inherited subject pairs with three repeated partitions; no new personal training.')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=report['total_checks'],failures=len(audit.failures),completed=len(summaries),out=str(out)),ensure_ascii=False))
    if not report['passed']:sys.exit(1)


if __name__=='__main__':main()
