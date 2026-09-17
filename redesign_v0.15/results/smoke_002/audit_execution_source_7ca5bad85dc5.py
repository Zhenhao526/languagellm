"""Independent v15 calibration, scaled-path and matched social execution audit.

No import of run_scaled or scaled_interface. Reuse independently implemented
historical audit math/replay, with an observation implementation written here.
"""
from pathlib import Path
from itertools import product
from datetime import datetime, timezone
import argparse, copy, importlib.util, json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
BASE_PATH=PROJECT/'redesign_v0.14/audit_reset.py'
spec=importlib.util.spec_from_file_location('v15_independent_v14_audit',BASE_PATH)
v14=importlib.util.module_from_spec(spec);spec.loader.exec_module(v14)
b=v14.b;read=b.read;sha=b.sha;load=b.load;same=b.same;arrays=b.arrays


def independent_replay():
    replay=b.import_replay();original=replay.remake_agents
    base=original.__globals__['CampAgent']
    class AuditScaledAgent(base):
        def observe(self,visual,delay=0,memory_mode='retain'):
            # Rebuild the old observation explicitly; do not call a production
            # scaled class, attach wrapper, or its observe implementation.
            n=len(visual);zero=visual.new_zeros(n,self.width)
            encoded=self.encode_slots(visual)
            seen=torch.cat((encoded,visual.new_ones(n,1)),-1)
            h=self.memory(seen,zero)
            for _ in range(delay):h=self.memory(torch.zeros_like(seen),h)
            if memory_mode=='reset':h=torch.zeros_like(h)
            elif memory_mode=='replay':h=self.memory(seen,zero)
            elif memory_mode!='retain':raise ValueError(memory_mode)
            return h*self.visual_scale
    def remake(seed,prepared,vocab=7,length=2,representation='identity'):
        agents=original(seed,prepared,vocab,length,representation)
        for a in agents:
            a.__class__=AuditScaledAgent
            a.register_buffer('visual_scale',a.memory.weight_ih.new_tensor(1.))
        return agents
    replay.remake_agents=remake
    replay.unscaled_remake_agents=original
    return replay


def archive_source(batch):
    target=batch/'audit_execution_source.py'
    if target.exists() and sha(target)!=sha(__file__):target.rename(target.with_name('audit_execution_source_'+sha(target)[:12]+'.py'))
    target.write_bytes(Path(__file__).read_bytes())


def strip_scale(states):
    return [{k:v for k,v in state.items() if k!='visual_scale'} for state in states]


@torch.no_grad()
def audit_calibration_pair(folder,source,reset_batch,bank,replay,audit):
    meta=read(folder/'calibration.json');seed=meta['seed'];p=meta['partition']
    audit.check(meta['status']=='complete' and meta['training_updates']==0,'calibration_complete_without_training',str(folder))
    for path,digest in meta['source_hashes'].items():audit.check(sha(path)==digest,'calibration_bound_source_unchanged',path)
    audit.check(meta['calibration_source_sha256']==sha(ROOT/'scaled_interface.py'),'calibration_executed_module_bound',str(folder))
    pools=[np.sort(np.asarray(bank.pools['train',k],dtype=np.int64)) for k in (0,1)]
    old=b.groups(p)['old'];pairs=np.array(list(product(*pools)),dtype=np.int64)
    mids=np.repeat(old,len(pairs));photos=np.tile(pairs,(len(old),1));positions=b.MAPS[mids]
    expected=dict(world_id=np.arange(len(mids)),map_ids=mids,photo_pair_ids=np.tile(np.arange(len(pairs)),len(old)),positions=positions,photo_ids=photos)
    audit.check([len(x) for x in pools]==[22,22] and len(mids)==8712,'calibration_exact_18_by_22_by_22_support',str(folder))
    audit.check(meta['old_map_ids']==old.tolist() and meta['train_photo_ids']==[x.tolist() for x in pools],'calibration_declared_old_and_train_pools',str(folder))
    audit.check(meta['chunk_size']==512 and meta['device']=='cpu' and meta['torch_threads']==1 and not meta['grad_enabled'] and meta['model_training'],'calibration_execution_flags',str(folder))
    audit.check(meta['no_test_or_held_maps'] and not np.isin(mids,np.concatenate([b.groups(p)[g] for g in ('added','sealed')])).any(),'calibration_no_added_or_sealed',str(folder))
    for k,pool in enumerate(pools):
        rows=[dict(feature_row=int(i),image_id=bank.entries[int(i)]['id'],sha256=bank.entries[int(i)]['sha256']) for i in pool]
        audit.check(rows==meta['photo_manifest'][k] and all(bank.entries[int(i)]['split']=='train' and bank.entries[int(i)]['category']==('food','water')[k] for i in pool),'calibration_feature_manifest_and_category',[str(folder),k])
    prepared=load(source/f'prepared_{seed}.pt')
    old_states=load(source/f'social_s{seed}_p{p}_control/initial.pt')
    reset_states=load(reset_batch/f'social_s{seed}_p{p}_reset/initial.pt')
    agents_by_name={}
    for name,states in [('retained',old_states),('reset',reset_states)]:
        agents=replay.unscaled_remake_agents(seed,prepared,7,2,'identity')
        for a,state in zip(agents,states):
            a.load_state_dict(state)
            for key,param in a.named_parameters():param.requires_grad_(key.startswith(b.SOCIAL))
            a.train()
        agents_by_name[name]=agents
    alphas=[]
    for who,entry in enumerate(meta['persons']):
        raw_path=folder/f'person{who}.npz';z=arrays(raw_path)
        audit.check(entry['person']==who and Path(entry['raw_path'])==raw_path and entry['raw_sha256']==sha(raw_path),'calibration_raw_hash_and_person',[str(folder),who])
        for key,value in expected.items():audit.check(np.array_equal(z[key],value),'calibration_complete_unique_world_order',[str(folder),who,key])
        ids=np.unique(np.concatenate(pools));a=agents_by_name['retained'][who]
        projected_values=a.project(bank.features[torch.from_numpy(ids)])
        projected=projected_values.new_zeros((len(bank.features),64));projected[torch.from_numpy(ids)]=projected_values
        audit.check(not projected[np.setdiff1d(np.arange(len(bank.features)),ids)].any(),'calibration_only_train_features_projected',[str(folder),who])
        norms={}
        for name in ('retained','reset'):
            agent=agents_by_name[name][who];chunks=[]
            for lo in range(0,len(mids),512):
                visual=replay.scene_visual(positions[lo:lo+512],photos[lo:lo+512],projected)
                # Independent original observation: single zero-history GRU.
                seen=torch.cat((agent.encode_slots(visual),visual.new_ones(len(visual),1)),-1)
                chunks.append(agent.memory(seen,visual.new_zeros(len(visual),96)).numpy())
            h=np.concatenate(chunks)
            audit.check(np.array_equal(h,z[name+'_h']) and h.dtype==np.float32 and h.shape==(8712,96),'calibration_raw_hidden_independent_forward_bitwise',[str(folder),who,name])
            audit.check(np.isfinite(h).all(),'calibration_hidden_finite',[str(folder),who,name])
            hf=h.astype(np.float64);norms[name]=float(np.sqrt(np.sum(hf*hf,axis=1)).mean())
            audit.check(norms[name]==entry[name+'_mean_l2'],'calibration_independent_mean_l2',[str(folder),who,name])
        ratio=norms['retained']/norms['reset'];effective=float(np.float32(ratio));scaled=z['reset_h']*np.float32(effective)
        sl2=float(np.sqrt(np.sum(scaled.astype(np.float64)**2,axis=1)).mean())
        audit.check(np.isfinite(effective) and effective>0 and ratio==entry['alpha_float64'] and effective==entry['alpha_float32'] and sl2==entry['scaled_mean_l2'],'calibration_ratio_single_float32_rounding',[str(folder),who])
        audit.check(entry['world_count']==8712 and entry['requires_grad']=={k:v.requires_grad for k,v in agents_by_name['reset'][who].named_parameters()},'calibration_person_budget_and_flags',[str(folder),who])
        alphas.append(effective);audit.counts['calibration_persons']+=1;audit.counts['calibration_raw_hidden_worlds_replayed']+=2*len(mids)
    return alphas


def write_result(path,audit,extra):
    result=dict(passed=not audit.failures,created_utc=datetime.now(timezone.utc).isoformat(),check_count=sum(audit.checks.values()),checks=dict(audit.checks),counts=dict(audit.counts),failures=audit.failures,script_sha256=sha(__file__),independent_helpers={str(p):sha(p) for p in (BASE_PATH,PROJECT/'redesign_v0.13/audit_v13.py',PROJECT/'redesign_v0.11/audit_query_replay.py',PROJECT/'redesign_v0.12/audit_receiver.py')},**extra)
    if path.exists():path.rename(path.with_name(path.stem+'_previous_'+datetime.now().strftime('%Y%m%d%H%M%S')+path.suffix))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),passed=result['passed'],checks=result['check_count'],counts=result['counts'],failures=result['failures'][:20]),ensure_ascii=False,indent=2),flush=True)
    assert result['passed']


def source_check(receipt,audit,formal):
    v14.source_check(receipt['retained'],audit,formal)
    data=receipt['reset'];source=Path(data['source_batch']);inv=read(source/'invocation.json')
    audit.check(read(source/'training_complete.json')['status']=='complete' and data['source_training_hashes']==inv['source_hashes'],'reset_source_complete_and_frozen_sources',str(source))
    for path,digest in data['files'].items():audit.check(sha(path)==digest,'consumed_reset_file_unchanged',path)
    for path,digest in data['source_training_hashes'].items():audit.check(sha(path)==digest,'reset_training_source_unchanged',path)
    if formal:
        manifest=read(source/'completion_manifest.json');qa=read(source/'audit_execution.json')
        audit.check(manifest['status']=='complete' and qa['passed'] and qa['script_sha256']==sha(BASE_PATH),'reset_completed_independent_audit_bound',str(source))
        for path,digest in data['files'].items():
            if Path(path).name=='completion_manifest.json':continue
            audit.check(manifest['artifacts'].get(str(Path(path).relative_to(PROJECT)))==digest,'reset_files_match_completion_manifest',path)


def check_source_archive(batch,hashes,audit):
    for i,(path,digest) in enumerate(hashes.items()):
        audit.check(sha(path)==digest,'current_source_matches_executed',path)
        snapshot=batch/'frozen_sources'/f'{i:02d}_{Path(path).name}'
        if snapshot.exists():audit.check(sha(snapshot)==digest,'archived_source_matches_executed',path)
        else:audit.check(Path(path).name=='features.npz','only_features_not_duplicated',path)


def audit_calibration(batch,audit,replay,bank):
    if not hasattr(audit,'calibrations'):audit.calibrations={}
    if str(batch) in audit.calibrations:return audit.calibrations[str(batch)]
    inv=read(batch/'invocation.json');args=inv['args'];done=read(batch/'calibration_complete.json');receipt=read(batch/'source_receipt.json')
    source=Path(args['source_batch']);reset=Path(args['reset_batch']);hashes=inv['source_hashes']
    audit.check(inv['phase']=='calibration' and done['status']=='complete' and done['source_hashes']==hashes and done['train_only'] and done['training_updates']==0,'calibration_batch_complete',str(batch))
    audit.check(done['seeds']==args['seeds'] and done['partitions']==args['partitions'] and done['persons']==2*len(args['seeds'])*len(args['partitions']),'calibration_all_planned_persons',str(batch))
    audit.check(done['source_receipt_sha256']==sha(batch/'source_receipt.json'),'calibration_dual_source_receipt_bound',str(batch))
    check_source_archive(batch,hashes,audit);source_check(receipt,audit,inv['formal'])
    expected={f's{s}_p{p}/{name}' for s,p in product(args['seeds'],args['partitions']) for name in ('calibration.json','person0.npz','person1.npz')}
    audit.check(set(done['files'])==expected,'calibration_manifest_exact_file_inventory',str(batch))
    for path,digest in done['files'].items():audit.check(sha(batch/path)==digest,'calibration_every_file_bound',path)
    alphas={}
    for s,p in product(args['seeds'],args['partitions']):
        audit.check(sha(batch/f'prepared_{s}.pt')==sha(source/f'prepared_{s}.pt'),'calibration_prepared_copy_from_retained',s)
        alphas[s,p]=audit_calibration_pair(batch/f's{s}_p{p}',source,reset,bank,replay,audit)
    audit.calibrations[str(batch)]=(alphas,inv)
    return alphas,inv


def audit_batch(batch,audit):
    initial_count=audit.counts['runs'];inv=read(batch/'invocation.json');args=inv['args'];hashes=inv['source_hashes'];done=read(batch/'training_complete.json')
    source=Path(args['source_batch']);reset_batch=Path(args['reset_batch']);calibration=Path(args['calibration']);receipt=read(batch/'source_receipt.json')
    audit.check(inv['phase']=='social' and done['status']=='complete' and done['new_private_runs']==0 and done['source_hashes']==hashes,'only_new_scaled_social_stage_complete',str(batch))
    check_source_archive(batch,hashes,audit);source_check(receipt,audit,inv['formal'])
    torch.set_num_threads(1);replay=independent_replay();bank=replay.ImageBank()
    alphas,cinv=audit_calibration(calibration,audit,replay,bank)
    audit.check(cinv['source_hashes']==hashes and read(calibration/'source_receipt.json')==receipt,'calibration_social_same_sources',str(batch))
    cr=read(batch/'calibration_receipt.json');ccomplete=read(calibration/'calibration_complete.json')
    audit.check(Path(cr['path'])==calibration/'calibration_complete.json' and cr['sha256']==sha(calibration/'calibration_complete.json') and cr['files']==ccomplete['files'],'social_consumes_exact_calibration_manifest',str(batch))
    if inv['formal']:
        audit.check(args['seeds']==[31101,31102,31103,31104] and args['partitions']==[1,2,3] and args['social_updates']==2400 and args['batch']==512 and args['eval_n']==9600 and inv['arms']==['reset_scaled'],'formal_single_condition_fixed_matrix')
        gate=read(batch/'frozen_sources/preflight_qa.json')
        audit.check(gate['passed'] and gate['source_hashes']==hashes and gate['source_receipt_sha256']==sha(ROOT/'source_receipt.json') and gate['calibration_complete_sha256']==cr['sha256'],'formal_source_calibration_and_gate_bound')
        audit.check(receipt==read(ROOT/'source_receipt.json'),'formal_exact_dual_source_receipt')
    fingerprints=[]
    for s in args['seeds']:
        f=batch/f'prepared_{s}.pt';audit.check(sha(f)==sha(source/f.name)==sha(reset_batch/f.name),'prepared_same_both_references',s);fingerprints.append(b.state_digest(load(f)))
    audit.check(len(set(fingerprints))==len(fingerprints),'four_distinct_original_preparations')
    for seed,p,arm in product(args['seeds'],args['partitions'],inv['arms']):
        folder=batch/f'social_s{seed}_p{p}_{arm}';old=source/f'social_s{seed}_p{p}_control';reset=reset_batch/f'social_s{seed}_p{p}_reset';cfg=read(folder/'config.json');oc=read(old/'config.json');zc=read(reset/'config.json')
        initial=load(folder/'initial.pt');baseline=load(reset/'initial.pt');scales=[1.,1.] if arm=='unit_replay' else alphas[seed,p]
        audit.check(arm in ('unit_replay','reset_scaled'),'legal_condition',folder.name)
        b.compare_tensors(audit,strip_scale(initial),baseline,'only_scale_added_to_exact_reset_initial',[folder.name])
        audit.check(same(initial,load(folder/'source_state.pt')) and cfg['source_checkpoint']['sha256']==sha(folder/'source_state.pt'),'source_scaled_state_bound',folder.name)
        audit.check(cfg['scales_float32']==scales and cfg['initial_sha256']==b.state_digest(initial),'config_effective_scale_and_initial_hash',folder.name)
        for who,state in enumerate(initial):
            value=state['visual_scale'];audit.check(set(state)==set(baseline[who])|{'visual_scale'} and value.shape==torch.Size([]) and value.dtype==torch.float32 and float(value)==scales[who] and float(value)>0,'one_positive_scalar_only',[folder.name,who])
        for key in ('batch','eval_n','learning_rate','entropy_off_after','entropy_coefficient','role_loss_weights','gradient_clip','training_pool','parameter_partition','rng_world_namespace','rng_policy_namespace','training_policy_stream','evaluation_policy_stream'):
            audit.check(cfg[key]==oc[key]==zc[key],'same_social_settings_both_references',[folder.name,key])
        audit.check(cfg['updates']==args['social_updates'] and cfg['source_hashes']==hashes and cfg['training_pool']==b.groups(p)['old'].tolist(),'run_budget_and_old_support',folder.name)
        audit.check(same(load(folder/'initial_optimizer.pt'),load(old/'initial_optimizer.pt')) and same(load(folder/'initial_optimizer.pt'),load(reset/'initial_optimizer.pt')),'fresh_Adam_same_both_references',folder.name)
        for key,path in [('reset_initial',reset/'initial.pt'),('retained_social_initial',old/'initial.pt'),('calibration',calibration/f's{seed}_p{p}/calibration.json')]:
            item=cfg['source_inputs'][key];audit.check(Path(item['path'])==path and item['sha256']==sha(path),'config_consumed_source_bound',[folder.name,key])
        constructed=replay.remake_agents(seed,load(batch/f'prepared_{seed}.pt'))
        for who,(a,state) in enumerate(zip(constructed,initial)):
            a.load_state_dict(state);audit.check('visual_scale' in dict(a.named_buffers()) and 'visual_scale' not in dict(a.named_parameters()) and not a.visual_scale.requires_grad,'scale_is_unoptimized_buffer',[folder.name,who])
        for cp in cfg['checkpoints']:
            state=load(folder/f'checkpoint_{cp:04d}.pt')
            for who in (0,1):
                for key,value in state[who].items():
                    if not key.startswith(b.SOCIAL):audit.check(torch.equal(value,initial[who][key]),'all_visual_weights_and_scale_frozen_every_checkpoint',[folder.name,cp,who,key])
            audit.counts['checkpoints']+=1
        audit.check(same(load(folder/'final.pt'),load(folder/f"checkpoint_{cfg['updates']:04d}.pt")),'final_equals_final_checkpoint',folder.name)
        lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];ref_lines=[[json.loads(x) for x in (q/'training.jsonl').read_text().splitlines()] for q in (old,reset)]
        audit.check(len(lines)==cfg['updates'],'full_training_budget',folder.name);paired=[0,0]
        for step,line in enumerate(lines):
            digest=b.world_digest(b.social_worlds(bank,seed,p,step,cfg['batch'],True))
            audit.check(line['stats']['world_sha256']==digest,'all_training_worlds_reconstructed',[folder.name,step])
            for j,ref in enumerate(ref_lines):
                if step<len(ref):audit.check(digest==ref[step]['stats']['world_sha256'],'training_worlds_identical_reference',[folder.name,step,j]);paired[j]+=1
            audit.check(line['update']==step+1 and line['entropy_weight']==(.02 if step<2100 else 0.),'fixed_update_entropy_schedule',[folder.name,step])
            audit.check(line['stats']['map_groups']['old']['n']==cfg['batch'] and line['stats']['map_groups']['added']['n']==line['stats']['map_groups']['sealed']['n']==0,'only_old_train_feedback',[folder.name,step])
            for who,person in enumerate(line['agents']):
                parts=person['parts'];audit.check([q['role'] for q in parts]==['sender','receiver'] and all(q['n']==cfg['batch']//2 for q in parts),'equal_roles_and_directions',[folder.name,step,who])
                totals=[q['policy_loss']+q['value_loss']-line['entropy_weight']*q['entropy'] for q in parts]
                audit.check(abs(np.mean(totals)-person['loss'])<2e-6 and np.isfinite(list(person['gradient_norm_by_role'].values())).all(),'equal_role_logged_loss_finite_gradients',[folder.name,step,who])
            audit.counts['updates']+=1
        for j,name in enumerate(('retained','reset')):audit.counts[name+'_reference_updates_paired']+=paired[j]
        if inv['formal']:audit.check(paired==[2400,2400],'every_formal_world_paired_both_references',folder.name)
        for step in (1,2101):
            if step<=cfg['updates']:v14.step_replay(folder,step,bank,replay,audit)
        b.social_protocol(folder,bank,replay,audit)
        agents=replay.remake_agents(seed,load(batch/f'prepared_{seed}.pt'))
        for a,state in zip(agents,load(folder/'final.pt')):a.load_state_dict(state);a.requires_grad_(False)
        projected=replay.projected_banks(agents,bank)
        for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
            path=folder/f'final_{mode}.npz';replay.replay(path,agents,agents,projected,cfg,0,91,mode,audit);z=arrays(path);worlds=b.social_worlds(bank,seed,p,0,cfg['eval_n'],False)
            for d in (0,1):
                ix=z['scout']==d
                for key,value in worlds[d].items():audit.check(np.array_equal(z[key][ix],value),'final_balanced_worlds_reconstructed',[folder.name,mode,d,key])
            for ref in (old,reset):
                rz=arrays(ref/f'final_{mode}.npz')
                for key in worlds[0]:audit.check(np.array_equal(z[key],rz[key]),'final_worlds_identical_both_references',[folder.name,mode,ref.name,key])
            b.final_statistics(folder,mode,audit);audit.counts['final_modes']+=1
        if arm=='unit_replay':
            audit.check(cfg['updates']==zc['updates'],'unit_replay_equals_reference_budget')
            states=('initial.pt','after_first_update.pt','final.pt',*[f'checkpoint_{u:04d}.pt' for u in cfg['checkpoints']])
            optimizers=('initial_optimizer.pt','after_first_optimizer.pt','final_optimizer.pt',*[f'optimizer_{u:04d}.pt' for u in cfg['checkpoints']])
            for name in states:audit.check(same(strip_scale(load(folder/name)),load(reset/name)),'unit_replay_parameters_bitwise_reference',name)
            for name in optimizers:audit.check(same(load(folder/name),load(reset/name)),'unit_replay_Adam_bitwise_reference',name)
            audit.check(lines==ref_lines[1],'unit_replay_all_training_log_exact')
            # State hashes include the additional persistent scalar; scores do not.
            c1=read(folder/'curve.json');c0=read(reset/'curve.json')
            audit.check([{k:v for k,v in q.items() if k!='state_sha256'} for q in c1]==[{k:v for k,v in q.items() if k!='state_sha256'} for q in c0],'unit_replay_all_curve_scores_exact')
            for u in cfg['checkpoints']:audit.check(read(folder/f'protocol_{u:04d}.json')==read(reset/f'protocol_{u:04d}.json'),'unit_replay_protocol_metrics_exact',u)
            for path in folder.glob('*.npz'):
                counterpart=reset/path.name
                if counterpart.exists():audit.check(same(arrays(path),arrays(counterpart)),'unit_replay_raw_arrays_exact',path.name)
        audit.counts['runs']+=1
    expected=len(args['seeds'])*len(args['partitions'])*len(inv['arms'])
    audit.check(audit.counts['runs']-initial_count==expected and done['new_social_runs']==expected,'all_new_runs_audited',str(batch))
    audit.counts['training_communications']+=expected*args['social_updates']*args['batch'];audit.counts['training_actions']+=2*expected*args['social_updates']*args['batch']
    return inv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/scaled_001');ap.add_argument('--preflight',action='store_true');args=ap.parse_args();audit=b.Audit();torch.set_num_threads(1)
    if args.preflight:
        batches=[ROOT/'results/smoke_001',ROOT/'results/smoke_002'];invs=[]
        for batch in batches:archive_source(batch);invs.append(audit_batch(batch,audit))
        audit.check(invs[0]['source_hashes']==invs[1]['source_hashes'],'smokes_share_frozen_sources')
        receipt=read(ROOT/'source_receipt.json');source_check(receipt,audit,True)
        replay=independent_replay();_,cinv=audit_calibration(ROOT/'calibration_001',audit,replay,replay.ImageBank())
        audit.check(cinv['source_hashes']==invs[0]['source_hashes'],'formal_calibration_matches_smoke_sources')
        write_result(ROOT/'preflight_qa.json',audit,dict(source_hashes=invs[0]['source_hashes'],source_receipt_sha256=sha(ROOT/'source_receipt.json'),calibration_complete_sha256=sha(ROOT/'calibration_001/calibration_complete.json'),smoke_batches=[str(p) for p in batches],scope='Independent all train-only calibration, unit replay of prior reset, nonunit scaled paths, first/2101 Adam exact replay, source/calibration gate; no new private fitting or outcome selection.'))
    else:
        batch=args.root.resolve();archive_source(batch);inv=audit_batch(batch,audit)
        write_result(batch/'audit_execution.json',audit,dict(source_hashes=inv['source_hashes'],source_receipt_sha256=sha(batch/'source_receipt.json'),calibration_complete_sha256=sha(Path(inv['args']['calibration'])/'calibration_complete.json'),scope='Independent source/calibration, scaled observation implementation, all frozen states/world pairing, first/2101 exact Adam, all endpoints and native/greedy protocol; 12 new social runs only, four previously observed sources.'))


if __name__=='__main__':main()
