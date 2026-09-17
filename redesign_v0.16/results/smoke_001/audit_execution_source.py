"""Independent v16 branch input-scale audit; never imports branch_interface.

The raw observer remains the old CampAgent. The two branch transforms below are
implemented independently, preserving exact parameter storage and sampling math.
"""
from pathlib import Path
from itertools import product
from datetime import datetime,timezone
import argparse,copy,importlib.util,json
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
PREVIOUS=PROJECT/'redesign_v0.15/audit_scaled.py'
spec=importlib.util.spec_from_file_location('v16_independent_previous',PREVIOUS)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
b=previous.b;v14=previous.v14;read=b.read;sha=b.sha;load=b.load;same=b.same;arrays=b.arrays
BUFFER_KEYS=('send_context.h_scale','send_value.h_scale')
FACTORS={'neither_replay':(0,0),'policy_only_scale':(1,0),'value_only_scale':(0,1),'both_replay':(1,1)}


def independent_replay():
    replay=b.import_replay();original=replay.remake_agents
    class IndependentPrefix(nn.Sequential):
        def forward(self,x):
            h=x[:, :96]*self.h_scale
            y=torch.cat((h,x[:,96:]),dim=1)
            for layer in self:y=layer(y)
            return y
    def remake(seed,prepared,vocab=7,length=2,representation='identity'):
        agents=original(seed,prepared,vocab,length,representation)
        for a in agents:
            for module in (a.send_context,a.send_value):
                module.__class__=IndependentPrefix
                module.register_buffer('h_scale',module[0].weight.new_tensor(1.))
        return agents
    replay.remake_agents=remake
    return replay


def archive_source(batch):
    path=batch/'audit_execution_source.py'
    if path.exists() and sha(path)!=sha(__file__):path.rename(path.with_name('audit_execution_source_'+sha(path)[:12]+'.py'))
    path.write_bytes(Path(__file__).read_bytes())


def strip_scale(states):
    return [type(s)((k,v) for k,v in s.items() if k not in (*BUFFER_KEYS,'visual_scale')) for s in states]


def source_check(receipt,audit,formal):
    previous.source_check(receipt,audit,formal)
    data=receipt['scaled'];source=Path(data['source_batch']);inv=read(source/'invocation.json')
    audit.check(read(source/'training_complete.json')['status']=='complete' and data['source_training_hashes']==inv['source_hashes'],'scaled_source_complete_and_frozen_sources',str(source))
    for path,digest in data['files'].items():audit.check(sha(path)==digest,'consumed_scaled_file_unchanged',path)
    for path,digest in data['source_training_hashes'].items():audit.check(sha(path)==digest,'scaled_training_source_unchanged',path)
    if formal:
        manifest=read(source/'completion_manifest.json');qa=read(source/'audit_execution.json')
        audit.check(manifest['status']=='complete' and qa['passed'] and qa['script_sha256']==sha(PREVIOUS),'scaled_completed_independent_audit_bound',str(source))
        for path,digest in data['files'].items():
            if Path(path).name=='completion_manifest.json':continue
            audit.check(manifest['artifacts'].get(str(Path(path).relative_to(PROJECT)))==digest,'scaled_files_match_completion_manifest',path)


check_source_archive=previous.check_source_archive


def inherited_calibration(calibration,scaled_batch,audit):
    done=read(calibration/'calibration_complete.json');inv=read(calibration/'invocation.json');old_receipt=read(scaled_batch/'calibration_receipt.json')
    audit.check(done['status']=='complete' and done['train_only'] and done['training_updates']==0 and done['source_hashes']==inv['source_hashes'],'existing_calibration_complete_no_new_fitting',str(calibration))
    audit.check(Path(old_receipt['path'])==calibration/'calibration_complete.json' and old_receipt['sha256']==sha(calibration/'calibration_complete.json') and old_receipt['files']==done['files'],'calibration_exactly_same_as_11_reference',str(calibration))
    for path,digest in done['source_hashes'].items():audit.check(sha(path)==digest,'inherited_calibration_sources_unchanged',path)
    for name,digest in done['files'].items():audit.check(sha(calibration/name)==digest,'inherited_calibration_files_unchanged',name)
    prior_qa=read(PROJECT/'redesign_v0.15/preflight_qa.json')
    audit.check(prior_qa['passed'] and prior_qa['script_sha256']==sha(PREVIOUS),'calibration_forward_has_original_independent_preflight_proof')
    result={}
    for seed,p in product(done['seeds'],done['partitions']):
        cfg=read(calibration/f's{seed}_p{p}/calibration.json')
        result[seed,p]=[q['alpha_float32'] for q in cfg['persons']]
        audit.check(len(cfg['persons'])==2 and cfg['old_map_ids']==b.groups(p)['old'].tolist() and all(q['world_count']==8712 for q in cfg['persons']),'inherited_calibration_same_two_people_old_support',[seed,p])
        audit.check(all(np.isfinite(x) and x>0 and float(np.float32(x))==x for x in result[seed,p]),'inherited_calibration_float32_exact',[seed,p])
    return result,inv


def recorded_branches(folder,reset,scaled,audit):
    cfg=read(folder/'config.json')
    for cp in cfg['checkpoints']:
        state=load(folder/f'checkpoint_{cp:04d}.pt')
        for who in (0,1):
            raw=arrays(folder/f'protocol_{cp:04d}_d{who}.npz');old=arrays(reset/f'protocol_0000_d{who}.npz')
            audit.check(np.array_equal(raw['h'],raw['raw_h']) and np.array_equal(raw['raw_h'],old['h']),'raw_h_same_as_original_reset_interface',[folder.name,cp,who])
            for name,key in [('policy',BUFFER_KEYS[0]),('value',BUFFER_KEYS[1])]:
                expected=raw['raw_h']*np.float32(state[who][key].item())
                audit.check(np.array_equal(raw[name+'_h'],expected),'recorded_effective_h_matches_actual_branch_scale',[folder.name,cp,who,name])
            old11=arrays(scaled/f'protocol_0000_d{who}.npz')
            alpha=cfg['calibration_scales_float32'][who]
            audit.check(np.array_equal(raw['raw_h']*np.float32(alpha),old11['h']),'old_whole_scale_effective_h_exactly_aligned',[folder.name,cp,who])
            audit.counts['raw_and_effective_h_tables_checked']+=1

def audit_batch(batch,audit):
    initial_count=audit.counts['runs'];inv=read(batch/'invocation.json');args=inv['args'];hashes=inv['source_hashes'];done=read(batch/'training_complete.json')
    source=Path(args['source_batch']);reset_batch=Path(args['reset_batch']);scaled_batch=Path(args['scaled_batch']);calibration=Path(args['calibration']);receipt=read(batch/'source_receipt.json')
    audit.check(inv['phase']=='social' and done['status']=='complete' and done['new_private_runs']==0 and done['source_hashes']==hashes,'only_new_scaled_social_stage_complete',str(batch))
    check_source_archive(batch,hashes,audit);source_check(receipt,audit,inv['formal'])
    torch.set_num_threads(1);replay=independent_replay();bank=replay.ImageBank()
    alphas,cinv=inherited_calibration(calibration,scaled_batch,audit)
    audit.check(read(calibration/'source_receipt.json')=={k:v for k,v in receipt.items() if k not in ('scaled','factorial_references')},'calibration_social_same_sources',str(batch))
    cr=read(batch/'calibration_receipt.json');ccomplete=read(calibration/'calibration_complete.json')
    audit.check(Path(cr['path'])==calibration/'calibration_complete.json' and cr['sha256']==sha(calibration/'calibration_complete.json') and cr['files']==ccomplete['files'],'social_consumes_exact_calibration_manifest',str(batch))
    if inv['formal']:
        audit.check(args['seeds']==[31101,31102,31103,31104] and args['partitions']==[1,2,3] and args['social_updates']==2400 and args['batch']==512 and args['eval_n']==9600 and inv['arms']==['policy_only_scale','value_only_scale'],'formal_two_new_conditions_fixed_matrix')
        gate=read(batch/'frozen_sources/preflight_qa.json')
        audit.check(gate['passed'] and gate['source_hashes']==hashes and gate['source_receipt_sha256']==sha(ROOT/'source_receipt.json') and gate['calibration_complete_sha256']==cr['sha256'],'formal_source_calibration_and_gate_bound')
        audit.check(receipt==read(ROOT/'source_receipt.json'),'formal_exact_dual_source_receipt')
    fingerprints=[]
    for s in args['seeds']:
        f=batch/f'prepared_{s}.pt';audit.check(sha(f)==sha(source/f.name)==sha(reset_batch/f.name),'prepared_same_both_references',s);fingerprints.append(b.state_digest(load(f)))
    audit.check(len(set(fingerprints))==len(fingerprints),'four_distinct_original_preparations')
    for seed,p,arm in product(args['seeds'],args['partitions'],inv['arms']):
        folder=batch/f'social_s{seed}_p{p}_{arm}';old=source/f'social_s{seed}_p{p}_control';reset=reset_batch/f'social_s{seed}_p{p}_reset';scaled=scaled_batch/f'social_s{seed}_p{p}_reset_scaled';cfg=read(folder/'config.json');oc=read(old/'config.json');zc=read(reset/'config.json')
        initial=load(folder/'initial.pt');baseline=load(reset/'initial.pt');factors=FACTORS[arm];alpha=alphas[seed,p];scales=[{'policy':a if factors[0] else 1.,'value':a if factors[1] else 1.} for a in alpha]
        audit.check(arm in FACTORS,'legal_condition',folder.name)
        b.compare_tensors(audit,strip_scale(initial),baseline,'only_scale_added_to_exact_reset_initial',[folder.name])
        audit.check(same(initial,load(folder/'source_state.pt')) and cfg['source_checkpoint']['sha256']==sha(folder/'source_state.pt'),'source_scaled_state_bound',folder.name)
        audit.check(cfg['branch_scales_float32']==scales and cfg['calibration_scales_float32']==alpha and cfg['factors']==list(factors) and cfg['initial_sha256']==b.state_digest(initial),'config_effective_scale_and_initial_hash',folder.name)
        for who,state in enumerate(initial):
            audit.check(set(state)==set(baseline[who])|set(BUFFER_KEYS) and 'visual_scale' not in state,'only_two_branch_buffers_added',[folder.name,who])
            for name,key in zip(('policy','value'),BUFFER_KEYS):
                value=state[key];audit.check(value.shape==torch.Size([]) and value.dtype==torch.float32 and float(value)==scales[who][name] and float(value)>0,'one_scalar_per_branch',[folder.name,who,name])
        for key in ('batch','eval_n','learning_rate','entropy_off_after','entropy_coefficient','role_loss_weights','gradient_clip','training_pool','parameter_partition','rng_world_namespace','rng_policy_namespace','training_policy_stream','evaluation_policy_stream'):
            audit.check(cfg[key]==oc[key]==zc[key],'same_social_settings_both_references',[folder.name,key])
        audit.check(cfg['updates']==args['social_updates'] and cfg['source_hashes']==hashes and cfg['training_pool']==b.groups(p)['old'].tolist(),'run_budget_and_old_support',folder.name)
        audit.check(same(load(folder/'initial_optimizer.pt'),load(old/'initial_optimizer.pt')) and same(load(folder/'initial_optimizer.pt'),load(reset/'initial_optimizer.pt')),'fresh_Adam_same_both_references',folder.name)
        for key,path in [('reset_initial',reset/'initial.pt'),('retained_social_initial',old/'initial.pt'),('calibration',calibration/f's{seed}_p{p}/calibration.json'),('scaled_initial',scaled/'initial.pt')]:
            item=cfg['source_inputs'][key];audit.check(Path(item['path'])==path and item['sha256']==sha(path),'config_consumed_source_bound',[folder.name,key])
        constructed=replay.remake_agents(seed,load(batch/f'prepared_{seed}.pt'))
        for who,(a,state) in enumerate(zip(constructed,initial)):
            a.load_state_dict(state)
            for key in BUFFER_KEYS:audit.check(key in dict(a.named_buffers()) and key not in dict(a.named_parameters()) and not dict(a.named_buffers())[key].requires_grad,'branch_scale_is_unoptimized_buffer',[folder.name,who,key])
        for cp in cfg['checkpoints']:
            state=load(folder/f'checkpoint_{cp:04d}.pt')
            for who in (0,1):
                for key,value in state[who].items():
                    if not key.startswith(b.SOCIAL) or key in BUFFER_KEYS:audit.check(torch.equal(value,initial[who][key]),'all_visual_weights_and_scale_frozen_every_checkpoint',[folder.name,cp,who,key])
            audit.counts['checkpoints']+=1
        audit.check(same(load(folder/'final.pt'),load(folder/f"checkpoint_{cfg['updates']:04d}.pt")),'final_equals_final_checkpoint',folder.name)
        lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];ref_lines=[[json.loads(x) for x in (q/'training.jsonl').read_text().splitlines()] for q in (old,reset,scaled)]
        audit.check(len(lines)==cfg['updates'],'full_training_budget',folder.name);paired=[0,0,0]
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
        for j,name in enumerate(('retained','reset','scaled')):audit.counts[name+'_reference_updates_paired']+=paired[j]
        if inv['formal']:audit.check(paired==[2400,2400,2400],'every_formal_world_paired_both_references',folder.name)
        for step in (1,2101):
            if step<=cfg['updates']:v14.step_replay(folder,step,bank,replay,audit)
        b.social_protocol(folder,bank,replay,audit)
        recorded_branches(folder,reset,scaled,audit)
        agents=replay.remake_agents(seed,load(batch/f'prepared_{seed}.pt'))
        for a,state in zip(agents,load(folder/'final.pt')):a.load_state_dict(state);a.requires_grad_(False)
        projected=replay.projected_banks(agents,bank)
        for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
            path=folder/f'final_{mode}.npz';replay.replay(path,agents,agents,projected,cfg,0,91,mode,audit);z=arrays(path);worlds=b.social_worlds(bank,seed,p,0,cfg['eval_n'],False)
            for d in (0,1):
                ix=z['scout']==d
                for key,value in worlds[d].items():audit.check(np.array_equal(z[key][ix],value),'final_balanced_worlds_reconstructed',[folder.name,mode,d,key])
            for ref in (old,reset,scaled):
                rz=arrays(ref/f'final_{mode}.npz')
                for key in worlds[0]:audit.check(np.array_equal(z[key],rz[key]),'final_worlds_identical_both_references',[folder.name,mode,ref.name,key])
            b.final_statistics(folder,mode,audit);audit.counts['final_modes']+=1
        if arm in ('neither_replay','both_replay'):
            reference=reset if arm=='neither_replay' else scaled;ref_cfg=read(reference/'config.json')
            audit.check(cfg['updates']==ref_cfg['updates'],'endpoint_replay_reference_same_budget',folder.name)
            states=['initial.pt','after_first_update.pt','final.pt',*[f'checkpoint_{u:04d}.pt' for u in cfg['checkpoints']]]
            optimizers=['initial_optimizer.pt','after_first_optimizer.pt','final_optimizer.pt',*[f'optimizer_{u:04d}.pt' for u in cfg['checkpoints']]]
            if cfg['updates']>=2101:states.append('after_2101_update.pt');optimizers.append('after_2101_optimizer.pt')
            for name in states:audit.check(same(strip_scale(load(folder/name)),strip_scale(load(reference/name))),'endpoint_replay_all_base_state_bitwise',[folder.name,name])
            for name in optimizers:audit.check(same(load(folder/name),load(reference/name)),'endpoint_replay_Adam_bitwise',[folder.name,name])
            ref_log=[json.loads(x) for x in (reference/'training.jsonl').read_text().splitlines()]
            audit.check(lines==ref_log,'endpoint_replay_full_training_log_exact',folder.name)
            c1=read(folder/'curve.json');c0=read(reference/'curve.json')
            audit.check([{k:v for k,v in q.items() if k!='state_sha256'} for q in c1]==[{k:v for k,v in q.items() if k!='state_sha256'} for q in c0],'endpoint_replay_curve_scores_exact',folder.name)
            for u in cfg['checkpoints']:audit.check(read(folder/f'protocol_{u:04d}.json')==read(reference/f'protocol_{u:04d}.json'),'endpoint_replay_protocol_metrics_exact',[folder.name,u])
            for path in folder.glob('*.npz'):
                counterpart=reference/path.name
                if counterpart.exists():
                    z=arrays(path);old_z=arrays(counterpart)
                    for key,expected in old_z.items():
                        actual=z['policy_h'] if arm=='both_replay' and key=='h' else z[key]
                        audit.check(np.array_equal(actual,expected),'endpoint_replay_raw_array_exact_or_explicit_effective_h',[folder.name,path.name,key])
        audit.counts['runs']+=1
    expected=len(args['seeds'])*len(args['partitions'])*len(inv['arms'])
    audit.check(audit.counts['runs']-initial_count==expected and done['new_social_runs']==expected,'all_new_runs_audited',str(batch))
    audit.counts['training_communications']+=expected*args['social_updates']*args['batch'];audit.counts['training_actions']+=2*expected*args['social_updates']*args['batch']
    return inv



def write_result(path,audit,extra):
    result=dict(passed=not audit.failures,created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),check_count=sum(audit.checks.values()),checks=dict(audit.checks),counts=dict(audit.counts),failures=audit.failures,independent_helpers={str(p):sha(p) for p in (PREVIOUS,PROJECT/'redesign_v0.14/audit_reset.py',PROJECT/'redesign_v0.13/audit_v13.py',PROJECT/'redesign_v0.11/audit_query_replay.py',PROJECT/'redesign_v0.12/audit_receiver.py')},**extra)
    if path.exists():path.rename(path.with_name(path.stem+'_previous_'+datetime.now().strftime('%Y%m%d%H%M%S')+path.suffix))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(path),passed=result['passed'],checks=result['check_count'],counts=result['counts'],failures=result['failures'][:20]),ensure_ascii=False,indent=2),flush=True);assert result['passed']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/branches_001');ap.add_argument('--preflight',action='store_true');args=ap.parse_args();audit=b.Audit();torch.set_num_threads(1)
    if args.preflight:
        batches=[ROOT/'results/smoke_001',ROOT/'results/smoke_002'];invs=[]
        for batch in batches:archive_source(batch);invs.append(audit_batch(batch,audit))
        audit.check(invs[0]['source_hashes']==invs[1]['source_hashes'],'smokes_same_frozen_sources')
        source_check(read(ROOT/'source_receipt.json'),audit,True)
        calibration=PROJECT/'redesign_v0.15/calibration_001';inherited_calibration(calibration,PROJECT/'redesign_v0.15/results/scaled_001',audit)
        write_result(ROOT/'preflight_qa.json',audit,dict(source_hashes=invs[0]['source_hashes'],source_receipt_sha256=sha(ROOT/'source_receipt.json'),calibration_complete_sha256=sha(calibration/'calibration_complete.json'),smoke_batches=[str(p) for p in batches],scope='Independent branch implementation; exact old00/11 parameter, Adam, behavior, effective/raw h replay; both new arms first/2101 loss/Adam and all protocol/final modes. Calibration forward proof inherited unchanged from v15.'))
    else:
        batch=args.root.resolve();archive_source(batch);inv=audit_batch(batch,audit)
        write_result(batch/'audit_execution.json',audit,dict(source_hashes=inv['source_hashes'],source_receipt_sha256=sha(batch/'source_receipt.json'),calibration_complete_sha256=sha(Path(inv['args']['calibration'])/'calibration_complete.json'),scope='24 new branch runs, all original sources and unchanged calibration, independent first96 branch transforms, first/2101 exact Adam, full frozen checkpoints/world pairing/endpoints/protocol. No new private fitting, alpha calibration, or independent source.'))


if __name__=='__main__':main()
