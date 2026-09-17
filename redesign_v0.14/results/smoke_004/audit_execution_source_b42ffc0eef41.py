"""Independent source, reset, matched-stream and policy audit for v0.14.

Reuse only the separately implemented v13 auditor's social replay/math helpers;
never call a training update or re-audit the unchanged private training stage.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from itertools import product
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
BASE_PATH=PROJECT/'redesign_v0.13/audit_v13.py'
spec=importlib.util.spec_from_file_location('v14_independent_v13_helpers',BASE_PATH)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
read=b.read;sha=b.sha;load=b.load;same=b.same;arrays=b.arrays
PREFIXES=('memory.','slot_phi.')


def archive_source(batch):
    path=batch/'audit_execution_source.py'
    if path.exists() and sha(path)!=sha(__file__):path.rename(path.with_name('audit_execution_source_'+sha(path)[:12]+'.py'))
    path.write_bytes(Path(__file__).read_bytes())


def source_check(receipt,audit,formal=False):
    source=Path(receipt['source_batch']);inv=read(source/'invocation.json');done=read(source/'training_complete.json')
    audit.check(done['status']=='complete' and receipt['source_training_hashes']==inv['source_hashes'],'inherited_source_complete_and_training_hashes',str(source))
    for path,digest in receipt['files'].items():audit.check(sha(path)==digest,'consumed_source_file_unchanged',path)
    for path,digest in receipt['source_training_hashes'].items():audit.check(sha(path)==digest,'inherited_training_source_unchanged',path)
    if formal:
        manifest=read(source/'completion_manifest.json');old_audit=read(source/'audit_execution.json')
        audit.check(manifest['status']=='complete' and old_audit['passed'],'source_completion_and_execution_audit_passed',str(source))
        for path,digest in receipt['files'].items():
            if Path(path).name=='completion_manifest.json':continue
            key=str(Path(path).relative_to(PROJECT));audit.check(manifest['artifacts'].get(key)==digest,'consumed_source_matches_completion_manifest',key)
        audit.check(old_audit['script_sha256']==sha(BASE_PATH),'reused_audit_helpers_match_v13_successful_formal_source')


def step_replay(folder,step,bank,replay,audit):
    cfg=read(folder/'config.json');prepared=load(folder.parent/f"prepared_{cfg['seed']}.pt");agents=replay.remake_agents(cfg['seed'],prepared,7,2,'identity')
    pre=folder/('initial.pt' if step==1 else 'checkpoint_2100.pt');before=load(pre)
    for a,state in zip(agents,before):
        a.load_state_dict(state)
        for key,param in a.named_parameters():param.requires_grad_(key.startswith(b.SOCIAL))
    opts=[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in agents]
    if step==1:audit.check(same([o.state_dict() for o in opts],load(folder/'initial_optimizer.pt')),'empty_fresh_Adam',folder.name)
    else:
        for o,state in zip(opts,load(folder/'optimizer_2100.pt')):o.load_state_dict(state)
    projected=replay.projected_banks(agents,bank);line=json.loads((folder/'training.jsonl').read_text().splitlines()[step-1]);weight=.02 if step<=2100 else 0.
    terms=replay.replay.__wrapped__(folder/f'train_{step:04d}.npz',agents,agents,projected,cfg,step-1,11,'stochastic',audit)
    for who,(a,opt) in enumerate(zip(agents,opts)):
        parts=[]
        for j,role in enumerate(('sender','receiver')):
            lp,en,value,target=terms[who][role];policy=-(lp*(target-value).detach()).mean();vl=.5*F.mse_loss(value,target);total=policy+vl-weight*en.mean();parts.append(total)
            values=dict(policy_loss=policy.item(),value_loss=vl.item(),entropy=en.mean().item(),total=total.item(),target_mean=target.mean().item())
            audit.check(all(abs(v-line['agents'][who]['parts'][j][k])<2e-6 for k,v in values.items()),'forensic_role_loss_independent',[folder.name,step,who,role])
        loss=torch.stack(parts).mean();opt.zero_grad(set_to_none=True);loss.backward()
        audit.check(abs(loss.item()-line['agents'][who]['loss'])<2e-6,'forensic_equal_role_loss',[folder.name,step,who])
        for role,prefix in (('sender',('send_',)),('receiver',b.SOCIAL[1:])):
            norm=torch.nn.utils.clip_grad_norm_([p for k,p in a.named_parameters() if k.startswith(prefix)],2.)
            audit.check(abs(norm.item()-line['agents'][who]['gradient_norm_by_role'][role])<2e-6,'forensic_separate_gradient_clip',[folder.name,step,who,role])
        audit.check(all(p.grad is None for p in a.parameters() if not p.requires_grad),'frozen_no_gradient',[folder.name,step,who])
    for opt in opts:opt.step()
    post=folder/('after_first_update.pt' if step==1 else 'after_2101_update.pt');po=folder/('after_first_optimizer.pt' if step==1 else 'after_2101_optimizer.pt')
    b.compare_tensors(audit,[a.state_dict() for a in agents],load(post),'forensic_Adam_parameter_bitwise',[folder.name,step])
    b.compare_tensors(audit,[o.state_dict() for o in opts],load(po),'forensic_Adam_moments_bitwise',[folder.name,step]);audit.counts['exact_optimizer_updates_replayed']+=2


def audit_batch(batch,audit):
    inv=read(batch/'invocation.json');args=inv['args'];receipt=read(batch/'source_receipt.json');source=Path(args['source_batch']);hashes=inv['source_hashes'];done=read(batch/'training_complete.json')
    audit.check(done['status']=='complete' and done['new_private_runs']==0 and done['source_hashes']==hashes,'new_social_only_complete',str(batch))
    if inv['formal']:
        audit.check(args['seeds']==[31101,31102,31103,31104] and args['partitions']==[1,2,3] and args['social_updates']==2400 and args['batch']==512 and args['eval_n']==9600 and inv['arms']==['reset'],'formal_matrix_fixed')
        gate=read(batch/'frozen_sources/preflight_qa.json');audit.check(gate['passed'] and gate['source_hashes']==hashes and gate['source_receipt_sha256']==sha(ROOT/'source_receipt.json'),'formal_success_gate_bound')
        audit.check(receipt==read(ROOT/'source_receipt.json'),'formal_exact_source_receipt')
    source_check(receipt,audit,inv['formal'])
    for i,(path,digest) in enumerate(hashes.items()):
        audit.check(sha(path)==digest,'current_source_matches_executed',path);snapshot=batch/'frozen_sources'/f'{i:02d}_{Path(path).name}'
        if snapshot.exists():audit.check(sha(snapshot)==digest,'archived_source_matches_executed',path)
        else:audit.check(Path(path).name=='features.npz','only_features_not_duplicated',path)
    torch.set_num_threads(1);replay=b.import_replay();bank=replay.ImageBank();fingerprints=[]
    for seed in args['seeds']:
        pfile=batch/f'prepared_{seed}.pt';audit.check(sha(pfile)==sha(source/pfile.name),'prepared_copy_bitwise_same_source',seed);fingerprints.append(b.state_digest(load(pfile)))
    audit.check(len(set(fingerprints))==len(args['seeds']),'distinct_original_source_preparations')
    for seed,p,arm in product(args['seeds'],args['partitions'],inv['arms']):
        folder=batch/f'social_s{seed}_p{p}_{arm}';old=source/f'social_s{seed}_p{p}_control';private=source/f'private_s{seed}_p{p}_control';cfg=read(folder/'config.json');oc=read(old/'config.json')
        initial=load(folder/'initial.pt');prior=load(private/'initial.pt')['agents'];retained=load(old/'initial.pt');expected=copy.deepcopy(retained)
        for who in (0,1):
            for key in expected[who]:
                if arm=='reset' and key.startswith(PREFIXES):expected[who][key]=prior[who][key]
        b.compare_tensors(audit,initial,expected,'initial_exact_source_prefix_mix',[folder.name])
        audit.check(same(initial,load(folder/'source_state.pt')),'source_state_equals_initial',folder.name)
        if arm=='reset':audit.check(same(initial,prior),'reset_complete_pair_equals_original_private_initial',folder.name)
        for who in (0,1):
            keys=[k for k in initial[who] if k.startswith(PREFIXES)];audit.check(len(keys)==6 and sum(initial[who][k].numel() for k in keys)==145122,'reset_prefix_six_tensors_145122', [folder.name,who])
            actual_changed=[k for k in keys if not torch.equal(retained[who][k],prior[who][k])];audit.check('memory.weight_hh' not in actual_changed,'single_step_zero_history_recurrent_matrix_unchanged',[folder.name,who])
        for key in ('batch','eval_n','learning_rate','entropy_off_after','entropy_coefficient','role_loss_weights','gradient_clip','training_pool','parameter_partition','rng_world_namespace','rng_policy_namespace','training_policy_stream','evaluation_policy_stream'):
            audit.check(cfg[key]==oc[key],'same_source_social_settings',[folder.name,key])
        audit.check(cfg['updates']==args['social_updates'] and cfg['source_hashes']==hashes and cfg['training_pool']==b.groups(p)['old'].tolist(),'run_budget_and_old_support',folder.name)
        audit.check(same(load(folder/'initial_optimizer.pt'),load(old/'initial_optimizer.pt')),'same_source_fresh_Adam_initial',folder.name)
        for key,expected_path in [('private_initial',private/'initial.pt'),('retained_social_initial',old/'initial.pt')]:
            entry=cfg['source_inputs'][key];audit.check(Path(entry['path'])==expected_path and entry['sha256']==sha(expected_path),'config_source_inputs_bound',[folder.name,key])
        audit.check(cfg['source_checkpoint']['sha256']==sha(folder/'source_state.pt'),'source_state_fingerprint',folder.name)
        for cp in cfg['checkpoints']:
            state=load(folder/f'checkpoint_{cp:04d}.pt')
            for who in (0,1):
                for key,value in state[who].items():
                    if not key.startswith(b.SOCIAL):audit.check(torch.equal(value,initial[who][key]),'frozen_every_checkpoint',[folder.name,cp,who,key])
            audit.counts['checkpoints']+=1
        audit.check(same(load(folder/'final.pt'),load(folder/f"checkpoint_{cfg['updates']:04d}.pt")),'final_equals_last_checkpoint',folder.name)
        lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];oldlines=[json.loads(x) for x in (old/'training.jsonl').read_text().splitlines()];audit.check(len(lines)==cfg['updates'],'full_update_budget',folder.name)
        paired=0
        for step,line in enumerate(lines):
            worlds=b.social_worlds(bank,seed,p,step,cfg['batch'],True);digest=b.world_digest(worlds)
            audit.check(line['stats']['world_sha256']==digest,'all_worlds_independently_reconstructed',[folder.name,step])
            if step<len(oldlines):audit.check(digest==oldlines[step]['stats']['world_sha256'],'all_worlds_exactly_match_control',[folder.name,step]);paired+=1
            audit.check(line['update']==step+1 and line['entropy_weight']==(.02 if step<2100 else 0.),'fixed_update_entropy_schedule',[folder.name,step])
            audit.check(line['stats']['map_groups']['old']['n']==cfg['batch'] and line['stats']['map_groups']['added']['n']==line['stats']['map_groups']['sealed']['n']==0,'old_only_exposure',[folder.name,step])
            for who,person in enumerate(line['agents']):
                audit.check([q['role'] for q in person['parts']]==['sender','receiver'] and all(q['n']==cfg['batch']//2 for q in person['parts']),'equal_roles_and_directions',[folder.name,step,who]);totals=[q['policy_loss']+q['value_loss']-line['entropy_weight']*q['entropy'] for q in person['parts']]
                audit.check(abs(np.mean(totals)-person['loss'])<2e-6 and np.isfinite(list(person['gradient_norm_by_role'].values())).all(),'logged_role_weights_and_finite',[folder.name,step,who])
            audit.counts['updates']+=1
        audit.counts['source_control_updates_paired']+=paired
        if inv['formal']:audit.check(paired==2400,'all_formal_steps_paired_with_completed_control',folder.name)
        for step in (1,2101):
            if step<=cfg['updates']:step_replay(folder,step,bank,replay,audit)
        b.social_protocol(folder,bank,replay,audit)
        agents=replay.remake_agents(seed,load(batch/f'prepared_{seed}.pt'),7,2,'identity')
        for a,state in zip(agents,load(folder/'final.pt')):a.load_state_dict(state);a.requires_grad_(False)
        projected=replay.projected_banks(agents,bank)
        for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
            path=folder/f'final_{mode}.npz';replay.replay(path,agents,agents,projected,cfg,0,91,mode,audit);z=arrays(path);oz=arrays(old/f'final_{mode}.npz');worlds=b.social_worlds(bank,seed,p,0,cfg['eval_n'],False)
            for d in (0,1):
                ix=z['scout']==d
                for k in worlds[d]:audit.check(np.array_equal(z[k][ix],worlds[d][k]),'final_balanced_worlds_independent',[folder.name,mode,d,k]);audit.check(np.array_equal(z[k],oz[k]),'final_worlds_identical_control',[folder.name,mode,k])
            b.final_statistics(folder,mode,audit);audit.counts['final_modes']+=1
        if arm=='retained_replay':
            audit.check(cfg['updates']==oc['updates'],'retained_replay_budget_equals_source',folder.name)
            for filename in ('initial.pt','initial_optimizer.pt','after_first_update.pt','after_first_optimizer.pt','final.pt','final_optimizer.pt',*[f'checkpoint_{t:04d}.pt' for t in cfg['checkpoints']],*[f'optimizer_{t:04d}.pt' for t in cfg['checkpoints']]):audit.check(same(load(folder/filename),load(old/filename)),'retained_replay_all_state_and_Adam_bitwise',filename)
            audit.check(lines==oldlines,'retained_replay_training_log_exact')
            for filename in ('curve.json',*[f'protocol_{t:04d}.json' for t in cfg['checkpoints']]):audit.check(read(folder/filename)==read(old/filename),'retained_replay_score_and_protocol_exact',filename)
            for path in folder.glob('*.npz'):
                counterpart=old/path.name
                if counterpart.exists():audit.check(same(arrays(path),arrays(counterpart)),'retained_replay_raw_arrays_exact',path.name)
        audit.counts['runs']+=1
    expected=len(args['seeds'])*len(args['partitions'])*len(inv['arms']);audit.check(audit.counts['runs']==expected and done['new_social_runs']==expected,'all_new_runs_audited',str(batch));audit.counts['training_communications']+=expected*args['social_updates']*args['batch'];audit.counts['training_actions']+=2*expected*args['social_updates']*args['batch']
    return inv


def write_result(path,audit,extra):
    result=dict(passed=not audit.failures,created_utc=datetime.now(timezone.utc).isoformat(),check_count=sum(audit.checks.values()),checks=dict(audit.checks),counts=dict(audit.counts),failures=audit.failures,script_sha256=sha(__file__),independent_helpers={str(p):sha(p) for p in (BASE_PATH,PROJECT/'redesign_v0.11/audit_query_replay.py',PROJECT/'redesign_v0.12/audit_receiver.py')},**extra)
    if path.exists():path.rename(path.with_name(path.stem+'_previous_'+datetime.now().strftime('%Y%m%d%H%M%S')+path.suffix))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(path),passed=result['passed'],checks=result['check_count'],counts=result['counts'],failures=result['failures'][:20]),ensure_ascii=False,indent=2));assert result['passed']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/reset_001');ap.add_argument('--preflight',action='store_true');args=ap.parse_args();audit=b.Audit()
    if args.preflight:
        batches=[ROOT/'results/smoke_003',ROOT/'results/smoke_004'];invs=[]
        for batch in batches:archive_source(batch);invs.append(audit_batch(batch,audit))
        audit.check(invs[0]['source_hashes']==invs[1]['source_hashes'],'both_smokes_same_frozen_source')
        receipt=read(ROOT/'source_receipt.json');source_check(receipt,audit,True)
        write_result(ROOT/'preflight_qa.json',audit,dict(source_hashes=invs[0]['source_hashes'],source_receipt_sha256=sha(ROOT/'source_receipt.json'),smoke_batches=[str(x) for x in batches],scope='Short exact retained-control replay plus reset prefix mix; long reset covers entropy-off step2101 and exact post-Adam. No repeated private-stage audit; formal-source integrity verified before gate.'))
    else:
        batch=args.root.resolve();archive_source(batch);inv=audit_batch(batch,audit)
        write_result(batch/'audit_execution.json',audit,dict(source_hashes=inv['source_hashes'],source_receipt_sha256=sha(batch/'source_receipt.json'),scope='New social reset runs only: source prefix replacement, every frozen checkpoint/Adam step, all exogenous worlds paired to existing controls, step1/2101 independent native loss/gradient/Adam, all final modes and all protocol logits/metrics. Inherited RNG namespace proof remains v13 evidence; actual pairing verified here.'))

if __name__=='__main__':main()
