"""Paired policy-gradient gain study; frozen v0.8 game and independent agents.

The training loop is copied from v0.8 with one loss-combination change plus gain
metadata. Native rewards, critic target/loss, entropy, RNG and Adam are unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import platform
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

STUDY = Path(__file__).resolve().parent
WORK = STUDY.parents[1]
ROOT = WORK / 'redesign_v0.8'
sys.path.insert(0, str(ROOT))
import run_experiment as base
from run_experiment import (remake_agents, projected_banks, state_sha, split_maps,
    write_json, SITES, HISTORY, MAPS, sha, training_schedule, evaluate,
    training_batch_plan, rollout, ImageBank)
sys.path.insert(0, str(ROOT))
from run_controls import run as run_personal_control

SEEDS = (28101, 28102, 28103, 28104)
KINDS = ('additive', 'joint')
GAINS = (1, 3)
CONDITIONS = {}
for split in (1, 2, 3, 0):
    for kind in KINDS:
        base_name = f'split{split}_{kind}' if split else f'full_{kind}'
        for gain in GAINS:
            CONDITIONS[f'{base_name}_gain{gain}'] = dict(base_condition=base_name,
                plan=dict(base.CONDITIONS[base_name]), policy_gain=gain)
UPDATES, BATCH, EVAL_N = 2400, 512, 9600
EXPECTED_V08 = {
    'camp.py': '21ad5ff2eecf871ff2e8e17e5623875f55d701e66302b741a46658c0cec81854',
    'run_experiment.py': '485221ac4c90f69b88047f380edfaf4922973cff986c8bfa5c47e700cab13d62',
    'run_controls.py': '38ef2921e9e57342bd79063af8bd33d0ee35db4b7352672937ac710f7b4ded73',
    '固定执行方案.md': 'cf3d92f5eb9755a83b11dcbd3f1792f5474203ce445624fc7b347ccf499996b4',
}


def combine_loss(policy, value, entropy, entropy_weight, gain):
    if gain not in GAINS:
        raise ValueError('Unknown policy gain')
    # Literal baseline branch preserves the original operation order.
    if gain == 1:
        return policy + value - entropy_weight * entropy
    return gain * policy + value - entropy_weight * entropy


def validate_base():
    for name, expected in EXPECTED_V08.items():
        assert sha(ROOT / name) == expected, 'Reviewed v0.8 source changed: ' + name
    for module, relative in [('run_experiment', 'redesign_v0.8/run_experiment.py'),
                             ('run_controls', 'redesign_v0.8/run_controls.py'),
                             ('camp', 'redesign_v0.8/camp.py'),
                             ('agents', 'redesign_v0.4/agents.py'),
                             ('run_pilot', 'redesign_v0.4/run_pilot.py'),
                             ('resource_env', 'redesign_v0.4/resource_env.py')]:
        assert Path(sys.modules[module].__file__).resolve() == WORK / relative


def train(seed, name, plan, prepared, bank, out, updates, batch, eval_n, policy_gain):
    out.mkdir(parents=True, exist_ok=False)
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'], plan['representation'])
    banks = projected_banks(agents, bank)
    frozen = [{k: v.clone() for k, v in a.state_dict().items() if k.startswith('project.') or k == 'input_transform'} for a in agents]
    init = state_sha(agents)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    boundary = updates - min(300, updates // 6)
    checkpoints = sorted(set(x for x in [0,100,300,600,1200,1800,boundary,updates] if x<=updates))
    train_ids, held_ids = split_maps(plan['split'])
    write_json(out / 'config.json', dict(seed=seed, condition=name, plan=plan, updates=updates, batch=batch,
        eval_n=eval_n, policy_gain=policy_gain, checkpoint_eval_n=1200, checkpoints=checkpoints, sites=SITES, history_dim=HISTORY,
        train_map_ids=train_ids.tolist(), heldout_map_ids=held_ids.tolist(), map_table=MAPS.tolist(),
        learning_rate=.0007, entropy_coefficient=.02, entropy_off_after=boundary, gamma=1,
        training_seed_offset=60000000, choices_per_world=2,
        reward_formula='(1-lambda)*mean(successes)+lambda*product(successes)',
        receiver_policy_score='sum of two action log probabilities', trace_schema='paired_world_v1', initial_sha256=init,
        trainable_initial_sha256=state_sha(agents, trainable_only=True),
        receiver_initial_sha256=state_sha(agents, receiver_only=True),
        input_transforms=[a.input_transform.tolist() for a in agents],
        source_hashes={p.name:sha(p) for p in (ROOT/'camp.py', ROOT/'run_experiment.py', ROOT/'固定执行方案.md')},
        prepared_source=dict(path=str(out.parent/f'prepared_{seed}.pt'), sha256=sha(out.parent/f'prepared_{seed}.pt')),
        trainable_parameters=[sum(p.numel() for p in a.parameters() if p.requires_grad) for a in agents]))
    levels, order = training_schedule(seed, updates, plan['schedule'])
    write_json(out / 'training_schedule.json', dict(levels=levels.tolist(), batch_identities=order.tolist(),
        sampling='access subset weighted by its number of legal training maps, then uniform map; independent access RNG'))
    optimizers = [torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=.0007) for a in agents]
    curve, started = [], time.monotonic()
    with (out/'training.jsonl').open('w') as log:
        for update in range(updates+1):
            if update in checkpoints:
                scores = evaluate(agents,banks,bank,plan,seed+69100000,1200)
                curve.append(dict(update=update,scores=scores)); write_json(out/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],out/f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(seed=seed,condition=name,update=update,
                    normal_single=round(scores['normal']['single_accuracy'],4),
                    normal_both=round(scores['normal']['both_accuracy'],4),
                    unseen_both=scores['normal']['map_groups']['unseen']['both_accuracy'])),flush=True)
            if update==updates: break
            bid=int(order[update]); level=int(levels[update])
            train_plan=training_batch_plan(seed,bid,level,plan)
            stats,learning,_=rollout(agents,banks,bank,train_plan,seed*100000+60000000+bid+1,
                                     batch,training=True,greedy=False,split='train')
            weight=.02 if update<boundary else 0.
            info=[]
            for who in range(2):
                losses=[]; components=[]
                for lp,ent,value,target in learning[who]:
                    ploss=-(lp*(target-value).detach()).mean()
                    vloss=.5*F.mse_loss(value,target)
                    losses.append(combine_loss(ploss, vloss, ent.mean(), weight, policy_gain))
                    components.append(dict(policy_loss=float(ploss.detach()),weighted_policy_loss=float((policy_gain*ploss).detach()),value_loss=float(vloss.detach()),entropy=float(ent.mean().detach())))
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                optimizers[who].zero_grad(); loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(agents[who].parameters(),2.))
                optimizers[who].step()
                info.append(dict(loss=float(loss.detach()),gradient_norm=norm,components=components))
            log.write(json.dumps(dict(update=update+1,batch_identity=bid,active_sites=level,
                allowed_sites=train_plan['allowed_sites'],map_pool=train_plan['map_pool'],
                world_sha256=stats['world_sha256'],reward=stats['mean_reward'],
                single_accuracy=stats['single_accuracy'],both_accuracy=stats['both_accuracy'],
                reward_variance=stats['reward_variance'],positive_rewards=stats['positive_rewards'],
                outcome_counts=stats['outcome_counts'],entropy_weight=weight,policy_gain=policy_gain,agents=info))+'\n')
            if (update+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,plan,seed+69200000,eval_n,out)
    for a,snapshot in zip(agents,frozen):
        assert all(torch.equal(v,a.state_dict()[k]) for k,v in snapshot.items())
    torch.save([a.state_dict() for a in agents],out/'final.pt')
    torch.save([o.state_dict() for o in optimizers],out/'final_optimizer.pt')
    result=dict(seed=seed,condition=name,plan=plan,policy_gain=policy_gain,updates=updates,batch=batch,scores=final,
        initial_sha256=init,final_sha256=state_sha(agents),seconds=time.monotonic()-started,frozen_projection_verified=True)
    write_json(out/'result.json',result)
    return result


def source_files():
    return [STUDY / n for n in ('runner.py', 'plan.md', 'analysis.py', 'probe.py', 'preflight.py')] + [
        WORK / 'research_program' / n for n in ('v08_formation_trajectory.py', 'v08_receiver_coverage.py')] + [
        ROOT / n for n in EXPECTED_V08] + [WORK / 'redesign_v0.4' / n for n in (
            'agents.py', 'run_pilot.py', 'resource_env.py', 'data/manifest.json', 'data/features.npz')]


def run_study(out):
    validate_base()
    preflight = json.loads((STUDY / 'preflight_result.json').read_text())
    assert preflight['status'] == 'passed' and preflight['runner_sha256'] == sha(Path(__file__))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p): sha(p) for p in source_files()}
    for path in source_hashes:
        src = Path(path)
        if src.suffix in ('.py', '.md'):
            dest = out / 'source_snapshot' / src.relative_to(WORK)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    write_json(out / 'manifest.json', dict(seeds=list(SEEDS), conditions=CONDITIONS,
        updates=UPDATES, batch=BATCH, eval_n=EVAL_N, expected_social_runs=64,
        expected_social_updates=153600, expected_social_worlds=78643200,
        expected_social_choices=157286400, native_reward_changed_by_gain=False,
        source_hashes=source_hashes, preflight_sha256=sha(STUDY/'preflight_result.json'),
        python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__),
        device='cpu', torch_num_threads=1, started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Four new initial seeds, old development photos; gain only multiplies REINFORCE terms.'))
    started=time.monotonic(); context={}
    try:
        bank=ImageBank(); personal=[]
        for seed in SEEDS:
            context=dict(phase='personal_control', seed=seed)
            prepared=base.prepare(seed,bank,out)
            personal.append(run_personal_control(seed,'identity',prepared,bank,out))
        summary=dict(complete=True,passed=all(r['passed'] for r in personal),runs=4,results=personal)
        write_json(out/'individual_controls/summary.json',summary)
        if not summary['passed']:
            raise RuntimeError('Inherited personal control gate failed; preserve all seeds and results')
        write_json(out/'social_launch_gate.json',dict(personal_runs=4,all_passed=True,
            personal_summary_sha256=sha(out/'individual_controls/summary.json'),
            social_runs_planned=64, diagnostic_weights_transferred=False))
        results=[]
        for seed in SEEDS:
            prepared=torch.load(out/f'prepared_{seed}.pt',weights_only=True,map_location='cpu')
            for name,condition in CONDITIONS.items():
                context=dict(phase='social',seed=seed,condition=name)
                result=train(seed,name,condition['plan'],prepared,bank,out/f's{seed}_{name}',
                             UPDATES,BATCH,EVAL_N,condition['policy_gain'])
                results.append(dict(seed=seed,condition=name,final_sha256=result['final_sha256']))
                write_json(out/'status.json',dict(status='running',completed_runs=len(results),expected_runs=64,context=context))
        assert len(results)==64
        assert all(sha(Path(p))==h for p,h in source_hashes.items()), 'Frozen source changed during execution'
        write_json(out/'status.json',dict(status='complete',completed_runs=64,expected_runs=64,
            elapsed_seconds=time.monotonic()-started,runs=results,finished_utc=datetime.now(timezone.utc).isoformat()))
        print('COMPLETE',flush=True)
    except Exception as e:
        write_json(out/'failure.json',dict(status='failed',context=context,error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run_study(args.out)
