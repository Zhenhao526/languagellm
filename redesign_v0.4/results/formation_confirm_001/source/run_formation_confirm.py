"""New-seed confirmation of formation, plus two explicitly bounded controls.

The old training and evaluation modules are imported without modification.
The blocked condition samples a sender but delivers constant zero from update 1.
The substitutable condition always obtains native reward 1; balanced collection
is only a common diagnostic in that condition, never its native objective.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from agents import draw
from run_pilot import ImageBank, make_agents, individual_practice, write_json
from resource_env import sample_scenes, transition
from run_curriculum import private_public, grad_norm, entropy
from curriculum_eval import sample_curriculum, evaluate, message_intervention, coordination_bounds

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('course_communication', 'direct_communication', 'course_blocked', 'course_substitutable')
MODES = (('normal', 'normal', True), ('shuffle', 'shuffle', True),
         ('blank', 'blank', True), ('stochastic', 'normal', False),
         ('stochastic_blank', 'blank', False))
CONFIG = {
    'seeds': list(range(1201, 1211)), 'conditions': list(CONDITIONS),
    'updates': 1200, 'course_updates': 600, 'exploration_updates': 900,
    'batch_size': 1024, 'horizon': 16, 'learning_rate': .0003,
    'action_entropy_coefficient': .05, 'signalling_coefficient': 0.,
    'gamma': 0., 'capacity': 1, 'initial_inventory': [0, 0],
    'practice_updates': 200, 'practice_batch_size': 64, 'practice_min_accuracy': .8,
    'evaluation_n': 8192, 'checkpoint_evaluation_n': 2048, 'intervention_n': 4096,
    'checkpoints': [0, 20, 50, 100, 200, 400, 600, 650, 750, 900, 1000, 1200],
    'engineering_criteria': {'course_success': .9, 'full_success': .85,
                             'full_shuffle_gap': .1, 'direction_shuffle_gap': .1},
    'fixed_budget_no_success_based_stopping': True,
    'backbone': 'frozen official DINOv2 ViT-L/14; unchanged cached photograph features',
    'architecture': 'unchanged ResourceAgent, independent parameters; no private history input',
    'blocked_loss': 'same sampled send_logp plus action_logp REINFORCE; only message delivery changes',
    'substitutable_native_reward': 'always 1: any two collected objects satisfy substitutable demand',
    'substitutable_scope': 'zero coordination-value boundary, not continuous ecological effect',
    'primary_comparisons': ['course_communication minus direct_communication',
                            'course_communication minus course_blocked native'],
    'evaluation_scope': 'same held-out photographs as earlier exploratory batches; new training seeds, no new image confirmation set',
    'training_device': 'cpu', 'torch': str(torch.__version__), 'python': platform.python_version(),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_digest(states):
    h = hashlib.sha256()
    for i, state in enumerate(states):
        for key in sorted(state):
            h.update(f'{i}/{key}'.encode())
            h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def schedule(update, condition, config=CONFIG):
    if condition not in CONDITIONS or not 1 <= update <= config['updates']:
        raise ValueError('invalid condition or update')
    task = 'curriculum' if condition != 'direct_communication' and update <= config['course_updates'] else 'full'
    return {'task': task,
            'stage': 'course' if task == 'curriculum' else ('full_exploration' if update <= config['exploration_updates'] else 'pure_reward'),
            'action_entropy_weight': config['action_entropy_coefficient'] if update <= config['exploration_updates'] else 0.,
            'aux_weight': 0., 'message_blocked': condition == 'course_blocked',
            'substitutable': condition == 'course_substitutable'}


def delivered_to_receivers(sent, condition):
    return [torch.zeros_like(sent[1 - i]) if condition == 'course_blocked' else sent[1 - i].detach()
            for i in range(2)]


def native_reward(balanced, condition):
    return np.ones_like(balanced) if condition == 'course_substitutable' else balanced.copy()


def load_agents(seed, states):
    agents = make_agents(seed)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
    pointers = [p.data_ptr() for agent in agents for p in agent.parameters()]
    assert len(set(pointers)) == len(pointers)
    return agents


@torch.no_grad()
def evaluate_bundle(agents, bank, seed, condition, n, horizon, out=None, prefix='final'):
    result = {}
    for task in ('curriculum', 'full'):
        result[task] = {}
        for key, mode, greedy in MODES:
            stats, rows = evaluate(agents, bank, seed, n=n, task=task, mode=mode,
                                   greedy=greedy, horizon=horizon, trace=out is not None)
            stats['assessment_scope'] = 'common balanced-resource diagnostic'
            result[task][key] = stats
            if out is not None:
                trace_file = out / f'{prefix}_{task}_{key}_trace.jsonl'
                with trace_file.open('w') as f:
                    for row in rows:
                        row['native_reward'] = 1. if condition == 'course_substitutable' else row['reward']
                        row['native_channel'] = condition != 'course_blocked' or mode == 'blank'
                        f.write(json.dumps(row) + '\n')
    native = {}
    for task in ('curriculum', 'full'):
        gkey = 'blank' if condition == 'course_blocked' else 'normal'
        skey = 'stochastic_blank' if condition == 'course_blocked' else 'stochastic'
        native[task] = {
            'greedy_success': 1. if condition == 'course_substitutable' else result[task][gkey]['mean_reward_per_step'],
            'stochastic_success': 1. if condition == 'course_substitutable' else result[task][skey]['mean_reward_per_step'],
            'greedy_diagnostic_key': gkey, 'stochastic_diagnostic_key': skey,
            'reward_rule': 'any resource pair succeeds' if condition == 'course_substitutable' else 'one food plus one water',
            'delivered_channel': 'constant symbol 0' if condition == 'course_blocked' else 'partner symbol',
        }
    result['native_task'] = native
    return result


def train(seed, condition, states, bank, out, config):
    out.mkdir(exist_ok=False)
    agents = load_agents(seed, states)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    initial_digest = state_digest(states)
    optimizers = [torch.optim.Adam(a.parameters(), lr=config['learning_rate']) for a in agents]
    world_rng = np.random.default_rng(seed * 10000 + 101)
    policy_rng = [np.random.default_rng(seed * 10000 + 102 + i) for i in range(2)]
    n, horizon = config['batch_size'], config['horizon']
    public, _ = private_public(n, horizon)
    inventory = np.zeros((n, 2), np.int64)
    curve, start = [], time.monotonic()
    with (out / 'training_metrics.jsonl').open('w') as f:
        for completed in range(config['updates'] + 1):
            if completed in config['checkpoints']:
                torch.save([a.state_dict() for a in agents], out / f'checkpoint_{completed:04d}.pt')
                record = {'update': completed, **schedule(max(1, completed), condition, config),
                          **evaluate_bundle(agents, bank, seed + 700000, condition,
                                            config['checkpoint_evaluation_n'], horizon)}
                curve.append(record)
                write_json(out / 'learning_curve.json', curve)
                print(json.dumps({'seed': seed, 'condition': condition, 'update': completed,
                                  'native_full': record['native_task']['full']['greedy_success'],
                                  'balanced_normal': record['full']['normal']['mean_reward_per_step'],
                                  'balanced_shuffle': record['full']['shuffle']['mean_reward_per_step']}), flush=True)
            if completed == config['updates']:
                break
            plan = schedule(completed + 1, condition, config)
            kinds = sample_curriculum(world_rng, n) if plan['task'] == 'curriculum' else sample_scenes(world_rng, n)
            features, ids = bank.sample(kinds, 'train', world_rng)
            reps = [a.observe(features[:, i], public) for i, a in enumerate(agents)]
            sends = [draw(a.send(reps[i][1]), policy_rng[i]) for i, a in enumerate(agents)]
            sent = [s[0].detach() for s in sends]
            received = delivered_to_receivers(sent, condition)
            acts = [draw(a.act(*reps[i], received[i]), policy_rng[i]) for i, a in enumerate(agents)]
            actions = np.column_stack([a[0].numpy() for a in acts])
            _, balanced, _ = transition(inventory, kinds, actions, capacity=1)
            reward = native_reward(balanced, condition)
            target = torch.from_numpy((reward - 1).astype(np.float32))
            metrics = {'update': completed + 1, **plan,
                       'native_training_success': float(reward.mean()),
                       'balanced_training_diagnostic': float(balanced.mean()),
                       'world_input_sha256': hashlib.sha256(kinds.tobytes() + ids.tobytes()).hexdigest(),
                       'world_rng_after': copy.deepcopy(world_rng.bit_generator.state), 'agents': []}
            for i, agent in enumerate(agents):
                value = agent.baseline(reps[i][1])
                advantage = (target - value).detach()
                policy_loss = -((sends[i][1] + acts[i][1]) * advantage).mean()
                value_loss = .5 * F.mse_loss(value, target)
                loss = policy_loss + value_loss - plan['action_entropy_weight'] * acts[i][2].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                optimizers[i].zero_grad()
                loss.backward()
                norms = {name: grad_norm(getattr(agent, name)) for name in ('project', 'sender', 'actor', 'value')}
                clip = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.))
                optimizers[i].step()
                mixed = torch.from_numpy(kinds[:, i, 0] != kinds[:, i, 1])
                metrics['agents'].append({
                    'agent': i, 'loss': float(loss.detach()), 'policy_loss': float(policy_loss.detach()),
                    'value_loss': float(value_loss.detach()), 'gradient_norms': norms,
                    'gradient_norm_before_clip': clip,
                    'sender_entropy': float(sends[i][2].mean().detach()),
                    'mixed_action_entropy': float(acts[i][2][mixed].mean().detach()),
                    'sent_sha256': tensor_digest(sent[i].numpy()),
                    'received_sha256': tensor_digest(received[i].numpy()),
                    'received_nonzero': int(torch.count_nonzero(received[i])),
                    'policy_rng_after': copy.deepcopy(policy_rng[i].bit_generator.state),
                })
            f.write(json.dumps(metrics) + '\n')
            if (completed + 1) % 100 == 0:
                f.flush()
    final = {'seed': seed, 'condition': condition, 'updates': config['updates'],
             'joint_training_steps': config['updates'] * n, 'initial_parameter_sha256': initial_digest,
             'final_aux_weight': 0., 'final_action_entropy_weight': 0.,
             'trainable_parameters_per_agent': sum(p.numel() for p in agents[0].parameters()),
             **evaluate_bundle(agents, bank, seed + 800000, condition, config['evaluation_n'], horizon, out)}
    final['intervention'] = message_intervention(agents, bank, seed + 900000,
                                                n=config['intervention_n'], task='full', horizon=horizon)
    final['intervention']['assessment_scope'] = 'common diagnostic with partner-generated baseline; counterfactual channel opening for blocked training'
    final['seconds'] = time.monotonic() - start
    torch.save({'agents': [a.state_dict() for a in agents],
                'optimizers': [o.state_dict() for o in optimizers],
                'world_rng': world_rng.bit_generator.state,
                'policy_rng': [r.bit_generator.state for r in policy_rng],
                'torch_rng': torch.get_rng_state(), 'completed_updates': config['updates']},
               out / 'training_state_final.pt')
    write_json(out / 'result.json', final)
    write_json(out / 'trace_hashes.json', {p.name: sha(p) for p in sorted(out.glob('*_trace.jsonl'))})
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='formation_confirm_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / 'results' / args.name
    out.mkdir(exist_ok=False)
    config = copy.deepcopy(CONFIG)
    config['task_bounds'] = {t: coordination_bounds(t) for t in ('curriculum', 'full')}
    write_json(out / 'config.json', config)
    shutil.copyfile(ROOT / '完整研究_执行方案.md', out / 'execution_plan.md')
    source = out / 'source'
    source.mkdir()
    for name in ('run_formation_confirm.py', 'test_formation_confirm.py', 'run_curriculum.py',
                 'curriculum_eval.py', 'run_pilot.py', 'agents.py', 'resource_env.py'):
        shutil.copyfile(ROOT / name, source / name)
    write_json(out / 'source_hashes.json', {p.name: sha(p) for p in source.iterdir()})
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        shutil.copyfile(ROOT / 'data' / name, out / ('data_' + name))
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    results, preparation, start = [], {}, time.monotonic()
    for seed in config['seeds']:
        agents = make_agents(seed)
        before_sender = [{k: v.clone() for k, v in a.sender.state_dict().items()} for a in agents]
        practice = individual_practice(agents, bank, seed, config['practice_updates'], config['practice_batch_size'])
        states = copy.deepcopy([a.state_dict() for a in agents])
        sender_unchanged = all(torch.equal(a.sender.state_dict()[k], old[k])
                               for a, old in zip(agents, before_sender) for k in old)
        assert sender_unchanged
        prepared_file = out / f'prepared_s{seed}.pt'
        torch.save(states, prepared_file)
        preparation[str(seed)] = {'practice': practice, 'sender_head_unchanged': sender_unchanged,
                                 'checkpoint_sha256': sha(prepared_file), 'parameter_sha256': state_digest(states)}
        write_json(out / 'preparation.json', preparation)
        if any(p['heldout_need_sensitive_choice'] < config['practice_min_accuracy'] for p in practice):
            write_json(out / 'failed_preparation.json', {'seed': seed, 'practice': practice,
                                                        'action': 'stop without replacement seeds'})
            raise RuntimeError(f'preparation failed for {seed}; no seed replacement')
        for condition in config['conditions']:
            results.append(train(seed, condition, states, bank, out / f'{condition}_s{seed}', config))
            write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results),
                                        'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
