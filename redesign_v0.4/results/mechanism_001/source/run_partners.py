"""Four independent resource agents: fixed versus balanced rotating partners.

The only condition-dependent operation is which agents occupy a dyad. Two
independent camps keep the original pair-level consequences. Scene/photo slots
are paired across conditions; per-agent observations need not be identical.
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

from agents import ResourceAgent, draw
from run_pilot import ImageBank, individual_practice, make_agents, write_json
from run_curriculum import private_public, grad_norm
from curriculum_eval import sample_curriculum, coordination_bounds
from resource_env import sample_scenes, transition
from partner_eval import checkpoint_population, evaluate_population


ROOT = Path(__file__).resolve().parent
MATCHINGS = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]], dtype=np.int64)
MATCHINGS.flags.writeable = False
CONFIG = {
    'seeds': [101, 202, 303], 'conditions': ['fixed', 'rotating'],
    'population_size': 4, 'course_updates': 600, 'exploratory_full_updates': 300,
    'pure_reward_updates': 300, 'batch_size_per_agent': 1024, 'horizon': 16,
    'learning_rate': .0003, 'action_entropy_coefficient': .05,
    'signalling_coefficient': 0., 'gamma': 0., 'capacity': 1,
    'initial_inventory': [0, 0], 'preparation_seed_offset': 1000,
    'preparation_updates': 200, 'preparation_batch_size': 64,
    'preparation_accuracy_gate': .8,
    'checkpoints': [0, 20, 50, 100, 200, 400, 600, 650, 750, 900, 1000, 1200],
    'checkpoint_evaluation_n': 512, 'evaluation_n': 4096,
    'intervention_n': 2048, 'alignment_n': 2048,
    'fixed_matching': [[0, 1], [2, 3]],
    'rotation': 'Each three-update block contains all three perfect matchings once, in random order.',
    'partner_identity_input': False, 'signal_vocabulary_size': 5,
    'architecture': 'unchanged ResourceAgent, 83527 trainable parameters per independent interface',
    'backbone': 'frozen official DINOv2 ViT-L/14, cached original-photo features',
    'training_device': 'cpu', 'torch': torch.__version__, 'python': platform.python_version(),
    'source_course_batch': 'curriculum_001; retain its baseline curriculum irrespective of course-control result',
    'fixed_budget_no_success_based_stopping': True,
    'primary_metric': 'equal-weight full-task normal success and normal-minus-shuffle across all six pairs',
    'replicate_unit': 'population seed, not pair, direction or evaluation case',
    'engineering_edge_criteria': {'full_success': .85, 'shuffle_gap': .1, 'direction_shuffle_gap': .1},
    'population_no_message_expected_upper_bounds': {
        'full': {'original': 5 / 7, 'cross': 5 / 7, 'all': 13 / 21},
        'curriculum': {'original': .5, 'cross': .5, 'all': .5},
        'scope': 'Uniform IID scenes; one fixed local policy per agent, no partner/slot identity input; 256 distinct resource-choice policy populations enumerated.',
    },
    'scope': 'Rotating populations train on all six edges. Cross-pair performance is not equal-exposure zero-shot generalization.',
}


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def schedule(update, config=CONFIG):
    course_end = config['course_updates']
    exploratory_end = course_end + config['exploratory_full_updates']
    total = exploratory_end + config['pure_reward_updates']
    if not 1 <= update <= total:
        raise ValueError('update outside fixed budget')
    if update <= course_end:
        stage, task = 'course', 'curriculum'
    elif update <= exploratory_end:
        stage, task = 'full_exploration', 'full'
    else:
        stage, task = 'pure_reward', 'full'
    return {'stage': stage, 'task': task, 'aux_weight': 0.,
            'action_entropy_weight': config['action_entropy_coefficient'] if update <= exploratory_end else 0.}


def matching_schedule(seed, total):
    if total % 3:
        raise ValueError('balanced rotation requires complete three-update blocks')
    rng = np.random.default_rng(seed * 10000 + 301)
    return np.concatenate([rng.permutation(3) for _ in range(total // 3)])


def assign_slots(matching_index, layout_bits):
    if matching_index not in (0, 1, 2):
        raise ValueError('invalid matching index')
    bits = np.asarray(layout_bits)
    if bits.shape != (3,) or np.any((bits != 0) & (bits != 1)):
        raise ValueError('three binary layout choices required')
    slots = MATCHINGS[matching_index].copy()
    if bits[0]:
        slots = slots[::-1].copy()
    for slot in range(2):
        if bits[slot + 1]:
            slots[slot] = slots[slot, ::-1]
    return slots


def partners_for(assignment):
    assignment = np.asarray(assignment)
    if assignment.shape != (2, 2) or sorted(assignment.flat) != [0, 1, 2, 3]:
        raise ValueError('assignment must contain four agents exactly once')
    partner = np.empty(4, dtype=np.int64)
    for left, right in assignment:
        partner[left], partner[right] = right, left
    return partner


def assert_independent(agents):
    if len(agents) != 4 or len({id(a) for a in agents}) != 4:
        raise ValueError('four distinct agents required')
    pointers = [p.data_ptr() for a in agents for p in a.parameters()]
    if len(set(pointers)) != len(pointers):
        raise ValueError('agents must not share parameter storage')


def load_population(path):
    states = torch.load(path, map_location='cpu', weights_only=True)
    if len(states) != 4:
        raise ValueError('population checkpoint must have four states')
    agents = [ResourceAgent() for _ in range(4)]
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
    assert_independent(agents)
    return agents


def prepare_population(seed, bank, out, config=CONFIG):
    """Reuse two pre-social agents and independently prepare two more, once."""
    original_path = ROOT / f'results/pilot_001/prepared_s{seed}.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    agents = make_agents(seed)
    for agent, state in zip(agents, original):
        agent.load_state_dict(state, strict=True)
    extra_seed = seed + config['preparation_seed_offset']
    extra_agents = make_agents(extra_seed)
    extra_initial_senders = [copy.deepcopy(a.sender.state_dict()) for a in extra_agents]
    practice = individual_practice(extra_agents, bank, extra_seed,
                                   updates=config['preparation_updates'], batch=config['preparation_batch_size'])
    for local_id, record in enumerate(practice):
        record['agent'] = local_id + 2
        record['sender_unchanged_during_preparation'] = all(
            torch.equal(value, extra_initial_senders[local_id][name])
            for name, value in extra_agents[local_id].sender.state_dict().items())
        if not record['sender_unchanged_during_preparation']:
            raise RuntimeError('individual practice unexpectedly trained the sender')
    agents.extend(extra_agents)
    assert_independent(agents)
    old_report = json.loads((ROOT / 'results/pilot_001/individual_practice.json').read_text())
    old_practice = next(entry['agents'] for entry in old_report if entry['seed'] == seed)
    checkpoint = out / f'prepared_s{seed}.pt'
    torch.save([a.state_dict() for a in agents], checkpoint)
    passed = all(r['heldout_need_sensitive_choice'] >= config['preparation_accuracy_gate']
                 for r in old_practice + practice)
    report = {'seed': seed, 'reused_agents': [0, 1], 'new_agents': [2, 3], 'new_preparation_seed': extra_seed,
              'old_source': str(original_path), 'old_source_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
              'reused_practice': old_practice, 'new_practice': practice,
              'prepared_path': str(checkpoint), 'prepared_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'gate': config['preparation_accuracy_gate'], 'passed': passed}
    write_json(out / f'preparation_s{seed}.json', report)
    return checkpoint, report


def rollout(agents, features, kinds, assignment, public, policy_rngs):
    """Build four private graphs; score the two camps separately.

    features: (batch, pair_slot, agent_slot, option, feature_dimension).
    kinds are simulator metadata used only in resource transitions and logging.
    """
    partner = partners_for(assignment)
    reps = [None] * 4
    for pair_slot in range(2):
        for agent_slot in range(2):
            who = int(assignment[pair_slot, agent_slot])
            reps[who] = agents[who].observe(features[:, pair_slot, agent_slot], public)
    send_outputs = [draw(a.send(reps[i][1]), policy_rngs[i]) for i, a in enumerate(agents)]
    sent = [s[0].detach() for s in send_outputs]
    received = [sent[partner[i]] for i in range(4)]
    act_outputs = [draw(a.act(*reps[i], received[i]), policy_rngs[i]) for i, a in enumerate(agents)]
    actions = np.column_stack([x[0].numpy() for x in act_outputs])
    targets = [None] * 4
    pair_outcomes = []
    inventory = np.zeros((len(features), 2), dtype=np.int64)
    for pair_slot, pair in enumerate(assignment):
        next_inventory, rewards, info = transition(inventory, kinds[:, pair_slot], actions[:, pair], capacity=1)
        if np.any(next_inventory):
            raise RuntimeError('independent trials require empty post-consumption inventory')
        target = torch.from_numpy((rewards - 1).astype(np.float32))
        for who in pair:
            targets[who] = target
        pair_outcomes.append({'slot': pair_slot, 'agents': pair.tolist(),
                              'success': float(rewards.mean()), 'successes': int(info['balanced_gathering'].sum())})
    return {'representations': reps, 'send_outputs': send_outputs, 'act_outputs': act_outputs,
            'sent': sent, 'received': received, 'actions': actions, 'targets': targets,
            'partners': partner, 'pair_outcomes': pair_outcomes}


def optimize(agents, optimizers, result, entropy_weight):
    metrics = []
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    for i, agent in enumerate(agents):
        local = result['representations'][i][1]
        value = agent.baseline(local)
        target = result['targets'][i]
        advantage = (target - value).detach()
        actor_loss = -((result['send_outputs'][i][1] + result['act_outputs'][i][1]) * advantage).mean()
        critic_loss = .5 * F.mse_loss(value, target)
        action_entropy = result['act_outputs'][i][2].mean()
        loss = actor_loss + critic_loss - entropy_weight * action_entropy
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        norms = {name: grad_norm(getattr(agent, name)) for name in ('project', 'sender', 'actor', 'value')}
        before_clip = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.))
        optimizers[i].step()
        metrics.append({'agent': i, 'partner': int(result['partners'][i]),
                        'sampled_success': float((target + 1).mean()), 'loss': float(loss.detach()),
                        'policy_loss': float(actor_loss.detach()), 'value_loss': float(critic_loss.detach()),
                        'action_entropy': float(action_entropy.detach()),
                        'conditional_sender_entropy': float(result['send_outputs'][i][2].mean().detach()),
                        'gradient_norms': norms, 'gradient_norm_before_clip': before_clip,
                        'sent_sha256': digest(result['sent'][i].numpy()),
                        'received_sha256': digest(result['received'][i].numpy())})
    return metrics


def train(seed, condition, bank, prepared, out, config=CONFIG):
    if condition not in ('fixed', 'rotating'):
        raise ValueError('condition must be fixed or rotating')
    out.mkdir(exist_ok=False)
    agents = load_population(prepared)
    # Byte-identical initial checkpoints are copied from the common prepared file.
    shutil.copyfile(prepared, out / 'initial.pt')
    optimizers = [torch.optim.Adam(a.parameters(), lr=config['learning_rate']) for a in agents]
    total = config['course_updates'] + config['exploratory_full_updates'] + config['pure_reward_updates']
    n = config['batch_size_per_agent']
    rotation = matching_schedule(seed, total)
    layout_rng = np.random.default_rng(seed * 10000 + 302)
    world_rng = np.random.default_rng(seed * 10000 + 101)
    policy_rngs = [np.random.default_rng(seed * 10000 + 102 + i) for i in range(4)]
    public, _ = private_public(n, config['horizon'])
    edges = np.zeros((4, 4), dtype=np.int64)
    curve = []
    start = time.monotonic()
    with (out / 'training_metrics.jsonl').open('x') as handle:
        for completed in range(total + 1):
            if completed in config['checkpoints']:
                torch.save([a.state_dict() for a in agents], out / f'checkpoint_{completed:04d}.pt')
                record = {'update': completed, **schedule(max(1, completed), config),
                          'edge_training_updates': edges.tolist(),
                          'evaluation': checkpoint_population(agents, bank, seed + 700000,
                                                               n=config['checkpoint_evaluation_n'], horizon=config['horizon'])}
                curve.append(record)
                write_json(out / 'learning_curve.json', curve)
                group = record['evaluation']['aggregates']
                print(json.dumps({'seed': seed, 'condition': condition, 'update': completed,
                                  'all': group['all']['tasks']['full']['normal']['mean_reward_per_step'],
                                  'all_shuffled': group['all']['tasks']['full']['shuffle']['mean_reward_per_step'],
                                  'original': group['original']['tasks']['full']['normal']['mean_reward_per_step'],
                                  'cross': group['cross']['tasks']['full']['normal']['mean_reward_per_step']}), flush=True)
            if completed == total:
                break
            update = completed + 1
            plan = schedule(update, config)
            matching_index = int(rotation[completed]) if condition == 'rotating' else 0
            layout_bits = layout_rng.integers(2, size=3)
            assignment = assign_slots(matching_index, layout_bits)
            sampler = sample_curriculum if plan['task'] == 'curriculum' else sample_scenes
            kinds = sampler(world_rng, 2 * n).reshape(n, 2, 2, 2)
            features, image_ids = bank.sample(kinds, 'train', world_rng)
            result = rollout(agents, features, kinds, assignment, public, policy_rngs)
            agent_metrics = optimize(agents, optimizers, result, plan['action_entropy_weight'])
            for pair_slot, pair in enumerate(assignment):
                edges[pair[0], pair[1]] += 1
                edges[pair[1], pair[0]] += 1
                for agent_slot, who in enumerate(pair):
                    agent_metrics[who]['private_input_sha256'] = digest(kinds[:, pair_slot, agent_slot],
                                                                       image_ids[:, pair_slot, agent_slot])
            metrics = {'update': update, **plan, 'matching_index': matching_index,
                       'layout_bits': layout_bits.tolist(), 'slot_assignment': assignment.tolist(),
                       'world_slots_sha256': digest(kinds, image_ids),
                       'sampled_training_success': float(np.mean([p['success'] for p in result['pair_outcomes']])),
                       'pair_outcomes': result['pair_outcomes'], 'agents': agent_metrics}
            handle.write(json.dumps(metrics, allow_nan=False) + '\n')
            if update % 100 == 0:
                handle.flush()
    final = {'seed': seed, 'condition': condition, 'updates': total,
             'individual_training_choices_per_agent': total * n,
             'joint_dyad_training_steps': total * n * 2,
             'edge_training_updates': edges.tolist(), 'edge_joint_cases': (edges * n).tolist(),
             'final_aux_weight': 0., 'final_action_entropy_weight': 0.,
             'trainable_parameters_per_agent': [sum(p.numel() for p in a.parameters()) for a in agents],
             'evaluation': evaluate_population(agents, bank, seed + 800000, n=config['evaluation_n'],
                                               trace_dir=out / 'traces', intervention_n=config['intervention_n'],
                                               alignment_n=config['alignment_n'], horizon=config['horizon'])}
    final['seconds'] = time.monotonic() - start
    write_json(out / 'result.json', final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='partners_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / 'results' / args.name
    out.mkdir(exist_ok=False)
    config = copy.deepcopy(CONFIG)
    config['task_bounds'] = {task: coordination_bounds(task) for task in ('curriculum', 'full')}
    write_json(out / 'config.json', config)
    source = out / 'source'
    source.mkdir()
    for name in ('run_partners.py', 'partner_eval.py', 'run_curriculum.py', 'curriculum_eval.py',
                 'run_pilot.py', 'agents.py', 'resource_env.py'):
        shutil.copyfile(ROOT / name, source / name)
    shutil.copyfile(ROOT / '对照与伙伴实验_执行方案.md', out / 'execution_plan.md')
    write_json(out / 'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        shutil.copyfile(ROOT / 'data' / name, out / ('data_' + name))
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    preparations = {}
    prepared_paths = {}
    start = time.monotonic()
    for seed in config['seeds']:
        path, report = prepare_population(seed, bank, out, config)
        preparations[str(seed)] = report
        prepared_paths[seed] = path
        write_json(out / 'preparations.json', preparations)
        if not report['passed']:
            write_json(out / 'blocked.json', {'reason': 'preparation accuracy gate failed', 'seed': seed,
                                             'no_social_training_started': True})
            raise RuntimeError(f'preparation gate failed for seed {seed}; results retained')
    results = []
    for seed in config['seeds']:
        for condition in config['conditions']:
            result = train(seed, condition, bank, prepared_paths[seed], out / f'{condition}_s{seed}', config)
            results.append(result)
            write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results),
                                       'seconds_including_new_individual_practice': time.monotonic() - start})


if __name__ == '__main__':
    main()
