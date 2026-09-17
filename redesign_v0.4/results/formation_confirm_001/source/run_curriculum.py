"""Predeclared curriculum experiment with independent discrete communicators.

Only the signalling condition receives a state-dependent signalling loss.
No class labels, reference dictionary, peer gradients, or privileged critic.
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, platform, shutil, time
import numpy as np
import torch
from torch.nn import functional as F
from agents import draw
from run_pilot import ImageBank, make_agents, public_tensor, write_json
from resource_env import sample_scenes, transition
from curriculum_eval import sample_curriculum, evaluate, message_intervention, coordination_bounds

ROOT = Path(__file__).resolve().parent

CONFIG = {
    'seeds': [101, 202, 303], 'conditions': ['baseline', 'signalling'],
    'course_updates': 600, 'transition_updates': 300, 'pure_reward_updates': 300,
    'batch_size': 1024, 'horizon': 16, 'learning_rate': .0003,
    'action_entropy_coefficient': .05, 'signalling_coefficient': .1,
    'signalling_target_entropy': math.log(5) / 2,
    'signalling_conditional_weight': 1., 'sender_max_entropy_coefficient': 0.,
    'gamma': 0., 'capacity': 1, 'initial_inventory': [0, 0],
    'evaluation_n': 8192, 'checkpoint_evaluation_n': 2048,
    'checkpoints': [0, 20, 50, 100, 200, 400, 600, 650, 750, 900, 1000, 1200],
    'engineering_criteria': {'course_success': .9, 'full_success': .85,
                             'full_shuffle_gap': .1, 'direction_shuffle_gap': .1},
    'fixed_budget_no_success_based_stopping': True,
    'architecture': 'unchanged ResourceAgent; 83527 trainable interface parameters per agent',
    'backbone': 'frozen official DINOv2 ViT-L/14; cached original photo features',
    'initialization': 'results/pilot_001/prepared_s{seed}.pt; before social learning',
    'training_device': 'cpu', 'torch': torch.__version__, 'python': platform.python_version(),
    'bias_reference': 'https://arxiv.org/abs/1912.05676',
    'bias_scope': 'adapted one-step target conditional entropy and marginal entropy loss; not an exact reproduction',
}

def schedule(update, condition, config=CONFIG):
    """One-based learning update; evaluations report the completed update."""
    if not 1 <= update <= sum(config[k] for k in ('course_updates', 'transition_updates', 'pure_reward_updates')):
        raise ValueError('update outside predeclared budget')
    course_end = config['course_updates']
    transition_end = course_end + config['transition_updates']
    if update <= course_end:
        stage, task, fraction = 'course', 'curriculum', 1.
        entropy = config['action_entropy_coefficient']
    elif update <= transition_end:
        stage, task = 'transition', 'full'
        fraction = 1. - (update - course_end) / config['transition_updates']
        entropy = config['action_entropy_coefficient']
    else:
        stage, task, fraction, entropy = 'pure_reward', 'full', 0., 0.
    coefficient = config['signalling_coefficient'] * fraction if condition == 'signalling' else 0.
    return {'stage': stage, 'task': task, 'aux_weight': coefficient, 'action_entropy_weight': entropy}

def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)

def signalling_loss(probabilities, public_groups, target_entropy, conditional_weight=1.):
    """Private policy probabilities only; condition marginal entropy on public time.

    E[(H(M|observation)-target)^2] - H(E[p(M|observation)]).
    Grouping prevents the publicly known round from earning an information bonus.
    There are no class or resource IDs in this function.
    """
    terms = []
    for group in torch.unique(public_groups):
        p = probabilities[public_groups == group]
        terms.append(conditional_weight * (entropy(p) - target_entropy).square().mean() - entropy(p.mean(0)))
    return torch.stack(terms).mean()

def initialize(seed):
    agents = make_agents(seed)
    states = torch.load(ROOT / f'results/pilot_001/prepared_s{seed}.pt', map_location='cpu', weights_only=True)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
    return agents

def private_public(n, horizon):
    if n % horizon:
        raise ValueError('batch size must divide into equal public-time groups')
    rounds = np.repeat(np.arange(horizon), n // horizon)
    public = np.column_stack([np.zeros((n, 2), np.float32), (horizon - rounds).astype(np.float32) / horizon])
    return torch.from_numpy(public), torch.from_numpy(rounds)

def grad_norm(module):
    return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None) ** .5

@torch.no_grad()
def diagnostic(agents, features, public, sent, kinds):
    result = []
    for i, agent in enumerate(agents):
        options, local = agent.observe(features[:, i], public)
        sp = agent.send(local).softmax(-1)
        ap = agent.act(options, local, sent[1 - i]).softmax(-1)
        mixed = torch.from_numpy(kinds[:, i, 0] != kinds[:, i, 1])
        result.append({'agent': i, 'sender_entropy': float(entropy(sp).mean()),
                       'action_entropy_mixed': float(entropy(ap)[mixed].mean()),
                       'max_action_probability_mixed': float(ap[mixed].max(-1).values.mean())})
    return result

@torch.no_grad()
def checkpoint_evaluation(agents, bank, seed, completed_update, config):
    record = {'update': completed_update,
              **schedule(max(1, completed_update), 'signalling', config)}
    for task in ('curriculum', 'full'):
        task_results = {}
        for key, mode, greedy in [('normal', 'normal', True), ('shuffle', 'shuffle', True), ('stochastic', 'normal', False)]:
            task_results[key] = evaluate(agents, bank, seed + 700000, n=config['checkpoint_evaluation_n'],
                                         task=task, mode=mode, greedy=greedy)[0]
        record[task] = task_results
    return record

def train(seed, condition, bank, out, config):
    out.mkdir(exist_ok=False)
    agents = initialize(seed)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    optimizers = [torch.optim.Adam(a.parameters(), lr=config['learning_rate']) for a in agents]
    world_rng = np.random.default_rng(seed * 10000 + 101)
    policy_rng = [np.random.default_rng(seed * 10000 + 102 + i) for i in range(2)]
    n, horizon = config['batch_size'], config['horizon']
    public, groups = private_public(n, horizon)
    inventory = np.zeros((n, 2), np.int64)
    total = sum(config[k] for k in ('course_updates', 'transition_updates', 'pure_reward_updates'))
    curve, start = [], time.monotonic()
    last_diagnostic = None
    last_kinds = None
    with (out / 'training_metrics.jsonl').open('w') as metrics_file:
        for completed in range(total + 1):
            if completed in config['checkpoints']:
                torch.save([a.state_dict() for a in agents], out / f'checkpoint_{completed:04d}.pt')
                record = checkpoint_evaluation(agents, bank, seed, completed, config)
                record.update(schedule(max(1, completed), condition, config))
                record['diagnostic_on_last_training_batch'] = last_diagnostic
                curve.append(record)
                write_json(out / 'learning_curve.json', curve)
                print(json.dumps({'seed': seed, 'condition': condition, 'update': completed,
                                  'stage': record['stage'], 'aux': record['aux_weight'],
                                  'course': record['curriculum']['normal']['mean_reward_per_step'],
                                  'full': record['full']['normal']['mean_reward_per_step'],
                                  'full_shuffled': record['full']['shuffle']['mean_reward_per_step'],
                                  'full_stochastic': record['full']['stochastic']['mean_reward_per_step']}), flush=True)
            if completed == total:
                break
            plan = schedule(completed + 1, condition, config)
            kinds = sample_curriculum(world_rng, n) if plan['task'] == 'curriculum' else sample_scenes(world_rng, n)
            features, image_ids = bank.sample(kinds, 'train', world_rng)
            reps = [a.observe(features[:, i], public) for i, a in enumerate(agents)]
            send_outputs = [draw(a.send(reps[i][1]), policy_rng[i]) for i, a in enumerate(agents)]
            sent = [s[0].detach() for s in send_outputs]
            act_outputs = [draw(a.act(*reps[i], sent[1 - i]), policy_rng[i]) for i, a in enumerate(agents)]
            actions = np.column_stack([o[0].numpy() for o in act_outputs])
            _, reward, info = transition(inventory, kinds, actions, capacity=1)
            target = torch.from_numpy((reward - 1).astype(np.float32))
            metrics = {'update': completed + 1, **plan, 'sampled_training_success': float(reward.mean()),
                       'world_input_sha256': hashlib.sha256(kinds.tobytes() + image_ids.tobytes()).hexdigest(),
                       'agents': []}
            for i, a in enumerate(agents):
                value = a.baseline(reps[i][1])
                advantage = (target - value).detach()
                actor_loss = -((act_outputs[i][1] + send_outputs[i][1]) * advantage).mean()
                critic_loss = .5 * F.mse_loss(value, target)
                p = a.send(reps[i][1]).softmax(-1)
                signal_loss = signalling_loss(p, groups, config['signalling_target_entropy'], config['signalling_conditional_weight'])
                loss = actor_loss + critic_loss - plan['action_entropy_weight'] * act_outputs[i][2].mean() + plan['aux_weight'] * signal_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                optimizers[i].zero_grad(); loss.backward()
                norms = {name: grad_norm(getattr(a, name)) for name in ('project', 'sender', 'actor', 'value')}
                before_clip = float(torch.nn.utils.clip_grad_norm_(a.parameters(), 2.))
                optimizers[i].step()
                metrics['agents'].append({'agent': i, 'loss': float(loss.detach()), 'policy_loss': float(actor_loss.detach()),
                                          'value_loss': float(critic_loss.detach()), 'signalling_loss': float(signal_loss.detach()),
                                          'conditional_sender_entropy': float(entropy(p).mean().detach()),
                                          'gradient_norms': norms, 'gradient_norm_before_clip': before_clip})
            if completed + 1 in config['checkpoints']:
                last_diagnostic = diagnostic(agents, features, public, sent, kinds)
            metrics_file.write(json.dumps(metrics) + '\n')
            if (completed + 1) % 100 == 0:
                metrics_file.flush()
    final = {'seed': seed, 'condition': condition, 'updates': total, 'joint_training_steps': total * n,
             'seconds': time.monotonic() - start, 'final_aux_weight': 0., 'final_action_entropy_weight': 0.,
             'trainable_parameters_per_agent': sum(p.numel() for p in agents[0].parameters())}
    for task in ('curriculum', 'full'):
        final[task] = {}
        for mode in ('normal', 'shuffle', 'blank'):
            stats, rows = evaluate(agents, bank, seed + 800000, n=config['evaluation_n'], task=task, mode=mode, trace=True)
            final[task][mode] = stats
            with (out / f'final_{task}_{mode}_trace.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
        final[task]['stochastic'] = evaluate(agents, bank, seed + 800000, n=config['evaluation_n'], task=task, greedy=False)[0]
    final['intervention'] = message_intervention(agents, bank, seed + 900000, n=4096, task='full')
    write_json(out / 'result.json', final)
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='curriculum_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / 'results' / args.name
    out.mkdir(exist_ok=False)
    config = copy.deepcopy(CONFIG)
    config['task_bounds'] = {t: coordination_bounds(t) for t in ('curriculum', 'full')}
    write_json(out / 'config.json', config)
    source = out / 'source'; source.mkdir()
    for name in ('run_curriculum.py', 'curriculum_eval.py', 'run_pilot.py', 'agents.py', 'resource_env.py'):
        shutil.copyfile(ROOT / name, source / name)
    write_json(out / 'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        shutil.copyfile(ROOT / 'data' / name, out / ('data_' + name))
    prepared = {str(s): hashlib.sha256((ROOT / f'results/pilot_001/prepared_s{s}.pt').read_bytes()).hexdigest() for s in config['seeds']}
    write_json(out / 'initial_checkpoint_hashes.json', prepared)
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    results, start = [], time.monotonic()
    for seed in config['seeds']:
        for condition in config['conditions']:
            result = train(seed, condition, bank, out / f'{condition}_s{seed}', config)
            results.append(result)
            write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results), 'seconds': time.monotonic() - start})

if __name__ == '__main__':
    main()
