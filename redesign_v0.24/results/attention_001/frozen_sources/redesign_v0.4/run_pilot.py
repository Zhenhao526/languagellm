"""Local pilot: frozen DINO features, independent reward-trained interfaces."""
from pathlib import Path
import argparse, copy, hashlib, json, platform, time
import numpy as np
import torch
from torch.nn import functional as F
from agents import ResourceAgent, draw
from resource_env import sample_scenes, transition, one_step_coordination_bounds

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)

class ImageBank:
    def __init__(self):
        self.entries = json.loads((ROOT / 'data/manifest.json').read_text())['images']
        z = np.load(ROOT / 'data/features.npz')['features'].astype(np.float32)
        train = np.array([e['split'] == 'train' for e in self.entries])
        # Unsupervised scaling fitted on training images only.
        self.center = z[train].mean(0)
        self.scale = float(np.sqrt(((z[train] - self.center) ** 2).mean()))
        self.features = torch.from_numpy((z - self.center) / max(self.scale, 1e-6))
        self.pools = {(split, kind): np.array([
            i for i, e in enumerate(self.entries)
            if e['split'] == split and e['category'] == name
        ]) for split in ('train', 'test') for kind, name in enumerate(('food', 'water'))}
        assert all(len(v) >= 4 for v in self.pools.values())
        hashes = [{e['sha256'] for e in self.entries if e['split'] == split}
                  for split in ('train', 'test')]
        assert not (hashes[0] & hashes[1]), 'same image bytes in both splits'

    def sample(self, kinds, split, rng):
        # Category metadata chooses simulator photographs; policies get only
        # the corresponding cached pixel features, never this array or IDs.
        ids = np.empty(kinds.shape, dtype=np.int64)
        for kind in (0, 1):
            mask = kinds == kind
            ids[mask] = rng.choice(self.pools[split, kind], int(mask.sum()))
        return self.features[torch.from_numpy(ids)], ids

def public_tensor(inventory, remaining, horizon):
    return torch.from_numpy(np.column_stack([
        inventory.astype(np.float32),
        np.full(len(inventory), remaining / horizon, dtype=np.float32),
    ]).astype(np.float32))

def make_agents(seed):
    agents = []
    for i in range(2):
        torch.manual_seed(seed * 1000 + i * 137)
        agents.append(ResourceAgent())
    return agents

def individual_practice(agents, bank, seed, updates=200, batch=64):
    """Learn resource consequences from own choices and scalar reward only."""
    report = []
    for who, agent in enumerate(agents):
        rng = np.random.default_rng(seed * 10000 + who * 131 + 7)
        policy_rng = np.random.default_rng(seed * 10000 + who * 131 + 8)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        for update in range(updates):
            need = rng.integers(2, size=batch)
            kinds = np.column_stack([rng.integers(2, size=batch), np.zeros(batch, dtype=np.int64)])
            kinds[:, 1] = 1 - kinds[:, 0]
            inventory = np.ones((batch, 2), dtype=np.int64)
            inventory[np.arange(batch), need] = 0
            own, _ = bank.sample(kinds, 'train', rng)
            options, local = agent.observe(own, public_tensor(inventory, 1, 1))
            choices, logp, entropy = draw(agent.act(options, local, torch.zeros(batch, dtype=torch.int64)), policy_rng)
            selected = kinds[np.arange(batch), choices.numpy()]
            after_pick = inventory.copy()
            after_pick[np.arange(batch), selected] += 1
            # Consequence: fraction of the two demands that can be consumed.
            reward = torch.from_numpy((np.minimum(after_pick, 1).sum(1) - 1).astype(np.float32))
            loss = -((reward - .5) * logp).mean() - .01 * entropy.mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 2)
            optimizer.step()
        # Fresh held-out source photos, no update.
        with torch.no_grad():
            n = 1024
            test_rng = np.random.default_rng(seed * 10000 + who * 131 + 9)
            kinds = np.column_stack([test_rng.integers(2, size=n), np.zeros(n, dtype=np.int64)])
            kinds[:, 1] = 1 - kinds[:, 0]
            need = test_rng.integers(2, size=n)
            inventory = np.ones((n, 2), dtype=np.int64)
            inventory[np.arange(n), need] = 0
            own, _ = bank.sample(kinds, 'test', test_rng)
            options, local = agent.observe(own, public_tensor(inventory, 1, 1))
            choices = agent.act(options, local, torch.zeros(n, dtype=torch.int64)).argmax(-1).numpy()
            accuracy = float((kinds[np.arange(n), choices] == need).mean())
        report.append({'agent': who, 'updates': updates, 'choices': updates * batch,
                       'heldout_need_sensitive_choice': accuracy,
                       'method': 'own choices + scalar resource consequence; no category target loss'})
    return report

def shuffled_messages(messages, inventory, rng):
    delivered = messages.copy()
    for key in np.unique(inventory, axis=0):
        idx = np.flatnonzero((inventory == key).all(1))
        for sender in range(2):
            # Independent reassignment across worlds with the same public state.
            delivered[idx, sender] = messages[rng.permutation(idx), sender]
    return delivered

@torch.no_grad()
def policy_diagnostics(agents, bank, condition, seed, n=512):
    """Inspect probabilities as well as argmax; never update a policy here."""
    rng = np.random.default_rng(seed)
    kinds = sample_scenes(rng, n)
    features, _ = bank.sample(kinds, 'test', rng)
    inventory = np.zeros((n, 2), dtype=np.int64)
    public = public_tensor(inventory, 1, 1)
    rep = [agents[i].observe(features[:, i], public) for i in range(2)]
    send_p = [agents[i].send(rep[i][1]).softmax(-1) for i in range(2)]
    diagnostics = []
    for i in range(2):
        sent = send_p[1 - i].argmax(-1) if condition == 'communicate' else torch.zeros(n, dtype=torch.int64)
        action_p = agents[i].act(*rep[i], sent).softmax(-1)
        mixed = torch.from_numpy(kinds[:, i, 0] != kinds[:, i, 1])
        # Duplicated resource options can have high positional entropy even
        # when the resource choice is certain, so inspect mixed options.
        entropy = -(action_p * action_p.clamp_min(1e-12).log()).sum(-1)
        sender_entropy = -(send_p[i] * send_p[i].clamp_min(1e-12).log()).sum(-1)
        diagnostics.append({'agent': i, 'mixed_samples': int(mixed.sum()),
                            'sender_entropy_nats': float(sender_entropy.mean()),
                            'mixed_action_entropy_nats': float(entropy[mixed].mean()),
                            'mixed_max_action_probability': float(action_p[mixed].max(-1).values.mean())})
    return diagnostics

@torch.no_grad()
def evaluate(agents, bank, condition, seed, n=512, horizon=16, mode='normal', split='test', greedy=True, trace=False):
    world_rng = np.random.default_rng(seed)
    policy_rng = [np.random.default_rng(seed + 11001 + i) for i in range(2)]
    intervention_rng = np.random.default_rng(seed + 22001)
    inventory = np.zeros((n, 2), dtype=np.int64)
    rows = []
    episodes = np.zeros((n, 4), dtype=np.float64)
    table = np.zeros((2, 3, 5), dtype=np.int64)
    step_rows = []
    for step in range(horizon):
        kinds = sample_scenes(world_rng, n)
        features, ids = bank.sample(kinds, split, world_rng)
        public = public_tensor(inventory, horizon - step, horizon)
        representations = [agents[i].observe(features[:, i], public) for i in range(2)]
        sent = np.column_stack([draw(agents[i].send(representations[i][1]), policy_rng[i], greedy)[0].numpy() for i in range(2)])
        if condition == 'silent':
            delivered = np.zeros_like(sent)
        elif mode == 'shuffle':
            delivered = shuffled_messages(sent, inventory, intervention_rng)
        else:
            delivered = sent.copy()
        actions = np.column_stack([
            draw(agents[i].act(*representations[i], torch.from_numpy(delivered[:, 1 - i])), policy_rng[i], greedy)[0].numpy()
            for i in range(2)
        ])
        for i in range(2):
            state = kinds[:, i].sum(1)  # Audit only: FF=0, mixed=1, WW=2.
            np.add.at(table[i], (state, sent[:, i]), 1)
        nxt, rewards, info = transition(inventory, kinds, actions, capacity=1)
        episodes[:, 0] += rewards
        episodes[:, 1] += info['shortage'].sum(1)
        episodes[:, 2] += info['overflow'].sum(1)
        episodes[:, 3] += info['balanced_gathering']
        step_rows.append({'step': step, 'mean_reward': float(rewards.mean()),
                          'balanced_gathering': float(info['balanced_gathering'].mean())})
        if trace:
            rows.append({'step': step, 'inventory': inventory.tolist(), 'kinds': kinds.tolist(),
                         'image_ids': ids.tolist(), 'sent': sent.tolist(), 'delivered': delivered.tolist(),
                         'actions': actions.tolist(), 'reward': rewards.tolist(),
                         'next_inventory': nxt.tolist()})
        inventory = nxt
    result = {'mean_reward_per_step': float(episodes[:, 0].mean() / horizon),
              'shortage_per_episode': float(episodes[:, 1].mean()),
              'overflow_per_episode': float(episodes[:, 2].mean()),
              'balanced_gathering': float(episodes[:, 3].mean() / horizon),
              'episodes_without_shortage': float((episodes[:, 1] == 0).mean()),
              'episodes': n, 'horizon': horizon, 'mode': mode, 'split': split, 'greedy': greedy,
              'symbols_by_local_resource_set': table.tolist(), 'per_step': step_rows}
    return result, rows

@torch.no_grad()
def message_intervention(agents, bank, seed, n=1024):
    """At identical recipient input, change only the discrete received symbol."""
    rng = np.random.default_rng(seed)
    kinds = sample_scenes(rng, n)
    features, _ = bank.sample(kinds, 'test', rng)
    inventory = np.zeros((n, 2), dtype=np.int64)
    public = public_tensor(inventory, 16, 16)
    rep = [agents[i].observe(features[:, i], public) for i in range(2)]
    sent = [agents[i].send(rep[i][1]).argmax(-1) for i in range(2)]
    directions = []
    for receiver in range(2):
        sender = 1 - receiver
        # Reference symbols come from an independent photo/scene sample.
        ref_kinds = sample_scenes(rng, n)
        ref_features, _ = bank.sample(ref_kinds, 'test', rng)
        _, ref_local = agents[sender].observe(ref_features[:, sender], public)
        ref_sent = agents[sender].send(ref_local).argmax(-1).numpy()
        counts = np.bincount(ref_sent, minlength=5)
        used = np.flatnonzero(counts >= max(5, n // 100))
        base_logits = agents[receiver].act(*rep[receiver], sent[sender])
        base_p = base_logits.softmax(-1).numpy()
        base_action = base_p.argmax(-1)
        effects = []
        for symbol in used:
            changed_p = agents[receiver].act(*rep[receiver], torch.full((n,), int(symbol), dtype=torch.int64)).softmax(-1).numpy()
            changed_action = changed_p.argmax(-1)
            mask = sent[sender].numpy() != symbol
            if not mask.any():
                continue
            original_kind = kinds[np.arange(n), receiver, base_action]
            changed_kind = kinds[np.arange(n), receiver, changed_action]
            effects.append({'replacement_symbol': int(symbol), 'n': int(mask.sum()),
                            'choice_flip_rate': float((base_action[mask] != changed_action[mask]).mean()),
                            'resource_flip_rate': float((original_kind[mask] != changed_kind[mask]).mean()),
                            'mean_action_total_variation': float((.5 * np.abs(base_p[mask] - changed_p[mask]).sum(1)).mean())})
        directions.append({'sender': sender, 'receiver': receiver, 'reference_counts': counts.tolist(),
                           'reference_min_count': max(5, n // 100), 'interventions': effects})
    return directions

def train_one(agents, bank, seed, condition, out, config):
    out.mkdir(parents=True, exist_ok=False)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    optimizers = [torch.optim.Adam(a.parameters(), lr=config['learning_rate']) for a in agents]
    rng = np.random.default_rng(seed * 10000 + 101)
    policy_rng = [np.random.default_rng(seed * 10000 + 102 + i) for i in range(2)]
    horizon, n = config['horizon'], config['batch_episodes']
    start = time.monotonic()
    curve = []
    training_window = []
    for update in range(config['updates'] + 1):
        if update % config['checkpoint_every'] == 0 or update == config['updates']:
            torch.save([a.state_dict() for a in agents], out / f'checkpoint_{update:04d}.pt')
            normal, _ = evaluate(agents, bank, condition, seed + 700000, n=128, horizon=horizon)
            disrupted, _ = evaluate(agents, bank, condition, seed + 700000, n=128, horizon=horizon, mode='shuffle')
            stochastic, _ = evaluate(agents, bank, condition, seed + 700000, n=128, horizon=horizon, greedy=False)
            record = {'update': update, 'normal': normal, 'shuffle': disrupted,
                      'stochastic': stochastic,
                      'policy_diagnostics': policy_diagnostics(agents, bank, condition, seed + 710000),
                      'training_window': training_window}
            curve.append(record)
            training_window = []
            write_json(out / 'learning_curve.json', curve)
            print(json.dumps({'seed': seed, 'condition': condition, 'update': update,
                              'reward': normal['mean_reward_per_step'], 'balanced': normal['balanced_gathering'],
                              'shuffle_reward': disrupted['mean_reward_per_step'],
                              'stochastic_reward': stochastic['mean_reward_per_step']}), flush=True)
        if update == config['updates']:
            break
        inventory = np.zeros((n, 2), dtype=np.int64)
        logs = [[], []]; entropies = [[], []]; values = [[], []]; rewards = []
        for step in range(horizon):
            kinds = sample_scenes(rng, n)
            features, _ = bank.sample(kinds, 'train', rng)
            public = public_tensor(inventory, horizon - step, horizon)
            rep = [agents[i].observe(features[:, i], public) for i in range(2)]
            symbols, slp, sentropy = [], [], []
            for i in range(2):
                symbol, lp, entropy = draw(agents[i].send(rep[i][1]), policy_rng[i])
                symbols.append(symbol.detach()); slp.append(lp); sentropy.append(entropy)
            actions = []
            for i in range(2):
                received = symbols[1 - i] if condition == 'communicate' else torch.zeros(n, dtype=torch.int64)
                action, lp, entropy = draw(agents[i].act(*rep[i], received), policy_rng[i])
                actions.append(action.numpy())
                active = 1.0 if condition == 'communicate' else 0.0
                logs[i].append(lp + active * slp[i])
                entropies[i].append(entropy + active * sentropy[i])
                values[i].append(agents[i].baseline(rep[i][1]))
            inventory, reward, _ = transition(inventory, kinds, np.column_stack(actions), capacity=1)
            # Remove a constant 1 per step; action rankings/reward differences
            # are unchanged, and the value function predicts resource costs.
            rewards.append(torch.from_numpy((reward - 1).astype(np.float32)))
        returns = []
        future = torch.zeros(n)
        for reward in reversed(rewards):
            future = reward + config['gamma'] * future
            returns.append(future)
        target = torch.stack(list(reversed(returns))).detach()
        update_metrics = {'update': update + 1,
                          'sampled_training_success': float(torch.stack(rewards).mean() + 1),
                          'agents': []}
        for i in range(2):
            value = torch.stack(values[i])
            advantage = (target - value).detach()
            actor_loss = -(torch.stack(logs[i]) * advantage).mean()
            critic_loss = .5 * F.mse_loss(value, target)
            loss = actor_loss + critic_loss - config['entropy'] * torch.stack(entropies[i]).mean()
            assert torch.isfinite(loss)
            optimizers[i].zero_grad(); loss.backward()
            grad_norms = {}
            for name in ('project', 'sender', 'actor', 'value'):
                total = sum(float(p.grad.detach().square().sum()) for p in getattr(agents[i], name).parameters() if p.grad is not None)
                grad_norms[name] = total ** .5
            unclipped_norm = torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 2)
            optimizers[i].step()
            update_metrics['agents'].append({'agent': i, 'loss': float(loss.detach()),
                                            'actor_loss': float(actor_loss.detach()),
                                            'critic_loss': float(critic_loss.detach()),
                                            'unclipped_gradient_norm': float(unclipped_norm),
                                            'gradient_norms': grad_norms})
        training_window.append(update_metrics)
    normal, rows = evaluate(agents, bank, condition, seed + 800000, n=config['evaluation_episodes'], horizon=horizon, trace=True)
    shuffled, _ = evaluate(agents, bank, condition, seed + 800000, n=config['evaluation_episodes'], horizon=horizon, mode='shuffle')
    stochastic, _ = evaluate(agents, bank, condition, seed + 800000, n=config['evaluation_episodes'], horizon=horizon, greedy=False)
    train_images, _ = evaluate(agents, bank, condition, seed + 800000, n=config['evaluation_episodes'], horizon=horizon, split='train')
    result = {'seed': seed, 'condition': condition, 'normal': normal, 'shuffle': shuffled,
              'stochastic': stochastic, 'train_images': train_images,
              'same_observation_intervention': message_intervention(agents, bank, seed + 900000),
              'seconds': time.monotonic() - start,
              'trainable_parameters_per_agent': sum(p.numel() for p in agents[0].parameters()),
              'training_joint_steps': config['updates'] * n * horizon}
    write_json(out / 'result.json', result)
    with (out / 'heldout_trace.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pilot_001')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--practice-threshold', type=float, default=.8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = {'seeds': args.seeds, 'updates': args.updates, 'practice_updates': args.warmup,
              'batch_episodes': 64, 'horizon': 16, 'capacity': 1, 'initial_inventory': [0, 0],
              'learning_rate': 3e-4,
              'gamma': 0.0, 'entropy': .01, 'checkpoint_every': 100,
              'evaluation_episodes': 512, 'practice_threshold': args.practice_threshold,
              'conditions': ['communicate', 'silent'],
              'torch': torch.__version__, 'python': platform.python_version(), 'training_device': 'cpu',
              'no_success_based_stopping': True, 'one_step_reference': one_step_coordination_bounds()}
    out = ROOT / 'results' / args.name
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', config)
    source = out / 'source'
    source.mkdir()
    for filename in ('run_pilot.py', 'agents.py', 'resource_env.py', 'encode_images.py'):
        (source / filename).write_bytes((ROOT / filename).read_bytes())
    write_json(out / 'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    results, practice = [], []
    start = time.monotonic()
    prepared = {}
    for seed in args.seeds:
        base = make_agents(seed)
        evidence = individual_practice(base, bank, seed, updates=args.warmup)
        practice.append({'seed': seed, 'agents': evidence})
        write_json(out / 'individual_practice.json', practice)
        print(json.dumps({'stage': 'individual_practice', 'seed': seed, 'evidence': evidence}), flush=True)
        torch.save([a.state_dict() for a in base], out / f'prepared_s{seed}.pt')
        prepared[seed] = base
    if any(a['heldout_need_sensitive_choice'] < args.practice_threshold for row in practice for a in row['agents']):
        write_json(out / 'stopped.json', {'reason': 'individual resource choice below predeclared engineering threshold',
                                         'seconds': time.monotonic() - start, 'social_training_started': False})
        raise RuntimeError('Individual resource-choice check failed; no social runs started.')
    for seed in args.seeds:
        base = prepared[seed]
        for condition in config['conditions']:
            agents = copy.deepcopy(base)
            result = train_one(agents, bank, seed, condition, out / f'{condition}_s{seed}', config)
            results.append(result)
            write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'seconds': time.monotonic() - start, 'runs': len(results), 'status': 'completed'})

if __name__ == '__main__':
    main()
