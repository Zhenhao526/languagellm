"""Independent audit of fixed-sender receiver fits and saved probability tables.

No training routine or production metric formula is reused by this auditor.
The original CampAgent definition may be used for read-only source replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SEEDS = (29101, 29102, 29103, 29104)
ARMS = ('warm_rl24', 'warm_ce24', 'fresh_ce24', 'fresh_ce30')
MAPS = np.array(list(product(range(6), repeat=2)), dtype=np.int64)
MAPS = MAPS[MAPS[:, 0] != MAPS[:, 1]]
MATCHINGS = (((0, 1), (2, 3), (4, 5)), ((0, 2), (1, 4), (3, 5)),
             ((0, 3), (1, 5), (2, 4)))
PHOTO_PREFIX = 'receiver-baseline|2026-09-15|photo-split|'
SEED_PREFIX = 'receiver-baseline|2026-09-15'


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.failures = []
        self.counts = Counter()

    def check(self, ok, name, context=None):
        self.checks[name] += 1
        if not bool(ok):
            self.failures.append({'check': name, 'context': context})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def tensor_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def digest_arrays(*items):
    h = hashlib.sha256()
    for item in items:
        h.update(np.ascontiguousarray(item).tobytes())
    return h.hexdigest()


def partitions(p):
    def matching(index):
        pairs = {z for a, b in MATCHINGS[index] for z in ((a, b), (b, a))}
        return np.array([i for i, z in enumerate(MAPS) if tuple(z) in pairs])
    added, sealed = matching(p - 1), matching(p % 3)
    old = np.setdiff1d(np.arange(30), np.r_[added, sealed])
    return {'old': old, 'added': added, 'sealed': sealed}


def photo_split():
    entries = read(PROJECT / 'redesign_v0.4/data/manifest.json')['images']
    result = {name: [] for name in ('fit', 'dev', 'evaluation')}
    for resource in ('food', 'water'):
        ids = [i for i, e in enumerate(entries) if e['category'] == resource and e['split'] == 'train']
        ids.sort(key=lambda i: hashlib.sha256((PHOTO_PREFIX + entries[i]['sha256']).encode()).hexdigest())
        result['fit'].append(np.array(ids[:16], dtype=np.int64))
        result['dev'].append(np.array(ids[16:], dtype=np.int64))
        result['evaluation'].append(np.array([i for i, e in enumerate(entries)
            if e['category'] == resource and e['split'] == 'test'], dtype=np.int64))
    return entries, result


def seed_for(seed, p, direction, kind, step):
    identity = f'{SEED_PREFIX}|{seed}|{p}|{direction}|{kind}|{step}'
    return identity, int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')


def log_probs(logits):
    x = np.asarray(logits, dtype=np.float64)
    return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)


def sender_policies(first, second):
    first_p, second_p = np.exp(log_probs(first)), np.exp(log_probs(second))
    p = (first_p[:, :, None] * second_p).reshape(-1, 49)
    a = np.asarray(first).argmax(-1)
    b = np.asarray(second)[np.arange(len(a)), a].argmax(-1)
    greedy = np.column_stack((a, b))
    onehot = np.eye(49)[a * 7 + b]
    return p, onehot, greedy


def contingency(probability, positions, mask=None):
    if mask is None:
        mask = np.ones(len(positions), dtype=bool)
    out = np.zeros((49, 6, 6), dtype=np.float64)
    selected = np.flatnonzero(mask)
    assert len(selected)
    # Each frozen visual context has the same fixed probability weight.
    for f, w in MAPS:
        ix = selected[(positions[selected, 0] == f) & (positions[selected, 1] == w)]
        out[:, f, w] = probability[ix].sum(0) / len(selected)
    return out


def independent_metrics(c, logits):
    """Reference by explicit code/action loops, distinct from production einsums."""
    c = np.asarray(c, dtype=np.float64)
    c = c / c.sum()  # Match the documented roundoff-only probability normalization.
    log_r = log_probs(logits)
    receiver = np.exp(log_r)
    mass = c.sum((1, 2))
    cf, cw = c.sum(2), c.sum(1)
    denom = np.where(mass > 0, mass, 1)
    pf, pw, joint = cf / denom[:, None], cw / denom[:, None], c / denom[:, None, None]
    def hconditional(count, posterior):
        positive = count > 0
        return float(-(count[positive] * np.log(posterior[positive])).sum())
    hf, hw, hj = hconditional(cf, pf), hconditional(cw, pw), hconditional(c, joint)
    nf, nw = float(-(cf * log_r[:, 0]).sum()), float(-(cw * log_r[:, 1]).sum())
    kl = []
    for goal, (weights, posterior) in enumerate(((cf, pf), (cw, pw))):
        positive = weights > 0
        kl.append(float((weights[positive] * (np.log(posterior[positive]) - log_r[:, goal][positive])).sum()))
    def action_stats(actions=None, stochastic=False):
        food = water = both = 0.
        for m in range(49):
            if stochastic:
                for a, b in product(range(6), repeat=2):
                    probability = receiver[m, 0, a] * receiver[m, 1, b]
                    food += probability * cf[m, a]
                    water += probability * cw[m, b]
                    both += probability * c[m, a, b]
            else:
                a, b = actions[m]
                food += cf[m, a]; water += cw[m, b]; both += c[m, a, b]
        return dict(J=float(both), single_accuracy=float((food + water) / 2),
            food_accuracy=float(food), water_accuracy=float(water),
            mixed_reward=float((food + water) / 4 + both / 2))
    actual_actions = np.asarray(logits).argmax(-1)
    ce = np.column_stack((cf.argmax(-1), cw.argmax(-1)))
    map_actions = np.array([np.unravel_index(x.argmax(), (6, 6)) for x in c])
    mixed_actions = []
    for m in range(49):
        values = [0.25 * (cf[m, a] + cw[m, b]) + 0.5 * c[m, a, b]
                  for a, b in product(range(6), repeat=2)]
        mixed_actions.append(divmod(int(np.argmax(values)), 6))
    mixed_actions = np.array(mixed_actions)
    result = dict(branch_nll=nf + nw, food_nll=nf, water_nll=nw,
        joint_conditional_entropy=hj, food_conditional_entropy=hf, water_conditional_entropy=hw,
        branch_conditional_entropy=hf + hw, conditional_mutual_information=hf + hw - hj,
        kl_food=kl[0], kl_water=kl[1], kl_sum=sum(kl),
        nll_entropy_gap=nf + nw - hf - hw,
        decomposition_error=nf + nw - (hj + hf + hw - hj + sum(kl)),
        stochastic=action_stats(stochastic=True), greedy=action_stats(actual_actions),
        receiver_greedy_actions=actual_actions.tolist(), message_probability=mass.tolist(),
        posterior_food=pf.tolist(), posterior_water=pw.tolist(), posterior_joint=joint.tolist())
    for name, actions in (('oracle_branch_ce', ce), ('oracle_joint_map', map_actions),
                          ('oracle_mixed_reward', mixed_actions)):
        result[name] = dict(action_stats(actions), actions=actions.tolist())
    return result


def compare_numbers(audit, actual, expected, label, atol=2e-10):
    for key, value in expected.items():
        if isinstance(value, dict):
            compare_numbers(audit, actual[key], value, [label, key], atol)
        else:
            audit.check(np.allclose(actual[key], value, atol=atol, rtol=2e-10),
                        'independent_probability_metric', [label, key])


def math_selftests(audit, implementation=None):
    c = np.zeros((49, 6, 6))
    c[0, MAPS[:, 0], MAPS[:, 1]] = 1 / 30
    zero = np.zeros((49, 2, 6))
    expected = independent_metrics(c, zero)
    for key, target in (('joint_conditional_entropy', np.log(30)),
                        ('branch_conditional_entropy', np.log(36))):
        audit.check(np.isclose(expected[key], target), 'constant_message_entropy', key)
    for key, metric, target in (('oracle_joint_map', 'J', 1 / 30),
                               ('oracle_mixed_reward', 'mixed_reward', 1 / 10),
                               ('stochastic', 'J', 1 / 36),
                               ('stochastic', 'mixed_reward', 7 / 72)):
        audit.check(np.isclose(expected[key][metric], target), 'constant_message_distinct_baselines', [key, metric])
    audit.check(expected['oracle_branch_ce']['J'] == 0, 'marginal_tie_can_choose_illegal_truth_pair')
    audit.check(expected['oracle_joint_map']['actions'][1:] == [[0, 0]] * 48,
                'zero_mass_codes_defined_without_nan')
    if implementation is not None:
        compare_numbers(audit, implementation(c, zero), expected, 'constant')
    perfect = np.zeros_like(c)
    for i, (f, w) in enumerate(MAPS):
        perfect[i, f, w] = 1 / 30
    pe = independent_metrics(perfect, zero)
    audit.check(pe['branch_conditional_entropy'] == 0 and np.isclose(pe['oracle_joint_map']['J'], 1),
                'perfect_code_entropy_and_joint')
    rng = np.random.default_rng(120120)
    for i in range(24):
        counts = rng.integers(0, 30, size=(49, 6, 6)).astype(float)
        counts[rng.random(counts.shape) < .6] = 0
        counts[rng.random(49) < .25] = 0
        counts /= counts.sum()
        logits = rng.normal(size=(49, 2, 6)) * 3
        truth = independent_metrics(counts, logits)
        audit.check(abs(truth['decomposition_error']) < 1e-12 and truth['kl_sum'] >= -1e-12,
                    'random_KL_entropy_identity', i)
        audit.check(truth['oracle_joint_map']['J'] + 1e-12 >= truth['greedy']['J'] and
                    truth['oracle_joint_map']['J'] + 1e-12 >= truth['stochastic']['J'],
                    'all_36_actions_joint_bound', i)
        audit.check(truth['oracle_mixed_reward']['mixed_reward'] + 1e-12 >= truth['stochastic']['mixed_reward'] and
                    truth['oracle_mixed_reward']['mixed_reward'] + 1e-12 >= truth['greedy']['mixed_reward'],
                    'all_36_actions_utility_bound', i)
        if implementation is not None:
            compare_numbers(audit, implementation(counts, logits), truth, ['random', i])


def state_hash(state, excluded=()):
    h = hashlib.sha256()
    for key, tensor in state.items():
        if not key.startswith(excluded):
            h.update(key.encode()); h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def active_prefixes(arm):
    return ('receive_embedding.', 'actor.', 'receive_value.') if arm == 'warm_rl24' else ('receive_embedding.', 'actor.')


def model_from_state(state):
    # Import the already frozen original architecture, never the v12 runner.
    sys.path.insert(0, str(PROJECT / 'redesign_v0.8'))
    from camp import CampAgent
    projection = torch.nn.Sequential(torch.nn.Linear(1024, 64), torch.nn.Tanh())
    model = CampAgent(projection, 7, 2, 96)
    model.load_state_dict(state)
    model.requires_grad_(False)
    return model


def configure_receiver(source, seed, p, direction, arm):
    receiver = model_from_state(source)
    if arm.startswith('fresh'):
        torch.manual_seed(seed_for(seed, p, direction, 'receiver_initialization', 0)[1])
        receiver.receive_embedding.reset_parameters()
        for module in receiver.actor.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    for name, param in receiver.named_parameters():
        param.requires_grad_(name.startswith(active_prefixes(arm)))
    return receiver


def native_draw(logits, rng):
    log = torch.nn.functional.log_softmax(logits, -1)
    p = log.exp()
    u = torch.from_numpy(rng.random((len(p), 1)).astype(np.float32))
    a = (p.detach().cumsum(-1) < u).sum(-1).clamp(max=p.shape[-1] - 1)
    return a, log.gather(1, a[:, None]).squeeze(1), -(p * log).sum(-1)


@torch.no_grad()
def independent_send(sender, h, rng):
    state = sender.send_context(torch.cat((h, torch.zeros(len(h), 4)), 1))
    first, _, _ = native_draw(sender.send_out(state), rng)
    state = sender.send_recur(sender.send_embedding(first), state)
    second, _, _ = native_draw(sender.send_out(state), rng)
    return torch.stack((first, second), 1)


@torch.no_grad()
def decode_table(receiver):
    messages = torch.tensor(list(product(range(7), repeat=2)))
    context = torch.cat((receiver.receive_embedding(messages).flatten(1), torch.zeros(49, 20)), 1)
    return receiver.actor(context).reshape(49, 2, 6).numpy()


def optimizer_check(audit, opt, names, step, label):
    groups = opt['param_groups']
    audit.check(len(groups) == 1 and len(groups[0]['params']) == len(names) and
                groups[0]['params'] == list(range(len(names))), 'optimizer_active_parameters', [label, step])
    audit.check(all(g['lr'] == .0007 and g['betas'] == (.9, .999) and g['eps'] == 1e-8 and
                    g['weight_decay'] == 0 for g in groups), 'optimizer_hyperparameters', [label, step])
    audit.check((not opt['state']) if step == 0 else len(opt['state']) == len(names) and
                all(float(v['step']) == step for v in opt['state'].values()),
                'fresh_adam_and_exact_update_budget', [label, step])


def source_audit(audit, root, source_root, seed, p, direction, detail):
    folder = root / f'source_s{seed}_p{p}_d{direction}'
    cfg = read(folder / 'config.json')
    original = tensor_load(source_root / f's{seed}_p{p}_base/final.pt')
    saved = tensor_load(folder / 'source.pt')
    audit.check(same(original, saved), 'source_full_pair_matches_base', folder.name)
    audit.check(cfg['sender'] == direction and cfg['receiver'] == 1 - direction and cfg['direction'] == direction,
                'source_direction_identity', folder.name)
    audit.check(cfg['source_state_file_sha256'] == sha(folder / 'source.pt') and
                cfg['frozen_sender_sha256'] == state_hash(original[direction]) and
                cfg['source_receiver_sha256'] == state_hash(original[1 - direction]),
                'source_state_digests', folder.name)
    for path, digest in cfg['source_hashes'].items():
        audit.check(sha(path) == digest, 'source_asset_fingerprint', path)
    audit.check(cfg['photo_split'] == detail, 'source_same_fixed_photo_split', folder.name)
    _, split = photo_split()
    sender = model_from_state(original[direction])
    raw = arrays(PROJECT / 'redesign_v0.4/data/features.npz')['features'].astype(np.float32)
    entries, _ = photo_split()
    train = np.array([e['split'] == 'train' for e in entries])
    center = raw[train].mean(0)
    scale = float(np.sqrt(((raw[train] - center) ** 2).mean()))
    with torch.no_grad():
        projected = sender.project(torch.from_numpy((raw - center) / max(scale, 1e-6)))
    contexts = {}
    for name, pools in split.items():
        path = folder / f'context_{name}.npz'; context = arrays(path)
        photos = np.array(list(product(*pools)), dtype=np.int64)
        mids = np.repeat(np.arange(30), len(photos)); ids = np.tile(photos, (30, 1))
        positions = MAPS[mids]
        audit.check(cfg['context_hashes'][name] == sha(path), 'context_saved_file_hash', [folder.name, name])
        audit.check(np.array_equal(context['map_ids'], mids) and np.array_equal(context['photo_ids'], ids) and
                    np.array_equal(context['positions'], positions), 'exact_cartesian_contexts_and_photo_isolation', [folder.name, name])
        n = len(mids)
        audit.check(context['h'].shape == (n, 96) and context['first_logits'].shape == (n, 7) and
                    context['second_logits'].shape == (n, 7, 7) and all(context[k].dtype == np.float32 and
                    np.isfinite(context[k]).all() for k in ('h', 'first_logits', 'second_logits')),
                    'source_context_float32_shapes', [folder.name, name])
        probs, _, greedy = sender_policies(context['first_logits'], context['second_logits'])
        audit.check(np.allclose(probs.sum(1), 1, atol=2e-14) and np.array_equal(greedy, context['greedy_message']),
                    'autoregressive_native_and_sequential_greedy', [folder.name, name])
        for lo in range(0, n, 512):
            pos, pix = positions[lo:lo + 512], ids[lo:lo + 512]
            size = len(pos)
            visual = torch.zeros(size, 6, 64); exists = torch.zeros(size, 6)
            for kind in (0, 1):
                visual[torch.arange(size), torch.from_numpy(pos[:, kind])] = projected[torch.from_numpy(pix[:, kind])]
                exists[torch.arange(size), torch.from_numpy(pos[:, kind])] = 1
            with torch.no_grad():
                h = sender.observe(torch.cat((visual.flatten(1), exists), 1))
                state = sender.send_context(torch.cat((h, torch.zeros(size, 4)), 1))
                first = sender.send_out(state)
                tokens = torch.arange(7).repeat(size)
                nxt = sender.send_recur(sender.send_embedding(tokens), state.repeat_interleave(7, 0))
                second = sender.send_out(nxt).reshape(size, 7, 7)
            audit.check(np.array_equal(context['h'][lo:lo + size], h.numpy()) and
                        np.array_equal(context['first_logits'][lo:lo + size], first.numpy()) and
                        np.array_equal(context['second_logits'][lo:lo + size], second.numpy()),
                        'source_cache_direct_model_replay_exact', [folder.name, name, lo])
        contexts[name] = context
        audit.counts['source_contexts'] += n
    return original, sender, contexts


def forensic_audit(audit, folder, sender, original, context, cfg, row, step):
    label = [folder.name, step]
    before = folder / ('initial.pt' if step == 0 else 'forensic_2100.pt')
    before_opt = folder / ('initial_optimizer.pt' if step == 0 else 'forensic_optimizer_2100.pt')
    receiver = model_from_state(tensor_load(before))
    for name, param in receiver.named_parameters():
        param.requires_grad_(name.startswith(active_prefixes(cfg['arm'])))
    opt = torch.optim.Adam([p for p in receiver.parameters() if p.requires_grad], lr=.0007)
    opt.load_state_dict(tensor_load(before_opt))
    saved = arrays(folder / f'train_{step + 1:04d}.npz')
    n = cfg['batch']; support = '30' if cfg['arm'].endswith('30') else '24'
    identity = lambda kind: seed_for(cfg['seed'], cfg['partition'], cfg['direction'], kind + support, step)[1]
    possible = np.flatnonzero(np.isin(context['map_ids'], cfg['training_pool']))
    indices = np.random.default_rng(identity('world')).choice(possible, n)
    audit.check(np.array_equal(saved['context_indices'], indices) and all(np.array_equal(saved[k], context[k][indices])
                for k in ('map_ids', 'positions', 'photo_ids', 'h')), 'forensic_exact_world_indices_and_context', label)
    sent = independent_send(sender, torch.from_numpy(context['h'][indices]), np.random.default_rng(identity('send')))
    audit.check(np.array_equal(saved['sent'], sent.numpy()), 'forensic_independent_native_message_draw', label)
    logits, values = [], []
    for kind in (0, 1):
        goal = torch.zeros(n, 2); goal[:, kind] = 1
        scores, value = receiver.receive(sent, goal, torch.zeros(n, 2), torch.zeros(n, 18), torch.arange(6).repeat(n, 1))
        logits.append(scores); values.append(value)
    audit.check(np.array_equal(saved['receiver_logits'], torch.stack(logits, 1).detach().numpy()) and
                np.array_equal(saved['values'], torch.stack(values, 1).detach().numpy()), 'forensic_legal_receiver_interface_replay', label)
    truth = torch.from_numpy(saved['positions'])
    ce = sum(torch.nn.functional.cross_entropy(logits[k], truth[:, k]) for k in (0, 1))
    if cfg['arm'] == 'warm_rl24':
        rng = np.random.default_rng(identity('action'))
        draws = [native_draw(x, rng) for x in logits]
        action = torch.stack([x[0] for x in draws], 1)
        success = (action == truth).float(); reward = success.sum(1) / 4 + success.prod(1) / 2
        target = reward - 1; value = torch.stack(values).mean(0)
        policy = -((draws[0][1] + draws[1][1]) * (target - value).detach()).mean()
        value_loss = .5 * torch.nn.functional.mse_loss(value, target)
        entropy = (draws[0][2] + draws[1][2]).mean()
        ent_weight = .02 if step < 2100 else 0.
        loss = policy + value_loss - ent_weight * entropy
        audit.check(all(np.array_equal(saved[k], v.detach().numpy()) for k, v in
                    (('action', action), ('successes', success), ('reward', reward), ('target', target))),
                    'forensic_independent_two_actions_reward_target', label)
        expected = dict(policy_loss=float(policy.detach()), value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                        entropy_weight=ent_weight, mean_reward=float(reward.mean()), J=float(success.prod(1).mean()))
    else:
        loss = ce; expected = dict(entropy_weight=0.)
        audit.check(not {'action', 'reward', 'target', 'successes'} & saved.keys(), 'CE_no_sampled_action_or_reward_target', label)
    expected.update(loss=float(loss.detach()), branch_ce_sample=float(ce.detach()))
    audit.check(row['components'] == expected, 'forensic_loss_components_exact', label)
    opt.zero_grad(set_to_none=True); loss.backward()
    audit.check(all(p.grad is None for p in sender.parameters()) and all(p.grad is None for p in receiver.parameters() if not p.requires_grad),
                'forensic_gradient_partition', label)
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in receiver.parameters() if p.requires_grad], 2.))
    audit.check(norm == row['gradient_norm'], 'forensic_clipped_gradient_norm', label)
    opt.step()
    if step == 0:
        audit.check(same(receiver.state_dict(), tensor_load(folder / 'after_first_update.pt')),
                    'independent_first_update_all_parameters_exact', label)
        audit.check(same(opt.state_dict(), tensor_load(folder / 'after_first_optimizer.pt')),
                    'independent_first_update_all_adam_exact', label)
    audit.counts['forensic_communications'] += n


def audit_fit(audit, root, seed, p, direction, arm, args, original, sender, contexts, hashes, full_messages):
    folder = root / f's{seed}_p{p}_d{direction}_{arm}'
    cfg = read(folder / 'config.json'); result = read(folder / 'result.json'); curve = read(folder / 'curve.json')
    points = sorted(set([0, args['updates']] + [x for x in (0, 100, 300, 600, 1200, 1800, 2400) if x < args['updates']]))
    groups = partitions(p); pool = np.arange(30) if arm.endswith('30') else np.sort(np.r_[groups['old'], groups['added']])
    model = configure_receiver(original[1 - direction], seed, p, direction, arm)
    initial, final = tensor_load(folder / 'initial.pt'), tensor_load(folder / 'final.pt')
    names = [k for k, param in model.named_parameters() if param.requires_grad]
    for key, value in dict(seed=seed, partition=p, direction=direction, sender=direction, receiver=1-direction,
            arm=arm, updates=args['updates'], batch=args['batch'], checkpoints=points, training_pool=pool.tolist(),
            learning_rate=.0007, entropy_off_after=2100, forensic_pre_update=2100, trainable_names=names,
            source_hashes=hashes, menu='identity', goal_order=[0, 1],
            extra_supervision=arm!='warm_rl24', sealed_supervised_exposure=arm=='fresh_ce30').items():
        audit.check(cfg[key] == value, 'fixed_fit_configuration', [folder.name, key])
    audit.check(same(model.state_dict(), initial), 'exact_original_or_fixed_fresh_initialization', folder.name)
    audit.check(cfg['initial_sha256'] == state_hash(initial) and result['final_sha256'] == state_hash(final),
                'fit_initial_final_digest', folder.name)
    audit.check(cfg['frozen_sender_sha256'] == state_hash(original[direction]) and
                cfg['frozen_sha256'] == state_hash(initial, active_prefixes(arm)), 'frozen_all_sender_and_unused_state_digest', folder.name)
    audit.check(result['complete'] and result['updates'] == args['updates'] and result['frozen_verified'] and
                result['frozen_sender_verified'], 'fit_complete_fixed_budget', folder.name)
    for step in points:
        state = tensor_load(folder / f'checkpoint_{step:04d}.pt')
        audit.check(all(torch.equal(v, initial[k]) for k, v in state.items() if not k.startswith(active_prefixes(arm))),
                    'all_checkpoint_frozen_parameters_and_buffers', [folder.name, step])
        audit.check(curve[points.index(step)]['state_sha256'] == state_hash(state), 'curve_exact_checkpoint_hash', [folder.name, step])
        opt = tensor_load(folder / f'optimizer_{step:04d}.pt')
        optimizer_check(audit, opt, names, step, folder.name)
        logits = arrays(folder / f'receiver_{step:04d}.npz')['logits']
        audit.check(np.array_equal(logits, decode_table(model_from_state(state))), 'checkpoint_receiver_logits_replayed', [folder.name, step])
        for split in ('fit', 'dev'):
            context = contexts[split]
            probability, _, _ = sender_policies(context['first_logits'], context['second_logits'])
            c = contingency(probability, context['positions'], np.isin(context['map_ids'], pool))
            expected = independent_metrics(c, logits)
            actual = curve[points.index(step)]['scores'][split]
            compact = {k: v for k, v in expected.items() if np.isscalar(v)}
            for key in ('stochastic', 'greedy', 'oracle_branch_ce', 'oracle_joint_map', 'oracle_mixed_reward'):
                compact[key] = {k: v for k, v in expected[key].items() if np.isscalar(v)}
            compare_numbers(audit, actual, compact, [folder.name, step, split])
        audit.counts['checkpoints'] += 1
    audit.check(same(initial, tensor_load(folder / 'checkpoint_0000.pt')) and
                same(final, tensor_load(folder / f'checkpoint_{args["updates"]:04d}.pt')),
                'initial_and_final_are_planned_checkpoints', folder.name)
    audit.check(same(tensor_load(folder/'initial_optimizer.pt'), tensor_load(folder/'optimizer_0000.pt')) and
                same(tensor_load(folder/'final_optimizer.pt'), tensor_load(folder/f'optimizer_{args["updates"]:04d}.pt')),
                'initial_final_optimizer_are_planned_checkpoints', folder.name)
    selected = None if arm == 'warm_rl24' else min(curve, key=lambda r: (r['scores']['dev']['branch_nll'], r['update']))['update']
    audit.check([r['update'] for r in curve] == points and result['selected_dev_update'] == selected and
                result['final_support_scores'] == curve[-1]['scores'], 'fixed_curve_budget_and_CE_dev_earliest_selection', folder.name)
    rows = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    audit.check(len(rows) == args['updates'], 'complete_training_update_count', folder.name)
    context = contexts['fit']; possible = np.flatnonzero(np.isin(context['map_ids'], pool))
    support = '30' if arm.endswith('30') else '24'
    for step, row in enumerate(rows):
        streams = {k:dict(zip(('label', 'seed'), seed_for(seed, p, direction, k+support, step)))
                   for k in ('world', 'send', 'action')}
        indices = np.random.default_rng(streams['world']['seed']).choice(possible, args['batch'])
        world = digest_arrays(indices, context['positions'][indices], context['photo_ids'][indices])
        exposure = {k:int(np.isin(context['map_ids'][indices], pool).sum()) for k,pool in groups.items()}
        audit.check(row['update'] == step+1 and row['streams'] == streams, 'exact_training_rng_identity', [folder.name, step])
        audit.check(row['world_sha256'] == world and row['exposure'] == exposure and
                    (arm=='fresh_ce30' or exposure['sealed']==0), 'all_worlds_photo_support_and_exposure', [folder.name, step])
        audit.check(row['components']['entropy_weight'] == (.02 if arm=='warm_rl24' and step<2100 else 0.),
                    'exact_entropy_schedule_and_CE_no_entropy', [folder.name, step])
        audit.check(all(np.isfinite(v) for v in row['components'].values()) and np.isfinite(row['gradient_norm']),
                    'finite_losses_and_gradient_norm', [folder.name, step])
        if full_messages:
            sent = independent_send(sender, torch.from_numpy(context['h'][indices]), np.random.default_rng(streams['send']['seed']))
            audit.check(digest_arrays(sent.numpy()) == row['message_sha256'], 'all_training_native_messages_independent_replay', [folder.name, step])
        if step in (0, 2100):
            forensic_audit(audit, folder, sender, original, context, cfg, row, step)
        audit.counts['training_updates'] += 1
        audit.counts['training_worlds'] += args['batch']
        if arm=='warm_rl24': audit.counts['RL_sampled_primitive_actions'] += 2*args['batch']
    audit.counts['fits'] += 1
    return dict(initial=initial, rows=rows, cfg=cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'results/receiver_001')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-full-message-replay', action='store_true', help='Development only; never a full formal pass')
    args = parser.parse_args(); root = args.out.resolve(); torch.set_num_threads(1)
    audit = Audit(); invocation = read(root/'invocation.json'); hashes = invocation['source_hashes']
    source = Path(invocation['source_root']).resolve()
    output = args.output or (ROOT/'preflight_qa.json' if args.preflight else root/'audit_execution.json')
    sys.path.insert(0, str(ROOT))
    import receiver_metrics
    math_selftests(audit, receiver_metrics.evaluate)
    expected = dict(seeds=[99510], partitions=[1], directions=[0], arms=list(ARMS), updates=4, batch=512) if args.preflight else dict(
        seeds=list(SEEDS), partitions=[1,2,3], directions=[0,1], arms=list(ARMS), updates=2400, batch=512)
    for key, value in expected.items(): audit.check(invocation[key] == value, 'fixed_invocation_budget', key)
    audit.check(invocation['formal'] != args.preflight and invocation['device']=='cpu' and invocation['torch_threads']==1,
                'fixed_execution_mode_device')
    for i, (path, digest) in enumerate(hashes.items()):
        audit.check(sha(path) == digest, 'current_frozen_source_hash', path)
        if Path(path).suffix in ('.py', '.md', '.json'):
            audit.check(sha(root/f'frozen_sources/{i:02d}_{Path(path).name}')==digest, 'executed_source_snapshot_hash', path)
    entries, split = photo_split()
    detail = {name:[[{ 'feature_row':int(i), 'image_id':entries[i]['id'], 'sha256':entries[i]['sha256'] }
                     for i in ids] for ids in pools] for name,pools in split.items()}
    audit.check(read(root/'photo_split.json') == detail, 'fixed_sha_sorted_photo_split')
    for kind in (0, 1):
        ids = [split[s][kind] for s in split]
        audit.check([len(x) for x in ids] == [16,6,8] and len(set(np.concatenate(ids)))==30,
                    'fit_dev_evaluation_disjoint_ids_and_sizes', kind)
        audit.check(len({entries[i]['sha256'] for i in np.concatenate(ids)})==30, 'photo_bytes_nonoverlap', kind)
    actual_dirs = {p.name for p in root.glob('s*_p*_d*_*') if p.is_dir()}
    expected_dirs = {f's{s}_p{p}_d{d}_{a}' for s,p,d,a in product(invocation['seeds'],invocation['partitions'],invocation['directions'],ARMS)}
    audit.check(actual_dirs == expected_dirs, 'exact_all_fit_directories')
    for seed,p,direction in product(invocation['seeds'],invocation['partitions'],invocation['directions']):
        original, sender, contexts = source_audit(audit,root,source,seed,p,direction,detail)
        group = {}
        for arm in ARMS:
            group[arm] = audit_fit(audit,root,seed,p,direction,arm,invocation,original,sender,contexts,hashes,
                                  not args.skip_full_message_replay)
        audit.check(same(group['warm_rl24']['initial'],group['warm_ce24']['initial']), 'two_warm_exact_same_initial', [seed,p,direction])
        audit.check(same(group['fresh_ce24']['initial'],group['fresh_ce30']['initial']), 'two_fresh_exact_same_initial', [seed,p,direction])
        for arm in ('warm_ce24','fresh_ce24'):
            a,b=group['warm_rl24']['rows'],group[arm]['rows']
            audit.check(all(x['streams']==y['streams'] and x['world_sha256']==y['world_sha256'] and
                            x['message_sha256']==y['message_sha256'] for x,y in zip(a,b)),
                        'all_24_support_actual_worlds_and_messages_paired', [seed,p,direction,arm])
        print(json.dumps(dict(audited_source=[seed,p,direction],fits=audit.counts['fits'],failures=len(audit.failures))),flush=True)
    completed = read(root/'training_complete.json')
    audit.check(completed['complete'] and completed['runs']==len(expected_dirs) and completed['source_hashes']==hashes,
                'batch_complete_receipt')
    if not args.preflight:
        audit.check(not args.skip_full_message_replay, 'formal_all_native_messages_replayed')
    result = dict(passed=not audit.failures, checks=dict(audit.checks), total_checks=sum(audit.checks.values()),
        failures=audit.failures, counts=dict(audit.counts), runner_sha256=sha(ROOT/'run_receiver.py'),
        frozen_source_hashes=hashes, audit_script_sha256=sha(__file__), root=str(root),
        preflight=args.preflight, all_training_native_messages_replayed=not args.skip_full_message_replay,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent source/model replay, frozen states, Adam steps, every world and native message, saved forensic gradients; no complete optimizer retraining. Endpoint evaluation analysis audited separately when available.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        prior=read(output)
        if not prior.get('passed',False):
            keep=output.with_name(output.stem+'_prior_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
            keep.write_text(json.dumps(prior,ensure_ascii=False,indent=2))
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'passed':result['passed'],'checks':result['total_checks'],'counts':result['counts'],'output':str(output)}),flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
