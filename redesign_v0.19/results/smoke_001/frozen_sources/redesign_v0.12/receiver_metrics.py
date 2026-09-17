"""NumPy-only, target-specific diagnostics for a fixed 49-code sender.

All entropy/NLL/KL quantities are nats per world, with two branch NLLs summed.
Oracle actions are deterministic joint actions and include all 36 site pairs.
This module neither imports a model nor constructs or changes sender policies.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np

MESSAGES = 49
SITES = 6
ATOL = 1e-10


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}; got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _log_softmax(logits):
    # Use the log probabilities directly in NLL/KL; never clip small r values.
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = logits-logits.max(axis=-1, keepdims=True)
        log_prob = shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    if not np.isfinite(log_prob).all():
        raise ValueError('logit differences must permit finite float64 log probabilities')
    return log_prob


def _entropy(mass, conditional):
    selected = mass > 0
    return float(-np.sum(mass[selected]*np.log(conditional[selected])))


def _deterministic_scores(c, food_mass, water_mass, actions):
    row = np.arange(MESSAGES)
    f = float(food_mass[row, actions[:, 0]].sum())
    w = float(water_mass[row, actions[:, 1]].sum())
    joint = float(c[row, actions[:, 0], actions[:, 1]].sum())
    return dict(J=joint, single_accuracy=(f+w)/2, food_accuracy=f, water_accuracy=w,
                mixed_reward=.25*(f+w)+.5*joint)


def evaluate(c, receiver_logits):
    """Evaluate one specified message/world distribution and one receiver.

    c[m,f,w] is a normalized joint probability, not a count or posterior.
    receiver_logits[m,goal,site] must be finite; goal 0 is food, 1 is water.
    Input mass is required within 1e-10 of one and normalized for roundoff only.
    Zero-mass messages have zero posterior rows and canonical (0,0) oracle actions.
    All outputs consist exclusively of JSON-compatible Python scalars/lists/dicts.
    """
    c = _finite_array(c, (MESSAGES, SITES, SITES), 'c')
    if (c < 0).any():
        raise ValueError('c must be nonnegative')
    input_mass = float(c.sum())
    if not math.isclose(input_mass, 1., rel_tol=0., abs_tol=ATOL):
        raise ValueError(f'c must sum to one; got {input_mass}')
    c = c/input_mass
    logits = _finite_array(receiver_logits, (MESSAGES, 2, SITES), 'receiver_logits')
    log_receiver = _log_softmax(logits)
    receiver = np.exp(log_receiver)
    message_mass = c.sum(axis=(1, 2))
    food_mass = c.sum(axis=2)
    water_mass = c.sum(axis=1)
    posterior_joint = np.divide(c, message_mass[:, None, None], out=np.zeros_like(c),
                                where=message_mass[:, None, None] > 0)
    posterior_food = np.divide(food_mass, message_mass[:, None], out=np.zeros_like(food_mass),
                               where=message_mass[:, None] > 0)
    posterior_water = np.divide(water_mass, message_mass[:, None], out=np.zeros_like(water_mass),
                                where=message_mass[:, None] > 0)
    joint_entropy = _entropy(c, posterior_joint)
    food_entropy = _entropy(food_mass, posterior_food)
    water_entropy = _entropy(water_mass, posterior_water)
    branch_entropy = food_entropy+water_entropy
    # CMI is calculated from the joint posterior, independently of H_F+H_W-H_FW.
    mi_terms = np.zeros_like(c)
    for m in np.flatnonzero(message_mass > 0):
        ix = c[m] > 0
        food_log = np.zeros(SITES)
        water_log = np.zeros(SITES)
        np.log(posterior_food[m], out=food_log, where=posterior_food[m] > 0)
        np.log(posterior_water[m], out=water_log, where=posterior_water[m] > 0)
        ratio_log = np.zeros((SITES, SITES))
        np.log(posterior_joint[m], out=ratio_log, where=ix)
        ratio_log -= food_log[:, None]+water_log[None, :]
        mi_terms[m, ix] = c[m, ix]*ratio_log[ix]
    cmi = float(mi_terms.sum())
    food_nll = float(-np.sum(food_mass*log_receiver[:, 0, :]))
    water_nll = float(-np.sum(water_mass*log_receiver[:, 1, :]))
    branch_nll = food_nll+water_nll
    selected_f = food_mass > 0
    selected_w = water_mass > 0
    kl_food = float(np.sum(food_mass[selected_f]*(np.log(posterior_food[selected_f])-log_receiver[:, 0, :][selected_f])))
    kl_water = float(np.sum(water_mass[selected_w]*(np.log(posterior_water[selected_w])-log_receiver[:, 1, :][selected_w])))
    kl_sum = kl_food+kl_water
    food_success = float(np.sum(food_mass*receiver[:, 0, :]))
    water_success = float(np.sum(water_mass*receiver[:, 1, :]))
    stochastic_j = float(np.sum(c*receiver[:, 0, :, None]*receiver[:, 1, None, :]))
    stochastic = dict(J=stochastic_j, single_accuracy=(food_success+water_success)/2,
                      food_accuracy=food_success, water_accuracy=water_success,
                      mixed_reward=.25*(food_success+water_success)+.5*stochastic_j)
    greedy_actions = logits.argmax(axis=2)
    greedy = _deterministic_scores(c, food_mass, water_mass, greedy_actions)
    branch_actions = np.column_stack((food_mass.argmax(axis=1), water_mass.argmax(axis=1)))
    joint_flat = c.reshape(MESSAGES, SITES*SITES).argmax(axis=1)
    joint_actions = np.column_stack((joint_flat//SITES, joint_flat % SITES))
    # Unnormalized expected reward per message; includes diagonal legal actions.
    reward_matrix = .25*(food_mass[:, :, None]+water_mass[:, None, :])+.5*c
    mixed_flat = reward_matrix.reshape(MESSAGES, SITES*SITES).argmax(axis=1)
    mixed_actions = np.column_stack((mixed_flat//SITES, mixed_flat % SITES))
    branch = dict(**_deterministic_scores(c, food_mass, water_mass, branch_actions), actions=branch_actions.tolist())
    joint = dict(**_deterministic_scores(c, food_mass, water_mass, joint_actions), actions=joint_actions.tolist())
    mixed = dict(**_deterministic_scores(c, food_mass, water_mass, mixed_actions), actions=mixed_actions.tolist())
    result = dict(input_mass=input_mass, positive_message_count=int(np.count_nonzero(message_mass)),
        branch_nll=branch_nll, food_nll=food_nll, water_nll=water_nll,
        joint_conditional_entropy=joint_entropy, food_conditional_entropy=food_entropy,
        water_conditional_entropy=water_entropy, branch_conditional_entropy=branch_entropy,
        conditional_mutual_information=cmi, kl_food=kl_food, kl_water=kl_water, kl_sum=kl_sum,
        nll_entropy_gap=branch_nll-branch_entropy,
        decomposition_error=branch_nll-joint_entropy-cmi-kl_sum,
        stochastic=stochastic, greedy=greedy, oracle_branch_ce=branch, oracle_joint_map=joint,
        oracle_mixed_reward=mixed, receiver_greedy_actions=greedy_actions.tolist(),
        message_probability=message_mass.tolist(), posterior_food=posterior_food.tolist(),
        posterior_water=posterior_water.tolist(), posterior_joint=posterior_joint.tolist())
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
        raise ArithmeticError('Information quantities exceeded finite float64 range')
    # Numerical consistency tolerances, not definitions or success thresholds.
    tolerance = 1e-9*max(1., branch_nll, kl_sum)
    if abs(result['decomposition_error']) > tolerance or abs(cmi-(branch_entropy-joint_entropy)) > tolerance:
        raise ArithmeticError('NLL/entropy/KL decomposition failed')
    if min(cmi, kl_food, kl_water, food_entropy, water_entropy, joint_entropy) < -tolerance:
        raise ArithmeticError('A nonnegative information quantity is below numerical tolerance')
    for outcome in (stochastic, greedy, branch, joint, mixed):
        if outcome['J'] > joint['J']+ATOL or outcome['mixed_reward'] > mixed['mixed_reward']+ATOL:
            raise ArithmeticError('An outcome exceeded its target-specific oracle')
    return result


def evaluate_groups(message_probs, positions, receiver_logits, masks):
    """Uniformly weight selected worlds, retaining their supplied message policy.

    Call separately for native stochastic sender probabilities and one-hot
    sequential-greedy sender messages. This function does not convert one to the
    other. Group-specific oracles may differ and cannot be combined into a single
    jointly deployable decoder. Empty groups are explicitly undefined.
    """
    probs = np.asarray(message_probs, dtype=np.float64)
    pos = np.asarray(positions)
    if probs.ndim != 2 or probs.shape[1] != MESSAGES:
        raise ValueError('message_probs must have shape [N,49]')
    n = len(probs)
    if pos.shape != (n, 2) or not np.issubdtype(pos.dtype, np.integer):
        raise ValueError('positions must contain integer physical sites and have shape [N,2]')
    if ((pos < 0)|(pos >= SITES)).any():
        raise ValueError('positions must be in 0..5')
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError('message_probs must be finite and nonnegative')
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1., rtol=0., atol=ATOL):
        raise ValueError('each message_probs row must sum to one')
    probs = probs/sums[:, None]
    _finite_array(receiver_logits, (MESSAGES, 2, SITES), 'receiver_logits')
    output = {}
    for name, raw_mask in masks.items():
        if not isinstance(name, str):
            raise ValueError('group names must be strings')
        mask = np.asarray(raw_mask)
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError(f'mask {name!r} must be a boolean vector of length N')
        count = int(mask.sum())
        if not count:
            output[name] = dict(world_count=0, defined=False)
            continue
        selected_pos = pos[mask]
        # Accumulate [map, message] then transpose to [message, food, water].
        by_map = np.zeros((SITES*SITES, MESSAGES), dtype=np.float64)
        np.add.at(by_map, selected_pos[:, 0]*SITES+selected_pos[:, 1], probs[mask]/count)
        c = by_map.T.reshape(MESSAGES, SITES, SITES)
        output[name] = dict(**evaluate(c, receiver_logits), world_count=count, defined=True)
    return output


def self_test():
    checks = []
    def close(actual, expected, tolerance=1e-10):
        assert abs(actual-expected) < tolerance, (actual, expected)
    logits = np.zeros((MESSAGES, 2, SITES))
    c = np.zeros((MESSAGES, SITES, SITES))
    for f in range(SITES):
        for w in range(SITES):
            if f != w:
                c[0, f, w] = 1/30
    r = evaluate(c, logits)
    close(r['joint_conditional_entropy'], math.log(30))
    close(r['branch_conditional_entropy'], math.log(36))
    close(r['branch_nll'], math.log(36))
    close(r['conditional_mutual_information'], math.log(36/30))
    close(r['kl_sum'], 0.)
    close(r['stochastic']['J'], 1/36)
    close(r['stochastic']['mixed_reward'], 7/72)
    close(r['oracle_joint_map']['J'], 1/30)
    close(r['oracle_mixed_reward']['mixed_reward'], 1/10)
    assert r['oracle_joint_map']['actions'][0] == r['oracle_mixed_reward']['actions'][0] == [0, 1]
    assert r['oracle_branch_ce']['actions'][0] == r['receiver_greedy_actions'][0] == [0, 0]
    close(r['oracle_branch_ce']['J'], 0.)
    close(r['oracle_branch_ce']['mixed_reward'], 1/12)
    checks.append('Constant-message uniform 30-map entropy and distinct oracle/random chance levels')
    assert r['positive_message_count'] == 1
    assert np.count_nonzero(np.array(r['posterior_joint'])[1:]) == 0
    assert all(row == [0, 0] for row in r['oracle_joint_map']['actions'][1:])
    assert math.isfinite(r['branch_nll']) and abs(r['decomposition_error']) < 1e-10
    checks.append('Zero-mass messages have finite metrics, zero posterior and canonical tied actions')
    counts = np.array([[0,4,6,5,1,2], [11,0,6,0,9,8], [0,5,0,9,0,1],
                       [1,8,2,0,8,1], [1,5,1,10,0,3], [2,4,5,6,10,0]], dtype=np.float64)
    c2 = np.zeros_like(c); c2[0] = counts/counts.sum()
    r2 = evaluate(c2, logits)
    assert r2['oracle_branch_ce']['actions'][0] == [1, 3]
    assert r2['oracle_joint_map']['actions'][0] == [1, 0]
    assert r2['oracle_mixed_reward']['actions'][0] == [1, 4]
    close(r2['oracle_branch_ce']['J'], 0.)
    close(r2['oracle_joint_map']['J'], 11/134)
    close(r2['oracle_mixed_reward']['J'], 9/134)
    close(r2['oracle_mixed_reward']['mixed_reward'], 80/536)
    checks.append('Nonuniform map distribution has three distinct unique target-optimal action pairs')
    perfect = np.zeros_like(c)
    for m, (f, w) in enumerate((f, w) for f in range(SITES) for w in range(SITES) if f != w):
        perfect[m, f, w] = 1/30
    rp = evaluate(perfect, logits)
    close(rp['joint_conditional_entropy'], 0.)
    close(rp['branch_conditional_entropy'], 0.)
    for name in ('oracle_branch_ce', 'oracle_joint_map', 'oracle_mixed_reward'):
        close(rp[name]['J'], 1.); close(rp[name]['mixed_reward'], 1.)
    checks.append('Perfect map-distinguishing messages give zero conditional entropy and unit oracles')
    rng = np.random.default_rng(1201201)
    for _ in range(20):
        random_c = rng.random(c.shape); random_c[rng.random(c.shape) < .4] = 0
        random_c /= random_c.sum(); random_logits = rng.normal(0, 3, logits.shape)
        rr = evaluate(random_c, random_logits)
        # Independent full-joint expansion of the branch cross entropy.
        shifted = random_logits-random_logits.max(-1, keepdims=True)
        log_r = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
        full_nll = -(random_c*(log_r[:, 0, :, None]+log_r[:, 1, None, :])).sum()
        close(rr['branch_nll'], float(full_nll))
        close(rr['nll_entropy_gap'], rr['kl_sum'])
        close(rr['decomposition_error'], 0.)
        # Exhaustive candidate scoring verifies both global objectives per message.
        by_m = np.array(rr['message_probability'])
        oracle_j = oracle_u = 0.
        for m in range(MESSAGES):
            scores = []
            for af in range(SITES):
                for aw in range(SITES):
                    j = random_c[m, af, aw]
                    u = .25*(random_c[m, af].sum()+random_c[m, :, aw].sum())+.5*j
                    scores.append((j, u))
            oracle_j += max(x[0] for x in scores)
            oracle_u += max(x[1] for x in scores)
        close(rr['oracle_joint_map']['J'], oracle_j)
        close(rr['oracle_mixed_reward']['mixed_reward'], oracle_u)
        assert abs(by_m.sum()-1) < 1e-10
    checks.append('20 random joint tables: full-joint NLL, KL identity and independent 36-action enumeration')
    independent = np.zeros_like(c)
    f = np.array([1, 2, 3, 4, 5, 6])/21
    w = f[::-1]
    independent[0] = f[:, None]*w[None, :]
    ri = evaluate(independent, logits)
    close(ri['conditional_mutual_information'], 0.)
    extreme = evaluate(c, np.tile(np.array([0., -10000., -2000., -700., -900., -3000.]), (49, 2, 1)))
    assert math.isfinite(extreme['branch_nll']) and abs(extreme['decomposition_error']) < 1e-9
    checks.append('Conditional independence and stable NLL despite underflowed receiver probabilities')
    positions = np.array([[0, 1], [0, 1], [2, 3]])
    native = np.zeros((3, 49)); native[:, 0] = .25; native[:, 1] = .75
    greedy = np.zeros((3, 49)); greedy[:, 0] = 1.
    masks = {'all': np.array([True, True, True]), 'first': np.array([True, True, False]),
             'empty': np.array([False, False, False])}
    groups = evaluate_groups(native, positions, logits, masks)
    direct = np.zeros_like(c); direct[0, 0, 1] = 1/6; direct[1, 0, 1] = .5
    direct[0, 2, 3] = 1/12; direct[1, 2, 3] = .25
    close(groups['all']['joint_conditional_entropy'], evaluate(direct, logits)['joint_conditional_entropy'])
    assert groups['empty'] == {'world_count': 0, 'defined': False}
    assert groups['all']['world_count'] == 3 and groups['first']['world_count'] == 2
    assert groups['all']['message_probability'] != evaluate_groups(greedy, positions, logits, masks)['all']['message_probability']
    json.dumps(groups, allow_nan=False); json.dumps(extreme, allow_nan=False)
    checks.append('Uniform-world group aggregation, empty groups, native/greedy separation and strict JSON output')
    for bad_c in (np.zeros_like(c), -c, c*2):
        try:
            evaluate(bad_c, logits); raise AssertionError('Malformed mass accepted')
        except ValueError:
            pass
    checks.append('Malformed probability mass rejected')
    return dict(passed=True, checks=checks, nonuniform_count_example=counts.astype(int).tolist(),
                nonuniform_actions={name: r2[name]['actions'][0]
                    for name in ('oracle_branch_ce', 'oracle_joint_map', 'oracle_mixed_reward')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run synthetic tests (the only CLI operation)')
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False, indent=2, allow_nan=False))
