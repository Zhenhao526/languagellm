"""Small deterministic tests run before preparation and independently audited."""
from __future__ import annotations

from itertools import product

import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_rule_communication_study import kernel as old_kernel

from research_program.triadic_global_reward_control_study import design, kernel, remap, runner


def test_design():
    prepared = design.make_prepared()
    assert len(design.RULES) == 3
    assert len(design.CONDITIONS) == 24
    assert len(prepared['runs']) == 8 * 24
    assert prepared['need_count'] == 1560
    assert prepared['pair_set_counts'] == {'AB_AC': 520, 'AB_BC': 520, 'AC_BC': 520}
    assert design.PENALTIES['c0'] == 0 and design.GLOBAL_SCALES['global25'] == .25
    for condition in design.CONDITIONS:
        rule, information, schedule, live = design.parse_condition(condition)
        assert rule in design.RULES and information in design.INFORMATIONS
        assert schedule in design.SCHEDULES and isinstance(live, bool)


def test_remap():
    static = design.make_prepared()
    spec = static['partitions']['train']
    ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    pids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int8)
    for information in design.INFORMATIONS:
        arrays = dataset.make_arrays(spec, information=information, indices=ids)
        x, rewards, states = arrays['x_' + information], arrays['rewards'], arrays['packed_states']
        out_x, out_rewards, out_states = remap.remap_batch(x, rewards, states, pids, information)
        for row, pidx in enumerate(pids):
            perm = remap.PERMS[int(pidx)]
            assert np.array_equal(out_states[row, :3], states[row, list(perm)])
            assert np.array_equal(out_rewards[row], rewards[row, remap.PLAN_MAPS[int(pidx)]])
            if information == 'FI':
                expected = np.concatenate([x[row, 0, 7 * old:7 * old + 7] for old in perm])
                assert np.array_equal(out_x[row, 0, :21], expected)
                assert np.all(out_x[row, :, 53] == 1)
            else:
                for new, old in enumerate(perm):
                    assert np.array_equal(out_x[row, new, 7 * new:7 * new + 7],
                                          x[row, old, 7 * old:7 * old + 7])
                assert np.all(out_x[row, :, 53] == 0)


def _brute_expected(prob_intent, prob_proposal, packed_state, rule):
    total = 0.0
    for actions in product(range(17), repeat=3):
        prob = 1.0
        for actor, action in enumerate(actions):
            if action == 0:
                prob *= prob_intent[actor, 0]
            else:
                prob *= prob_intent[actor, 1] * prob_proposal[actor, action - 1]
        settled = runner.settle_factorized(
            packed_state[None], np.asarray(actions, dtype=np.int16)[None], rule)
        total += prob * float(settled['penalized_reward'][0])
    return total


def test_rule_objective():
    static = design.make_prepared()
    spec = static['partitions']['train']
    arrays = runner.make_arrays(spec, 'PL')
    arrays['rewards'] = arrays['rewards'][:2]
    arrays['packed_states'] = arrays['packed_states'][:2]
    rng = np.random.default_rng(991)
    logits = rng.normal(size=(2, 3, 18))
    for rule in design.RULES:
        terms = kernel.objective_terms(logits, arrays['rewards'], rule)
        for row in range(2):
            expected = _brute_expected(terms['intent_probabilities'][row],
                                       terms['proposal_probabilities'][row],
                                       arrays['packed_states'][row], rule)
            assert abs(expected - terms['J'][row]) < 1e-12
            eps = 1e-6
            plus = logits.copy(); minus = logits.copy()
            plus[row, 0, 0] += eps; minus[row, 0, 0] -= eps
            numeric = (kernel.objective_terms(plus, arrays['rewards'], rule)['log_J'][row]
                       - kernel.objective_terms(minus, arrays['rewards'], rule)['log_J'][row]) / (2 * eps)
            assert abs(numeric - terms['intent_log_gradient'][row, 0, 0]) < 1e-5
            plus = logits.copy(); minus = logits.copy()
            plus[row, 1, 7] += eps; minus[row, 1, 7] -= eps
            numeric = (kernel.objective_terms(plus, arrays['rewards'], rule)['log_J'][row]
                       - kernel.objective_terms(minus, arrays['rewards'], rule)['log_J'][row]) / (2 * eps)
            assert abs(numeric - terms['proposal_log_gradient'][row, 1, 5]) < 1e-5

    # c0 is exactly the old reciprocal objective.
    old = old_kernel.objective_terms(logits, arrays['rewards'], 'reciprocal')
    zero = kernel.objective_terms(logits, arrays['rewards'], 'c0')
    for key in ('J', 'log_J', 'intent_log_gradient', 'proposal_log_gradient'):
        assert np.max(np.abs(old[key] - zero[key])) < 1e-12
    # A global reward scale leaves the normalized policy gradient unchanged.
    scaled = kernel.objective_terms(logits, arrays['rewards'], 'global25')
    for key in ('intent_log_gradient', 'proposal_log_gradient'):
        assert np.max(np.abs(zero[key] - scaled[key])) < 1e-12
    assert np.max(np.abs(scaled['log_J'] - zero['log_J'] - np.log(.25))) < 1e-12
    assert np.max(np.abs(scaled['J'] - zero['J'] * .25)) < 1e-12


def test_evaluation_shapes():
    static = design.make_prepared()
    spec = static['partitions']['train']
    arrays = runner.make_arrays(spec, 'PL')
    networks = runner.make_networks(67001)
    for rule in design.RULES:
        result = runner.evaluate_factorized(networks, arrays, spec, 'PL', False, rule)
        assert result['worlds'] == spec['world_count']
        assert result['physical_worlds'] <= result['worlds']
        assert len(result['legal_plan_selection_counts']) == 24
        assert 0 <= result['reward_rate'] <= 1


def test_global_training_gradient_null():
    static = design.make_prepared()
    arrays = runner.make_arrays(static['partitions']['train'], 'PL')
    batch = slice(0, 5)
    networks = runner.make_networks(67001)
    uniforms = np.random.default_rng(670991).random((2, 5, 2, 3, 4))
    g0, row0 = kernel.training_gradients(
        networks, arrays['x_PL'][batch], arrays['rewards'][batch], True, uniforms, 1, 'c0')
    gg, rowg = kernel.training_gradients(
        networks, arrays['x_PL'][batch], arrays['rewards'][batch], True, uniforms, 1, 'global25')
    max_error = max(float(np.max(np.abs(a[k] - b[k])))
                    for a, b in zip(g0, gg) for k in a)
    assert max_error < 1e-12
    assert abs(rowg['mean_log_expected_utility'] - row0['mean_log_expected_utility'] - np.log(.25)) < 1e-12


if __name__ == '__main__':
    test_design(); test_remap(); test_rule_objective(); test_evaluation_shapes(); test_global_training_gradient_null()
    print('triadic_global_reward_control_study tests passed')
