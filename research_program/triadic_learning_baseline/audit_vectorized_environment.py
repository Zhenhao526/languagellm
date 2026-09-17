"""Independent finite environment and exact-objective audit; no training.

The candidate runner is imported only inside audit(), after manual source
review. Explicit random-array finite differences exercise MLP mathematics;
no study-seed actor, trained weights, optimizer or training is run.
"""
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np

from research_program.triadic_task import environment as env

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actions(agent):
    return [{"kind": "wait"}] + [
        {"kind": "transport", "site": f"S{site}", "destination": dest, "partner": partner}
        for site, dest, partner in product(range(4), ("L", "R"), [a for a in env.AGENTS if a != agent])]


def matching(triple):
    selected = {a: actions(a)[int(i)] for a, i in zip(env.AGENTS, triple)}
    active = [a for a in env.AGENTS if selected[a]["kind"] == "transport"]
    if len(active) != 2:
        return (), selected
    a, b = active
    x, y = selected[a], selected[b]
    matched = (x["partner"] == b and y["partner"] == a and
               x["site"] == y["site"] and x["destination"] == y["destination"])
    return tuple(active) if matched else (), selected


def own_reward(state, triple):
    active, selected = matching(triple)
    if not active:
        return 0.0
    action = selected[active[0]]
    material = state.layout[int(action["site"][1:])]
    dest = ("L", "R").index(action["destination"])
    units = 0
    for agent in active:
        resource, destinations = divmod(state.needs[env.AGENTS.index(agent)], 3)
        units += (material in ((0, 1), (2, 3), (0, 2), (1, 3))[resource]
                  and dest in ((0,), (1,), (0, 1))[destinations])
    return units / 2


def independent_rewards(states, triples):
    """Vectorize only states; resolve physical legality independently per triple."""
    needs = np.asarray([s.needs for s in states], dtype=np.int64)
    layouts = np.asarray([s.layout for s in states], dtype=np.int64)
    values = np.zeros((len(states), len(triples)), dtype=np.float64)
    resource_accept = np.asarray([[m in v for m in range(4)] for v in ((0, 1), (2, 3), (0, 2), (1, 3))])
    destination_accept = np.asarray([[d in v for d in range(2)] for v in ((0,), (1,), (0, 1))])
    for column, triple in enumerate(triples):
        active, selected = matching(triple)
        if active:
            example = selected[active[0]]
            material = layouts[:, int(example["site"][1:])]
            dest = ("L", "R").index(example["destination"])
            for agent in active:
                ids = needs[:, env.AGENTS.index(agent)]
                values[:, column] += .5 * (resource_accept[ids // 3, material] & destination_accept[ids % 3, dest])
    return values


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def dense_expectation(probs, rewards, triples):
    joint = probs[:, 0, triples[:, 0]] * probs[:, 1, triples[:, 1]] * probs[:, 2, triples[:, 2]]
    return (joint * rewards).sum(axis=-1)


def audit():
    from research_program.triadic_learning_baseline import runner as candidate
    dense = np.asarray(list(product(range(17), repeat=3)), dtype=np.int64)
    sparse = np.asarray(candidate.JOINT_ACTIONS, dtype=np.int64)
    own_legal = [tuple(t) for t in dense if matching(t)[0]]
    require(dense.shape == (4913, 3) and sparse.shape == (24, 3), "Wrong joint action sizes")
    require(len(set(map(tuple, sparse))) == 24 and set(map(tuple, sparse)) == set(own_legal), "Sparse actions omit or add a physical matching")
    require(all(actions(a) == env.all_actions(a) for a in env.AGENTS), "Original17 ordering changed")

    # Every structural joint action is checked against the immutable scalar env.
    reference_state = env.State((0, 0, 0), (0, 1, 2, 3))
    for triple in dense:
        active, selected = matching(triple)
        original = env.settle(reference_state, selected, require_match=True)
        executed = tuple(a for a in env.AGENTS if original["individual_feedback"][a]["executed"])
        require(executed == active, "4913-way scalar structural execution differs")
        require(original["reward"] == own_reward(reference_state, triple), "Scalar reward reference differs")

    states = [env.State(needs, layout) for needs in env.support() for layout in permutations(range(4))]
    require(len(states) == 23904, "Full demand/layout support differs")
    checked, candidate_max_error = 0, 0.0
    uniform_sum = 0.0
    for offset in range(0, len(states), 384):
        batch = states[offset:offset + 384]
        expected = independent_rewards(batch, sparse)
        actual = np.asarray(candidate.reward_terms(batch), dtype=np.float64)
        require(actual.shape == expected.shape and np.array_equal(actual, expected), "Full-support vector reward differs")
        require(np.isfinite(actual).all() and set(np.unique(actual)) <= {0., .5, 1.}, "Bad reward values")
        # Independent identity: sum all24 rewards = 2*sum destination cardinalities.
        destination_cards = np.asarray([[2 if n % 3 == 2 else 1 for n in s.needs] for s in batch])
        require(np.array_equal(expected.sum(axis=1), 2 * destination_cards.sum(axis=1)), "Uniform reward identity differs")
        candidate_max_error = max(candidate_max_error, float(np.max(np.abs(actual-expected))))
        uniform_sum += float(expected.sum()) / 4913
        checked += expected.size

    # Public private-site assignment changes observation allocation, not physics.
    variants = [env.State(states[i].needs, states[i].layout, owners)
                for i in (0, 397, 4000, 23903) for owners in permutations((1, 2, 3))]
    v = np.asarray(candidate.reward_terms(variants))
    require(np.array_equal(v, independent_rewards(variants, sparse)), "Private-site assignment changed reward")
    for start in range(0, len(variants), 6):
        require(np.array_equal(v[start:start + 6], np.repeat(v[start:start + 1], 6, axis=0)), "Reward depends on observer assignment")

    rng = np.random.default_rng(20260918031)
    probes = [states[i] for i in (0, 17, 395, 4086, 19153, 23903)]
    logits = rng.normal(0., .7, (len(probes), 3, 17))
    probs = softmax(logits)
    rdense = independent_rewards(probes, dense)
    rsparse = independent_rewards(probes, sparse)
    expected_E = dense_expectation(probs, rdense, dense)
    actual_E, actual_gradient = candidate.expected_reward_and_logit_gradient(probs, rsparse)
    actual_E, actual_gradient = np.asarray(actual_E), np.asarray(actual_gradient)
    expected_joint = probs[:, 0, sparse[:, 0]] * probs[:, 1, sparse[:, 1]] * probs[:, 2, sparse[:, 2]]
    require(np.allclose(actual_E, expected_E, rtol=1e-12, atol=1e-14), "Exact sparse expectation differs from full4913")
    require(actual_gradient.shape == probs.shape and np.isfinite(actual_gradient).all(), "Gradient shape/finite mismatch")
    require(np.allclose(actual_gradient.sum(axis=-1), 0., rtol=0, atol=1e-13), "Softmax gradient sums nonzero")
    h, fd = 1e-5, np.empty_like(logits)
    for row, actor, action in product(range(len(probes)), range(3), range(17)):
        high, low = logits[row:row+1].copy(), logits[row:row+1].copy()
        high[0, actor, action] += h; low[0, actor, action] -= h
        fd[row, actor, action] = (dense_expectation(softmax(high), rdense[row:row+1], dense)[0]
                                  - dense_expectation(softmax(low), rdense[row:row+1], dense)[0]) / (2*h)
    require(np.allclose(actual_gradient, fd, rtol=2e-6, atol=2e-11), "Analytic gradient differs from independent finite difference")
    uniform = np.full_like(probs, 1/17)
    uE, _ = candidate.expected_reward_and_logit_gradient(uniform, rsparse)
    require(np.allclose(uE, rdense.sum(axis=1)/4913, rtol=1e-12, atol=1e-14), "Matched terms were renormalized")
    require(np.all(actual_E <= 1) and np.all(actual_E >= 0), "Expected reward outside [0,1]")

    # Full and partial encoders are checked without ever calling a policy.
    encoder_rows = []
    for state in variants:
        views = {a: env.observe(state, a, shared_needs=True, full_information=True) for a in env.AGENTS}
        encoder_rows.append(views)
    encoded = candidate.encode_observations(encoder_rows)
    require(encoded.shape == (24, 3, 54) and set(np.unique(encoded)) <= {0., 1.}, "Unexpected encoder shape/value")
    for b, state in enumerate(variants):
        for actor in range(3):
            expected = np.zeros(54)
            for who, need in enumerate(state.needs):
                resource, destination = divmod(need, 3)
                expected[7*who] = 1
                expected[7*who+1+resource] = 1
                for d in ((0,), (1,), (0, 1))[destination]:
                    expected[7*who+5+d] = 1
            for site, material in enumerate(state.layout):
                expected[21+5*site] = 1
                expected[21+5*site+1+material//2] = 1
                expected[21+5*site+3+material%2] = 1
            for who, site in enumerate(state.private_sites):
                expected[41+3*(site-1)+who] = 1
            expected[50+actor] = expected[53] = 1
            require(np.array_equal(encoded[b, actor], expected), "Full feature semantic encoding differs")
    local_pairs_checked = 0
    for actor, name in enumerate(env.AGENTS):
        base = env.State((0, 0, 0), (0, 1, 2, 3))
        hidden = [s for s in range(4) if s not in (0, base.private_sites[actor])]
        layout = list(base.layout); layout[hidden[0]], layout[hidden[1]] = layout[hidden[1]], layout[hidden[0]]
        needs = [4, 7, 10]; needs[actor] = 0
        other = env.State(tuple(needs), tuple(layout))
        views = [{a: env.observe(s, a, shared_needs=False, full_information=False) for a in env.AGENTS} for s in (base, other)]
        require(views[0][name] == views[1][name], "Local hidden-state fixture invalid")
        x = candidate.encode_observations(views)
        require(np.array_equal(x[0, actor], x[1, actor]), "Other actors' observations leak into own features")
        for who in range(3):
            if who != actor:
                require(not x[0, actor, 7*who:7*who+7].any(), "Hidden demand populated")
        for site in hidden:
            require(not x[0, actor, 21+5*site:21+5*site+5].any(), "Hidden material populated")
        require(x[0, actor, 53] == 0, "Local view flagged complete")
        local_pairs_checked += 1
    contaminated = {a: env.observe(base, a, shared_needs=True, full_information=True) for a in env.AGENTS}
    contaminated["A"]["researcher_witness"] = {"reward": 1}
    try:
        candidate.encode_observations([contaminated])
    except ValueError:
        pass
    else:
        raise AssertionError("Researcher field accepted by encoder")

    # Entropy gradient is the mean over actors, checked independently.
    logs = np.log(probs)
    entropy, entropy_gradient = candidate.entropy_and_logit_gradient(probs, logs)
    require(np.allclose(entropy, -(probs*logs).sum(-1).mean(-1), rtol=0, atol=1e-14), "Entropy reduction differs")
    hfd = np.empty_like(logits)
    for row, actor, action in product(range(len(probes)), range(3), range(17)):
        hi, lo = logits[row:row+1].copy(), logits[row:row+1].copy()
        hi[0, actor, action] += h; lo[0, actor, action] -= h
        pp, pm = softmax(hi), softmax(lo)
        hfd[row, actor, action] = (-(pp*np.log(pp)).sum(-1).mean() + (pm*np.log(pm)).sum(-1).mean())/(2*h)
    require(np.allclose(entropy_gradient, hfd, rtol=2e-6, atol=2e-10), "Entropy logit gradient differs")

    # Scratch random matrices test the shared MLP algebra, not task learning.
    scratch = {}
    for i, (left, right) in enumerate(((54, 64), (64, 64), (64, 17)), 1):
        scratch[f"W{i}"] = rng.normal(0, .07, (left, right))
        scratch[f"b{i}"] = rng.normal(0, .03, right)
    inputs, upstream = rng.normal(0, .4, (4, 54)), rng.normal(0, .2, (4, 17))
    _, cache = candidate.actor_forward(scratch, inputs)
    analytic = candidate.actor_backward(scratch, cache, upstream)
    parameter_errors, forwards = [], 1
    for key, values in scratch.items():
        for flat_index in (0, values.size//2, values.size-1):
            index = np.unravel_index(flat_index, values.shape)
            old = values[index]
            values[index] = old+h
            hi = float((candidate.actor_forward(scratch, inputs)[0] * upstream).sum())
            values[index] = old-h
            lo = float((candidate.actor_forward(scratch, inputs)[0] * upstream).sum())
            values[index] = old
            numerical = (hi-lo)/(2*h)
            require(np.isclose(analytic[key][index], numerical, rtol=2e-6, atol=2e-10), "MLP parameter gradient differs: "+key)
            parameter_errors.append(abs(float(analytic[key][index])-numerical)); forwards += 2

    # Validate the four Cartesian partitions without creating policy objects.
    prepared = candidate.make_prepared()
    tr = set(map(tuple, prepared["train_need_multisets"]))
    ho = set(map(tuple, prepared["heldout_need_multisets"]))
    require(len(tr) == 159 and len(ho) == 53 and not tr & ho
            and tr | ho == {tuple(sorted(s)) for s in env.support()}, "Multiset split incomplete or overlapping")
    assigned, partition_counts = set(), {}
    for name in candidate.PARTITIONS:
        spec = prepared["partitions"][name]
        expected_need_set = tr if name in ("train", "new_layouts") else ho
        require({tuple(sorted(n)) for n in spec["needs"]} == expected_need_set, "Needs escaped multiset split")
        require(set(map(tuple, spec["private_sites"])) == set(permutations((1, 2, 3))), "Observation allocations omitted")
        keys = {tuple(n)+tuple(l)+tuple(p) for n, l, p in product(spec["needs"], spec["layouts"], spec["private_sites"])}
        require(not keys & assigned and len(keys) == spec["world_count"], "World partition overlap or duplicate")
        assigned.update(keys); partition_counts[name] = len(keys)
        require(len(spec["monitor_indices"]) == len(set(spec["monitor_indices"])) == 1024 and
                all(0 <= i < len(keys) for i in spec["monitor_indices"]), "Invalid fixed monitor sample")
    parts = prepared["partitions"]
    train_layouts, held_layouts = set(map(tuple, parts["train"]["layouts"])), set(map(tuple, parts["new_layouts"]["layouts"]))
    require(len(train_layouts) == 18 and len(held_layouts) == 6 and not train_layouts & held_layouts
            and train_layouts | held_layouts == set(permutations(range(4))), "Layout split differs")
    require(set(map(tuple, parts["new_needs"]["layouts"])) == train_layouts and
            set(map(tuple, parts["new_needs_and_layouts"]["layouts"])) == held_layouts, "Crossed layout split differs")
    require(len(assigned) == 996*24*6, "Full world support not covered")
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
            "audit_source_sha256": sha(__file__), "runner_sha256": sha(candidate.__file__), "environment_sha256": sha(env.__file__),
            "trained_or_pretrained_model_calls": 0, "training_updates": 0, "optimizer_updates": 0,
            "study_seed_actor_objects_created": 0, "random_matrix_forward_checks": forwards,
            "joint_action_structure_checked_against_scalar_environment": len(dense),
            "structurally_executable_terms": len(sparse), "zero_reward_structural_terms": len(dense)-len(sparse),
            "full_semantic_states": len(states), "full_support_reward_cells_exact": checked,
            "full_support_reward_max_abs_error": candidate_max_error,
            "observer_assignment_reward_cells_exact": int(v.size),
            "dense_expectation_probe_states": len(probes), "finite_difference_logits": int(logits.size),
            "expectation_max_abs_error": float(np.max(np.abs(actual_E-expected_E))),
            "gradient_max_abs_error": float(np.max(np.abs(actual_gradient-fd))),
            "entropy_gradient_max_abs_error": float(np.max(np.abs(entropy_gradient-hfd))),
            "random_matrix_parameter_coordinates_checked": len(parameter_errors),
            "random_matrix_parameter_gradient_max_abs_error": max(parameter_errors),
            "full_encoder_vectors_checked": 72, "local_hidden_state_pairs_checked": local_pairs_checked,
            "researcher_field_rejected": True, "partition_counts": partition_counts,
            "partitions_disjoint_and_full_support": True,
            "uniform_structural_execution_probability": 24/4913,
            "uniform_expected_reward_full_support": uniform_sum / len(states),
            "gradient_tolerance": {"relative": 2e-6, "absolute": 2e-11, "difference_step": h},
            "scope": "Finite reward/logit/entropy/encoder/split checks plus explicitly authorized scratch-matrix MLP finite differences; no training, optimizer update, pretrained model, study-seed actor, or training-success claim"}


if __name__ == "__main__":
    out = HERE / "independent_environment_audit.json"
    require(not out.exists(), "Do not overwrite prior audit")
    result = audit()
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
