"""Post-hoc aligned/placebo transfer probe on the factorial holdout policies.

The source policies are frozen outputs of ``factorial_001_corrected``.  Every
heldout-resource edge is expanded in both directions and evaluated on the
same heldout layouts and site owners.  The probe only performs forward
rollouts and settlement; it never updates a parameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as message_runner
from research_program.triadic_factorial_formation_study import runner as factorial_runner
from research_program.triadic_reciprocal_execution_study import environment


SEEDS = tuple(range(62101, 62117))
REGIMES = ("factorial_holdout", "saturated")
RULES = ("strict", "reciprocal")
CHANNELS = ("live", "silent")
AXES = ("kind", "length", "destination")
PARTITION = "heldout_both"
CASES_PER_POLICY = 2568
BACKGROUND_COUNT = 36
CHUNK_CASES = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def condition_name(regime, rule, channel):
    require(regime in REGIMES and rule in RULES and channel in CHANNELS, "Invalid policy factors")
    return f"{regime}_{rule}_PL_{channel}"


def build_cases(static):
    specs = static["partitions"]
    left = specs[REGIMES[0]][PARTITION]
    right = specs[REGIMES[1]][PARTITION]
    require(left["needs"] == right["needs"] and left["layouts"] == right["layouts"] and left["private_sites"] == right["private_sites"], "Regime heldout domain mismatch")
    raw = static["need_response_cases"][REGIMES[0]][PARTITION]
    needs = [tuple(map(int, row)) for row in left["needs"]]
    axis_names = AXES
    cases = []
    edges = raw["edge_need_indices"]
    changed = raw["changed_person"]
    axes = raw["axis_index"]
    pairs = raw["target_pairs"]
    for edge_index, ((before, after), who, axis_index, pair_pair) in enumerate(zip(edges, changed, axes, pairs)):
        source_pair, target_pair = map(int, pair_pair)
        require(source_pair != target_pair, "Edge pair labels must differ")
        axis = axis_names[int(axis_index)]
        for direction, source_index, target_index, source_label, target_label in (
            ("before_to_after", int(before), int(after), source_pair, target_pair),
            ("after_to_before", int(after), int(before), target_pair, source_pair),
        ):
            cases.append(dict(
                case_id=f"edge_{edge_index:04d}__{direction}", edge_index=edge_index,
                axis=axis, axis_index=int(axis_index), changed_person=int(who), direction=direction,
                source_need_index=source_index, target_need_index=target_index,
                source_needs=list(needs[source_index]), target_needs=list(needs[target_index]),
                source_pair=int(source_label), target_pair=int(target_label),
            ))
    cases.sort(key=lambda row: (row["axis_index"], row["changed_person"], row["direction"], tuple(row["source_needs"]), tuple(row["target_needs"])))
    require(len(cases) == CASES_PER_POLICY, "Expected 2568 directed heldout edges")
    counts = {axis: sum(row["axis"] == axis for row in cases) for axis in AXES}
    require(counts == {"kind": 1008, "length": 1008, "destination": 552}, "Unexpected directed axis counts")
    strata = defaultdict(list)
    for index, row in enumerate(cases):
        strata[(row["axis"], row["changed_person"], row["direction"])].append(index)
    require(len(strata) == 18 and all(len(indices) >= 2 for indices in strata.values()), "Placebo strata incomplete")
    mapping = list(range(len(cases)))
    for indices in strata.values():
        for pos, index in enumerate(indices):
            mapping[index] = indices[(pos + 1) % len(indices)]
    require(all(mapping[i] != i for i in range(len(cases))), "Placebo mapping identity")
    return cases, mapping, {"axis_case_counts": counts, "strata": {str(key): len(value) for key, value in sorted(strata.items(), key=lambda item: str(item[0]))}}


def prepare(out, source):
    out = Path(out).resolve(); source = Path(source).resolve(); require(not out.exists(), "Never overwrite preparation")
    static = json.loads((source / "prepared.json").read_text(encoding="utf8"))
    require((source / "freeze.json").is_file(), "Missing source freeze")
    cases, mapping, meta = build_cases(static)
    out.mkdir(parents=True)
    write_new(out / "cases.json", dict(schema="factorial_heldout_semantic_transfer_cases_v1", cases=cases, placebo_mapping=mapping, meta=meta))
    manifest = dict(
        schema="factorial_heldout_semantic_transfer_manifest_v1", source=str(source), partition=PARTITION,
        seeds=list(SEEDS), regimes=list(REGIMES), rules=list(RULES), channels=list(CHANNELS),
        policy_blocks=len(SEEDS) * len(REGIMES) * len(RULES) * len(CHANNELS), cases=CASES_PER_POLICY,
        backgrounds_per_case=BACKGROUND_COUNT, directed_edges=CASES_PER_POLICY,
        axis_case_counts=meta["axis_case_counts"],
        aligned="source endpoint first-window message shown to target cross-viewers; target self-view retained; W2/action and settlement recomputed",
        placebo="cyclic source-message donor within axis×changed_person×direction; message marginal preserved and source-demand alignment broken",
        no_training_by_probe=True,
    )
    write_new(out / "manifest.json", manifest)
    prepared = dict(
        schema="factorial_heldout_semantic_transfer_prepared_v1", source=str(source), partition=PARTITION,
        source_prepared_sha256=sha(source / "prepared.json"), source_freeze_sha256=sha(source / "freeze.json"),
        candidate_sha256=sha(out / "cases.json"), manifest_sha256=sha(out / "manifest.json"),
        seeds=list(SEEDS), regimes=list(REGIMES), rules=list(RULES), channels=list(CHANNELS),
        cases=CASES_PER_POLICY, backgrounds_per_case=BACKGROUND_COUNT,
        axis_case_counts=meta["axis_case_counts"], no_training=True, no_optimizer_updates=True,
    )
    write_new(out / "prepared.json", prepared)
    plan = dict(
        status="prepared_without_policy_reads", created_at=message_runner.base.now(), runtime=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={"source_prepared.json": prepared["source_prepared_sha256"], "source_freeze.json": prepared["source_freeze_sha256"], "cases.json": prepared["candidate_sha256"], "manifest.json": sha(out / "manifest.json"), "prepared.json": sha(out / "prepared.json")},
        prepared_sha256=sha(out / "prepared.json"), policy_blocks=128, cases=CASES_PER_POLICY, no_training=True, no_optimizer_updates=True,
    )
    write_new(out / "plan.json", plan); write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), policy_blocks=128, cases=CASES_PER_POLICY, model_calls=0, optimizer_updates=0))
    return dict(status="prepared_static_only", output=str(out), policy_blocks=128, cases=CASES_PER_POLICY)


def verify(out):
    out = Path(out).resolve(); plan = json.loads((out / "plan.json").read_text()); prepared = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan mismatch"); require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared mismatch")
    require(prepared["manifest_sha256"] == sha(out / "manifest.json"), "Manifest mismatch")
    source = Path(prepared["source"]); require(sha(source / "prepared.json") == prepared["source_prepared_sha256"] and sha(source / "freeze.json") == prepared["source_freeze_sha256"], "Source freeze changed")
    require(sha(out / "cases.json") == prepared["candidate_sha256"], "Cases changed")
    return plan, prepared, json.loads((out / "cases.json").read_text())


def route_transfer(target_tokens, source_sender_tokens, sender, live=True):
    target_tokens = np.asarray(target_tokens); source_sender_tokens = np.asarray(source_sender_tokens)
    require(target_tokens.ndim == 3 and target_tokens.shape[1:] == (3, 4) and source_sender_tokens.shape == (len(target_tokens), 4), "Token shape")
    visibility = np.ones((3, 3), dtype=np.float64) if live else np.eye(3, dtype=np.float64)
    visible = np.eye(8, dtype=np.float64)[target_tokens][:, None] * visibility[None, :, :, None, None]
    if live:
        source_onehot = np.eye(8, dtype=np.float64)[source_sender_tokens]
        for viewer in range(3):
            if viewer != int(sender):
                visible[:, viewer, int(sender)] = source_onehot
    bits = np.broadcast_to(visibility[None], (len(target_tokens), 3, 3)).copy()
    return np.concatenate((visible.reshape(len(target_tokens), 3, 96), bits), axis=-1)


def action_probs_after_transfer(networks, x, target_first, source_sender, sender, live=True):
    routed_first = route_transfer(target_first, source_sender, sender, live)
    second_input = np.concatenate((x, routed_first), axis=-1); second_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 1], second_input[:, actor]); second_logits.append(z.reshape(len(x), 4, 8))
    second_probabilities, _ = message_runner.base.policy_distribution(np.stack(second_logits, axis=1)); second_tokens = message_runner.categorical_tokens(second_probabilities)
    action_input = np.concatenate((x, routed_first, message_runner.routed_window(second_tokens, live)), axis=-1); action_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 2], action_input[:, actor]); action_logits.append(z)
    action_probabilities, _ = message_runner.base.policy_distribution(np.stack(action_logits, axis=1))
    return action_probabilities


def pair_metrics(intervened_pair, natural_pair, source_pair, target_pair, intervened_actions, natural_actions):
    intervened_pair = np.asarray(intervened_pair); natural_pair = np.asarray(natural_pair)
    source_pair = int(source_pair); target_pair = int(target_pair)
    natural_source = float(np.mean(natural_pair == source_pair)); natural_target = float(np.mean(natural_pair == target_pair))
    intervention_source = float(np.mean(intervened_pair == source_pair)); intervention_target = float(np.mean(intervened_pair == target_pair))
    natural_physical = float(np.mean(natural_pair >= 0)); intervention_physical = float(np.mean(intervened_pair >= 0))
    return dict(
        natural=dict(source_pair_hit=natural_source, target_pair_hit=natural_target, physical_execution_rate=natural_physical, pair_change_rate=0.0),
        intervened=dict(source_pair_hit=intervention_source, target_pair_hit=intervention_target, physical_execution_rate=intervention_physical, pair_change_rate=float(np.mean(intervened_pair != natural_pair))),
        delta=dict(source_pair_hit=intervention_source - natural_source, target_pair_hit=intervention_target - natural_target, physical_execution_rate=intervention_physical - natural_physical, pair_change_rate=float(np.mean(intervened_pair != natural_pair))),
        source_pull=float((intervention_source - natural_source) - (intervention_target - natural_target)), worlds=len(intervened_pair),
        action_change_rate=float(np.mean(np.any(np.asarray(intervened_actions) != np.asarray(natural_actions), axis=1))),
    )


def run_policy(source, prepared_static, cases, mapping, seed, regime, rule, channel):
    source = Path(source); condition = condition_name(regime, rule, channel); folder = source / "execution" / f"seed_{seed}_{condition}"; checkpoint = folder / "checkpoint_6000.npz"; result_path = folder / "result.json"; data_path = folder / f"final_{PARTITION}.npz"
    require(checkpoint.is_file() and result_path.is_file() and data_path.is_file(), "Missing frozen policy files")
    networks = message_runner.load_networks(checkpoint)
    spec = prepared_static["partitions"][regime][PARTITION]; arrays = factorial_runner.make_arrays(spec); nphysical = len(spec["layouts"]) * len(spec["private_sites"]); lookup = {tuple(need): index for index, need in enumerate(spec["needs"])}
    with np.load(data_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in ("states", "messages", "action_indices", "action_probabilities", "actual_pair_index")}
    n = int(spec["world_count"]); require(n == CASES_PER_POLICY * 0 or saved["states"].shape == (n, 10), "State shape mismatch"); require(saved["messages"].shape == (n, 2, 3, 4) and saved["action_indices"].shape == (n, 3), "Natural arrays shape mismatch")
    first_all = saved["messages"][:, 0]; natural_actions_all = saved["action_indices"]; natural_pair_all = saved["actual_pair_index"]; live = channel == "live"; rows = []
    # Grouping by sender permits one batched network replay per chunk instead of one call per edge.
    indices_by_sender = defaultdict(list)
    for index, case in enumerate(cases): indices_by_sender[case["changed_person"]].append(index)
    for sender, case_indices in sorted(indices_by_sender.items()):
        for start in range(0, len(case_indices), CHUNK_CASES):
            selected = case_indices[start:start + CHUNK_CASES]
            target_ids_list = []; source_ids_list = []; donor_ids_list = []
            for index in selected:
                case = cases[index]; backgrounds = np.arange(nphysical, dtype=np.int64); source_ids = lookup[tuple(case["source_needs"])] * nphysical + backgrounds; target_ids = lookup[tuple(case["target_needs"])] * nphysical + backgrounds; donor_case = cases[mapping[index]]; donor_ids = lookup[tuple(donor_case["source_needs"])] * nphysical + backgrounds
                require(np.array_equal(saved["states"][source_ids, 3:], saved["states"][target_ids, 3:]), "Background mismatch")
                target_ids_list.append(target_ids); source_ids_list.append(source_ids); donor_ids_list.append(donor_ids)
            target_ids = np.concatenate(target_ids_list); source_ids = np.concatenate(source_ids_list); donor_ids = np.concatenate(donor_ids_list)
            target_first = first_all[target_ids]; source_sender = first_all[source_ids, sender]; placebo_sender = first_all[donor_ids, sender]; natural_actions = natural_actions_all[target_ids]; natural_pair = natural_pair_all[target_ids]
            if live:
                aligned_actions = action_probs_after_transfer(networks, arrays["x_PL"][target_ids], target_first, source_sender, sender, True).argmax(axis=-1).astype(np.int16)
                placebo_actions = action_probs_after_transfer(networks, arrays["x_PL"][target_ids], target_first, placebo_sender, sender, True).argmax(axis=-1).astype(np.int16)
                aligned_pair = environment.settle(saved["states"][target_ids], aligned_actions, rule)["actual_pair_index"]
                placebo_pair = environment.settle(saved["states"][target_ids], placebo_actions, rule)["actual_pair_index"]
                forwards = 12 * len(target_ids); alias = False
            else:
                aligned_actions = placebo_actions = natural_actions.copy(); aligned_pair = placebo_pair = natural_pair.copy(); forwards = 0; alias = True
            for position, index in enumerate(selected):
                sl = slice(position * nphysical, (position + 1) * nphysical); case = cases[index]; donor_case = cases[mapping[index]]
                aligned_values = pair_metrics(aligned_pair[sl], natural_pair[sl], case["source_pair"], case["target_pair"], aligned_actions[sl], natural_actions[sl]); placebo_values = pair_metrics(placebo_pair[sl], natural_pair[sl], case["source_pair"], case["target_pair"], placebo_actions[sl], natural_actions[sl])
                rows.append(dict(
                    **case, placebo_donor_case_id=donor_case["case_id"], placebo_donor_source_need_index=donor_case["source_need_index"],
                    checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), data_sha256=sha(data_path),
                    target_indices_sha256=array_sha(target_ids[sl]), source_indices_sha256=array_sha(source_ids[sl]), donor_indices_sha256=array_sha(donor_ids[sl]),
                    target_first_messages_sha256=array_sha(target_first[sl]), source_sender_messages_sha256=array_sha(source_sender[sl]), placebo_sender_messages_sha256=array_sha(placebo_sender[sl]),
                    trained_channel=condition, rule=rule, alias_of_natural=alias, model_forward_samples=forwards // len(selected),
                    intervention="aligned source endpoint first-window message or cyclically misaligned same-stratum donor shown to target cross-viewers; target self-view retained; W2/action/settlement recomputed",
                    aligned=aligned_values, placebo=placebo_values,
                ))
    case_order = {case["case_id"]: index for index, case in enumerate(cases)}
    rows.sort(key=lambda row: case_order[row["case_id"]])
    require(len(rows) == CASES_PER_POLICY, "Policy row count")
    return dict(seed=seed, regime=regime, rule=rule, channel=channel, condition=condition, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), data_sha256=sha(data_path), target_first_messages_sha256=array_sha(first_all), policy_rows=rows, alias_of_natural=not live)


def worker(payload):
    source, prepared_static, cases, mapping, seed, regime, rule, channel = payload
    return run_policy(source, prepared_static, cases, mapping, int(seed), regime, rule, channel)


def execute(out, workers=4):
    out = Path(out).resolve(); _, prepared, case_doc = verify(out); cases = case_doc["cases"]; mapping = case_doc["placebo_mapping"]; execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir(); started = time.perf_counter()
    tasks = [(prepared["source"], json.loads((Path(prepared["source"]) / "prepared.json").read_text()), cases, mapping, seed, regime, rule, channel) for seed in SEEDS for regime in REGIMES for rule in RULES for channel in CHANNELS]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool: policies = pool.map(worker, tasks)
    require(len(policies) == 128 and all(len(policy["policy_rows"]) == CASES_PER_POLICY for policy in policies), "Incomplete policy grid")
    rows = [row for policy in policies for row in policy["policy_rows"]]; live_blocks = sum(policy["channel"] == "live" for policy in policies); expected_rows = 128 * CASES_PER_POLICY
    result = dict(status="completed_json_only_factorial_heldout_semantic_transfer_probe", completed_at=message_runner.base.now(), elapsed_seconds=time.perf_counter() - started, source=prepared["source"], partition=PARTITION, seeds=list(SEEDS), regimes=list(REGIMES), rules=list(RULES), channels=list(CHANNELS), policy_blocks=128, cases_per_policy=CASES_PER_POLICY, intervention_rows=len(rows), live_rows=64 * CASES_PER_POLICY, alias_rows=64 * CASES_PER_POLICY, backgrounds_per_case=BACKGROUND_COUNT, worlds=len(rows) * BACKGROUND_COUNT, model_forward_samples=sum(row["model_forward_samples"] for row in rows), optimizer_updates=0, training_updates=0, model_calls=0, axis_case_counts={axis: sum(row["axis"] == axis for row in rows) for axis in AXES}, policies=policies, interpretation_boundary="Post-hoc heldout-combination aligned/placebo transfer diagnostic; not lexical meaning or language-origin evidence.")
    write_new(execution / "results.json", result); result_sha = sha(execution / "results.json"); write_new(execution / "status.json", dict(status="completed", results_sha256=result_sha, completed_at=message_runner.base.now(), elapsed_seconds=result["elapsed_seconds"])); write_new(execution / "receipt.json", dict(status="passed", results_sha256=result_sha, policy_blocks=128, intervention_rows=len(rows), live_rows=64 * CASES_PER_POLICY, alias_rows=64 * CASES_PER_POLICY, worlds=result["worlds"], model_forward_samples=result["model_forward_samples"], optimizer_updates=0))
    return dict(status=result["status"], results_sha256=result_sha, rows=len(rows), model_forward_samples=result["model_forward_samples"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); parser.add_argument("--source"); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    if args.command == "prepare": require(args.source, "prepare requires --source"); value = prepare(args.out, args.source)
    elif args.command == "verify": verify(args.out); value = dict(status="verified", output=str(Path(args.out).resolve()))
    else: value = execute(args.out, args.workers)
    print(json.dumps(value, ensure_ascii=False))
