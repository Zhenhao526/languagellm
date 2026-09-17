"""Bounded researcher optimization of legal I1-D1 decision tables, not training.

Prepare first, then execute only after the frozen plan is reviewed. The complete
state is used to evaluate an expectation; each deployed agent reads only its
own demand and two visible resources through a fixed 144-row action table.
No models, messages, gradients, policy networks, or current-state shared seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np


OUT = Path(__file__).resolve().parent
BASE = OUT.parent
WORK = BASE.parents[1]
RANDOM_SEEDS = (2026091501, 2026091502, 2026091503, 2026091504, 2026091505, 2026091506, 2026091507)
MAX_ROUNDS = 20
AGENTS = (0, 1, 2)
RESOURCE_SETS = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATION_SETS = ({0}, {1}, {0, 1})
DEMANDS = tuple(product(range(4), range(3)))
OBSERVATIONS = tuple((d, p, own) for d in range(12) for p in range(4) for own in range(4) if own != p)
UPPER = Fraction(75, 83)
SOURCE_FILES = (
    BASE / "fixed_pair_baseline.py", BASE / "fixed_pair_plan.json", BASE / "fixed_pair_results.json",
    BASE / "relaxation_001/ecology_information_bounds.py", BASE / "relaxation_001/plan.json",
    BASE / "relaxation_001/freeze.json", BASE / "relaxation_001/execution/results.json",
    BASE / "relaxation_001/execution/I1_D1_agent0.npz",
    BASE / "relaxation_001/execution/I1_D1_agent1.npz",
    BASE / "relaxation_001/execution/I1_D1_agent2.npz",
    BASE.parent / "ecology_draft_counts.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


def action_catalog(agent):
    # Same 17 semantic options at every observation, including wait and all partners.
    return [(-1, -1, -1)] + list(product(range(4), range(2), [p for p in AGENTS if p != agent]))


def compatible(a, b):
    ra, da = DEMANDS[a]
    rb, db = DEMANDS[b]
    return bool(RESOURCE_SETS[ra] & RESOURCE_SETS[rb]) and bool(DESTINATION_SETS[da] & DESTINATION_SETS[db])


def build_distribution():
    tables = [t for t in product(range(12), repeat=3)
              if sum(compatible(t[a], t[b]) for a, b in combinations(AGENTS, 2)) >= 2]
    layouts = list(permutations(range(4)))
    require(len(tables) == 996 and len(layouts) == 24, "Unexpected demand/layout support")
    demands = np.repeat(np.asarray(tables, dtype=np.int16), 24, axis=0)
    worlds = np.tile(np.asarray(layouts, dtype=np.int8), (996, 1))
    indices = {obs: i for i, obs in enumerate(OBSERVATIONS)}
    observations = np.asarray([[indices[(int(d[a]), int(w[0]), int(w[a + 1]))] for a in AGENTS]
                               for d, w in zip(demands, worlds)], dtype=np.int16)
    counts = [np.bincount(observations[:, a], minlength=144) for a in AGENTS]
    require(observations.shape == (23904, 3) and all((c > 0).all() for c in counts), "Missing local observations")
    return demands, worlds, observations, counts


def resolve_actions(tables, observations):
    tables = np.asarray(tables)
    require(tables.shape == (3, 144) and np.issubdtype(tables.dtype, np.integer)
            and ((0 <= tables) & (tables < 17)).all(), "Invalid local policy table")
    return tables[np.arange(3)[None, :], observations]


def score_units(demands, layouts, action_ids):
    """Exact simultaneous D1 settlement for N complete states; integer 0/1/2."""
    require(action_ids.shape == (len(layouts), 3), "Action batch has wrong shape")
    require(np.issubdtype(action_ids.dtype, np.integer) and ((action_ids >= 0) & (action_ids < 17)).all(),
            "Action ids outside the unpruned 17 choices")
    catalogs = np.asarray([action_catalog(a) for a in AGENTS], dtype=np.int8)
    actions = catalogs[np.arange(3)[None, :], action_ids]
    sites, destinations, partners = actions[:, :, 0], actions[:, :, 1], actions[:, :, 2]
    active = action_ids != 0
    rows = np.arange(len(layouts))[:, None]
    partner_actions = actions[rows, np.maximum(partners, 0)]
    matching = (active & (active.sum(1)[:, None] == 2)
                & (partner_actions[:, :, 0] == sites)
                & (partner_actions[:, :, 1] == destinations)
                & (partner_actions[:, :, 2] == np.arange(3)[None, :]))
    material = layouts[rows, np.maximum(sites, 0)]
    accepted_material = np.asarray([[r in RESOURCE_SETS[DEMANDS[d][0]] for r in range(4)] for d in range(12)])
    accepted_destination = np.asarray([[p in DESTINATION_SETS[DEMANDS[d][1]] for p in range(2)] for d in range(12)])
    own_correct = accepted_material[demands, material] & accepted_destination[demands, np.maximum(destinations, 0)]
    return (matching & own_correct).sum(1, dtype=np.int8)


def summarize_policy(tables, demands, layouts, observations):
    ids = resolve_actions(tables, observations)
    units = score_units(demands, layouts, ids)
    numerator, denominator = int(units.sum()), 2 * len(units)
    require(Fraction(numerator, denominator) <= UPPER, "Legal policy exceeded the frozen 75/83 upper bound")
    # Check separately from advanced-index action resolution that no observation
    # gets a different action in a different full world.
    for a in AGENTS:
        for obs in range(144):
            selected = ids[observations[:, a] == obs, a]
            require(len(selected) > 0 and np.all(selected == tables[a, obs]), "Local consistency failed")
    return {"success_units": numerator, "score_denominator": denominator,
            "score_fraction": str(Fraction(numerator, denominator)), "score": numerator / denominator,
            "reward_histogram_success_units": {str(k): int(v) for k, v in sorted(Counter(units.tolist()).items())},
            "local_observation_consistency": True, "at_or_below_75_over_83": True}


def best_response(tables, agent, demands, layouts, observations):
    """Optimize all 144 rows for one agent, with both other tables held fixed."""
    fixed_ids = resolve_actions(tables, observations)
    rewards = np.zeros((144, 17), dtype=np.int64)
    for candidate in range(17):
        submitted = fixed_ids.copy()
        submitted[:, agent] = candidate
        np.add.at(rewards[:, candidate], observations[:, agent], score_units(demands, layouts, submitted))
    old = tables[agent].copy()
    old_values = rewards[np.arange(144), old]
    best_values = rewards.max(1)
    replacement = rewards.argmax(1).astype(np.int16)
    # Equal-valued alternatives never displace the current action.
    replacement[old_values == best_values] = old[old_values == best_values]
    new_tables = tables.copy()
    new_tables[agent] = replacement
    before = int(score_units(demands, layouts, fixed_ids).sum())
    after = int(score_units(demands, layouts, resolve_actions(new_tables, observations)).sum())
    require(before == int(old_values.sum()) and after == int(best_values.sum()), "Best-response objective mismatch")
    require(after >= before, "Coordinate update decreased reward")
    require(Fraction(after, 2 * len(layouts)) <= UPPER, "Coordinate update exceeds upper bound")
    changes = int((old != replacement).sum())
    require((changes == 0) == (after == before), "Tie preservation or strict update rule failed")
    detail = {"agent": agent, "agent_name": "ABC"[agent], "before_success_units": before,
              "after_success_units": after, "strictly_improved": after > before,
              "changed_observation_rows": changes, "candidate_actions_per_observation": 17,
              "states_per_candidate": len(layouts), "old_action_kept_on_tie": True}
    return new_tables, detail


def prepare():
    require(not (OUT / "plan.json").exists() and not (OUT / "execution").exists(), "Already prepared or executed")
    require(len(OBSERVATIONS) == 144 and len(RANDOM_SEEDS) == 7, "Frozen dimensions differ")
    baseline = read(BASE / "fixed_pair_results.json")
    require(baseline["evaluations"]["I1_D1_public_L"]["score"]["reduced_fraction"] == "179/498", "Baseline source differs")
    bounds = read(BASE / "relaxation_001/execution/results.json")
    bound_rows = [r for r in bounds["results"] if r["condition"] == "I1_D1"]
    require(bounds["status"] == "completed" and len(bound_rows) == 3
            and all(Fraction(r["upper_bound_numerator"], r["upper_bound_denominator"]) == UPPER for r in bound_rows),
            "Missing certified I1 upper bound")
    for row in bound_rows:
        require(digest(BASE / "relaxation_001/execution" / row["certificate"]) == row["certificate_sha256"],
                "Upper bound certificate changed")
    initial = np.zeros((3, 144), dtype=np.int16)
    initial[0] = action_catalog(0).index((0, 0, 1))
    initial[1] = action_catalog(1).index((0, 0, 0))
    starts = [{"start_id": 0, "name": "fixed_AB_public_L", "rng_seed": None, "tables": initial.tolist()}]
    for i, seed in enumerate(RANDOM_SEEDS, 1):
        generator = np.random.Generator(np.random.PCG64(seed))
        starts.append({"start_id": i, "name": f"uniform_table_seed_{seed}", "rng_seed": seed,
                       "tables": generator.integers(0, 17, size=(3, 144), dtype=np.int16).tolist()})
    new_json(OUT / "initial_policies.json", {"starts": starts, "observation_rows": OBSERVATIONS,
                                            "action_catalogs": [action_catalog(a) for a in AGENTS]})
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    sources = {str(path): digest(path) for path in SOURCE_FILES}
    for path in SOURCE_FILES:
        target = snapshot / path.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (OUT / "coordinate_search_snapshot.py").write_bytes(Path(__file__).read_bytes())
    plan = {"status_at_freeze": "prepared_not_executed", "created_at": datetime.now().astimezone().isoformat(),
        "script_sha256": digest(__file__), "sources_sha256": sources,
        "initial_policies_sha256": digest(OUT / "initial_policies.json"), "prepare_runtime": runtime(),
        "condition": "I1_D1", "demand_support": "uniform_996_tables_at_least_two_compatible_pairs",
        "layout_support": "independent_uniform_24_permutations", "canonical_states": 23904,
        "fixed_private_assignment": [1, 2, 3], "represented_assignments": 6,
        "observation_schema": ["own_demand_id", "public_resource_id", "own_private_resource_id"],
        "observation_order": "own demand 0..11, public resource 0..3, own private resource 0..3 excluding public",
        "observation_rows_per_agent": 144, "actions_per_observation": 17,
        "action_order": "wait; then Cartesian site0..3,destinationL/R,other-agent indices ascending",
        "resource_order": ["short_wood", "long_wood", "short_fiber", "long_fiber"],
        "demand_order": "product(wood,fiber,short,long; L,R,either)",
        "starts": [{k: s[k] for k in ("start_id", "name", "rng_seed")} for s in starts],
        "random_table_generator": "NumPy PCG64(seed), independent uniform action0..16 for all3x144 rows; saved before optimization",
        "max_full_rounds_per_start": MAX_ROUNDS, "coordinate_order": [0, 1, 2],
        "max_coordinate_updates": 8 * MAX_ROUNDS * 3,
        "max_candidate_state_settlements": 8 * MAX_ROUNDS * 3 * 17 * 23904,
        "early_stop": "Only after a complete A/B/C round with no strict total reward increase",
        "tie_break": "Keep the old action whenever it attains the maximum; otherwise smallest maximizing action id",
        "objective": "Exact uniformly averaged team score over all23904 states, integer success-unit denominator47808",
        "reward": "D1 simultaneous reciprocal partner/same site/same destination, at most two active, each need scored separately",
        "upper_bound": {"fraction": "75/83", "max_success_units": 43200, "denominator": 47808},
        "saved_outputs": ["all8 initial policies", "every coordinate update", "every full round", "all8 final3x144 policies", "all8 exact scores and consistency checks"],
        "constraints": ["No full-state action pruning", "No mixed per-state policy selection", "No extra starts or rounds after seeing results", "No automatic retry or output overwrite"],
        "interpretation": "Researcher optimizes legal decision tables using the complete known distribution; this is a feasible lower-bound search, not exact unrestricted optimization or agents forming language by interaction.",
        "randomness_limit": "Initial random tables are fixed before states are evaluated. Neither shared/private randomization nor prior histories may carry current private state.",
        "prepare_payoff_evaluations": 0, "prepare_policy_coordinate_updates": 0,
        "new_model_calls": 0, "agent_training_updates": 0}
    new_json(OUT / "plan.json", plan)
    new_json(OUT / "freeze.json", {"plan_sha256": digest(OUT / "plan.json"),
                                    "status": "prepared_not_executed"})
    return {"status": "prepared_not_executed", "starts": len(starts), "max_rounds_per_start": MAX_ROUNDS,
            "payoff_evaluations": 0, "coordinate_updates": 0}


def verify_plan():
    plan, frozen = read(OUT / "plan.json"), read(OUT / "freeze.json")
    require(digest(OUT / "plan.json") == frozen["plan_sha256"], "Plan changed")
    require(digest(__file__) == plan["script_sha256"] == digest(OUT / "coordinate_search_snapshot.py"), "Script changed")
    require(digest(OUT / "initial_policies.json") == plan["initial_policies_sha256"], "Initial tables changed")
    for path, expected in plan["sources_sha256"].items():
        require(digest(path) == expected == digest(OUT / "source_snapshot" / Path(path).relative_to(WORK)), "Frozen source changed: " + path)
    require(plan["max_full_rounds_per_start"] == MAX_ROUNDS and plan["coordinate_order"] == list(AGENTS)
            and [r["rng_seed"] for r in plan["starts"]] == [None] + list(RANDOM_SEEDS), "Search budget differs")
    return plan


def execute():
    plan = verify_plan()
    execution = OUT / "execution"
    execution.mkdir(exist_ok=False)
    new_json(execution / "started.json", {"started_at": datetime.now().astimezone().isoformat(),
                                          "plan_sha256": digest(OUT / "plan.json"), "runtime": runtime()})
    finished = []
    try:
        demands, layouts, observations, counts = build_distribution()
        saved = read(OUT / "initial_policies.json")
        require(saved["observation_rows"] == [list(o) for o in OBSERVATIONS]
                and saved["action_catalogs"] == [[list(x) for x in action_catalog(a)] for a in AGENTS], "Input encoding differs")
        starts = saved["starts"]
        require(len(starts) == 8 and [{k: s[k] for k in ("start_id", "name", "rng_seed")} for s in starts] == plan["starts"], "Start list differs")
        with (execution / "coordinate_updates.jsonl").open("x", encoding="utf-8") as updates_log, \
             (execution / "rounds.jsonl").open("x", encoding="utf-8") as rounds_log:
            for start in starts:
                tables = np.asarray(start["tables"], dtype=np.int16)
                initial = summarize_policy(tables, demands, layouts, observations)
                if start["start_id"] == 0:
                    require(initial["score_fraction"] == "179/498", "Verified public-L starting score did not reproduce")
                before_round = initial["success_units"]
                trajectory = [{"round": 0, **initial}]
                rounds_log.write(json.dumps({"start_id": start["start_id"], **trajectory[0]}) + "\n")
                rounds_log.flush()
                stop_reason = "round_budget"
                for round_index in range(1, MAX_ROUNDS + 1):
                    for agent in AGENTS:
                        tables, detail = best_response(tables, agent, demands, layouts, observations)
                        updates_log.write(json.dumps({"start_id": start["start_id"], "round": round_index, **detail}) + "\n")
                        updates_log.flush()
                    measured = summarize_policy(tables, demands, layouts, observations)
                    require(measured["success_units"] >= before_round, "Full round decreased score")
                    trajectory.append({"round": round_index, **measured})
                    rounds_log.write(json.dumps({"start_id": start["start_id"], **trajectory[-1]}) + "\n")
                    rounds_log.flush()
                    if measured["success_units"] == before_round:
                        stop_reason = "no_strict_improvement_over_full_round"
                        break
                    before_round = measured["success_units"]
                policy_path = execution / f"start_{start['start_id']:02d}_final_policy.json"
                new_json(policy_path, {"start_id": start["start_id"], "observation_rows": OBSERVATIONS,
                                      "action_catalogs": [action_catalog(a) for a in AGENTS], "tables": tables.tolist()})
                entry = {"start_id": start["start_id"], "name": start["name"], "rng_seed": start["rng_seed"],
                    "initial": initial, "final": trajectory[-1], "full_rounds": len(trajectory) - 1,
                    "stop_reason": stop_reason, "trajectory": trajectory,
                    "final_policy_file": policy_path.name, "final_policy_sha256": digest(policy_path),
                    "coordinate_updates": 3 * (len(trajectory) - 1)}
                finished.append(entry)
                new_json(execution / f"start_{start['start_id']:02d}_result.json", entry)
                print(json.dumps({"start_id": start["start_id"], "rounds": entry["full_rounds"],
                                  "score": entry["final"]["score"], "stop_reason": stop_reason}), flush=True)
        require(len(finished) == 8, "Not all starts completed")
        best = max(finished, key=lambda e: e["final"]["success_units"])
        result = {"status": "completed", "finished_at": datetime.now().astimezone().isoformat(),
            "plan_sha256": digest(OUT / "plan.json"), "initial_policies_sha256": plan["initial_policies_sha256"],
            "starts": finished, "best_start_id": best["start_id"], "achievable_lower_bound": best["final"],
            "known_upper_bound": plan["upper_bound"], "state_count": len(layouts),
            "local_observation_counts": [c.tolist() for c in counts],
            "coordinate_updates": sum(e["coordinate_updates"] for e in finished),
            "source_logs_sha256": {name: digest(execution / name) for name in ("coordinate_updates.jsonl", "rounds.jsonl")},
            "exact_unrestricted_optimum_claimed": False, "researcher_policy_optimization": True,
            "new_model_calls": 0, "agent_training_updates": 0,
            "limits": ["Eight starts and coordinate maxima do not certify the global optimum.",
                "All state evaluation optimizes one fixed local table per agent, never selects joint actions from full state at deployment.",
                "No messages or learned language, no model competence, no evolutionary or ecological claim.",
                "Public site-assignment renaming represents six symmetric contexts, not six independent samples."]}
        new_json(execution / "results.json", result)
        new_json(execution / "status.json", {"status": "completed", "completed_starts": 8,
                                              "best_start_id": best["start_id"], "coordinate_updates": result["coordinate_updates"]})
        return {"status": "completed", "starts": 8, "best_start_id": best["start_id"], "score": best["final"]["score"]}
    except BaseException as error:
        new_json(execution / "failure.json", {"status": "failed", "error": str(error),
                                              "traceback": traceback.format_exc(), "completed_starts": len(finished), "automatic_retry": False})
        new_json(execution / "status.json", {"status": "failed", "completed_starts": len(finished)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    args = parser.parse_args()
    output = prepare() if args.command == "prepare" else verify_plan() if args.command == "verify" else execute()
    print(json.dumps(output if args.command != "verify" else {"status": "frozen_sources_match", "executed": (OUT / "execution").exists()}, ensure_ascii=False))
