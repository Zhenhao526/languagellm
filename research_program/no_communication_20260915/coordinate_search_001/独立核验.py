"""Read-only independent audit of completed finite I1-D1 policy search.

Does not import or call coordinate_search, rerun search starts, or update tables.
Independently scores each final policy and all final unilateral alternatives.
"""
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
RMASK = np.array([3, 12, 5, 10], dtype=np.int16)
DMASK = np.array([1, 2, 3], dtype=np.int16)


def check(test, message):
    if not test:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def compatible(a, b):
    return bool(RMASK[a // 3] & RMASK[b // 3]) and bool(DMASK[a % 3] & DMASK[b % 3])


def build():
    need_tables = [t for t in product(range(12), repeat=3)
                   if sum(compatible(t[i], t[j]) for i, j in combinations(range(3), 2)) >= 2]
    check(len(need_tables) == 996, "Demand support differs")
    demands = np.repeat(np.asarray(need_tables, dtype=np.int16), 24, axis=0)
    layouts = np.tile(np.asarray(list(permutations(range(4))), dtype=np.int16), (996, 1))
    # Algebraic observation encoding, independent of the search's dict lookup.
    public, private = layouts[:, 0:1], layouts[:, 1:]
    observations = demands * 12 + public * 3 + private - (private > public).astype(np.int16)
    schema = [(d, p, own) for d in range(12) for p in range(4) for own in range(4) if own != p]
    catalogs = [[(-1, -1, -1)] + list(product(range(4), range(2), [j for j in range(3) if j != i]))
                for i in range(3)]
    check(observations.shape == (23904, 3), "Observation grid differs")
    for i in range(3):
        check(set(observations[:, i]) == set(range(144)), "Missing observations")
        for row, (need, p, own) in enumerate(schema):
            selected = observations[:, i] == row
            check(np.all(demands[selected, i] == need) and np.all(layouts[selected, 0] == p)
                  and np.all(layouts[selected, i + 1] == own), "Observation encoding leaked or mixed states")
    return demands, layouts, observations, schema, np.asarray(catalogs, dtype=np.int16)


def deployed(tables, observations):
    tables = np.asarray(tables)
    check(tables.shape == (3, 144) and np.issubdtype(tables.dtype, np.integer)
          and ((tables >= 0) & (tables < 17)).all(), "Invalid decision table")
    return np.column_stack([tables[i][observations[:, i]] for i in range(3)])


def independent_score(demands, layouts, ids, catalogs):
    """Settle by unordered active pair, rather than the search's partner gather."""
    check(ids.shape == (len(layouts), 3), "Wrong action shape")
    actions = np.stack([catalogs[i][ids[:, i]] for i in range(3)], axis=1)
    units = np.zeros(len(layouts), dtype=np.int16)
    rows = np.arange(len(layouts))
    for i, j in combinations(range(3), 2):
        k = 3 - i - j
        ai, aj = actions[:, i], actions[:, j]
        paired = ((ids[:, i] != 0) & (ids[:, j] != 0) & (ids[:, k] == 0)
                  & (ai[:, 2] == j) & (aj[:, 2] == i)
                  & (ai[:, 0] == aj[:, 0]) & (ai[:, 1] == aj[:, 1]))
        material = layouts[rows, np.maximum(ai[:, 0], 0)]
        destination = np.maximum(ai[:, 1], 0)
        material_bit, destination_bit = np.left_shift(1, material), np.left_shift(1, destination)
        for agent in (i, j):
            correct = ((RMASK[demands[:, agent] // 3] & material_bit) != 0) & (
                (DMASK[demands[:, agent] % 3] & destination_bit) != 0)
            units += (paired & correct).astype(np.int16)
    check(((units >= 0) & (units <= 2)).all(), "Score is outside 0/1/2 success units")
    return units


def summary(units):
    n, denominator = int(units.sum()), 2 * len(units)
    check(n <= 43200 and denominator == 47808, "Legal policy exceeds the certified upper bound")
    return {"success_units": n, "score_denominator": denominator,
            "score_fraction": str(Fraction(n, denominator)), "score": n / denominator,
            "reward_histogram_success_units": {str(k): int(v) for k, v in sorted(Counter(units.tolist()).items())}}


def compare_summary(measured, recorded):
    for key, value in measured.items():
        check(recorded[key] == value, f"Score summary mismatch: {key}")


def final_response_certificate(tables, agent, demands, layouts, obs, catalogs):
    ids = deployed(tables, obs)
    values = np.empty((144, 17), dtype=np.int64)
    for candidate in range(17):
        alternative = ids.copy()
        alternative[:, agent] = candidate
        units = independent_score(demands, layouts, alternative, catalogs)
        # Counts are small integers, hence exactly represented in bincount's
        # float accumulator. Convert back only after checking integrality.
        totals = np.bincount(obs[:, agent], weights=units, minlength=144)
        check(np.array_equal(totals, totals.astype(np.int64)), "Nonintegral response totals")
        values[:, candidate] = totals.astype(np.int64)
    old = tables[agent]
    old_values = values[np.arange(144), old]
    maxima = values.max(axis=1)
    current = int(independent_score(demands, layouts, ids, catalogs).sum())
    check(int(old_values.sum()) == current, "Observation aggregation differs from deployed score")
    check(np.array_equal(old_values, maxima), "A final single-agent improvement still exists")
    tied = (values == maxima[:, None]).sum(axis=1)
    keep_on_tie = np.where(old_values == maxima, old, values.argmax(axis=1))
    check(np.array_equal(keep_on_tie, old), "Frozen tie rule would change a final policy")
    return {"agent": agent, "local_observations": 144, "actions_per_observation": 17,
            "exact_current_success_units": current, "exact_best_response_success_units": int(maxima.sum()),
            "improvement_success_units": 0, "observations_with_multiple_maxima": int((tied > 1).sum()),
            "old_is_maximizer_in_every_observation": True, "keep_old_on_tie_returns_identical_table": True,
            "candidate_state_settlements": 17 * len(layouts),
            "response_matrix_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "response_values_by_observation_action": values.tolist()}


def load_environment():
    path = WORK / "research_program/triadic_task/environment.py"
    name = "independent_triadic_environment"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def main():
    target = OUT / "独立核验.json"
    check(not target.exists(), "Audit output exists; preserve it instead of overwriting")
    plan, freeze = read(OUT / "plan.json"), read(OUT / "freeze.json")
    check(sha(OUT / "plan.json") == freeze["plan_sha256"], "Frozen plan changed")
    check(sha(OUT / "coordinate_search.py") == sha(OUT / "coordinate_search_snapshot.py")
          == plan["script_sha256"], "Search source changed")
    for path, expected in plan["sources_sha256"].items():
        check(sha(path) == expected == sha(OUT / "source_snapshot" / Path(path).relative_to(WORK)),
              "Frozen upstream source or snapshot changed")
    check(sha(OUT / "initial_policies.json") == plan["initial_policies_sha256"], "Initial policies changed")
    execution = OUT / "execution"
    result, status, started = read(execution / "results.json"), read(execution / "status.json"), read(execution / "started.json")
    check(result["status"] == status["status"] == "completed", "Search is incomplete")
    check(result["plan_sha256"] == started["plan_sha256"] == freeze["plan_sha256"], "Execution plan differs")
    check(result["initial_policies_sha256"] == plan["initial_policies_sha256"], "Executed initial policies differ")
    for filename, expected in result["source_logs_sha256"].items():
        check(sha(execution / filename) == expected, "Trajectory log SHA differs")
    saved = read(OUT / "initial_policies.json")
    demands, layouts, observations, schema, catalogs = build()
    check(saved["observation_rows"] == [list(x) for x in schema]
          and saved["action_catalogs"] == catalogs.tolist(), "Saved encoding differs")
    check([x["start_id"] for x in saved["starts"]] == list(range(8)), "Initial start set differs")
    check([x["start_id"] for x in result["starts"]] == list(range(8)), "Result start set differs")
    for initial, planned in zip(saved["starts"], plan["starts"]):
        check({k: initial[k] for k in ("start_id", "name", "rng_seed")} == planned, "Start metadata differs")
        if initial["rng_seed"] is not None:
            reproduced = np.random.Generator(np.random.PCG64(initial["rng_seed"])).integers(0, 17, (3, 144), dtype=np.int16)
            check(np.array_equal(reproduced, initial["tables"]), "Seed does not reproduce frozen initial table")
    baseline = np.zeros((3, 144), np.int16)
    baseline[0] = baseline[1] = 1
    check(np.array_equal(saved["starts"][0]["tables"], baseline), "Start 0 is not fixed AB/public/L")

    updates, rounds = read_lines(execution / "coordinate_updates.jsonl"), read_lines(execution / "rounds.jsonl")
    check(len(updates) == result["coordinate_updates"] == status["coordinate_updates"] == 135,
          "Update count differs")
    check(len(rounds) == 53, "Expected 45 completed rounds plus eight initial rows")
    expected_update_order, expected_round_order = [], []
    verified = []
    env, env_path = load_environment()
    env_catalogs = [env.all_actions(a) for a in env.AGENTS]
    for a in range(3):
        converted = [(-1, -1, -1) if action['kind'] == 'wait' else
                     (env.SITES.index(action['site']), env.DESTINATIONS.index(action['destination']),
                      env.AGENTS.index(action['partner'])) for action in env_catalogs[a]]
        check(converted == [tuple(row) for row in catalogs[a]], "Native action catalog differs")
    env_states = [env.State(tuple(map(int, d)), tuple(map(int, w))) for d, w in zip(demands, layouts)]
    exact_environment_states_checked = 0
    for initial, entry in zip(saved["starts"], result["starts"]):
        sid = entry["start_id"]
        check(all(initial[key] == entry[key] for key in ("start_id", "name", "rng_seed")),
              "Completed start metadata differs from frozen initial start")
        own_result_path = execution / f"start_{sid:02d}_result.json"
        check(read(own_result_path) == entry, "Individual result differs from consolidated result")
        check(entry["stop_reason"] == "no_strict_improvement_over_full_round"
              and 1 <= entry["full_rounds"] <= 20
              and entry["coordinate_updates"] == entry["full_rounds"] * 3, "Budget/stop metadata differs")
        policy_path = execution / entry["final_policy_file"]
        check(sha(policy_path) == entry["final_policy_sha256"], "Final policy SHA differs")
        policy = read(policy_path)
        check(policy["start_id"] == sid and policy["observation_rows"] == saved["observation_rows"]
              and policy["action_catalogs"] == saved["action_catalogs"], "Final policy encoding differs")
        tables = np.asarray(policy["tables"], dtype=np.int16)
        ids = deployed(tables, observations)
        units = independent_score(demands, layouts, ids, catalogs)
        measured = summary(units)
        compare_summary(measured, entry["final"])
        initial_units = independent_score(demands, layouts, deployed(initial["tables"], observations), catalogs)
        compare_summary(summary(initial_units), entry["initial"])
        for a in range(3):
            for obs in range(144):
                same = ids[observations[:, a] == obs, a]
                check(len(same) > 0 and np.all(same == tables[a, obs]), "Final policy violates local consistency")

        # Cross-check every final state through the separate native environment.
        for row, state in enumerate(env_states):
            actions = {env.AGENTS[a]: env_catalogs[a][int(ids[row, a])] for a in range(3)}
            settled = env.settle(state, actions, require_match=True)
            check(settled["satisfied_units"] == int(units[row]), "Native environment and independent scorer disagree")
            exact_environment_states_checked += 1

        observed_rounds = [r for r in rounds if r["start_id"] == sid]
        trajectory = entry["trajectory"]
        check(observed_rounds == [{"start_id": sid, **r} for r in trajectory], "Round log differs from embedded trajectory")
        check([r["round"] for r in trajectory] == list(range(entry["full_rounds"] + 1)), "Missing or duplicate rounds")
        compare_summary(summary(initial_units), trajectory[0])
        compare_summary(measured, trajectory[-1])
        for index, row in enumerate(trajectory):
            expected_round_order.append((sid, index))
            histogram = {int(k): v for k, v in row["reward_histogram_success_units"].items()}
            check(set(histogram) <= {0, 1, 2} and sum(histogram.values()) == 23904,
                  "Historical histogram denominator differs")
            check(sum(k * v for k, v in histogram.items()) == row["success_units"]
                  and 0 <= row["success_units"] <= 43200, "Historical score differs")
            check(row["score_denominator"] == 47808
                  and row["score_fraction"] == str(Fraction(row["success_units"], 47808))
                  and row["score"] == row["success_units"] / 47808, "Historical fraction differs")
        for round_id in range(1, entry["full_rounds"] + 1):
            trio = [u for u in updates if u["start_id"] == sid and u["round"] == round_id]
            check([u["agent"] for u in trio] == [0, 1, 2], "Coordinate order/coverage differs")
            previous = trajectory[round_id - 1]["success_units"]
            for u in trio:
                expected_update_order.append((sid, round_id, u["agent"]))
                check(u["before_success_units"] == previous and previous <= u["after_success_units"] <= 43200,
                      "Coordinate chain decreases or breaks")
                gain = u["after_success_units"] > previous
                check(u["strictly_improved"] == gain and (u["changed_observation_rows"] > 0) == gain
                      and 0 <= u["changed_observation_rows"] <= 144, "Change/gain log contract differs")
                check(u["old_action_kept_on_tie"] and u["candidate_actions_per_observation"] == 17
                      and u["states_per_candidate"] == 23904, "Historical update scope differs")
                previous = u["after_success_units"]
            check(previous == trajectory[round_id]["success_units"], "Round score differs from coordinate chain")
            if round_id == entry["full_rounds"]:
                check(previous == trajectory[round_id - 1]["success_units"]
                      and all(u["changed_observation_rows"] == 0 for u in trio), "Final round has a strict update")
            else:
                check(previous > trajectory[round_id - 1]["success_units"], "Search continued after a zero-gain full round")

        responses = [final_response_certificate(tables, a, demands, layouts, observations, catalogs) for a in range(3)]
        verified.append({"start_id": sid, "initial": summary(initial_units), "final": measured,
                         "full_rounds": entry["full_rounds"], "final_policy_sha256": sha(policy_path),
                         "individual_result_sha256": sha(own_result_path),
                         "native_environment_state_checks": 23904,
                         "final_best_responses": responses,
                         "single_agent_best_response_gain_zero": True})
        print(json.dumps({"verified_start": sid, "success_units": measured["success_units"],
                          "all_three_final_best_response_gains": [0, 0, 0]}, ensure_ascii=False), flush=True)

    check([(u["start_id"], u["round"], u["agent"]) for u in updates] == expected_update_order,
          "Global coordinate log order differs")
    check([(r["start_id"], r["round"]) for r in rounds] == expected_round_order,
          "Global round log order differs")
    check(sum(r["full_rounds"] for r in verified) == 45 and exact_environment_states_checked == 191232,
          "Completed-round or native validation counts differ")
    best = max(verified, key=lambda r: r["final"]["success_units"])
    check(best["start_id"] == result["best_start_id"] == status["best_start_id"] == 0
          and best["final"]["score_fraction"] == "179/498", "Best reported lower bound differs")
    compare_summary(best["final"], result["achievable_lower_bound"])
    check("torch" not in sys.modules, "Audit imported a model runtime")
    all_source_paths = [OUT / name for name in ("plan.json", "freeze.json", "initial_policies.json", "coordinate_search.py")]
    all_source_paths += [execution / name for name in ("results.json", "status.json", "started.json", "rounds.jsonl", "coordinate_updates.jsonl")]
    audit = {
        "status": "passed", "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "audit_script_sha256": sha(__file__), "numpy": np.__version__,
        "method": "Independent unordered-pair/bitmask settlement, cross-checked against native environment for all final states; no import/call of frozen search functions.",
        "source_files_sha256": {str(path): sha(path) for path in all_source_paths},
        "native_environment_path": str(env_path), "native_environment_sha256": sha(env_path),
        "starts_verified": 8, "initial_seed_tables_reproduced": 7, "canonical_states_per_policy": 23904,
        "native_environment_final_state_checks": exact_environment_states_checked,
        "local_observations_per_agent": 144, "all_final_policies_locally_executable": True,
        "historical_coordinate_log_records_checked": len(updates), "full_rounds_checked": 45,
        "round_rows_including_initials": len(rounds), "all_updates_report_nondecreasing_reward": True,
        "all_eight_stopped_after_zero_gain_full_round": True,
        "final_exact_single_agent_response_checks": 24,
        "final_response_candidate_state_settlements": 24 * 17 * 23904,
        "all_final_single_agent_best_response_gains_zero": True,
        "tie_audit_scope": "All 24 final response matrices exhaustively checked: every current action is maximizing and the frozen keep-old tie rule returns unchanged tables. Historical 135 log records satisfy flags/change/gain contracts, but intermediate action tables were not saved, so historical per-row tie handling is not independently reconstructed.",
        "starts": verified, "best_start_id": 0, "verified_existing_lower_bound": best["final"],
        "global_optimum_proved": False, "original_search_replayed": False,
        "new_search_starts": 0, "decision_tables_updated": 0, "model_calls": 0, "torch_imported": False,
        "limitations": ["Coordinate optimality is not global optimality or a certificate that coordinated multi-agent changes cannot improve payoff.",
                       "Validation of final counterfactual responses does not add starts or update policies.",
                       "The 135 historical updates are checked as a complete recorded chain, not re-executed from intermediate policies.",
                       "This is researcher optimization of finite local decision tables, not agent language formation or a model learning failure."]}
    with target.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "passed", "output": str(target), "updates_checked": 135,
                      "final_response_checks": 24, "new_search_starts": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
