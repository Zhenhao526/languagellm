"""Paired action-snapshot diagnostic; only execute loads the unchanged Qwen.

All36 CAP actions are replayed, then all36 receive a researcher-supplied plan.
No new history or partner action is fed into any probe. This is a post-hoc joint
intervention on plan information and instruction to follow it, not language
formation or an update to the original capability gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from . import environment as env
from . import qwen_capability as cap


WORK = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).resolve().parent / "results/20260915_CAP_01"
STUDY_PLAN = Path(__file__).resolve().parent / "共享计划执行诊断_方案.md"
ARMS = ("original_replay", "shared_plan_added")
MAX_CALLS = 144
REPRO_FIELDS = ("messages", "output", "prompt_sha256", "seed", "temperature", "mode",
                "valid", "fresh_cache", "native_thinking", "finish_reason", "prompt_tokens", "generation_tokens")


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_source_cap(source_run):
    source = Path(source_run).resolve()
    plan, prepared = cap.verify_prepared(source)
    audit = read(source / "独立核验.json")
    cap.require(audit["status"] == "passed_real_execution_audit" and audit["real_recorded_model_calls_checked"] == 216,
                "A completed independently audited CAP source is required")
    for name, digest in audit["source_artifact_sha256"].items():
        cap.require(cap.sha(name) == digest, "Audited CAP artifact changed: " + name)
    cap.require(read(source / "execution/status.json")["status"] == "completed", "CAP source incomplete")
    return plan, prepared, audit


def shared_plan_message(plan):
    text = ("实验控制：本次由研究者提供以下有效共同方案，三人现在均收到完全相同的方案。"
            "本次请按此共同方案，为你自己选择当前私人菜单中的编号，不改选其他方案。"
            "原观察、历史和六条广播均保留；原广播中的其他提议在本次只作为旧提议。"
            "以下仅列语义动作，不提供任何人的本地动作编号：\n")
    text += "\n".join(f"{agent}：{cap.action_text(plan[agent])}。" for agent in env.AGENTS)
    return {"role": "user", "content": text}


def build_cases(source_run=DEFAULT_SOURCE):
    source = Path(source_run).resolve()
    source_plan, cap_cases, audit = verify_source_cap(source)
    trials = lines(source / "execution/trials.jsonl")
    calls = lines(source / "execution/inference.jsonl")
    cap.require(len(trials) == 12 and len(calls) == 216, "Complete source grid required")
    index = {(c["label"]["group"], c["label"]["episode"], c["label"]["agent"], c["label"]["stage"]): c
             for c in calls if c["label"]["window"] is None}
    cap.require(len(index) == 72, "Expected exactly72 source action-stage records")
    cases, trial_plans = [], []
    for row in trials:
        state = env.State(**row["state"])
        plan = env.sufficient_information_witness(state)
        cap.require(env.settle(state, plan, require_match=True)["reward"] == 1, "Given plan is not full success")
        addition = shared_plan_message(plan)
        trial_plans.append({"source_trial_id": row["case_id"], "group": row["group"], "episode": row["episode"],
                            "scene_seed": row["scene_seed"], "state": deepcopy(row["state"]), "shared_plan": deepcopy(plan)})
        for agent in env.AGENTS:
            key = (row["group"], row["episode"], agent)
            analysis, formal = index[key + ("analysis",)], index[key + ("formal",)]
            base = cap.build_prompt(agent, row["observations"][agent], row["private_histories_before"][agent],
                                    row["messages"], menu=row["menus"][agent])
            cap.require(base == analysis["messages"][:-1], "Original action base did not reconstruct")
            cap.require(formal["messages"][:-2] == base and formal["messages"][-2]
                        == {"role": "assistant", "content": analysis["output"]}, "Formal premise is not same-decision analysis")
            seed = cap.stable_seed("generation", row["group"], row["episode"], 0, agent, "action")
            cap.require(analysis["seed"] == formal["seed"] == seed and analysis["temperature"] == formal["temperature"] == 0,
                        "Original generation parameters differ")
            cap.require(analysis["mode"] == "private_analysis" and formal["mode"] == "action"
                        and formal["call"] == analysis["call"] + 1, "Source stages not adjacent")
            menu = row["menus"][agent]
            cap.require([m["id"] for m in menu] == list(range(17)) and formal["output"] in tuple(map(str, range(17))),
                        "Original full private action menu differs")
            cases.append({"case_id": row["case_id"] + "_" + agent, "source_trial_id": row["case_id"],
                "group": row["group"], "episode": row["episode"], "agent": agent, "scene_seed": row["scene_seed"],
                "state": deepcopy(row["state"]), "menu": deepcopy(menu), "shared_plan": deepcopy(plan),
                "shared_plan_message": deepcopy(addition), "original_base_messages": deepcopy(base),
                "plan_base_messages": deepcopy(base) + [deepcopy(addition)], "seed": seed,
                "choices": list(range(17)), "temperature": 0, "expected_plan_action": deepcopy(plan[agent]),
                "source_call_ids": [analysis["call"], formal["call"]],
                "original_analysis": deepcopy(analysis), "original_formal": deepcopy(formal)})
    cap.require(len(cases) == 36 and len({c["seed"] for c in cases}) == 36, "Expected36 fixed decisions")
    return {"source_run": str(source), "source_cap_plan_sha256": cap.sha(source / "plan.json"),
            "source_cap_audit_sha256": cap.sha(source / "独立核验.json"), "cases": cases, "trial_plans": trial_plans}


def run_cases(backend, prepared, *, on_decision=lambda row: None, on_trial=lambda row: None):
    cap.require(backend.calls == 0, "Fresh backend/log required")
    decisions, trials = [], []
    for arm in ARMS:
        for start in range(0, 36, 3):
            bundle = prepared["cases"][start:start + 3]
            cap.require([c["agent"] for c in bundle] == list(env.AGENTS)
                        and len({c["source_trial_id"] for c in bundle}) == 1, "Incomplete source trial")
            selected_actions = {}
            for case in bundle:
                before = backend.calls
                cap.require(before + 2 <= MAX_CALLS, "Diagnostic cap reached")
                messages = case["original_base_messages"] if arm == "original_replay" else case["plan_base_messages"]
                label = {"phase": "triadic_execution_diagnostic_v1", "arm": arm,
                         **{k: case[k] for k in ("case_id", "source_trial_id", "group", "episode", "agent", "source_call_ids")}}
                answer = backend.decide(deepcopy(messages), mode="action", label=label, seed=case["seed"],
                                        choices=deepcopy(case["choices"]), temperature=0)
                cap.require(backend.calls == before + 2 and answer in tuple(map(str, case["choices"])),
                            "Malformed two-stage action decision")
                selected = next(m for m in case["menu"] if str(m["id"]) == answer)
                selected_actions[case["agent"]] = deepcopy(selected["action"])
                record = {"arm": arm, "case_id": case["case_id"], "source_trial_id": case["source_trial_id"],
                    "agent": case["agent"], "group": case["group"], "episode": case["episode"], "seed": case["seed"],
                    "source_call_ids": case["source_call_ids"], "call_range": [before + 1, backend.calls],
                    "output": answer, "selected": deepcopy(selected), "reference_plan_action": deepcopy(case["expected_plan_action"]),
                    "plan_was_shown": arm == "shared_plan_added", "matches_reference_plan": selected["action"] == case["expected_plan_action"]}
                decisions.append(record)
                on_decision(deepcopy(record))
            case = bundle[0]
            outcome = env.settle(env.State(**case["state"]), selected_actions, require_match=True)
            all_match = selected_actions == case["shared_plan"]
            trial = {"arm": arm, "source_trial_id": case["source_trial_id"], "group": case["group"],
                "episode": case["episode"], "actions": selected_actions, "outcome": outcome,
                "plan_was_shown": arm == "shared_plan_added", "all_actions_match_reference_plan": all_match,
                "full_success_via_different_plan": outcome["reward"] == 1 and not all_match}
            trials.append(trial)
            on_trial(deepcopy(trial))
    cap.require(backend.calls == 144 and len(decisions) == 72 and len(trials) == 24, "Incomplete fixed diagnostic budget")
    summaries = []
    for arm in ARMS:
        ds = [d for d in decisions if d["arm"] == arm]
        ts = [t for t in trials if t["arm"] == arm]
        summaries.append({"arm": arm, "decisions": 36, "calls": 72, "trials": 12,
            "rewards": [t["outcome"]["reward"] for t in ts], "full_success_trials": sum(t["outcome"]["full_success"] for t in ts),
            "individual_actions_matching_reference_plan": sum(d["matches_reference_plan"] for d in ds),
            "joint_actions_matching_reference_plan": sum(t["all_actions_match_reference_plan"] for t in ts),
            "full_success_via_different_plan": sum(t["full_success_via_different_plan"] for t in ts),
            "reference_plan_was_shown": arm == "shared_plan_added"})
    return {"decisions": decisions, "trials": trials, "arms": summaries, "model_calls": backend.calls}


def reproduction_report(actual_calls, prepared):
    cap.require(len(actual_calls) == 144 and [r["call"] for r in actual_calls] == list(range(1, 145)), "Incomplete inference log")
    checks = []
    for i, case in enumerate(prepared["cases"]):
        for j, source_name in enumerate(("original_analysis", "original_formal")):
            actual, source = actual_calls[2 * i + j], case[source_name]
            cap.require(actual["label"]["arm"] == "original_replay" and actual["label"]["case_id"] == case["case_id"], "Replay log order differs")
            checks.append({"new_call": actual["call"], "source_call": source["call"], "case_id": case["case_id"],
                           "equal": {field: actual[field] == source[field] for field in REPRO_FIELDS}})
    exact = all(all(c["equal"].values()) for c in checks)
    return {"source_calls": 72, "checks": checks, "complete_exact_reproduction": exact,
            "effect_interpretable": exact, "effect_status": "paired_posthoc_diagnostic" if exact else "effect_uninterpretable",
            "rule": "All144 calls still run after replay output differences; true generation/program exceptions terminate without retry."}


def plan_distribution(prepared):
    rows = prepared["trial_plans"]
    unique = {r["scene_seed"]: r for r in rows}
    def count(selected):
        return {"trials": len(selected),
            "active_pairs": dict(Counter("".join(a for a in env.AGENTS if r["shared_plan"][a]["kind"] == "transport") for r in selected)),
            "waiting_agents": dict(Counter(a for r in selected for a in env.AGENTS if r["shared_plan"][a]["kind"] == "wait"))}
    return {"context_trials": count(rows), "unique_semantic_draws": count(list(unique.values())),
            "selection": "Unmodified environment first sufficient_information_witness; not selected by model results."}


def prepare(output_dir, source_run=DEFAULT_SOURCE):
    output, source = Path(output_dir).resolve(), Path(source_run).resolve()
    cap.require(not output.exists(), "Never overwrite a diagnostic pack")
    prepared = build_cases(source)
    source_plan, _, audit = verify_source_cap(source)
    source_hashes = dict(audit["source_artifact_sha256"])
    for path in (Path(__file__).resolve(), STUDY_PLAN, source / "独立核验.json",
                 Path(__file__).resolve().parent / "audit_qwen_capability_results.py"):
        source_hashes[str(path)] = cap.sha(path)
    output.mkdir(parents=True, exist_ok=False)
    # The full model/source chain remains in the immutable CAP pack, not copied again.
    for path in (Path(__file__).resolve(), STUDY_PLAN):
        (output / path.name).write_bytes(path.read_bytes())
    cap.new_json(output / "prepared_cases.json", prepared)
    plan = {"status_at_freeze": "prepared_not_executed", "created_at": datetime.now(timezone.utc).isoformat(), "source_run": str(source),
        "sources_sha256": source_hashes, "prepared_cases_sha256": cap.sha(output / "prepared_cases.json"),
        "model": source_plan["model"], "model_revision": source_plan["model_revision"], "prepare_runtime": cap.runtime_versions(),
        "source_cap_plan_sha256": cap.sha(source / "plan.json"), "arms": list(ARMS), "source_decisions": 36,
        "calls_per_arm": 72, "max_backend_calls": MAX_CALLS, "all_source_actions_included": True,
        "unique_decision_seeds": 36, "seed_policy": "Reuse each source seed in both arms and both analysis/formal stages; all temperatures0",
        "case_order": "Source CAP group/episode/A-B-C order; finish all original_replay before all shared_plan_added",
        "intervention": "Append exactly one researcher user message giving the same valid semantic plan to all three and instructing adherence. Preserve every original base message, all6 broadcasts, history, observation and private17-action menu; no correct local ID in addition.",
        "new_probe_history": False, "probe_actions_visible_to_partners": False,
        "reproduction_requirement": "Source72 action-stage messages, outputs and rendered SHA must all exactly reproduce; differences do not stop144 but mark effect_uninterpretable",
        "plan_distribution": plan_distribution(prepared), "measurements": ["exact individual reference-plan match", "exact joint reference-plan match", "actual joint reward", "different valid full-success solution"],
        "scope": "Posthoc researcher-plan plus adherence-instruction joint intervention;12 snapshots from4 semantic draws x3 contexts, not new independent worlds or all-role capability",
        "modifies_original_CAP_gate": False, "modifies_original_v3_gate": False, "starts_symbolic_experiment": False,
        "weight_provenance": "Verify and reference CAP frozen metadata/source chain and weight byte sizes; no new full weight hash or copy",
        "prepare_model_calls": 0}
    cap.new_json(output / "plan.json", plan)
    cap.new_json(output / "freeze.json", {"plan_sha256": cap.sha(output / "plan.json"), "status": "prepared_not_executed"})
    return {"status": "prepared_not_executed", "cases": 36, "calls_planned": 144, "plan_distribution": plan["plan_distribution"]}


def verify(output_dir):
    output = Path(output_dir).resolve()
    plan, frozen = read(output / "plan.json"), read(output / "freeze.json")
    cap.require(cap.sha(output / "plan.json") == frozen["plan_sha256"], "Plan changed")
    for filename, digest in plan["sources_sha256"].items():
        cap.require(cap.sha(filename) == digest, "Frozen source changed: " + filename)
    for path in (Path(__file__).resolve(), STUDY_PLAN):
        cap.require(cap.sha(output / path.name) == plan["sources_sha256"][str(path)], "New source snapshot changed")
    cap.require(cap.sha(output / "prepared_cases.json") == plan["prepared_cases_sha256"], "Prepared cases changed")
    prepared = read(output / "prepared_cases.json")
    cap.require(build_cases(plan["source_run"]) == prepared and plan["arms"] == list(ARMS)
                and plan["max_backend_calls"] == 144, "Source reconstruction or budget differs")
    return plan, prepared


def execute(output_dir):
    output = Path(output_dir).resolve()
    plan, prepared = verify(output)
    execution = output / "execution"
    execution.mkdir(exist_ok=False)
    started_clock = time.perf_counter()
    cap.new_json(execution / "started.json", {"started_at": datetime.now(timezone.utc).isoformat(), "runtime": cap.runtime_versions(), "plan_sha256": cap.sha(output / "plan.json")})
    backend = None
    try:
        from qwen_language_v3.backend import Backend
        backend = Backend(Path(plan["model"]), execution / "inference.jsonl")
        with (execution / "decisions.jsonl").open("x", encoding="utf-8") as decisions_log, \
             (execution / "trials.jsonl").open("x", encoding="utf-8") as trials_log:
            def save_decision(row):
                decisions_log.write(json.dumps(row, ensure_ascii=False) + "\n"); decisions_log.flush()
            def save_trial(row):
                trials_log.write(json.dumps(row, ensure_ascii=False) + "\n"); trials_log.flush()
                print(json.dumps({"arm": row["arm"], "trial": row["source_trial_id"], "reward": row["outcome"]["reward"],
                                  "all_actions_match_reference_plan": row["all_actions_match_reference_plan"], "calls": backend.calls}), flush=True)
            result = run_cases(backend, prepared, on_decision=save_decision, on_trial=save_trial)
        reproduction = reproduction_report(lines(execution / "inference.jsonl"), prepared)
        cap.new_json(execution / "reproduction.json", reproduction)
        cap.new_json(execution / "results.json", {"status": "completed", "arms": result["arms"], "calls": backend.calls,
            "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started_clock,
            "reproduction": reproduction, "plan_distribution": plan["plan_distribution"], "runtime": cap.runtime_versions(),
            "plan_sha256": cap.sha(output / "plan.json"), "modifies_old_gates": False, "symbolic_started": False})
        cap.new_json(execution / "status.json", {"status": "completed", "calls": backend.calls,
                                                 "effect_status": reproduction["effect_status"]})
        return {"status": "completed", "arms": result["arms"], "effect_status": reproduction["effect_status"]}
    except BaseException as error:
        cap.new_json(execution / "failure.json", {"error": str(error), "traceback": traceback.format_exc(), "automatic_retry": False})
        cap.new_json(execution / "status.json", {"status": "failed", "calls": backend.calls if backend else 0})
        raise
    finally:
        if backend is not None:
            cap.new_json(execution / "backend_stats.json", backend.stats())
            backend.log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    result = prepare(args.out, args.source) if args.command == "prepare" else execute(args.out) if args.command == "execute" else {"status": "verified", "cases": len(verify(args.out)[1]["cases"])}
    print(json.dumps(result, ensure_ascii=False))
