"""New full-information capability batch, shared dialogue and paired action outputs.

Only execute loads Qwen. Preparation reads the local tokenizer, source metadata,
and old receipts; no old gate/history or result is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from random import Random
import time
import traceback

from research_program.triadic_task import environment as env
from research_program.triadic_task import qwen_capability as cap
from research_program.triadic_task.qwen_execution_diagnostic import verify_source_cap
from .interface_backend import ANALYSIS_REQUEST, FORMAL_INSTRUCTIONS, allowed_next


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PLAN = ROOT / "plan.md"
ROUTE = "triadic_action_interface_v1"
GROUPS = (401, 402, 403)
SCENE_SEEDS = (20260917011, 20260917012, 20260917013, 20260917014)
ARMS = ("number", "semantic")
MAX_CALLS = 288
DEFAULT_SOURCE = WORK / "research_program/triadic_task/results/20260915_CAP_01"
READ = lambda path: json.loads(Path(path).read_text())
RULES = cap.RULES.replace(
    "每轮重新抽取需求和布局；过去的物资、需求和编号不能代替本轮观察。",
    "每次检查都从空白个人历史开始，需求和布局重新抽取。"
).replace(
    "编号分别随机打乱，只供本人操作，不能直接照抄伙伴的动作编号。",
    "编号分别随机打乱，只供本人查看，不能直接照抄伙伴的动作编号。正式选择格式在行动阶段单独说明。"
)


def seed(*parts):
    return cap.stable_seed(ROUTE, *parts)


def build_prompt(agent, observation, transcript, *, window=None, menu=None):
    cap.require(observation["self"] == agent and observation.get("information_control") == "full_information",
                "This batch requires the recipient's explicit full observation")
    data = {"你自己的历史": [], "本轮完整信息观察": cap.literal_observation(observation),
            "本轮已公开广播": deepcopy(transcript)}
    if menu is None:
        cap.require(window in (1, 2), "Exactly two synchronous windows")
        data["当前决策"] = {"广播窗口": window, "说明": "世界尚未结算；可以提出方案或回应已公开消息。"}
    else:
        cap.require(window is None and len(menu) == 17, "All17 actions required")
        data["本次可选动作"] = [{"编号": m["id"], "动作": cap.action_text(m["action"])} for m in menu]
        data["当前决策"] = "两个广播窗口已结束。独立选择一个动作；其他人的未执行动作不可见。"
    return [{"role": "system", "content": f"你是合作群体中的{agent}，伙伴是另外两位A/B/C。\n" + RULES},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]


def build_cases():
    support = env.support()
    cap.require(len(support) == 996, "Demand support changed")
    scenarios = [{"episode": i, "scene_seed": s, "state": asdict(env.draw_state(Random(s), support))}
                 for i, s in enumerate(SCENE_SEEDS, 1)]
    trials = []
    for group in GROUPS:
        for scene in scenarios:
            state = env.State(**scene["state"])
            # Solvability only: no witness, expected actor, or good-action list is saved in model inputs.
            cap.require(env.settle(state, env.sufficient_information_witness(state), require_match=True)["full_success"],
                        "A sampled support state is not solvable; do not replace it")
            menus, menu_seeds = {}, {}
            for agent in env.AGENTS:
                menu_seeds[agent] = seed("menu", group, scene["episode"], scene["scene_seed"], agent)
                order = list(range(17)); Random(menu_seeds[agent]).shuffle(order)
                menus[agent] = env.action_menu(agent, order)
                cap.require({json.dumps(x["action"], sort_keys=True) for x in menus[agent]}
                            == {json.dumps(x, sort_keys=True) for x in env.all_actions(agent)}, "A legal action was filtered")
            trials.append({"case_id": f"g{group}_e{scene['episode']}", "group": group, **deepcopy(scene),
                "observations": {a: env.observe(state, a, shared_needs=True, full_information=True) for a in env.AGENTS},
                "menus": menus, "menu_seeds": menu_seeds})
    return json.loads(json.dumps({"scenarios": scenarios, "trials": trials}))


def choice_strings(menu, arm):
    values = [str(m["id"]) if arm == "number" else cap.action_text(m["action"]) for m in menu]
    cap.require(arm in ARMS and len(values) == len(set(values)) == 17, "All17 unique outputs required")
    return values


def build_tokenization(tokenizer, prepared, *, eos_token_ids=None):
    eos = sorted(set(eos_token_ids if eos_token_ids is not None else [tokenizer.eos_token_id]))
    cap.require(eos and all(isinstance(x, int) for x in eos), "Missing integer EOS")
    digits = {c: tokenizer.encode(c, add_special_tokens=False) for c in "0123456789"}
    cap.require(all(len(v) == 1 and tokenizer.decode(v) == k for k, v in digits.items()), "Original numeric tokens changed")
    numbers = {str(i): [digits[c][0] for c in str(i)] for i in range(17)}
    texts = sorted({cap.action_text(m["action"]) for case in prepared["trials"] for menu in case["menus"].values() for m in menu})
    semantic = {text: tokenizer.encode(text, add_special_tokens=False) for text in texts}
    cap.require(all(tokenizer.decode(v) == k for k, v in {**numbers, **semantic}.items()), "Candidate does not round-trip")
    cap.require(len({tuple(v) for v in semantic.values()}) == len(semantic), "Semantic encodings collided")
    for case in prepared["trials"]:
        for agent in env.AGENTS:
            for arm, mapping in (("number", numbers), ("semantic", semantic)):
                seqs = [mapping[c] for c in choice_strings(case["menus"][agent], arm)]
                for seq in seqs:
                    for n, token in enumerate(seq):
                        cap.require(token in allowed_next(seq[:n], seqs, eos), "A legal output is unreachable")
                    cap.require(set(eos) <= set(allowed_next(seq, seqs, eos)), "Complete output cannot end")
    maximum = max(map(len, list(numbers.values()) + list(semantic.values()))) + 1
    return {"eos_token_ids": eos, "number_sequences": numbers, "semantic_sequences": semantic,
            "formal_max_tokens": maximum, "number_encoding": "original per-digit token path",
            "semantic_encoding": "one frozen canonical tokenizer token sequence per complete action text",
            "all_17_choices_prefix_reachable_for_every_menu": True}


def load_tokenizer_only(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)


def run_cases(backend, prepared, *, on_snapshot=lambda row: None, on_decision=lambda row: None,
              on_trial=lambda row: None):
    cap.require(backend.calls == 0 and prepared["trials"] == build_cases()["trials"], "Fresh backend and complete fixed batch required")
    snapshots, decisions, trials = [], [], []
    for case in prepared["trials"]:
        transcript = []
        for window in (1, 2):
            prompts = {a: build_prompt(a, case["observations"][a], transcript, window=window) for a in env.AGENTS}
            pending = []
            for agent in env.AGENTS:
                before = backend.calls
                label = {"phase": ROUTE, "arm": "common", "case_id": case["case_id"], "group": case["group"],
                         "episode": case["episode"], "agent": agent, "window": window}
                decision_seed = seed("generation", case["group"], case["episode"], window, agent, "natural_message")
                answer = backend.decide(deepcopy(prompts[agent]), mode="natural_message", label=label,
                                        seed=decision_seed, temperature=.7)
                cap.require(backend.calls == before + 2 and backend.calls <= MAX_CALLS, "Two-stage call count changed")
                pending.append({"agent": agent, "window": window, "text": answer})
                record = {**label, "seed": decision_seed, "output": answer, "call_range": [before + 1, backend.calls]}
                decisions.append(record); on_decision(deepcopy(record))
            transcript.extend(pending)
        snapshot = {**deepcopy(case), "private_histories_before": {a: [] for a in env.AGENTS}, "messages": deepcopy(transcript),
                    "action_prompts": {a: build_prompt(a, case["observations"][a], transcript, menu=case["menus"][a]) for a in env.AGENTS}}
        snapshots.append(snapshot); on_snapshot(deepcopy(snapshot))
    cap.require(backend.calls == 144, "Common dialogue budget differs")
    for arm in ARMS:
        for snapshot in snapshots:
            actions, selections = {}, {}
            for agent in env.AGENTS:
                before = backend.calls
                label = {"phase": ROUTE, "arm": arm, "case_id": snapshot["case_id"], "group": snapshot["group"],
                         "episode": snapshot["episode"], "agent": agent, "window": None}
                decision_seed = seed("generation", snapshot["group"], snapshot["episode"], 0, agent, "action")
                choices = choice_strings(snapshot["menus"][agent], arm)
                answer = backend.decide_action(deepcopy(snapshot["action_prompts"][agent]), arm=arm,
                    choices=deepcopy(choices), label=label, seed=decision_seed)
                cap.require(backend.calls == before + 2 and backend.calls <= MAX_CALLS and answer in choices,
                            "Invalid two-stage formal selection; do not retry")
                selected = deepcopy(snapshot["menus"][agent][choices.index(answer)])
                actions[agent], selections[agent] = selected["action"], selected
                record = {**label, "seed": decision_seed, "choices": choices, "output": answer, "selected": selected,
                          "call_range": [before + 1, backend.calls]}
                decisions.append(record); on_decision(deepcopy(record))
            outcome = env.settle(env.State(**snapshot["state"]), actions, require_match=True)
            row = {"arm": arm, "case_id": snapshot["case_id"], "group": snapshot["group"], "episode": snapshot["episode"],
                   "scene_seed": snapshot["scene_seed"], "state": deepcopy(snapshot["state"]),
                   "actions": deepcopy(actions), "selections": deepcopy(selections), "outcome": outcome}
            trials.append(row); on_trial(deepcopy(row))
    cap.require(backend.calls == MAX_CALLS and len(decisions) == 144 and len(trials) == 24, "Incomplete fixed budget")
    summaries = [{"arm": arm, "trials": 12, "action_calls": 72, "common_calls_reused": 144,
                  "full_success_trials": sum(t["outcome"]["full_success"] for t in trials if t["arm"] == arm),
                  "rewards": [t["outcome"]["reward"] for t in trials if t["arm"] == arm],
                  "by_group": [{"group": group, "rewards": [t["outcome"]["reward"] for t in trials if t["arm"] == arm and t["group"] == group]} for group in GROUPS]}
                 for arm in ARMS]
    pairs = [{"case_id": snapshots[i]["case_id"], "number_reward": trials[i]["outcome"]["reward"],
              "semantic_reward": trials[i+12]["outcome"]["reward"],
              "semantic_minus_number": trials[i+12]["outcome"]["reward"] - trials[i]["outcome"]["reward"],
              "changed_agents": [a for a in env.AGENTS if trials[i]["actions"][a] != trials[i+12]["actions"][a]]} for i in range(12)]
    return {"snapshots": snapshots, "decisions": decisions, "trials": trials, "arms": summaries, "pairs": pairs, "calls": backend.calls}


def analysis_reproduction(calls):
    cap.require(len(calls) == MAX_CALLS and [x["call"] for x in calls] == list(range(1, MAX_CALLS+1)), "Complete ordered inference log required")
    fields = ("messages", "output", "prompt_sha256", "seed", "temperature", "mode", "valid", "fresh_cache",
              "native_thinking", "finish_reason", "prompt_tokens", "generation_tokens")
    by_key = {(x["label"]["arm"], x["label"]["case_id"], x["label"]["agent"]): x for x in calls
              if x["label"]["arm"] in ARMS and x["label"]["stage"] == "analysis"}
    cap.require(len(by_key) == 72, "Expected36 matched private analyses per arm")
    rows = []
    for case in build_cases()["trials"]:
        for agent in env.AGENTS:
            a, b = [by_key[(arm, case["case_id"], agent)] for arm in ARMS]
            rows.append({"case_id": case["case_id"], "agent": agent, "calls": [a["call"], b["call"]],
                         "equal": {f: a[f] == b[f] for f in fields}})
    return {"comparisons": rows, "exact": all(all(r["equal"].values()) for r in rows),
            "scope": "Only these36 private analyses; formal prompts/grammars intentionally differ. No retry on mismatch."}


def old_state_paths(source):
    return [source / "execution/histories_after.json", source / "execution/results.json", source / "execution/status.json",
            WORK / "research_program/triadic_task/results/20260915_EXEC_01/execution/results.json",
            WORK / "qwen_language_v3/results/20260915_152343/capability_gates.json"]


def prepare(output_dir, source_run=DEFAULT_SOURCE):
    output, source = Path(output_dir).resolve(), Path(source_run).resolve()
    cap.require(not output.exists(), "Refuse to overwrite prepared or executed output")
    source_plan, _, audit = verify_source_cap(source)
    prepared = build_cases()
    tokenizer = load_tokenizer_only(source_plan["model"])
    config = READ(Path(source_plan["model"]) / "config.json")
    eos = config.get("eos_token_id", tokenizer.eos_token_id)
    prepared["tokenization"] = build_tokenization(tokenizer, prepared, eos_token_ids=eos if isinstance(eos, list) else [eos])
    local_sources = [Path(__file__).resolve(), ROOT / "interface_backend.py", PLAN, ROOT / "tests/test_action_interface.py",
                     Path(env.__file__).resolve(), Path(cap.__file__).resolve(),
                     WORK / "research_program/triadic_task/qwen_execution_diagnostic.py", cap.BACKEND]
    sources = {str(p): cap.sha(p) for p in local_sources}
    prior = {str(p): cap.sha(p) for p in old_state_paths(source)}
    output.mkdir(parents=True, exist_ok=False)
    for filename in sources:
        path = Path(filename); target = output / "source_snapshot" / path.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    cap.new_json(output / "prepared_cases.json", prepared)
    cap.new_json(output / "prior_state_receipt.json", {"files_sha256": prior, "recorded_at": datetime.now(timezone.utc).isoformat()})
    plan = {"status_at_freeze": "prepared_not_executed", "created_at": datetime.now(timezone.utc).isoformat(),
        "route": ROUTE, "source_run": str(source), "source_cap_plan_sha256": cap.sha(source / "plan.json"),
        "source_cap_audit_sha256": cap.sha(source / "独立核验.json"), "sources_sha256": sources,
        "source_artifact_sha256": audit["source_artifact_sha256"], "prior_state_sha256": prior,
        "prior_state_receipt_sha256": cap.sha(output / "prior_state_receipt.json"),
        "prepared_cases_sha256": cap.sha(output / "prepared_cases.json"), "model": source_plan["model"],
        "model_revision": source_plan["model_revision"], "prepare_runtime": cap.runtime_versions(),
        "groups": list(GROUPS), "scene_seeds": list(SCENE_SEEDS), "arms": list(ARMS), "trials_per_arm": 12,
        "max_calls": MAX_CALLS, "common_broadcast_calls": 144, "action_calls_per_arm": 72,
        "formal_max_tokens": prepared["tokenization"]["formal_max_tokens"], "private_analysis_max_tokens": 256,
        "natural_message_max_tokens": 96, "private_analysis_request": ANALYSIS_REQUEST,
        "formal_instructions": FORMAL_INSTRUCTIONS, "history_mode": "empty_each_trial_no_feedback_to_future",
        "execution_order": "All12 common two-window dialogues, then all number actions, then all semantic actions; fixed A/B/C order with synchronous barriers",
        "analysis_policy": "Repeated same input, seed and temperature0 in each arm; exact36-pair postcheck. No shared cache or new memory.",
        "semantic_gate": "12/12 full_success plus exact analysis replay and integrity checks; never updates old gates or starts symbols",
        "scope": "Whole formal output interface including tokenization/length/greedy branching, not a pure index-translation cause;4 semantic draws x3 generation repetitions",
        "model_calls_at_prepare": 0, "new_weight_loads_at_prepare": 0, "tokenizer_only_at_prepare": True,
        "weight_provenance": "Bound audited CAP metadata and download receipt; sizes checked by source verifier, no full weight rehash here"}
    cap.new_json(output / "plan.json", plan)
    cap.new_json(output / "freeze.json", {"plan_sha256": cap.sha(output / "plan.json"), "status": "prepared_not_executed"})
    return {"status": "prepared_not_executed", "calls_planned": MAX_CALLS,
            "formal_max_tokens": plan["formal_max_tokens"], "trials_per_arm": 12}


def verify(output_dir):
    output = Path(output_dir).resolve()
    plan, frozen = READ(output / "plan.json"), READ(output / "freeze.json")
    cap.require(cap.sha(output / "plan.json") == frozen["plan_sha256"], "Plan changed")
    for name, digest in plan["sources_sha256"].items():
        cap.require(cap.sha(name) == digest == cap.sha(output / "source_snapshot" / Path(name).relative_to(WORK)), "Source/snapshot changed: " + name)
    for name, digest in {**plan["source_artifact_sha256"], **plan["prior_state_sha256"]}.items():
        cap.require(cap.sha(name) == digest, "Old source/gate/history changed: " + name)
    cap.require(cap.sha(output / "prior_state_receipt.json") == plan["prior_state_receipt_sha256"], "Prior receipt changed")
    cap.require(cap.sha(output / "prepared_cases.json") == plan["prepared_cases_sha256"], "Prepared cases changed")
    source_plan, _, _ = verify_source_cap(plan["source_run"])
    cap.require(cap.sha(Path(plan["source_run"]) / "plan.json") == plan["source_cap_plan_sha256"]
                and cap.sha(Path(plan["source_run"]) / "独立核验.json") == plan["source_cap_audit_sha256"], "Prior anchor changed")
    prepared = READ(output / "prepared_cases.json")
    base = build_cases()
    cap.require({k: v for k, v in prepared.items() if k != "tokenization"} == base, "Fixed IID cases differ")
    tokenizer = load_tokenizer_only(plan["model"])
    config = READ(Path(plan["model"]) / "config.json"); eos = config.get("eos_token_id", tokenizer.eos_token_id)
    rebuilt = build_tokenization(tokenizer, base, eos_token_ids=eos if isinstance(eos, list) else [eos])
    cap.require(rebuilt == prepared["tokenization"] and rebuilt["formal_max_tokens"] == plan["formal_max_tokens"], "Frozen tokenization/budget changed")
    cap.require(plan["max_calls"] == MAX_CALLS and plan["arms"] == list(ARMS) and plan["scene_seeds"] == list(SCENE_SEEDS)
                and plan["model"] == source_plan["model"] and plan["model_revision"] == source_plan["model_revision"], "Identity/budget differs")
    return plan, prepared


def execute(output_dir):
    output = Path(output_dir).resolve()
    plan, prepared = verify(output)
    execution = output / "execution"; execution.mkdir(exist_ok=False)
    started = time.perf_counter(); backend = None
    cap.new_json(execution / "started.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                 "runtime": cap.runtime_versions(), "plan_sha256": cap.sha(output / "plan.json")})
    try:
        from .interface_backend import InterfaceBackend
        backend = InterfaceBackend(Path(plan["model"]), execution / "inference.jsonl",
            formal_max_tokens=plan["formal_max_tokens"], tokenization=prepared["tokenization"])
        streams = {}
        try:
            for name in ("snapshots", "decisions", "trials"):
                streams[name] = (execution / (name + ".jsonl")).open("x", encoding="utf-8")
            def save(name, row):
                streams[name].write(json.dumps(row, ensure_ascii=False) + "\n"); streams[name].flush()
                if name == "trials":
                    print(json.dumps({"arm": row["arm"], "case_id": row["case_id"], "reward": row["outcome"]["reward"], "calls": backend.calls}), flush=True)
            result = run_cases(backend, prepared, on_snapshot=lambda r: save("snapshots", r),
                on_decision=lambda r: save("decisions", r), on_trial=lambda r: save("trials", r))
        finally:
            for stream in streams.values(): stream.close()
        calls = [json.loads(x) for x in (execution / "inference.jsonl").read_text().splitlines() if x.strip()]
        reproduction = analysis_reproduction(calls)
        cap.new_json(execution / "analysis_reproduction.json", reproduction)
        verify(output)  # Check all frozen inputs/sources/tokenization again, not just old gates.
        unchanged = {name: cap.sha(name) == digest for name, digest in plan["prior_state_sha256"].items()}
        cap.require(all(unchanged.values()), "Old gate/history bytes changed during execution")
        gate = {"configuration": ROUTE, "required_full_success": 12,
            "observed_full_success": result["arms"][1]["full_success_trials"], "analysis_exact": reproduction["exact"],
            "passed": result["arms"][1]["full_success_trials"] == 12 and reproduction["exact"],
            "old_gates_modified": False, "symbolic_stage_started": False}
        report = {"status": "completed", "calls": backend.calls, "arms": result["arms"], "pairs": result["pairs"],
            "gate": gate, "prior_state_unchanged": unchanged, "analysis_reproduction_exact": reproduction["exact"],
            "effect_interpretable": reproduction["exact"], "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter()-started, "runtime": cap.runtime_versions(),
            "plan_sha256": cap.sha(output / "plan.json"), "scope": plan["scope"]}
        cap.new_json(execution / "results.json", report)
        cap.new_json(execution / "status.json", {"status": "completed", "calls": backend.calls, "gate_passed": gate["passed"]})
        return report
    except BaseException as error:
        cap.new_json(execution / "failure.json", {"error": str(error), "traceback": traceback.format_exc(), "automatic_retry": False})
        cap.new_json(execution / "status.json", {"status": "failed", "calls": backend.calls if backend else 0})
        raise
    finally:
        if backend is not None:
            cap.new_json(execution / "backend_stats.json", backend.stats()); backend.log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out) if args.command == "prepare" else execute(args.out) if args.command == "execute" else {"status": "verified", "calls_planned": MAX_CALLS, "trials": len(verify(args.out)[1]["trials"])}
    print(json.dumps(result, ensure_ascii=False))
