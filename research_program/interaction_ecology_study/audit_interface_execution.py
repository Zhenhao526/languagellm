"""Read-only audit of a completed interface_001; no model imports or forwards.

Reconstructs the entire fixed call schedule from saved raw broadcasts. Uses
independent state/menu/settlement formulae rather than the runner's aggregates.
Only a local tokenizer is loaded to render and count input tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
from random import Random
import traceback

from research_program.triadic_task.qwen_capability import RULES as OLD_RULES

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
AGENTS = ("A", "B", "C")
ARMS = ("number", "semantic")
GROUPS = (401, 402, 403)
SCENE_SEEDS = (20260917011, 20260917012, 20260917013, 20260917014)
ROUTE = "triadic_action_interface_v1"
RULES = OLD_RULES.replace("每轮重新抽取需求和布局；过去的物资、需求和编号不能代替本轮观察。",
                         "每次检查都从空白个人历史开始，需求和布局重新抽取。").replace(
    "编号分别随机打乱，只供本人操作，不能直接照抄伙伴的动作编号。",
    "编号分别随机打乱，只供本人查看，不能直接照抄伙伴的动作编号。正式选择格式在行动阶段单独说明。")
COMMON_ANALYSIS = ("先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。"
                   "他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。")
ACTION_ANALYSIS = COMMON_ANALYSIS + "核对可选动作编号与动作描述；只依据自己的可见信息选择。"
BROADCAST_ANALYSIS = COMMON_ANALYSIS + "考虑此次要表达什么、是否需回应伙伴，或是否沉默。"
FORMAL = {"number": "只输出你选择的一个本地动作编号，不要解释。",
          "semantic": "只输出你选择的那一项的完整动作原文，与菜单中的动作文字完全一致；不要编号或解释。"}
BROADCAST_FORMAL = "现在输出正式广播消息；最多约50个汉字，也可以沉默。不输出私有分析。"
REPLAY_FIELDS = ("messages", "output", "prompt_sha256", "seed", "temperature", "mode", "valid", "fresh_cache",
                 "native_thinking", "finish_reason", "prompt_tokens", "generation_tokens")


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    raw = "/".join(map(str, ("triadic_qwen_capability_v1", ROUTE) + parts))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4], "big") % (2**31 - 1)


def accepts(need, material, destination):
    resource, dest = divmod(need, 3)
    return material in ((0, 1), (2, 3), (0, 2), (1, 3))[resource] and destination in ((0,), (1,), (0, 1))[dest]


def demand_support():
    return [list(needs) for needs in product(range(12), repeat=3)
            if sum(any(accepts(needs[i], m, d) and accepts(needs[j], m, d)
                       for m, d in product(range(4), range(2)))
                   for i, j in combinations(range(3), 2)) >= 2]


def all_actions(agent):
    return [{"kind": "wait"}] + [{"kind": "transport", "site": f"S{s}", "destination": d, "partner": p}
         for s, d, p in product(range(4), ("L", "R"), [a for a in AGENTS if a != agent])]


def action_text(action):
    return "等待" if action["kind"] == "wait" else f"与{action['partner']}互选合作，从{action['site']}取物资并运送到{action['destination']}"


def need_view(n):
    resource, destination = divmod(n, 3)
    return {"factor": "kind" if resource < 2 else "length", "value": ("wood", "fiber", "short", "long")[resource],
            "destinations": [["L"], ["R"], ["L", "R"]][destination]}


def observation(state, agent):
    who = AGENTS.index(agent)
    return {"self": agent, "public_site": "S0",
            "private_view_owners": {f"S{s}": AGENTS[i] for i, s in enumerate(state["private_sites"])},
            "own_need": need_view(state["needs"][who]),
            "visible_materials": [{"site": f"S{i}", "kind": "wood" if m < 2 else "fiber",
                                   "length": "short" if m % 2 == 0 else "long"}
                                  for i, m in enumerate(state["layout"])],
            "shared_needs": {a: need_view(n) for a, n in zip(AGENTS, state["needs"])},
            "information_control": "full_information"}


def need_text(need):
    names = {"wood": "木材", "fiber": "纤维", "short": "短", "long": "长"}
    criterion = "种类为" + names[need["value"]] + "，长短不限" if need["factor"] == "kind" else "长度为" + names[need["value"]] + "，种类不限"
    return criterion + "；可接受目的地：" + "或".join(need["destinations"])


def prompt(case, agent, transcript, *, window=None):
    obs = observation(case["state"], agent)
    literal = {"你是": agent, "本轮你自己的需求": need_text(obs["own_need"]),
               "本轮三人的需求": {a: need_text(obs["shared_needs"][a]) for a in AGENTS},
               "本轮四处物资": [{"位置": m["site"], "物资": {"short": "短", "long": "长"}[m["length"]]
                                + {"wood": "木材", "fiber": "纤维"}[m["kind"]]} for m in obs["visible_materials"]],
               "通常的私人观察分工": obs["private_view_owners"],
               "本次信息条件": "完整信息：四处物资和三人需求均可见"}
    data = {"你自己的历史": [], "本轮完整信息观察": literal, "本轮已公开广播": deepcopy(transcript)}
    if window is not None:
        data["当前决策"] = {"广播窗口": window, "说明": "世界尚未结算；可以提出方案或回应已公开消息。"}
    else:
        data["本次可选动作"] = [{"编号": m["id"], "动作": action_text(m["action"])} for m in case["menus"][agent]]
        data["当前决策"] = "两个广播窗口已结束。独立选择一个动作；其他人的未执行动作不可见。"
    return [{"role": "system", "content": f"你是合作群体中的{agent}，伙伴是另外两位A/B/C。\n" + RULES},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]


def settle(state, actions):
    active = [a for a in AGENTS if actions[a]["kind"] == "transport"]
    executed, success = {}, {}
    for agent in AGENTS:
        action = actions[agent]
        require(action in all_actions(agent), "Unknown physical action")
        matched = False
        if len(active) == 2 and agent in active:
            other = actions[action["partner"]]
            matched = (other["kind"] == "transport" and other["partner"] == agent
                       and other["site"] == action["site"] and other["destination"] == action["destination"])
        executed[agent] = matched
        success[agent] = bool(matched and accepts(state["needs"][AGENTS.index(agent)],
            state["layout"][int(action["site"][1:])], ("L", "R").index(action["destination"])))
    units = sum(success.values())
    return {"reward": units / 2, "satisfied_units": units, "full_success": units == 2,
            "individual_feedback": {a: {"executed": executed[a], "own_need_satisfied": success[a], "team_reward": units / 2} for a in AGENTS},
            "researcher": {"overload": len(active) > 2, "active_agents": active, "matching_required": True}}


def audit(run):
    run = Path(run).resolve(); execution = run / "execution"
    status, result = read(execution / "status.json"), read(execution / "results.json")
    require(status["status"] == result["status"] == "completed", "Wait for complete execution")
    plan, freeze, prepared = read(run / "plan.json"), read(run / "freeze.json"), read(run / "prepared_cases.json")
    require(sha(run / "plan.json") == freeze["plan_sha256"], "Plan hash differs")
    require(sha(run / "prepared_cases.json") == plan["prepared_cases_sha256"], "Cases hash differs")
    require(sha(run / "prior_state_receipt.json") == plan["prior_state_receipt_sha256"], "Prior receipt differs")
    require(read(run / "prior_state_receipt.json")["files_sha256"] == plan["prior_state_sha256"], "Prior hash inventory differs")
    sources_checked = {}
    def check_source(name, digest, snapshot=None):
        require(sha(name) == digest, "Source changed: " + str(name))
        sources_checked[str(name)] = digest
        if snapshot:
            require(sha(snapshot) == digest, "Snapshot changed: " + str(snapshot))
            sources_checked[str(snapshot)] = digest
    for name, digest in plan["sources_sha256"].items():
        check_source(name, digest, run / "source_snapshot" / Path(name).relative_to(WORK))
    for name, digest in {**plan["source_artifact_sha256"], **plan["prior_state_sha256"]}.items():
        check_source(name, digest)
    source_cap = Path(plan["source_run"])
    require(sha(source_cap / "plan.json") == plan["source_cap_plan_sha256"] and
            sha(source_cap / "独立核验.json") == plan["source_cap_audit_sha256"], "CAP anchor differs")
    cap_plan = read(source_cap / "plan.json")
    for name, digest in cap_plan["sources_sha256"].items():
        check_source(name, digest, source_cap / "source_snapshot" / Path(name).relative_to(WORK))
    for row in cap_plan["weight_files_size_checked"]:
        require(Path(row["file"]).stat().st_size == row["bytes"], "Weight file size differs")
    require(plan["model"] == cap_plan["model"] and plan["model_revision"] == cap_plan["model_revision"], "Model identity differs")
    require(plan["max_calls"] == 288 and plan["arms"] == list(ARMS) and plan["groups"] == list(GROUPS)
            and plan["scene_seeds"] == list(SCENE_SEEDS), "Fixed batch differs")
    require(plan["private_analysis_request"] == ACTION_ANALYSIS and plan["formal_instructions"] == FORMAL, "Planned requests differ")

    support = demand_support(); require(len(support) == 996, "Independent support differs")
    scenes = []
    for episode, scene_seed in enumerate(SCENE_SEEDS, 1):
        rng = Random(scene_seed); layout, sites = list(range(4)), [1, 2, 3]
        rng.shuffle(layout); rng.shuffle(sites)
        scenes.append({"episode": episode, "scene_seed": scene_seed,
                       "state": {"needs": rng.choice(support), "layout": layout, "private_sites": sites}})
    require(prepared["scenarios"] == scenes, "IID source worlds differ")
    expected_cases = []
    for group in GROUPS:
        for scene in scenes:
            menus, menu_seeds = {}, {}
            for agent in AGENTS:
                menu_seeds[agent] = seed("menu", group, scene["episode"], scene["scene_seed"], agent)
                order = list(range(17)); Random(menu_seeds[agent]).shuffle(order)
                menu = all_actions(agent)
                menus[agent] = [{"id": i, "action": menu[j]} for i, j in enumerate(order)]
            expected_cases.append({"case_id": f"g{group}_e{scene['episode']}", "group": group, **scene,
                "observations": {a: observation(scene["state"], a) for a in AGENTS}, "menus": menus, "menu_seeds": menu_seeds})
    require(prepared["trials"] == expected_cases, "Scenes/observations/private menus differ")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True, trust_remote_code=False)
    tok = prepared["tokenization"]
    digits = {c: tokenizer.encode(c, add_special_tokens=False) for c in "0123456789"}
    require(all(len(v) == 1 for v in digits.values()), "Digits no longer single-token")
    numeric = {str(i): [digits[c][0] for c in str(i)] for i in range(17)}
    semantic = {text: tokenizer.encode(text, add_special_tokens=False)
                for text in sorted({action_text(a) for who in AGENTS for a in all_actions(who)})}
    require(tok["number_sequences"] == numeric and tok["semantic_sequences"] == semantic, "Frozen token paths differ")
    require(all(tokenizer.decode(v) == k for k, v in {**numeric, **semantic}.items()), "Candidate token roundtrip differs")
    require(tok["eos_token_ids"] == [tokenizer.eos_token_id], "EOS differs")
    require(plan["formal_max_tokens"] == tok["formal_max_tokens"] == max(map(len, list(numeric.values())+list(semantic.values()))) + 1 == 16,
            "Common formal budget differs")
    require(len(set(map(tuple, semantic.values()))) == len(semantic), "Semantic token collision")
    for who in AGENTS:
        require(len({action_text(a) for a in all_actions(who)}) == 17, "Semantic menu incomplete")
    eos = tok["eos_token_ids"][0]
    require(all(eos not in seq for seq in list(numeric.values())+list(semantic.values())), "EOS inside a candidate")

    calls, decisions = jsonl(execution / "inference.jsonl"), jsonl(execution / "decisions.jsonl")
    snapshots, trials = jsonl(execution / "snapshots.jsonl"), jsonl(execution / "trials.jsonl")
    require([c["call"] for c in calls] == list(range(1, 289)) and len(decisions) == 144
            and len(snapshots) == 12 and len(trials) == 24, "Incomplete/duplicate call or record grid")
    call_index, expected_decisions = 0, []
    token_counts, finishes, truncations = Counter(), Counter(), []
    def check_call(expected_messages, label, mode, decision_seed, temperature, choices=None):
        nonlocal call_index
        c = calls[call_index]; call_index += 1
        require(c["label"] == label and c["mode"] == mode, f"call {call_index}: schedule/label differs")
        require(c["messages"] == expected_messages, f"call {call_index}: actual input differs")
        require(c["seed"] == decision_seed and c["temperature"] == temperature, f"call {call_index}: seed/temp differs")
        require(c["valid"] is True and c["fresh_cache"] is True and c["native_thinking"] is False,
                f"call {call_index}: valid/cache/thinking flags differ")
        rendered = tokenizer.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(c["prompt_sha256"] == hashlib.sha256(rendered.encode()).hexdigest(), f"call {call_index}: rendered SHA differs")
        add_special = tokenizer.bos_token is None or not rendered.startswith(tokenizer.bos_token)
        require(c["prompt_tokens"] == len(tokenizer.encode(rendered, add_special_tokens=add_special)), f"call {call_index}: prompt token count differs")
        limit = 256 if mode == "private_analysis" else 96 if mode == "natural_message" else 16
        require(type(c["generation_tokens"]) is int and 1 <= c["generation_tokens"] <= limit, f"call {call_index}: generation budget differs")
        require(c["finish_reason"] in ("stop", "length"), f"call {call_index}: unknown stop reason")
        if c["finish_reason"] == "length":
            require(c["generation_tokens"] == limit, f"call {call_index}: inconsistent length stop")
            truncations.append({"call": c["call"], "mode": mode, "label": label, "generation_tokens": c["generation_tokens"]})
        for field in ("seconds", "prompt_tps", "generation_tps", "peak_memory_gb"):
            require(math.isfinite(c[field]) and c[field] >= 0, f"call {call_index}: invalid numeric {field}")
        require(isinstance(c["output"], str), f"call {call_index}: output type differs")
        if choices is not None:
            mapping = numeric if mode == "action_number" else semantic
            seqs = [mapping[s] for s in choices]
            require(len(choices) == len(set(choices)) == 17 and c["choices"] == choices
                    and c["allowed_token_sequences"] == seqs and c["max_generation_tokens"] == 16,
                    f"call {call_index}: incomplete or changed formal paths")
            require(c["output"] in choices and c["finish_reason"] == "stop"
                    and c["generation_tokens"] == len(mapping[c["output"]]) + 1, f"call {call_index}: formal output/token stop differs")
            require(datetime.fromisoformat(c["completed_at"]) >= datetime.fromisoformat(c["started_at"]), "Reversed formal time")
        token_counts[mode] += c["generation_tokens"]
        finishes[(mode, c["finish_reason"])] += 1
        return c

    rebuilt_snapshots = []
    for case in expected_cases:
        transcript = []
        for window in (1, 2):
            pending = []
            for agent in AGENTS:
                label = {"phase": ROUTE, "arm": "common", "case_id": case["case_id"], "group": case["group"],
                         "episode": case["episode"], "agent": agent, "window": window}
                ds = seed("generation", case["group"], case["episode"], window, agent, "natural_message")
                base = prompt(case, agent, transcript, window=window)
                analysis = check_call(base + [{"role": "user", "content": BROADCAST_ANALYSIS}],
                                      {**label, "stage": "analysis"}, "private_analysis", ds, 0)
                formal = check_call(base + [{"role": "assistant", "content": analysis["output"]},
                                            {"role": "user", "content": BROADCAST_FORMAL}],
                                    {**label, "stage": "formal"}, "natural_message", ds, .7)
                expected_decisions.append({**label, "seed": ds, "output": formal["output"], "call_range": [analysis["call"], formal["call"]]})
                pending.append({"agent": agent, "window": window, "text": formal["output"]})
            transcript.extend(pending)
        rebuilt_snapshots.append({**deepcopy(case), "private_histories_before": {a: [] for a in AGENTS},
            "messages": transcript, "action_prompts": {a: prompt(case, a, transcript) for a in AGENTS}})
    require(snapshots == rebuilt_snapshots and call_index == 144, "Snapshot/current broadcasts differ")
    rebuilt_trials, analyses = [], {}
    for arm in ARMS:
        for snapshot in rebuilt_snapshots:
            actions, selections = {}, {}
            for agent in AGENTS:
                label = {"phase": ROUTE, "arm": arm, "case_id": snapshot["case_id"], "group": snapshot["group"],
                         "episode": snapshot["episode"], "agent": agent, "window": None}
                ds = seed("generation", snapshot["group"], snapshot["episode"], 0, agent, "action")
                base = snapshot["action_prompts"][agent]
                choices = [str(m["id"]) if arm == "number" else action_text(m["action"]) for m in snapshot["menus"][agent]]
                analysis = check_call(base + [{"role": "user", "content": ACTION_ANALYSIS}],
                                      {**label, "stage": "analysis"}, "private_analysis", ds, 0)
                formal = check_call(base + [{"role": "assistant", "content": analysis["output"]}, {"role": "user", "content": FORMAL[arm]}],
                                    {**label, "stage": "formal"}, "action_" + arm, ds, 0, choices)
                selected = deepcopy(snapshot["menus"][agent][choices.index(formal["output"])])
                selections[agent], actions[agent] = selected, selected["action"]
                analyses[(arm, snapshot["case_id"], agent)] = analysis
                expected_decisions.append({**label, "seed": ds, "choices": choices, "output": formal["output"],
                    "selected": selected, "call_range": [analysis["call"], formal["call"]]})
            rebuilt_trials.append({"arm": arm, "case_id": snapshot["case_id"], "group": snapshot["group"], "episode": snapshot["episode"],
                "scene_seed": snapshot["scene_seed"], "state": snapshot["state"], "actions": actions,
                "selections": selections, "outcome": settle(snapshot["state"], actions)})
    require(trials == rebuilt_trials and decisions == expected_decisions and call_index == 288, "Decision or independent settlement differs")
    comparisons = []
    for case in expected_cases:
        for agent in AGENTS:
            a, b = [analyses[(arm, case["case_id"], agent)] for arm in ARMS]
            comparisons.append({"case_id": case["case_id"], "agent": agent, "calls": [a["call"], b["call"]],
                                "equal": {field: a[field] == b[field] for field in REPLAY_FIELDS}})
    exact = all(all(x["equal"].values()) for x in comparisons)
    saved_replay = read(execution / "analysis_reproduction.json")
    require(saved_replay["comparisons"] == comparisons and saved_replay["exact"] == exact, "Analysis replay summary differs")
    arms = [{"arm": arm, "trials": 12, "action_calls": 72, "common_calls_reused": 144,
             "full_success_trials": sum(t["outcome"]["full_success"] for t in trials if t["arm"] == arm),
             "rewards": [t["outcome"]["reward"] for t in trials if t["arm"] == arm],
             "by_group": [{"group": group, "rewards": [t["outcome"]["reward"] for t in trials if t["arm"] == arm and t["group"] == group]}
                          for group in GROUPS]} for arm in ARMS]
    pairs = [{"case_id": expected_cases[i]["case_id"], "number_reward": trials[i]["outcome"]["reward"],
              "semantic_reward": trials[i+12]["outcome"]["reward"],
              "semantic_minus_number": trials[i+12]["outcome"]["reward"] - trials[i]["outcome"]["reward"],
              "changed_agents": [a for a in AGENTS if trials[i]["actions"][a] != trials[i+12]["actions"][a]]} for i in range(12)]
    require(result["arms"] == arms and result["pairs"] == pairs, "Reported arm or paired scores differ")
    gate = {"configuration": ROUTE, "required_full_success": 12, "observed_full_success": arms[1]["full_success_trials"],
            "analysis_exact": exact, "passed": arms[1]["full_success_trials"] == 12 and exact,
            "old_gates_modified": False, "symbolic_stage_started": False}
    require(result["gate"] == gate and status["gate_passed"] == gate["passed"] and
            result["effect_interpretable"] == result["analysis_reproduction_exact"] == exact, "Gate/effect flag differs")
    require(result["prior_state_unchanged"] == {name: True for name in plan["prior_state_sha256"]}, "Prior state flag differs")
    counts = Counter(c["seed"] for c in calls)
    require(len(counts) == 108 and Counter(counts.values()) == {2: 72, 4: 36}, "Seed reuse differs")
    require(Counter(c["mode"] for c in calls) == {"private_analysis": 144, "natural_message": 72, "action_number": 36, "action_semantic": 36}, "Mode denominators differ")
    stats, started = read(execution / "backend_stats.json"), read(execution / "started.json")
    require(stats["calls"] == result["calls"] == status["calls"] == 288 and stats["generation_tokens_by_mode"] == dict(token_counts), "Backend count totals differ")
    require(math.isclose(stats["inference_seconds"], sum(c["seconds"] for c in calls), rel_tol=1e-12, abs_tol=1e-6), "Inference time sum differs")
    require(stats["max_prompt_tokens"] == max(c["prompt_tokens"] for c in calls), "Max prompt length differs")
    require(stats["peak_mlx_memory_gb"] >= max(c["peak_memory_gb"] for c in calls), "Peak memory summary smaller than a call")
    require(started["runtime"] == result["runtime"] == plan["prepare_runtime"], "Runtime changed")
    require(started["plan_sha256"] == result["plan_sha256"] == freeze["plan_sha256"], "Run plan hash differs")
    require(math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= stats["inference_seconds"], "Elapsed budget invalid")
    artifacts = [p for p in execution.iterdir() if p.is_file()] + [run / f for f in ("plan.json", "freeze.json", "prepared_cases.json", "prior_state_receipt.json")]
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
            "source_sha256": sources_checked, "run_artifact_sha256": {str(p): sha(p) for p in artifacts},
            "scope": {"model_calls_by_auditor": 0, "weights_loaded_by_auditor": 0, "tokenizer_only": True,
                      "actual_calls": 288, "decisions": 144, "formal_broadcasts": 72, "action_formals": 72,
                      "snapshots": 12, "settlements": 24, "private_analysis_pairs": 36, "source_current_and_snapshot_checks": len(sources_checked)},
            "checks": {"actual_messages_exact": True, "rendered_sha_and_prompt_tokens_exact": True,
                       "same_window_cross_trial_cross_arm_isolation": True, "all17_action_candidates_retained": True,
                       "formal_paths_and_generation_token_counts_exact": True, "physical_settlement_independent": True,
                       "seeds_108_with_72x2_and_36x4": True, "old_state_hashes_unchanged": True},
            "private_analysis_exact": exact, "private_analysis_comparisons": comparisons,
            "formal_budget": 16, "finish_counts": [{"mode": k[0], "finish_reason": k[1], "count": v} for k, v in sorted(finishes.items())],
            "truncations": truncations, "arms": arms, "pairs": pairs, "gate": gate,
            "elapsed_seconds": result["elapsed_seconds"], "inference_seconds": stats["inference_seconds"],
            "peak_mlx_memory_gb": stats["peak_mlx_memory_gb"], "generation_tokens_by_mode": dict(token_counts),
            "limits": ["No current full weight hash; preserved CAP provenance and weight sizes only",
                       "Cache/thinking flags and fresh-cache source code checked; internal runtime cache memory not directly inspected",
                       "Free-text generated token IDs not logged; checked reported budgets/stops, not exact token reconstruction from decoded prose",
                       "Four IID semantic worlds with three repetitions; no symbol convention or language formation tested"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "results/interface_001")
    args = parser.parse_args(); run = args.run.resolve()
    out = run / "独立核验.json"; md = run / "独立核验.md"
    require(not out.exists() and not md.exists() and not (run / "audit_failure.json").exists(), "Refuse audit overwrite/retry")
    try:
        result = audit(run)
        with out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
        body = f'''# 动作接口执行独立核验

核验通过。独立重建全部288次实际输入、渲染SHA与提示词元数；72条真实广播、144个决定、12个同窗隔离快照、两臂24份物理结算全部吻合。审计没有载入模型权重或执行新推断。

两臂各36个正式动作均保留完整17候选，规范词元路径及输出序列长度加EOS对应的生成token数精确吻合，两臂正式上限16。108个不同seed的复用为72个广播seed各两次、36个动作seed各四次。所有原输入历史为空，同窗消息仅在下一窗可见；另一场、另一臂和尚未执行的物理动作没有进入提示。

36对私有分析的完整输入／输出及日志字段精确重现：{result['private_analysis_exact']}。报告不展示分析全文。number完整成功{result['arms'][0]['full_success_trials']}/12，semantic完整成功{result['arms'][1]['full_success_trials']}/12；新配置门槛passed={result['gate']['passed']}。实际长度触顶终止{len(result['truncations'])}次，具体call／mode记录在JSON。

冻结源码、快照、旧CAP链和旧状态共{len(result['source_sha256'])}项当前文件／快照SHA核验通过。旧v3门槛、CAP历史与结果、EXEC结果未改变。模型沿用已审计来源与文件大小，本次未重新全量hash权重。实际耗时{result['elapsed_seconds']:.3f}秒，峰值MLX内存{result['peak_mlx_memory_gb']:.3f}GB。

这是整个正式输出接口的配对检查：指令、词元化、长度和受限贪心分叉共同变化，不能唯一归因于编号映射。4个语义世界×3次重复不是12个独立世界；结果不检验符号约定或语言形成。私有分析内容不能直接证明模型内部机制。缓存标记和代码的新cache路径已核验，未直接检查运行时缓存内存；自由文本原始输出token ID未记录，不能从重新编码文本严格重建其生成轨迹。

详细证据与固定范围见[独立核验.json](独立核验.json)，审计器是[../../audit_interface_execution.py](../../audit_interface_execution.py)。
'''
        with md.open("x", encoding="utf-8") as stream:
            stream.write(body)
        print(json.dumps({k: result[k] for k in ("status", "scope", "private_analysis_exact", "finish_counts", "gate")}, ensure_ascii=False, indent=2))
    except BaseException as error:
        with (run / "audit_failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"failed_at": datetime.now(timezone.utc).isoformat(), "error": str(error),
                       "traceback": traceback.format_exc(), "audit_source_sha256": sha(__file__), "automatic_retry": False}, stream, ensure_ascii=False, indent=2)
        raise


if __name__ == "__main__":
    main()
