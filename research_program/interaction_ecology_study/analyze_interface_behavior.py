"""Read-only behavior analysis of a complete action-interface batch.

Reads public snapshots, decisions and physical outcomes; never reads inference
private analyses. Optional agent text coding must cite the exact public text.
This file is independent of the frozen runner/environment and loads no model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

AGENTS = ("A", "B", "C")
ARMS = ("number", "semantic")
ARM_NAMES = {"number": "编号", "semantic": "动作原文"}
CATEGORY_NAMES = {"all_wait": "全体等待", "single_transport": "仅一人运输", "overload": "三人运输超载",
                  "two_unmatched": "两人运输但未匹配", "matched_incomplete_needs": "实际匹配但需求未全部满足",
                  "matched_full_success": "实际匹配且两人需求满足"}


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def action_text(action):
    if action["kind"] == "wait": return "等待"
    return f"与{action['partner']}互选合作，从{action['site']}取物资并运送到{action['destination']}"


def legal_actions(agent):
    return [{"kind": "wait"}] + [{"kind": "transport", "site": f"S{s}", "destination": d, "partner": p}
        for s in range(4) for d in ("L", "R") for p in AGENTS if p != agent]


def acceptance(need, material, destination):
    resource_sets = ((0, 1), (2, 3), (0, 2), (1, 3))
    destination_sets = (("L",), ("R",), ("L", "R"))
    resource_ok = material in resource_sets[need // 3]
    destination_ok = destination in destination_sets[need % 3]
    return resource_ok, destination_ok


def replay(state, actions):
    """Independent D1 rules; does not import the frozen environment."""
    require(set(actions) == set(AGENTS), "Three actions required")
    require(sorted(state["layout"]) == list(range(4)) and len(state["needs"]) == 3
            and all(n in range(12) for n in state["needs"]), "Invalid state")
    for a in AGENTS: require(actions[a] in legal_actions(a), "Illegal physical action")
    active = [a for a in AGENTS if actions[a]["kind"] == "transport"]
    mismatch = {"partner": False, "site": False, "destination": False}
    physical_match = False
    if len(active) == 2:
        a, b = active
        mismatch = {"partner": actions[a]["partner"] != b or actions[b]["partner"] != a,
                    "site": actions[a]["site"] != actions[b]["site"],
                    "destination": actions[a]["destination"] != actions[b]["destination"]}
        physical_match = not any(mismatch.values())
    detail = {}
    for i, a in enumerate(AGENTS):
        action = actions[a]
        if action["kind"] == "wait":
            resource_ok = destination_ok = compatible = None
        else:
            material = state["layout"][int(action["site"][1:])]
            resource_ok, destination_ok = acceptance(state["needs"][i], material, action["destination"])
            compatible = resource_ok and destination_ok
        executed = physical_match and a in active
        detail[a] = {"attempted_transport": a in active, "executed": executed,
                     "resource_acceptable_if_executed": resource_ok, "destination_acceptable_if_executed": destination_ok,
                     "own_need_compatible_if_executed": compatible, "actually_satisfied": bool(executed and compatible)}
    units = sum(d["actually_satisfied"] for d in detail.values())
    category = ("all_wait" if len(active) == 0 else "single_transport" if len(active) == 1 else "overload" if len(active) == 3
                else "two_unmatched" if not physical_match else "matched_full_success" if units == 2 else "matched_incomplete_needs")
    outcome = {"reward": units / 2, "satisfied_units": units, "full_success": units == 2,
        "individual_feedback": {a: {"executed": detail[a]["executed"], "own_need_satisfied": detail[a]["actually_satisfied"],
                                     "team_reward": units / 2} for a in AGENTS},
        "researcher": {"overload": len(active) > 2, "active_agents": active, "matching_required": True}}
    return {"category": category, "active_agents": active, "mismatch_dimensions": mismatch,
            "physical_match": physical_match, "per_agent": detail, "outcome": outcome}


def load_complete(run):
    run = Path(run).resolve()
    status, result = read(run / "execution/status.json"), read(run / "execution/results.json")
    require(status["status"] == result["status"] == "completed" and status["calls"] == result["calls"] == 288,
            "Only the complete fixed288-call batch may be analyzed")
    plan, freeze = read(run / "plan.json"), read(run / "freeze.json")
    require(sha(run / "plan.json") == freeze["plan_sha256"] == result["plan_sha256"], "Frozen plan anchor changed")
    prepared = read(run / "prepared_cases.json")
    require(sha(run / "prepared_cases.json") == plan["prepared_cases_sha256"], "Prepared cases changed")
    snapshots = lines(run / "execution/snapshots.jsonl")
    trials = lines(run / "execution/trials.jsonl")
    decisions = lines(run / "execution/decisions.jsonl")
    require(len(snapshots) == 12 and len(trials) == 24 and len(decisions) == 144, "Incomplete behavior logs")
    require([s["case_id"] for s in snapshots] == [s["case_id"] for s in prepared["trials"]], "Case order changed")
    expected = [(arm, s["case_id"]) for arm in ARMS for s in snapshots]
    require([(t["arm"], t["case_id"]) for t in trials] == expected, "Trial order/cells changed")
    for snapshot, case in zip(snapshots, prepared["trials"]):
        require(all(snapshot[k] == case[k] for k in case), "Snapshot altered a frozen observation/menu/state")
        require(snapshot["private_histories_before"] == {a: [] for a in AGENTS}, "History is not independent per trial")
        require([(m["window"], m["agent"]) for m in snapshot["messages"]] == [(w, a) for w in (1, 2) for a in AGENTS], "Public-message sequence differs")
        require(all(set(m) == {"window", "agent", "text"} and isinstance(m["text"], str) for m in snapshot["messages"]), "Unexpected public-message fields")
    return run, plan, prepared, snapshots, trials, decisions, result


def validate_codings(snapshots, codings):
    require(codings["method"] == "agent_literal_text_coding", "Text coding must not claim human/inter-rater annotation")
    require(codings.get("status") == "agent_coded_complete", "An unreviewed template is not completed text coding")
    index = {(s["case_id"], m["window"], m["agent"]): m["text"] for s in snapshots for m in s["messages"]}
    rows = codings["messages"]
    require(len(rows) == 72 and len({(r["case_id"], r["window"], r["agent"]) for r in rows}) == 72, "All72 public messages require a coding, including uncertainty")
    for row in rows:
        key = row["case_id"], row["window"], row["agent"]
        require(key in index and row["source_text"] == index[key], "Coding source does not match recorded public text")
        require(row["classification"] in ("complete_plan", "own_action", "unresolved_or_other", "silent"), "Unknown coding type")
        for quote in row.get("evidence_quotes", []):
            require(quote and quote in row["source_text"], "Evidence quote is not in this message")
        if row["classification"] == "complete_plan":
            require(row.get("evidence_quotes") and set(row["actions"]) == set(AGENTS), "Complete plan requires all three explicit actions and quotes")
            require(all(row["actions"][a] in legal_actions(a) for a in AGENTS), "Text plan has an invalid physical action")
        elif row["classification"] == "own_action":
            require(row.get("evidence_quotes") and row["action"] in legal_actions(row["agent"]), "Own-action coding incomplete")
        elif row["classification"] == "silent":
            require(not row["source_text"], "Silent coding is not an empty broadcast")
    return {(r["case_id"], r["window"], r["agent"]): r for r in rows}


def public_plans(snapshots, trials, codings):
    index = validate_codings(snapshots, codings)
    trial_index = {(t["arm"], t["case_id"]): t for t in trials}
    result = []
    for snapshot in snapshots:
        for window in (1, 2):
            rows = [index[(snapshot["case_id"], window, a)] for a in AGENTS]
            complete = [r for r in rows if r["classification"] == "complete_plan"]
            unique = {json.dumps(r["actions"], sort_keys=True) for r in complete}
            status = ("all_three_explicit_same_plan" if len(complete) == 3 and len(unique) == 1 else
                      "different_explicit_proposals" if len(unique) > 1 else
                      "one_or_two_explicit_plans" if complete else "no_complete_plan_explicit_in_a_single_message")
            plan_rows = []
            for r in complete:
                plan_rows.append({"speaker": r["agent"], "text_actions": deepcopy(r["actions"]),
                    "if_executed": replay(snapshot["state"], r["actions"]),
                    "actual_matches": {arm: {a: trial_index[(arm, snapshot["case_id"])]["actions"][a] == r["actions"][a]
                                              for a in AGENTS} for arm in ARMS}})
            own = {}
            for r in rows:
                if r["classification"] == "own_action": own[r["agent"]] = r["action"]
                elif r["classification"] == "complete_plan": own[r["agent"]] = r["actions"][r["agent"]]
            own_bundle = None
            if len(own) == 3:
                own_bundle = {"actions_explicit_for_each_speaker": own, "if_executed": replay(snapshot["state"], own),
                    "actual_matches": {arm: {a: trial_index[(arm, snapshot["case_id"])]["actions"][a] == own[a]
                                              for a in AGENTS} for arm in ARMS},
                    "scope": "Three explicit own-action descriptions, not evidence of shared acceptance or commitment"}
            result.append({"case_id": snapshot["case_id"], "window": window, "status": status,
                           "complete_plan_speakers": [r["agent"] for r in complete], "plan_rows": plan_rows,
                           "own_action_bundle": own_bundle})
    return result


def analyze(run, *, coding_path=None):
    run, plan, prepared, snapshots, trials, decisions, result = load_complete(run)
    snaps = {s["case_id"]: s for s in snapshots}
    ds = {(d["arm"], d["case_id"], d["agent"]): d for d in decisions if d["arm"] in ARMS}
    require(len(ds) == 72, "Expected72 real action decisions")
    public_ds = {(d["case_id"], d["window"], d["agent"]): d for d in decisions if d["arm"] == "common"}
    require(len(public_ds) == 72, "Expected72 common message decisions")
    classified = []
    for trial in trials:
        snapshot = snaps[trial["case_id"]]
        require(trial["state"] == snapshot["state"], "Settlement state differs from snapshot")
        for message in snapshot["messages"]:
            require(public_ds[(trial["case_id"], message["window"], message["agent"])]["output"] == message["text"], "Recorded public text differs from decision")
        for agent in AGENTS:
            d = ds[(trial["arm"], trial["case_id"], agent)]
            require(trial["actions"][agent] == d["selected"]["action"] == trial["selections"][agent]["action"], "Action decision/settlement differ")
            menu = snapshot["menus"][agent]
            require(len(menu) == 17 and len({json.dumps(m["action"], sort_keys=True) for m in menu}) == 17,
                    "Duplicate/filtered menu")
            require({json.dumps(m["action"], sort_keys=True) for m in menu}
                    == {json.dumps(a, sort_keys=True) for a in legal_actions(agent)}, "Menu misses a legal bad action")
            require(d["selected"] in menu and d["selected"] == trial["selections"][agent], "Selection not in original private menu")
            expected = [str(m["id"]) if trial["arm"] == "number" else action_text(m["action"]) for m in menu]
            require(d["choices"] == expected and d["output"] == expected[menu.index(d["selected"])], "Formal output maps to different semantic action")
        independent = replay(trial["state"], trial["actions"])
        require(independent["outcome"] == trial["outcome"], "Independent physical/demand score differs")
        classified.append({"arm": trial["arm"], "case_id": trial["case_id"], "group": trial["group"],
                           "episode": trial["episode"], "actions": deepcopy(trial["actions"]), **independent})
    idx = {(t["arm"], t["case_id"]): t for t in classified}
    pairs, action_pairs = [], []
    for snapshot in snapshots:
        case_id = snapshot["case_id"]
        n, s = [idx[(arm, case_id)] for arm in ARMS]
        pairs.append({"case_id": case_id, "group": snapshot["group"], "episode": snapshot["episode"],
                      "number_reward": n["outcome"]["reward"], "semantic_reward": s["outcome"]["reward"],
                      "difference": s["outcome"]["reward"]-n["outcome"]["reward"],
                      "number_category": n["category"], "semantic_category": s["category"]})
        for agent in AGENTS:
            action_pairs.append({"case_id": case_id, "group": snapshot["group"], "episode": snapshot["episode"], "agent": agent,
                "number_action": n["actions"][agent], "semantic_action": s["actions"][agent],
                "number_menu_id": ds[("number", case_id, agent)]["selected"]["id"],
                "semantic_menu_id": ds[("semantic", case_id, agent)]["selected"]["id"],
                "changed": n["actions"][agent] != s["actions"][agent],
                "changed_fields": [k for k in ("kind", "partner", "site", "destination") if n["actions"][agent].get(k) != s["actions"][agent].get(k)]})
    summaries = []
    for arm in ARMS:
        rows = [r for r in classified if r["arm"] == arm]
        details = [v for r in rows for v in r["per_agent"].values()]
        summaries.append({"arm": arm, "trials": 12, "reward_sum": sum(r["outcome"]["reward"] for r in rows),
            "full_success_trials": sum(r["outcome"]["full_success"] for r in rows),
            "exclusive_categories": dict(Counter(r["category"] for r in rows)),
            "unmatched_pair_dimensions_overlapping": {k: sum(r["mismatch_dimensions"][k] for r in rows) for k in ("partner", "site", "destination")},
            "actions": {"wait": sum(not d["attempted_transport"] for d in details),
                        "transport_attempts": sum(d["attempted_transport"] for d in details),
                        "executed_transport_actions": sum(d["executed"] for d in details),
                        "satisfied_agents": sum(d["actually_satisfied"] for d in details),
                        "transport_need_incompatible_even_if_executed": sum(d["own_need_compatible_if_executed"] is False for d in details)}})
    for arm, logged in zip(summaries, result["arms"]):
        require(arm["arm"] == logged["arm"] and arm["full_success_trials"] == logged["full_success_trials"], "Stored summary differs")
    coding = read(coding_path) if coding_path else None
    text_result = public_plans(snapshots, trials, coding) if coding else None
    source_files = [run / f for f in ("plan.json", "freeze.json", "prepared_cases.json", "execution/status.json",
        "execution/results.json", "execution/snapshots.jsonl", "execution/decisions.jsonl", "execution/trials.jsonl")]
    if coding_path: source_files.append(Path(coding_path).resolve())
    return {"status": "complete_behavior_analysis", "completed_at": datetime.now(timezone.utc).isoformat(),
            "analysis_source_sha256": sha(__file__), "source_sha256": {str(p): sha(p) for p in source_files},
            "arms": summaries, "classified_trials": classified, "trial_pairs": pairs, "action_pairs": action_pairs,
            "changed_action_count": sum(r["changed"] for r in action_pairs),
            "transitions": dict(Counter(f"{r['number_reward']:g}→{r['semantic_reward']:g}" for r in pairs)),
            "by_group": [{"group": g, "pairs": [r for r in pairs if r["group"] == g]} for g in plan["groups"]],
            "by_semantic_scene": [{"episode": e, "scene_seed": plan["scene_seeds"][e-1],
                                   "pairs": [r for r in pairs if r["episode"] == e]} for e in range(1, 5)],
            "public_messages": [{"case_id": s["case_id"], **deepcopy(m)} for s in snapshots for m in s["messages"]],
            "text_coding": coding, "public_plan_comparison": text_result,
            "checks": {"independent_settlements": 24, "action_menu_and_decision_mappings": 72,
                       "all_action_pairs": 36, "all_trial_pairs": 12, "public_message_decisions": 72},
            "limitations": ["Behavior analysis, not an independent full288-prompt/token audit; that is a separate task.",
                "Only four semantic draws reused across three generation repetitions; not twelve independent worlds.",
                "Text coding is one agent's literal reading, not human annotation or inter-rater agreement.",
                "Different explicit proposals are not automatically incompatible commitments; silence/assent alone is not an action plan.",
                "No private inference analysis file was read; plan/action correspondence is not a claim about hidden intention.",
                "The output interface changes tokenization, sequence length and constrained greedy branching together.",
                "Old capability gates remain failed as recorded; no symbolic stage is started by this analysis."],
            "model_calls": 0, "neural_forward_calls": 0, "old_files_modified": False}


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown(result):
    out = ["# 完整24场行为核对", "", "本文件只读完整结果，独立重算物理匹配和需求满足。没有读取私有分析；文本部分仅依据原始公开广播。", "",
           f"全部36个动作配对中，{result['changed_action_count']}个语义动作改变。12场收益转移为：" + "、".join(f"{k}：{v}场" for k, v in sorted(result["transitions"].items())) + "。", "",
           "## 物理成败与需求", "", "以下类别互斥，每臂合计12场。未匹配的伙伴／位置／目的地维度另行并列统计，一场可以同时有多个问题。", "",
           "| 互斥结算类别 | 编号 | 动作原文 |", "| --- | ---: | ---: |"]
    for key, label in CATEGORY_NAMES.items():
        out.append(f"| {label} | {result['arms'][0]['exclusive_categories'].get(key,0)} | {result['arms'][1]['exclusive_categories'].get(key,0)} |")
    for arm in result["arms"]:
        out += ["", f"{ARM_NAMES[arm['arm']]}：两人未匹配的各维度为{arm['unmatched_pair_dimensions_overlapping']}；行动计数为{arm['actions']}。需求不兼容计数针对已经提交的运输提议，不能与实际执行或交付数互换。"]
    out += ["", "## 12场配对", "", "| 场次 | 编号收益／类型 | 原文收益／类型 | 差值 |", "| --- | --- | --- | ---: |"]
    for row in result["trial_pairs"]:
        out.append(f"| {row['case_id']} | {row['number_reward']:g}／{CATEGORY_NAMES[row['number_category']]} | {row['semantic_reward']:g}／{CATEGORY_NAMES[row['semantic_category']]} | {row['difference']:+g} |")
    out += ["", "四个场景在三个生成重复中复用；上表不是12个独立世界。逐重复、逐场景和全部原始整数见JSON。", "",
            "## 全部36个动作配对", "", "| 场次／主体 | 编号臂动作 | 原文臂动作 | 改变字段 |", "| --- | --- | --- | --- |"]
    for row in result["action_pairs"]:
        out.append(f"| {row['case_id']}／{row['agent']} | {escape(action_text(row['number_action']))} | {escape(action_text(row['semantic_action']))} | {','.join(row['changed_fields']) or '无'} |")
    out += ["", "## 公开广播与实际动作", "", "逐条编码属于agent文本判读，不是人工标注或独立双人一致性。只按明确文字编码完整方案或本人动作；单独‘同意’、含糊指代或多个未决选项不补齐为承诺。完整提议不同仅报告提议差异，不能自动断言已经互相承诺了不相容方案。"]
    coding_index = {(r["case_id"], r["window"], r["agent"]): r for r in result["text_coding"]["messages"]} if result["text_coding"] else {}
    if not coding_index: out += ["", "当前输出仅收录原文，尚未进行逐条文本编码；不能从本段推出公开方案是否一致。"]
    comparisons = {(r["case_id"], r["window"]): r for r in result["public_plan_comparison"]} if result["public_plan_comparison"] else {}
    for case_id in dict.fromkeys(m["case_id"] for m in result["public_messages"]):
        out += ["", f"### {case_id}", "", "| 窗口／主体 | 实际广播 | 逐条编码 |", "| --- | --- | --- |"]
        for m in result["public_messages"]:
            if m["case_id"] != case_id: continue
            row = coding_index.get((case_id, m["window"], m["agent"]))
            label = row["classification"] + ("；" + row.get("note", "") if row.get("note") else "") if row else "未编码"
            out.append(f"| {m['window']}／{m['agent']} | {escape(m['text'])} | {escape(label)} |")
        for window in (1, 2):
            comparison = comparisons.get((case_id, window))
            if comparison:
                out += ["", f"第{window}窗明确文本状态：{comparison['status']}。"]
                for p in comparison["plan_rows"]:
                    matches = "；".join(f"{ARM_NAMES[arm]}匹配主体：{','.join(a for a,v in p['actual_matches'][arm].items() if v) or '无'}" for arm in ARMS)
                    out.append(f"{p['speaker']}的完整文本方案若执行收益为{p['if_executed']['outcome']['reward']:g}；{matches}。")
                if comparison["own_action_bundle"]:
                    bundle = comparison["own_action_bundle"]
                    out.append(f"仅拼合三人各自明确的本人动作，若执行收益为{bundle['if_executed']['outcome']['reward']:g}；这不是共同承诺的判定。")
    out += ["", "## 核对范围", "", "24次结算、72项正式输出到完整私人菜单的映射、36动作配对与12场转移全部核对；未重跑模型。全288条推断输入与token验证由独立执行审计另行完成。本文件不替代它，不修改任何门槛。", "",
            "代码SHA：`" + result["analysis_source_sha256"] + "`。全部输入SHA和未解决文本项保存在同目录JSON。", ""]
    return "\n".join(out)


def coding_template(run):
    _, _, _, snapshots, _, _, _ = load_complete(run)
    return {"method": "agent_literal_text_coding", "status": "uncoded_template", "messages": [
        {"case_id": s["case_id"], **deepcopy(m), "source_text": m["text"], "classification": "unresolved_or_other",
         "evidence_quotes": [], "note": "待逐条判读"} for s in snapshots for m in s["messages"]]}


def execute(run, out, coding_path=None):
    out = Path(out).resolve(); require(not out.exists(), "Never overwrite a behavior output")
    started = time.perf_counter()
    result = analyze(run, coding_path=coding_path)
    result["elapsed_seconds"] = time.perf_counter()-started
    out.mkdir(parents=True, exist_ok=False)
    new_json(out / "behavior.json", result)
    with (out / "行为核对.md").open("x", encoding="utf-8") as stream: stream.write(markdown(result))
    return {"status": result["status"], "output": str(out), "changed_actions": result["changed_action_count"], "transitions": result["transitions"]}


def self_test():
    state = {"needs": [2, 2, 7], "layout": [0, 3, 1, 2]}
    wait = {a: {"kind": "wait"} for a in AGENTS}
    ab = deepcopy(wait); ab["A"] = {"kind": "transport", "site": "S0", "destination": "R", "partner": "B"}
    ab["B"] = {"kind": "transport", "site": "S0", "destination": "R", "partner": "A"}
    require(replay(state, wait)["category"] == "all_wait", "all wait")
    single = deepcopy(wait); single["A"] = deepcopy(ab["A"])
    require(replay(state, single)["category"] == "single_transport", "single")
    require(replay(state, ab)["outcome"]["reward"] == 1, "full match")
    over = deepcopy(ab); over["C"] = {"kind": "transport", "site": "S0", "destination": "R", "partner": "A"}
    require(replay(state, over)["category"] == "overload", "overload")
    wrong = deepcopy(ab); wrong["B"] = {"kind": "transport", "site": "S1", "destination": "L", "partner": "C"}
    r = replay(state, wrong)
    require(r["category"] == "two_unmatched" and all(r["mismatch_dimensions"].values()), "overlapping mismatch")
    ac = deepcopy(wait); ac["A"] = {"kind": "transport", "site": "S2", "destination": "R", "partner": "C"}
    ac["C"] = {"kind": "transport", "site": "S2", "destination": "R", "partner": "A"}
    require(replay(state, ac)["physical_match"] and replay(state, ac)["outcome"]["reward"] == .5, "one need fails after real match")
    neither = deepcopy(ab)
    for a in ("A", "B"): neither[a]["site"] = "S1"
    require(replay(state, neither)["physical_match"] and replay(state, neither)["outcome"]["reward"] == 0, "both needs fail after match")
    require(replay(state, ac)["per_agent"]["C"]["resource_acceptable_if_executed"] is False, "attribute compatibility")
    require(len(legal_actions("A")) == 17 and len({action_text(a) for a in legal_actions("A")}) == 17, "full interface")
    return {"status": "passed", "synthetic_checks": 9, "real_result_reads": 0, "model_calls": 0, "analysis_source_sha256": sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "template", "analyze"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--codings", type=Path)
    args = parser.parse_args()
    if args.command == "self-test": result = self_test()
    else:
        require(args.run is not None and args.out is not None, "--run and --out are required")
        if args.command == "template":
            new_json(args.out, coding_template(args.run)); result = {"status": "template_written", "path": str(args.out)}
        else: result = execute(args.run, args.out, args.codings)
    print(json.dumps(result, ensure_ascii=False))
