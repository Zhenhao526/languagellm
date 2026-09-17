"""Completed-run behavior summary from broadcasts and final actions only.

Never reads inference/private-analysis text, imports model code, or modifies the
run. Natural-language extraction is conservative and explicitly not a complete
semantic annotation. Outputs go outside the frozen CAP directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


AGENTS = ("A", "B", "C")
NAMES = {"full_success": "两人匹配且需求全满足", "matched_partial_need": "已匹配，仅一人需求满足",
         "matched_no_need": "已匹配，两人需求均未满足", "overload_three_active": "三人运输超载",
         "no_active": "全体等待", "single_active": "仅一人运输", "two_active_unmatched": "两人运输但字段未匹配"}
PAIR_PATTERN = re.compile(r"([ABC])\s*(?:与|和|跟|、|[-—–])\s*([ABC])")
UNCERTAINTY = re.compile(r"如果|否则|若|或者|或|不要|不应|不建议|反对|拒绝|[?？]")
WAIT_PATTERN = re.compile(r"([ABC])\s*(?:请|应|先|继续|保持|仍|就)?\s*(?:等待|等候|不运输|不参与)")
RESPONSE_PATTERN = re.compile(r"支持|同意|接受|采用|维持|坚持|改为|改选|调整")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def action_text(action):
    return "等待" if action["kind"] == "wait" else f"与{action['partner']}，{action['site']}→{action['destination']}"


def extract_plan(text, speaker):
    """Surface-only candidate: unique pair/site/destination and explicit third wait.

    Multiple alternatives or negation/conditional cues remain unclassified. A
    detected plan is a quoted proposal, never proof that partners accepted it.
    """
    normalized = text.replace("我", speaker)
    pairs = sorted({"".join(sorted(pair)) for pair in PAIR_PATTERN.findall(normalized) if pair[0] != pair[1]})
    sites = sorted(set(re.findall(r"(?<![A-Za-z0-9])S[0-3](?![0-9])", normalized)))
    destinations = sorted(set(re.findall(r"(?<![A-Za-z0-9])[LR](?![A-Za-z0-9])", normalized)))
    waiters = sorted(set(WAIT_PATTERN.findall(normalized)))
    cues = UNCERTAINTY.findall(normalized)
    plan = None
    if len(pairs) == len(sites) == len(destinations) == len(waiters) == 1 and not cues:
        pair = list(pairs[0])
        if set(pair) | set(waiters) == set(AGENTS) and not set(pair) & set(waiters):
            plan = {"pair": pair, "site": sites[0], "destination": destinations[0], "waiter": waiters[0]}
    return {"pair_mentions": pairs, "site_mentions": sites, "destination_mentions": destinations,
            "waiting_agent_mentions": waiters, "conditional_alternative_or_negation_cues": cues,
            "response_word_mentions": RESPONSE_PATTERN.findall(normalized),
            "single_explicit_surface_plan": plan,
            "scope": "Conservative surface extraction; null means not classified, not no proposal."}


def plan_key(plan):
    return None if plan is None else (tuple(plan["pair"]), plan["site"], plan["destination"], plan["waiter"])


def proposed_own_action(plan, agent):
    if agent == plan["waiter"]:
        return {"kind": "wait"}
    return {"kind": "transport", "site": plan["site"], "destination": plan["destination"],
            "partner": next(a for a in plan["pair"] if a != agent)}


def diagnose(row):
    actions = row["actions"]
    require(set(actions) == set(AGENTS), "Missing final action")
    materials = {r["site"]: {"kind": r["kind"], "length": r["length"]}
                 for r in row["observations"]["A"]["visible_materials"]}
    require(set(materials) == {"S0", "S1", "S2", "S3"}, "This summary requires the complete-information control")
    for agent in AGENTS:
        seen = {r["site"]: {"kind": r["kind"], "length": r["length"]}
                for r in row["observations"][agent]["visible_materials"]}
        require(seen == materials, "Full-information observers disagree on current material")
    active = [a for a in AGENTS if actions[a]["kind"] == "transport"]
    require(all(action["kind"] in ("wait", "transport") for action in actions.values()), "Unknown action kind")
    pair_fields = None
    if len(active) == 2:
        a, b = active
        pair_fields = {"mutual_partners": actions[a]["partner"] == b and actions[b]["partner"] == a,
                       "same_site": actions[a]["site"] == actions[b]["site"],
                       "same_destination": actions[a]["destination"] == actions[b]["destination"]}
    matched = pair_fields is not None and all(pair_fields.values())
    proposals, executed, satisfied = {}, {}, {}
    for index, agent in enumerate(AGENTS):
        action = actions[agent]
        need = row["observations"][agent]["own_need"]
        if action["kind"] == "wait":
            proposals[agent] = {"active": False, "own_need": need, "proposed_material": None,
                                "material_requirement_met": None, "destination_requirement_met": None}
            executed[agent] = satisfied[agent] = False
            continue
        require(action["site"] in materials and action["destination"] in ("L", "R")
                and action["partner"] in set(AGENTS) - {agent}, "Invalid semantic action")
        material = materials[action["site"]]
        resource_ok = material[need["factor"]] == need["value"]
        destination_ok = action["destination"] in need["destinations"]
        executed[agent] = bool(matched)
        satisfied[agent] = bool(matched and resource_ok and destination_ok)
        proposals[agent] = {"active": True, "own_need": need, "proposed_material": material,
            "material_requirement_met": resource_ok, "destination_requirement_met": destination_ok,
            "would_satisfy_own_need_if_transport_matched": resource_ok and destination_ok}
    score = sum(satisfied.values()) / 2
    require(score == row["outcome"]["reward"] and sum(satisfied.values()) == row["outcome"]["satisfied_units"],
            "Independent physical/need score differs from recorded outcome")
    for agent in AGENTS:
        feedback = row["outcome"]["individual_feedback"][agent]
        require(feedback["executed"] == executed[agent] and feedback["own_need_satisfied"] == satisfied[agent],
                "Own execution/need feedback differs")
    if matched:
        category = {1.: "full_success", .5: "matched_partial_need", 0.: "matched_no_need"}[score]
    else:
        category = {0: "no_active", 1: "single_active", 2: "two_active_unmatched", 3: "overload_three_active"}[len(active)]
    messages = row["messages"]
    require([(m["window"], m["agent"]) for m in messages] == [(w, a) for w in (1, 2) for a in AGENTS],
            "Expected two complete synchronous broadcast windows")
    first_call = row["call_range"][0]
    evidence = [{**m, "formal_call": first_call + 1 + 2 * i, "surface": extract_plan(m["text"], m["agent"])}
                for i, m in enumerate(messages)]
    window_details = []
    for window in (1, 2):
        entries = [m for m in evidence if m["window"] == window]
        keys = [plan_key(m["surface"]["single_explicit_surface_plan"]) for m in entries]
        distinct = set(k for k in keys if k is not None)
        window_details.append({"window": window, "classified_messages": sum(k is not None for k in keys),
            "distinct_complete_surface_plans": len(distinct),
            "competing_explicit_plans_detected": len(distinct) > 1,
            "all_three_explicit_plans_same": None if any(k is None for k in keys) else len(distinct) == 1,
            "interpretation": "Detected different proposals are observable; matching proposals alone do not prove agreement or causal response."})
    responses = []
    for index, agent in enumerate(AGENTS):
        old = evidence[index]["surface"]["single_explicit_surface_plan"]
        current = evidence[3 + index]["surface"]["single_explicit_surface_plan"]
        current_key = plan_key(current)
        other_first = [plan_key(m["surface"]["single_explicit_surface_plan"]) for m in evidence[:3] if m["agent"] != agent]
        responses.append({"agent": agent, "second_window_formal_call": evidence[3 + index]["formal_call"],
            "response_word_mentions": evidence[3 + index]["surface"]["response_word_mentions"],
            "changed_from_own_first_window_explicit_plan": None if old is None or current is None else plan_key(old) != current_key,
            "second_plan_matches_other_first_window_plan": None if current is None else current_key in other_first,
            "final_action_formal_call": first_call + 13 + index * 2,
            "final_action_matches_own_second_window_explicit_plan": None if current is None else actions[agent] == proposed_own_action(current, agent),
            "plan_implied_own_action": None if current is None else proposed_own_action(current, agent),
            "actual_action": actions[agent]})
    return {"case_id": row["case_id"], "group": row["group"], "episode": row["episode"],
            "reward": score, "active_agents": active, "active_count": len(active), "primary_category": category,
            "two_active_matching_fields": pair_fields, "physically_matched_pair": matched,
            "individual_proposal_checks": proposals, "actual_actions": actions,
            "broadcast_evidence": evidence, "windows": window_details, "second_window_and_action_checks": responses}


def summarize(run_dir, output_dir):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    require(run != output and run not in output.parents, "Behavior outputs must remain outside the frozen CAP directory")
    status_path = run / "execution/status.json"
    require(status_path.is_file() and load(status_path).get("status") == "completed", "Wait for the completed run; no partial summary emitted")
    records_path = run / "execution/trials.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    plan = load(run / "plan.json")
    expected = [(g, e) for g in plan["groups"] for e in range(1, 5)]
    require(len(rows) == 12 and [(r["group"], r["episode"]) for r in rows] == expected, "Incomplete or reordered trials")
    analyses = [diagnose(row) for row in rows]
    mismatches = Counter()
    for row in analyses:
        if row["two_active_matching_fields"] is not None:
            mismatches.update(k for k, matched in row["two_active_matching_fields"].items() if not matched)
    second_checks = [check for row in analyses for check in row["second_window_and_action_checks"]]
    annotated = [c for c in second_checks if c["final_action_matches_own_second_window_explicit_plan"] is not None]
    result = {"created_at": datetime.now().astimezone().isoformat(), "status": "completed_behavior_summary",
        "source_run": str(run), "scope": {"completed_trials": 12, "semantic_draws": 4, "context_groups": 3,
            "broadcasts_read": 72, "final_actions_read": 36, "private_analysis_calls_read": 0, "new_model_calls": 0},
        "file_sha256": {str(p): sha(p) for p in (records_path, run / "plan.json", Path(__file__))},
        "reward_histogram": dict(Counter(str(r["reward"]) for r in analyses)),
        "primary_categories": dict(Counter(r["primary_category"] for r in analyses)),
        "active_count_histogram": dict(Counter(str(r["active_count"]) for r in analyses)),
        "overlapping_field_mismatch_counts_among_two_active_trials": dict(mismatches),
        "two_active_trial_denominator": sum(r["active_count"] == 2 for r in analyses),
        "broadcast_surface_summary": {
            "windows_with_competing_explicit_plans_detected": sum(w["competing_explicit_plans_detected"] for r in analyses for w in r["windows"]),
            "second_windows_with_competing_explicit_plans_detected": sum(r["windows"][1]["competing_explicit_plans_detected"] for r in analyses),
            "second_window_explicit_plans_with_comparable_own_action": len(annotated),
            "own_formal_action_different_from_second_window_explicit_plan": sum(not c["final_action_matches_own_second_window_explicit_plan"] for c in annotated),
            "not_complete_semantic_coding": True},
        "trials": analyses,
        "limits": ["Physical classification is independently calculated from the current full-information observation and final actions.",
            "Text parsing detects only an explicit unique pair/site/destination/third-waiter with no recognized conditional/negative cues; null is unclassified, not absence.",
            "Second-window matching or changed wording is descriptive evidence, not proof a particular message caused an action.",
            "A final action different from an explicit proposal is not automatically a mask error, internal-plan reversal or failure to understand a rule.",
            "No private analysis read, no inference about internal mechanism, no language-formation judgment.",
            "The same4 scenarios repeat in3 context groups; these12 rows are not12 independent semantic draws."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "behavior.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    text = ["# 一步能力控制：行为摘要", "", f"来源：{run}。覆盖完整12场、72条广播、36个最终动作；未读取私有分析。", "",
            "物理错误按实际动作和当前需求独立结算；广播部分只是保守的表面方案提取，所有原文保留供逐条核对。", "",
            "|场景|收益|活跃者|物理结果|A最终动作|B最终动作|C最终动作|", "|---|---:|---|---|---|---|---|"]
    for row in analyses:
        actions = row["actual_actions"]
        text.append(f"|{row['case_id']}|{row['reward']}|{','.join(row['active_agents']) or '无'}|{NAMES[row['primary_category']]}|{action_text(actions['A'])}|{action_text(actions['B'])}|{action_text(actions['C'])}|")
    text += ["", "错误字段可重叠，不能把其计数相加当成失败场数。需求正确但未完成物理匹配的动作不记作交付成功。", "",
             "以下广播引用中的方案或支持措辞不代表伙伴已经接受。‘与首窗方案相同’不证明因果回应；‘正式动作不同’也不证明生成约束改变了原计划。未被程序识别的文本保留为未分类。"]
    for row in analyses:
        text += ["", f"## {row['case_id']}：{NAMES[row['primary_category']]}", "",
                 f"两个活跃者的字段匹配：{row['two_active_matching_fields']}。"]
        for message in row["broadcast_evidence"]:
            text += ["", f"- 窗{message['window']}，{message['agent']}，正式call{message['formal_call']}：{message['text']}"]
        for check in row["second_window_and_action_checks"]:
            if check["final_action_matches_own_second_window_explicit_plan"] is False:
                text += ["", f"{check['agent']}的正式call{check['final_action_formal_call']}与本人第二窗可识别方案字段不同："
                         f"方案对应“{action_text(check['plan_implied_own_action'])}”，实际“{action_text(check['actual_action'])}”。"]
        for agent, proposal in row["individual_proposal_checks"].items():
            if proposal["active"] and not proposal["would_satisfy_own_need_if_transport_matched"]:
                text += ["", f"{agent}选择的物资/目的地不完全符合本人需求：物资合格={proposal['material_requirement_met']}，目的地合格={proposal['destination_requirement_met']}。"]
    text += ["", "本报告不判断内部机制或语言形成。群体与场景的重复结构、生成截断和完整提示审计应结合各自记录报告。"]
    (output / "behavior.md").write_text("\n".join(text) + "\n")
    return {"status": result["status"], "output": str(output), "primary_categories": result["primary_categories"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.output_dir), ensure_ascii=False))
