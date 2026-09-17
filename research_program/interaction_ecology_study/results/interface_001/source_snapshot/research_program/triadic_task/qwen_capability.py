"""Reviewable single-round full-information Qwen capability control.

No backend/model import occurs at import, case building, fake audit, or prepare.
Only execute instantiates the unchanged audited v3 Backend. This module neither
changes the old v3 gate nor starts a symbolic-language experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from random import Random
import sys
import time
import traceback

from . import environment as env


WORK = Path(__file__).resolve().parents[2]
ROUTE = "triadic_qwen_capability_v1"
GROUPS = (301, 302, 303)
SCENE_SEEDS = (20260916011, 20260916012, 20260916013, 20260916014)
MAX_CALLS = 216
DEFAULT_MODEL = WORK / "qwen_collect_pilot/models/Qwen3.5-9B-8bit"
MODEL_REVISION = "16daa4818c54ce5f5436f929d52542eb65bbed9d"
BACKEND = WORK / "qwen_language_v3/backend.py"
BACKEND_SHA = "0aa38e9c5a53c9f3275a8208912bfc356181839e7d338083570dd32c332c7e23"
ENVIRONMENT_SHA = "b2d49ab829edc92e77df527f5b470602c289cdf3ad7728d5e71f411c53582ef6"

RULES = (
    "这是三人共同分拣与交付的一轮任务。每轮只有一次同时提交物理动作的机会。"
    "地点S0、S1、S2、S3各放一种物资：短木材、长木材、短纤维、长纤维各出现一次，每堆至少两份。"
    "S0至S3是所有人都能共同引用的位置名称。目的地是L和R。"
    "每人的物资需求只限制一个因素：木材或纤维限制种类、长短不限；短或长限制长度、种类不限。"
    "每人另外有自己的可接受目的地，可以是仅L、仅R或两者均可。"
    "所有物资无论种类、长短都使用同一个D1匹配规则：要执行运输，必须恰好两人选运输，"
    "双方互选对方作为伙伴，选择同一个物资位置和同一个目的地，第三人等待。"
    "只有一人运输或双方不匹配，运输不执行；三人都运输则超载，全体本轮不得分。"
    "一次匹配运输直接从选中位置取两份并送到选中目的地，两人各计一份；"
    "不需要先拿取、加工、移动或再提交交付动作，也没有工具、道路事件或跨步搬运。"
    "匹配后分别检查每人的需求：若只满足其中一人的需求，该人成功，另一人不成功。"
    "团队每满足一人得0.5，最高1；目标是本轮让两人需求都满足。"
    "不预先指定谁领导、谁参与；三人能力相同。"
    "本次是完整信息能力控制：每人都能看到四处物资和三人的需求；"
    "观察分工字段只说明通常谁看哪处，在本次控制中不限制可见物资。"
    "先进行两个同步自然语言广播窗口：同一窗口全部消息收齐后一起公开，下一窗口可以回应。"
    "两个窗口结束后，三人分别选择动作再同时结算；其他人尚未执行的动作不可见。"
    "广播每次至多约50个汉字，可以沉默。广播里的提议不等于动作已经执行。"
    "每人每轮有17个本地动作编号：等待，或四位置、两目的地、两位可选伙伴的全部组合。"
    "编号分别随机打乱，只供本人操作，不能直接照抄伙伴的动作编号。"
    "每轮重新抽取需求和布局；过去的物资、需求和编号不能代替本轮观察。"
    "历史中的伙伴广播是对方说的话，不是环境保证的事实。"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stable_seed(*parts):
    key = "/".join(map(str, (ROUTE,) + parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (2**31 - 1)


def runtime_versions():
    packages = {}
    for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def need_text(need):
    names = {"wood": "木材", "fiber": "纤维", "short": "短", "long": "长"}
    criterion = ("种类为" + names[need["value"]] + "，长短不限" if need["factor"] == "kind"
                 else "长度为" + names[need["value"]] + "，种类不限")
    return criterion + "；可接受目的地：" + "或".join(need["destinations"])


def literal_observation(observation):
    return {"你是": observation["self"], "本轮你自己的需求": need_text(observation["own_need"]),
        "本轮三人的需求": {a: need_text(observation["shared_needs"][a]) for a in env.AGENTS},
        "本轮四处物资": [{"位置": r["site"], "物资": {"short": "短", "long": "长"}[r["length"]]
                          + {"wood": "木材", "fiber": "纤维"}[r["kind"]]} for r in observation["visible_materials"]],
        "通常的私人观察分工": deepcopy(observation["private_view_owners"]),
        "本次信息条件": "完整信息：四处物资和三人需求均可见"}


def action_text(action):
    if action["kind"] == "wait":
        return "等待"
    return f"与{action['partner']}互选合作，从{action['site']}取物资并运送到{action['destination']}"


def build_prompt(agent, observation, history, transcript, *, window=None, menu=None):
    require(agent in env.AGENTS and observation["self"] == agent, "Prompt owner differs")
    require(observation.get("information_control") == "full_information"
            and len(observation["visible_materials"]) == 4 and set(observation["shared_needs"]) == set(env.AGENTS),
            "This capability prompt requires explicit full information")
    data = {"你自己的历史": deepcopy(history), "本轮完整信息观察": literal_observation(observation),
            "本轮已公开广播": deepcopy(transcript)}
    if menu is None:
        require(window in (1, 2), "Exactly two communication windows")
        data["当前决策"] = {"广播窗口": window, "说明": "世界尚未结算；可以提出方案或回应已公开消息。"}
    else:
        require(window is None and len(menu) == 17, "Action stage requires complete menu")
        data["本次可选动作"] = [{"编号": m["id"], "动作": action_text(m["action"])} for m in menu]
        data["当前决策"] = "两个广播窗口已结束。独立选择一个动作；其他人的未执行动作不可见。"
    return [{"role": "system", "content": f"你是合作群体中的{agent}，伙伴是另外两位A/B/C。\n" + RULES},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]


def build_cases():
    """Only draw the four preset IID semantic states; do not search their outcomes."""
    support = env.support()
    require(len(support) == 996, "Unexpected support")
    scenarios = []
    for episode, seed in enumerate(SCENE_SEEDS, 1):
        state = env.draw_state(Random(seed), support)
        scenarios.append({"episode": episode, "scene_seed": seed, "state": asdict(state)})
    trials = []
    first_prompts = {}
    for group in GROUPS:
        for scene in scenarios:
            state = env.State(**scene["state"])
            observations = {a: env.observe(state, a, shared_needs=True, full_information=True) for a in env.AGENTS}
            menus, menu_seeds = {}, {}
            for agent in env.AGENTS:
                menu_seed = stable_seed("menu", group, scene["episode"], scene["scene_seed"], agent)
                order = list(range(17))
                Random(menu_seed).shuffle(order)
                menus[agent] = env.action_menu(agent, order)
                menu_seeds[agent] = menu_seed
            trial = {"case_id": f"g{group}_e{scene['episode']}", "group": group, **deepcopy(scene),
                     "observations": observations, "menus": menus, "menu_seeds": menu_seeds}
            trials.append(trial)
            if scene["episode"] == 1:
                first_prompts[str(group)] = {a: build_prompt(a, observations[a], [], [], window=1) for a in env.AGENTS}
    # JSON round-trip makes the prepared form identical in memory and on disk.
    return json.loads(json.dumps({"scenarios": scenarios, "trials": trials, "first_prompts": first_prompts}, ensure_ascii=False))


def _decide(backend, prompt, *, mode, label, choices=None):
    before = backend.calls
    require(before + 2 <= MAX_CALLS, "Capability call cap reached")
    seed = stable_seed("generation", label["group"], label["episode"], label["window"] or 0, label["agent"], mode)
    answer = backend.decide(deepcopy(prompt), mode=mode, choices=choices, seed=seed,
                            temperature=.7 if mode == "natural_message" else 0, label=deepcopy(label))
    require(backend.calls == before + 2, "Each decision must perform exactly two backend calls")
    if mode == "action":
        require(answer in tuple(map(str, choices)), "Formal action outside its private menu")
    return answer


def run_trial(backend, case, histories):
    before = backend.calls
    history_before = deepcopy(histories)
    transcript = []
    label = {"phase": ROUTE, "condition": "full_information_natural", "dependency": "D1",
             "group": case["group"], "episode": case["episode"], "scene_seed": case["scene_seed"], "step": 1}
    for window in (1, 2):
        # All three inputs are frozen before generating any same-window output.
        prompts = {a: build_prompt(a, case["observations"][a], histories[a], transcript, window=window) for a in env.AGENTS}
        pending = []
        for agent in env.AGENTS:
            text = _decide(backend, prompts[agent], mode="natural_message", label={**label, "agent": agent, "window": window})
            pending.append({"agent": agent, "window": window, "text": text})
        transcript.extend(pending)
    # No submitted action can appear in a partner's current input.
    action_prompts = {a: build_prompt(a, case["observations"][a], histories[a], transcript, menu=case["menus"][a])
                      for a in env.AGENTS}
    actions, selections = {}, {}
    for agent in env.AGENTS:
        choice = int(_decide(backend, action_prompts[agent], mode="action", choices=list(range(17)),
                             label={**label, "agent": agent, "window": None}))
        selected = next(m for m in case["menus"][agent] if m["id"] == choice)
        selections[agent] = deepcopy(selected)
        actions[agent] = deepcopy(selected["action"])
    outcome = env.settle(env.State(**case["state"]), actions, require_match=True)
    for agent in env.AGENTS:
        histories[agent].append({"本轮观察": deepcopy(case["observations"][agent]),
            "公开广播": deepcopy(transcript), "自己提交的动作": deepcopy(actions[agent]),
            "自己可见结果": deepcopy(outcome["individual_feedback"][agent])})
    require(backend.calls - before == 18, "One trial must contain eighteen actual calls")
    return {"case_id": case["case_id"], **label, "state": deepcopy(case["state"]),
            "observations": deepcopy(case["observations"]), "menus": deepcopy(case["menus"]),
            "private_histories_before": history_before, "messages": transcript, "actions": actions,
            "selections": selections, "outcome": outcome,
            "history_lengths_after": {a: len(histories[a]) for a in env.AGENTS},
            "call_range": [before + 1, backend.calls]}


def capability_gate(records, prepared):
    require(len(records) == len(prepared["trials"]) == 12, "Gate requires all twelve complete trials")
    replayed = []
    for index, (row, case) in enumerate(zip(records, prepared["trials"])):
        require(row["case_id"] == case["case_id"] and row["state"] == case["state"]
                and row["group"] == case["group"] and row["episode"] == case["episode"]
                and row["scene_seed"] == case["scene_seed"], "Gate trial identity differs")
        require(row["observations"] == case["observations"] and row["menus"] == case["menus"], "Gate inputs differ")
        require(row["call_range"] == [18 * index + 1, 18 * (index + 1)], "Gate call range differs")
        require(len(row["messages"]) == 6, "Gate missing broadcasts")
        for agent in env.AGENTS:
            selected = row["selections"][agent]
            require(selected in case["menus"][agent] and row["actions"][agent] == selected["action"], "Gate action/menu mismatch")
        actual = env.settle(env.State(**case["state"]), row["actions"], require_match=True)
        require(actual == row["outcome"], "Gate settlement differs")
        replayed.append(actual)
    passed = all(outcome["reward"] == 1 and outcome["full_success"] is True for outcome in replayed)
    return {"rule": "all_twelve_full_success", "passed": passed, "full_success_trials": sum(o["full_success"] for o in replayed),
            "trials": 12, "new_task_only": True, "modifies_original_v3_gate": False, "starts_symbolic_experiment": False}


def run_cases(backend, prepared, *, on_trial=lambda record: None):
    require(backend.calls == 0, "A fresh backend log is required")
    memories = {g: {a: [] for a in env.AGENTS} for g in GROUPS}
    records = []
    for case in prepared["trials"]:
        row = run_trial(backend, case, memories[case["group"]])
        records.append(row)
        on_trial(deepcopy(row))
    gate = capability_gate(records, prepared)
    require(backend.calls == MAX_CALLS, "Completed control must contain exactly216 calls")
    return {"records": records, "histories_after": deepcopy(memories), "gate": gate, "model_calls": backend.calls}


class FakeBackend:
    """Deterministic audit double; failure on every trial tests continuation."""
    def __init__(self):
        self.calls, self.decisions = 0, []

    def decide(self, messages, *, mode, label, seed, temperature, choices=None):
        self.decisions.append({"messages": deepcopy(messages), "mode": mode, "label": deepcopy(label),
                               "seed": seed, "temperature": temperature, "choices": deepcopy(choices)})
        self.calls += 2
        if mode == "natural_message":
            return f"假广播g{label['group']}e{label['episode']}w{label['window']}{label['agent']}"
        data = json.loads(messages[-1]["content"])
        return str(next(m["编号"] for m in data["本次可选动作"] if m["动作"] == "等待"))


def fake_audit(prepared=None):
    prepared = build_cases() if prepared is None else prepared
    original = deepcopy(prepared)
    backend = FakeBackend()
    result = run_cases(backend, prepared)
    require(prepared == original, "Runner modified prepared cases")
    require(result["gate"]["passed"] is False and len(result["records"]) == 12, "A failed trial stopped later cases")
    require(len(backend.decisions) == 108 and backend.calls == 216, "Fake call accounting differs")
    require(len({decision["seed"] for decision in backend.decisions}) == 108, "Decision seed collision")
    cursor = 0
    for row, case in zip(result["records"], prepared["trials"]):
        for window in (1, 2, None):
            for agent in env.AGENTS:
                decision = backend.decisions[cursor]
                cursor += 1
                mode = "action" if window is None else "natural_message"
                require(decision["mode"] == mode and decision["label"]["agent"] == agent
                        and decision["label"]["window"] == window and decision["label"]["group"] == case["group"]
                        and decision["label"]["episode"] == case["episode"], "Decision label/mode differs")
                require(decision["seed"] == stable_seed("generation", case["group"], case["episode"], window or 0, agent, mode)
                        and decision["temperature"] == (0 if window is None else .7), "Seed/temperature differs")
                history = row["private_histories_before"][agent]
                require(len(history) == case["episode"] - 1, "History crossed group or scene boundary")
                visible = [] if window == 1 else row["messages"][:3] if window == 2 else row["messages"]
                expected = build_prompt(agent, case["observations"][agent], history, visible,
                                        window=window, menu=case["menus"][agent] if window is None else None)
                require(decision["messages"] == expected, "Same-window/action information leaked")
                user_data = json.loads(expected[-1]["content"])
                require(set(user_data) == {"你自己的历史", "本轮完整信息观察", "本轮已公开广播", "当前决策"}
                        | ({"本次可选动作"} if window is None else set()), "Unexpected model input fields")
                if case["episode"] == 1 and window == 1:
                    require(expected == prepared["first_prompts"][str(case["group"])][agent], "First prompt differs")
                text = json.dumps(history, ensure_ascii=False)
                for other in GROUPS:
                    if other != case["group"]:
                        require(f"假广播g{other}" not in text, "Another group entered history")
    return {"status": "passed", "real_model_calls": 0, "simulated_calls": 216, "decisions": 108,
            "unique_decision_seeds": 108, "analysis_and_formal_share_seed": True,
            "trials": 12, "windows_per_trial": 2, "all_failures_retained": True,
            "same_window_outputs_hidden": True, "current_actions_hidden_until_settlement": True,
            "group_histories_isolated": True, "within_group_history_lengths": [0, 1, 2, 3],
            "prepared_inputs_unmodified": True, "first_prompts_match": True}


def prepare(output_dir, model_dir=DEFAULT_MODEL):
    """Call only after environment review ends; no freeze is performed at import."""
    output, model = Path(output_dir).resolve(), Path(model_dir).resolve()
    require(not output.exists(), "Never overwrite a prepared or completed capability run")
    require(sha(BACKEND) == BACKEND_SHA, "Frozen v3 backend differs")
    environment_path = Path(env.__file__).resolve()
    audit_path = environment_path.parent / "audit_environment.json"
    environment_audit = json.loads(audit_path.read_text())
    require(sha(environment_path) == ENVIRONMENT_SHA
            and environment_audit["status"] == "passed_independent_environment_audit"
            and environment_audit["source_sha256"][str(environment_path)] == ENVIRONMENT_SHA,
            "Independent environment audit is missing or stale")
    receipt_path = WORK / "qwen_collect_pilot/model_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["revision"] == MODEL_REVISION and Path(receipt["local_path"]).resolve() == model, "Model provenance differs")
    model_metadata, weight_files = [], []
    for item in receipt["files"]:
        file = model / item["name"]
        require(file.is_file() and file.stat().st_size == item["bytes"], "Model file missing or byte size differs")
        if item["name"].endswith(".safetensors"):
            weight_files.append({"file": str(file), "bytes": item["bytes"], "download_receipt_sha256": item["sha256"]})
        else:
            if item["sha256"] is not None:
                require(sha(file) == item["sha256"], "Model metadata differs from download receipt")
            model_metadata.append(file)
    prepared = build_cases()
    audit = fake_audit(prepared)
    source_paths = [Path(__file__).resolve(), environment_path, BACKEND, audit_path,
                    environment_path.parent / "audit_environment.py", receipt_path,
                    environment_path.parent / "__init__.py", BACKEND.parent / "__init__.py"] + model_metadata
    if (WORK / "research_program/__init__.py").is_file():
        source_paths.append(WORK / "research_program/__init__.py")
    sources = {str(p): sha(p) for p in source_paths}
    output.mkdir(parents=True, exist_ok=False)
    for source in source_paths:
        target = output / "source_snapshot" / source.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    new_json(output / "prepared_cases.json", prepared)
    new_json(output / "fake_backend_audit.json", audit)
    plan = {"status_at_freeze": "prepared_not_executed", "created_at": datetime.now().astimezone().isoformat(),
        "route": ROUTE, "groups": list(GROUPS), "scene_seeds": list(SCENE_SEEDS), "trials": 12,
        "independent_semantic_draws": 4, "shared_scenarios_across_groups": True,
        "model": str(model), "model_revision": MODEL_REVISION, "sources_sha256": sources,
        "prepare_runtime": runtime_versions(),
        "weight_files_size_checked": weight_files,
        "prepared_cases_sha256": sha(output / "prepared_cases.json"), "fake_audit_sha256": sha(output / "fake_backend_audit.json"),
        "scenario_sampling": "Four preselected seed draws, uniform996 demand support and uniform24 layouts, same four states across3 context groups; no outcome-based selection",
        "execution_order": "group301/302/303 then episode1/2/3/4; A/B/C call order within synchronous windows",
        "condition": "full_information_natural", "dependency": "D1", "physical_rounds_per_trial": 1,
        "synchronous_windows": 2, "actions_per_agent": 17, "max_backend_calls": MAX_CALLS,
        "private_analysis_max_tokens": 256, "private_analysis_temperature": 0,
        "natural_message_max_tokens": 96, "natural_message_temperature": .7,
        "action_temperature": 0, "native_thinking": False, "fresh_cache_every_call": True,
        "unique_decision_seeds": audit["unique_decision_seeds"],
        "two_stage_seed_policy": "108 distinct decision seeds; the unchanged Backend uses the same seed for analysis and formal within each decision. 216 calls do not mean216 distinct seeds.",
        "history": "Own raw full observation, own submitted action and individual feedback, all public broadcasts; no private analysis. Fresh world each scene, own history retained within group and isolated across groups.",
        "gate": "All12 completed trials have replayed reward1; failure never stops remaining trials or changes prompts/budget",
        "new_task_gate_only": True, "modifies_original_v3_gate": False, "starts_symbolic_experiment": False,
        "scope": "One-action complete-information natural-language capability; no multistep planning, symbol emergence or original collection-task pass claim",
        "weight_files_fully_hashed_here": False, "prepare_real_model_calls": 0}
    new_json(output / "plan.json", plan)
    new_json(output / "freeze.json", {"plan_sha256": sha(output / "plan.json"), "status": "prepared_not_executed"})
    return {"status": "prepared_not_executed", "output": str(output), "fake_audit": audit}


def verify_prepared(output_dir):
    output = Path(output_dir).resolve()
    plan = json.loads((output / "plan.json").read_text())
    frozen = json.loads((output / "freeze.json").read_text())
    require(sha(output / "plan.json") == frozen["plan_sha256"], "Frozen plan differs")
    for name, digest in plan["sources_sha256"].items():
        require(sha(name) == digest == sha(output / "source_snapshot" / Path(name).relative_to(WORK)), "Source changed: " + name)
    require(sha(output / "prepared_cases.json") == plan["prepared_cases_sha256"]
            and sha(output / "fake_backend_audit.json") == plan["fake_audit_sha256"], "Prepared input or audit changed")
    require(all(Path(item["file"]).stat().st_size == item["bytes"] for item in plan["weight_files_size_checked"]),
            "Frozen weight byte size differs")
    prepared = json.loads((output / "prepared_cases.json").read_text())
    require(prepared == build_cases(), "Scenario/prompt reconstruction differs")
    require(plan["groups"] == list(GROUPS) and plan["scene_seeds"] == list(SCENE_SEEDS)
            and plan["max_backend_calls"] == MAX_CALLS, "Fixed grid or budget differs")
    return plan, prepared


def execute(output_dir):
    plan, prepared = verify_prepared(output_dir)
    output = Path(output_dir).resolve()
    execution = output / "execution"
    execution.mkdir(exist_ok=False)
    new_json(execution / "started.json", {"started_at": datetime.now().astimezone().isoformat(),
                                          "plan_sha256": sha(output / "plan.json"), "execute_runtime": runtime_versions()})
    backend = None
    completed = []
    started = time.perf_counter()
    try:
        # The only backend/model import and instantiation path in this module.
        from qwen_language_v3.backend import Backend
        backend = Backend(Path(plan["model"]), execution / "inference.jsonl")
        with (execution / "trials.jsonl").open("x", encoding="utf-8") as log:
            def save_trial(row):
                completed.append(row)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps({"event": "trial_completed", "case_id": row["case_id"],
                                  "reward": row["outcome"]["reward"], "calls": backend.calls}), flush=True)
            result = run_cases(backend, prepared, on_trial=save_trial)
        new_json(execution / "histories_after.json", result["histories_after"])
        new_json(execution / "results.json", {"status": "completed", "plan_sha256": sha(output / "plan.json"),
            "model_calls": backend.calls, "elapsed_seconds": time.perf_counter() - started,
            "execute_runtime": runtime_versions(),
            "gate": result["gate"], "by_group": [{"group": g, "rewards": [r["outcome"]["reward"] for r in completed if r["group"] == g]} for g in GROUPS],
            "trials_sha256": sha(execution / "trials.jsonl"), "symbolic_started": False, "original_v3_gate_modified": False})
        new_json(execution / "status.json", {"status": "completed", "trials": 12, "calls": backend.calls, "gate": result["gate"]})
        return {"status": "completed", "calls": backend.calls, "gate": result["gate"]}
    except BaseException as error:
        new_json(execution / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                                              "completed_trials": len(completed), "automatic_retry": False})
        new_json(execution / "status.json", {"status": "failed", "completed_trials": len(completed), "calls": backend.calls if backend else 0})
        raise
    finally:
        if backend is not None:
            new_json(execution / "backend_stats.json", backend.stats())
            backend.log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit-fake", "prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command != "audit-fake" and args.out is None:
        parser.error("--out is required")
    result = fake_audit() if args.command == "audit-fake" else prepare(args.out) if args.command == "prepare" else execute(args.out) if args.command == "execute" else {"status": "verified", "trials": len(verify_prepared(args.out)[1]["trials"])}
    print(json.dumps(result, ensure_ascii=False))
