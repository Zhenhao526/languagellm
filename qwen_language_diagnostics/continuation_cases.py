"""Reconstruct the frozen three-person history continuation without inference.

The runner may use ``state``, an arm's ``histories`` and the original v3 prompt
builder. ``first_prompts`` are model-facing first-window inputs. All source
records, calls, decisions and checks are researcher-only reproduction evidence;
in particular, source step 9 broadcasts must never initialize a new transcript.
Import performs no I/O. Construction reads frozen files and replays World only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


WORK = Path(__file__).resolve().parents[1]
SOURCE_RUN = WORK / "qwen_language_v3/results/20260915_152343/capability_full_information"
MAP_RUN = WORK / "qwen_language_diagnostics/results/20260915_MAP_01"
SOURCE_GROUP, SOURCE_EPISODE, WORLD_SEED, VARIANT = 17, 1, 92015, 0
SOURCE_STEPS = (9, 10, 11)
ANALYSIS_PREFIX = (
    "先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。"
    "他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。"
)
ANALYSIS_SUFFIX = {
    "action": "核对可选动作编号与动作描述；只依据自己的可见信息选择。",
    "natural_message": "考虑此次要表达什么、是否需回应伙伴，或是否沉默。",
}
FINAL_REQUEST = {
    "action": "只输出你选择的一个本地动作编号，不要解释。",
    "natural_message": "现在输出正式广播消息；最多约50个汉字，也可以沉默。不输出私有分析。",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, hashes, *, jsonl=False):
    path = path.resolve()
    raw = path.read_bytes()
    hashes[str(path)] = _sha(raw)
    if jsonl:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def _source_label(row):
    label = row.get("label", {})
    return all(label.get(key) == value for key, value in {
        "phase": "calibration", "condition": "full_information",
        "group": SOURCE_GROUP, "episode": SOURCE_EPISODE, "seed": WORLD_SEED,
    }.items())


def _check_pair(index, *, step, window, agent, mode, seed, prompt, expected_output):
    rows = [index[(step, window, agent, stage)] for stage in ("analysis", "formal")]
    analysis, formal = rows
    context = f"step {step}, window {window}, {agent}, {mode}"
    _require(analysis["mode"] == "private_analysis" and formal["mode"] == mode,
             f"Unexpected modes: {context}")
    _require(formal["call"] == analysis["call"] + 1,
             f"Nonconsecutive source stages: {context}")
    _require(analysis["messages"] == prompt + [{
        "role": "user", "content": ANALYSIS_PREFIX + ANALYSIS_SUFFIX[mode],
    }], f"Rebuilt analysis input differs from source: {context}")
    _require(formal["messages"] == prompt + [
        {"role": "assistant", "content": analysis["output"]},
        {"role": "user", "content": FINAL_REQUEST[mode]},
    ], f"Rebuilt formal input differs from source: {context}")
    _require(formal["output"] == expected_output,
             f"Formal output differs from recorded message/action: {context}")
    for row, temperature in ((analysis, 0), (formal, .7 if mode == "natural_message" else 0)):
        _require(row["seed"] == seed and row["temperature"] == temperature,
                 f"Source seed/temperature mismatch: {context}")
        _require(row["valid"] is True and row["fresh_cache"] is True
                 and row["native_thinking"] is False and row["symbol_limit"] is None,
                 f"Unexpected source inference flags: {context}")
    return {
        "step": step, "window": window, "agent": agent, "mode": mode,
        "seed": seed, "prompt": deepcopy(prompt),
        "call_ids": {"analysis": analysis["call"], "formal": formal["call"]},
    }


def build_continuation_cases(source_run=None):
    """Return two initial histories and exact source steps/calls for comparison.

    ``arms`` only vary in initial ``histories``. The runner must independently
    clone ``state`` for each arm, use newly generated broadcasts, and append
    each arm's new experiences normally. It must not read the reference calls
    as new model output, except in a separately labelled pure replay test.

    Source inference seeds are retained in every call and source decision.
    The live runner uses frozen ``inference_seed(17, 1, absolute_step,
    window_or_0, agent, 'message' or 'action')``. No new deadline, diagnostic
    statement, target action or role assignment is added to any prompt.
    """
    from qwen_language_diagnostics.history_cases import _check_sources
    from qwen_language_v3.agents import build_prompt, remember
    from qwen_language_v3.environment import AGENTS, RULES, World
    from qwen_language_v3.protocol import run_natural_communication
    from qwen_language_v3.run import inference_seed

    run_dir = Path(source_run).resolve() if source_run is not None else SOURCE_RUN
    hashes = {}
    manifest = _read(run_dir / "manifest.json", hashes)
    frozen = _read(run_dir / "core_source_frozen.json", hashes)
    checks = _check_sources(run_dir, WORK / "qwen_language_v3", manifest, frozen)
    # Preserve both the seven environment/runtime sources and all fourteen
    # sources already frozen by MAP. None is edited or executed as a script.
    map_manifest = _read(MAP_RUN / "manifest.json", hashes)
    _require(len(map_manifest["source_sha256"]) == 14, "Unexpected MAP frozen source set")
    for relative, expected in map_manifest["source_sha256"].items():
        for path in (WORK / relative, MAP_RUN / "code_snapshot" / relative):
            raw = path.read_bytes()
            hashes[str(path.resolve())] = _sha(raw)
            _require(_sha(raw) == expected, f"Frozen MAP source mismatch: {path}")
    _require(manifest["phase"] == "calibration"
             and manifest["conditions"] == ["full_information"]
             and manifest["groups"] == [SOURCE_GROUP]
             and manifest["history_mode"] == "independent_episodes"
             and manifest["scenarios"][0] == {
                 "episode": SOURCE_EPISODE, "seed": WORLD_SEED, "variant": VARIANT},
             "Source does not match the registered first v3 capability episode")
    status = _read(run_dir / "status.json", hashes)
    _require(status["status"] == "completed", "Source run is unfinished")
    checkpoint = _read(run_dir / "checkpoint_calibration_full_information_17_1_before.json", hashes)
    _require(checkpoint == {a: [] for a in AGENTS}, "Source episode checkpoint is not empty")
    all_records = _read(run_dir / "steps.jsonl", hashes, jsonl=True)
    all_calls = _read(run_dir / "inference.jsonl", hashes, jsonl=True)
    for filename in ("manifest.json", "steps.jsonl", "inference.jsonl"):
        _require(hashes[str((run_dir / filename).resolve())]
                 == map_manifest["source_records_sha256"][filename],
                 f"Source data differs from the frozen MAP provenance: {filename}")

    records = [r for r in all_records if all(r.get(k) == v for k, v in {
        "phase": "calibration", "condition": "full_information", "group": SOURCE_GROUP,
        "episode": SOURCE_EPISODE, "seed": WORLD_SEED,
    }.items())]
    _require([r["step"] for r in records] == list(range(1, 13)),
             "Expected exactly the twelve ordered source episode steps")
    calls = [r for r in all_calls if _source_label(r)
             and r["label"].get("step") in SOURCE_STEPS]
    _require([r["call"] for r in calls] == list(range(241, 331)),
             "Expected source calls 241 through 330 in order")
    index = {}
    for row in calls:
        label = row["label"]
        key = tuple(label[k] for k in ("step", "window", "agent", "stage"))
        _require(key not in index, "Duplicate source inference decision/stage")
        index[key] = row

    world = World(WORLD_SEED, VARIANT)
    histories = deepcopy(checkpoint)
    decisions = []
    for record in records[:11]:
        step = record["step"]
        _require(record["state_before"] == world.state_dict(),
                 f"Source pre-state does not replay at step {step}")
        for agent in AGENTS:
            _require(record["observations"][agent] == world.full_information_observe(agent),
                     f"Source observation differs at step {step}, {agent}")
            _require(record["menus"][agent] == world.action_menu(agent),
                     f"Source action menu differs at step {step}, {agent}")
        _require([(m["window"], m["sender"]) for m in record["messages"]]
                 == [(w, a) for w in range(1, 5) for a in AGENTS]
                 and all(m["step"] == step for m in record["messages"]),
                 f"Source broadcasts are not in frozen window/agent order: step {step}")
        if step == SOURCE_STEPS[0]:
            state = world.state_dict()
            initial_histories = deepcopy(histories)
            _require(world.t == 8 and world.max_steps == 12 and world.status == "running"
                     and world.score == 0, "S0 clock, deadline or task progress changed")

        if step in SOURCE_STEPS:
            message_index = {(m["window"], m["sender"]): m for m in record["messages"]}

            def original_message(agent, window, remaining, visible):
                _require(remaining is None, "Natural-language callback unexpectedly has a symbol budget")
                _require(visible == [m for m in record["messages"] if m["window"] < window],
                         "Current same-window or later broadcasts leaked into source reconstruction")
                prompt = build_prompt(agent, rules=RULES, observation=record["observations"][agent],
                                      history=histories[agent], transcript=visible,
                                      condition="full_information", window=window, remaining=None)
                output = message_index[(window, agent)]["text"]
                decisions.append(_check_pair(
                    index, step=step, window=window, agent=agent, mode="natural_message",
                    seed=inference_seed(SOURCE_GROUP, SOURCE_EPISODE, step, window, agent, "message"),
                    prompt=prompt, expected_output=output))
                return output

            communication = run_natural_communication(original_message, step=step)
            _require(communication["messages"] == record["messages"]
                     and communication["usage"] == record["channel_usage"],
                     f"Source communication did not reconstruct at step {step}")
            for agent in AGENTS:
                prompt = build_prompt(agent, rules=RULES, observation=record["observations"][agent],
                                      history=histories[agent], transcript=communication["transcripts"][agent],
                                      condition="full_information", menu=record["menus"][agent])
                selection = record["selections"][agent]
                _require(selection in record["menus"][agent]
                         and selection["action"] == record["actions"][agent],
                         f"Source local action selection differs at step {step}, {agent}")
                decisions.append(_check_pair(
                    index, step=step, window=None, agent=agent, mode="action",
                    seed=inference_seed(SOURCE_GROUP, SOURCE_EPISODE, step, 0, agent, "action"),
                    prompt=prompt, expected_output=str(selection["id"])))

        feedback = world.step(deepcopy(record["actions"]))
        _require(feedback == record["feedback"] and world.state_dict() == record["state_after"]
                 and world.score == record["score"], f"Source settlement differs at step {step}")
        _require(not world.done, f"Source prefix ended unexpectedly at step {step}")
        remember(histories, observations=record["observations"], messages=record["messages"],
                 actions=record["actions"], feedback=record["feedback"], episode_end=None)

    _require(len(decisions) == 45 and world.t == 11 and world.status == "running",
             "Reference continuation is not 45 decisions ending before the original deadline")
    for owner in AGENTS:
        _require(len(initial_histories[owner]) == 8, "Wrong initial private history length")
        for history, record in zip(initial_histories[owner], records[:8]):
            _require(history == {
                "观察": record["observations"][owner], "公开广播": record["messages"],
                "自己执行的动作": record["actions"][owner], "自己可见结果": record["feedback"][owner],
            }, "A history contains unfinished, inferred or non-owner private material")

    arms = [
        {"arm_id": "original_history", "histories": deepcopy(initial_histories)},
        {"arm_id": "reset_history", "histories": {a: [] for a in AGENTS}},
    ]
    start_world = World.from_state_dict(state)
    first_prompts = {
        arm["arm_id"]: {a: build_prompt(
            a, rules=RULES, observation=start_world.full_information_observe(a),
            history=arm["histories"][a], transcript=[], condition="full_information",
            window=1, remaining=None) for a in AGENTS}
        for arm in arms
    }
    for agent in AGENTS:
        old = first_prompts["original_history"][agent]
        reset = first_prompts["reset_history"][agent]
        old_data, reset_data = json.loads(old[1]["content"]), json.loads(reset[1]["content"])
        _require(old[0] == reset[0] and old_data["你自己的历史"] == initial_histories[agent]
                 and reset_data["你自己的历史"] == [], "Incorrect first-window history intervention")
        old_data["你自己的历史"] = []
        _require(old_data == reset_data and old_data["当前已可见广播"] == [],
                 "A first-window field other than initial history changed")
        source_first = index[(9, 1, agent, "analysis")]["messages"][:-1]
        _require(old == source_first, "Original arm's first prompt is not the exact source prompt")

    checks.update({
        "source_run_dir": str(run_dir), "map_freeze_source_count": 14,
        "checkpoint_all_agents_empty": True, "replayed_steps": list(range(1, 12)),
        "initial_history_lengths": {a: len(initial_histories[a]) for a in AGENTS},
        "source_steps": list(SOURCE_STEPS), "source_call_ids": list(range(241, 331)),
        "source_decisions_verified": len(decisions), "source_inference_inputs_verified": len(calls),
        "source_seeds_and_generation_settings_exact": True,
        "source_observations_menus_actions_feedback_states_exact": True,
        "first_prompts_original_source_exact": True,
        "only_changed_initial_prompt_field": "你自己的历史",
        "initial_current_broadcasts": {a: [] for a in AGENTS},
        "last_step_feedback_in_current_observation_preserved": True,
        "source_analysis_used_for_history_reconstruction": False,
        "initial_t": 8, "original_deadline": 12, "reference_end_t": 11,
        "reference_end_status": world.status, "reference_end_score": world.score,
        "new_inference_calls": 0, "model_weights_loaded": False,
    })
    result = {
        "state": deepcopy(state), "arms": arms,
        "source_records": deepcopy([r for r in records if r["step"] in SOURCE_STEPS]),
        "source_calls": deepcopy(calls), "source_decisions": decisions,
        "source_files_sha256": hashes, "first_prompts": first_prompts,
        "source_checks": checks,
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
