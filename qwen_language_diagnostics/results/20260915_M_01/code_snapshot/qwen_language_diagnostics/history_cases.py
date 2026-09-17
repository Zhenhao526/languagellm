"""Read-only reconstruction of the two preregistered S diagnostic inputs.

Only ``prompt`` or ``empty_history_prompt`` is to be sent to the model.
``original_record`` and the original inference rows are researcher evidence,
not additional prompt material.  Importing this module performs no I/O and
does not import an inference runtime or load model weights.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


CASE_SPECS = (("S_B_step2", "B", 2), ("S_C_step9", "C", 9))
CORE_FILES = (
    "environment.py", "agents.py", "backend.py", "protocol.py", "run.py",
    "probe.py", "execute_locked_pipeline.py",
)
ANALYSIS_REQUEST = (
    "先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。"
    "他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。"
    "核对可选动作编号与动作描述；只依据自己的可见信息选择。"
)
FINAL_REQUEST = "只输出你选择的一个本地动作编号，不要解释。"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, hashes, *, jsonl=False):
    raw = path.read_bytes()
    hashes[path.name] = _sha(raw)
    if jsonl:
        return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return json.loads(raw)


def _check_sources(run_dir, source_root, manifest, frozen):
    expected = frozen["source_sha256"]
    _require(set(expected) == set(CORE_FILES), "Unexpected frozen core file set")
    checked = {}
    for filename in CORE_FILES:
        current = _sha((source_root / filename).read_bytes())
        snapshot = _sha((run_dir / "code_snapshot" / filename).read_bytes())
        _require(current == snapshot == expected[filename]
                 == manifest["source_sha256"][filename],
                 f"Frozen source mismatch: {filename}")
        checked[filename] = current
    config_hash = _sha((run_dir / "experiment_locked.json").read_bytes())
    _require(config_hash == frozen["experiment_sha256"], "Run configuration digest mismatch")
    return {"core_sha256": checked, "run_config_sha256": config_hash}


def _same_label(row, *, episode, step, agent):
    label = row.get("label", {})
    return all(label.get(key) == value for key, value in {
        "phase": "calibration", "condition": "full_information", "group": 17,
        "episode": episode, "seed": 92015, "step": step, "agent": agent,
    }.items()) and "window" in label and label["window"] is None


def _action_pair(inferences, *, episode, step, agent, prompt, seed, record):
    rows = [row for row in inferences
            if _same_label(row, episode=episode, step=step, agent=agent)]
    _require(len(rows) == 2, f"Expected exactly one action pair for {agent} step {step}")
    analysis = next((r for r in rows if r["label"].get("stage") == "analysis"), None)
    formal = next((r for r in rows if r["label"].get("stage") == "formal"), None)
    _require(analysis is not None and formal is not None, "Missing analysis/formal stage")
    _require(analysis["mode"] == "private_analysis" and formal["mode"] == "action",
             "Unexpected action pair modes")
    _require(formal["call"] == analysis["call"] + 1, "Action pair calls are not consecutive")
    _require(analysis["messages"] == prompt + [{"role": "user", "content": ANALYSIS_REQUEST}],
             "Rebuilt prompt does not exactly match the original analysis input")
    _require(formal["messages"] == prompt + [
        {"role": "assistant", "content": analysis["output"]},
        {"role": "user", "content": FINAL_REQUEST},
    ], "Original formal input does not contain exactly its own same-decision analysis")
    for row in (analysis, formal):
        _require(row["seed"] == seed, "Inference seed mismatch")
        _require(row["temperature"] == 0, "Unexpected action/analysis temperature")
        _require(row["valid"] is True and row["fresh_cache"] is True
                 and row["native_thinking"] is False, "Unexpected backend validity/cache flags")
    candidates = [m for m in record["menus"][agent] if str(m["id"]) == formal["output"]]
    _require(len(candidates) == 1, "Original formal output is not a unique local option")
    _require(candidates[0] == record["selections"][agent]
             and candidates[0]["action"] == record["actions"][agent],
             "Original formal option differs from the recorded executed action")
    return analysis, formal


def _reconstruct_case(spec, *, run_dir, rows, inferences, checkpoint, manifest,
                      source_checks, hashes):
    # These imports define pure Python functions/classes. Backend's model/runtime
    # imports are confined to Backend.__init__/infer; neither is called here.
    from qwen_language_v3.agents import build_prompt, remember
    from qwen_language_v3.environment import AGENTS, RULES, World
    from qwen_language_v3.run import inference_seed

    case_id, agent, target_step = spec
    history = deepcopy(checkpoint)
    prefix = [r for r in rows if r["step"] <= target_step]
    _require([r["step"] for r in prefix] == list(range(1, target_step + 1)),
             f"Missing, duplicate, or unordered prefix for {case_id}")
    world = World(seed=92015, variant=0)
    prior_steps = []
    for row in prefix:
        _require(row["state_before"] == world.state_dict(), "State prefix does not replay exactly")
        for owner in AGENTS:
            _require(row["observations"][owner] == world.full_information_observe(owner),
                     f"Observation mismatch at step {row['step']} owner {owner}")
            _require(row["menus"][owner] == world.action_menu(owner),
                     f"Menu mismatch at step {row['step']} owner {owner}")
        _require([(m["window"], m["sender"]) for m in row["messages"]]
                 == [(window, owner) for window in range(1, 5) for owner in AGENTS]
                 and all(m["step"] == row["step"] for m in row["messages"]),
                 "Current or previous broadcast order/step differs from the frozen protocol")
        if row["step"] == target_step:
            record = row
            break
        feedback = world.step(deepcopy(row["actions"]))
        _require(feedback == row["feedback"] and world.state_dict() == row["state_after"]
                 and world.score == row["score"], "Prior actions/feedback do not replay exactly")
        _require(not world.done, "History prefix continued past a completed episode")
        remember(history, observations=row["observations"], messages=row["messages"],
                 actions=row["actions"], feedback=row["feedback"], episode_end=None)
        prior_steps.append(row["step"])

    probe_name = f"probe_calibration_full_information_17_1_{target_step}.json"
    probe_path = run_dir / probe_name
    probe_equal = None
    if target_step == 2 or probe_path.exists():
        probe = _read(probe_path, hashes)
        _require(probe == {**record, "private_histories_before_step": history},
                 "Reconstructed case/history differs from retained frozen probe")
        probe_equal = True

    # Frozen remember stores only this owner's completed actions and feedback.
    # Other owners' current actions stay in researcher evidence, never in prompt.
    for owner in AGENTS:
        _require(len(history[owner]) == target_step - 1, "Wrong history prefix length")
        for old, raw in zip(history[owner], prefix[:-1]):
            _require(set(old) == {"观察", "公开广播", "自己执行的动作", "自己可见结果"}
                     and old["自己执行的动作"] == raw["actions"][owner]
                     and old["自己可见结果"] == raw["feedback"][owner],
                     "History includes non-owner, unfinished, or derived records")

    prompt = build_prompt(agent, rules=RULES, observation=record["observations"][agent],
                          history=history[agent], transcript=record["messages"],
                          condition=record["condition"], menu=record["menus"][agent])
    empty_prompt = build_prompt(agent, rules=RULES, observation=record["observations"][agent],
                                history=[], transcript=record["messages"],
                                condition=record["condition"], menu=record["menus"][agent])
    _require(prompt[0] == empty_prompt[0], "History intervention changed the system message")
    original_data, empty_data = json.loads(prompt[1]["content"]), json.loads(empty_prompt[1]["content"])
    _require(original_data["你自己的历史"] == history[agent]
             and empty_data["你自己的历史"] == [], "Incorrect history arm contents")
    original_data["你自己的历史"] = []
    _require(original_data == empty_data, "History intervention changed another input field")
    seed = inference_seed(17, 1, target_step, 0, agent, "action")
    analysis, formal = _action_pair(inferences, episode=1, step=target_step, agent=agent,
                                    prompt=prompt, seed=seed, record=record)
    checks = deepcopy(source_checks)
    checks.update({
        "checkpoint_all_agents_empty": True,
        "prefix_steps_replayed": prior_steps,
        "history_owner_counts": {a: len(history[a]) for a in AGENTS},
        "frozen_probe_equal": probe_equal,
        "frozen_probe_path": str(probe_path) if probe_equal else None,
        "observation_and_menu_match_state_before": True,
        "original_analysis_prompt_exact": True,
        "formal_prompt_has_same_decision_analysis_only": True,
        "inference_seed_matches_original_pair": True,
        "original_formal_matches_recorded_action": True,
        "only_changed_prompt_field": "你自己的历史",
        "current_broadcasts_unchanged": True,
        "other_agents_current_actions_in_prompt": False,
        "analysis_used_for_history_reconstruction": False,
        "model_weights_loaded": False,
        "new_inference_calls": 0,
    })
    return {
        "case_id": case_id, "phase": "calibration", "condition": "full_information",
        "group": 17, "episode": 1, "step": target_step, "agent": agent,
        "seed": seed, "world_seed": 92015, "variant": 0,
        "model": manifest["model"], "model_commit": manifest["model_commit"],
        "history": deepcopy(history[agent]), "prompt": prompt,
        "empty_history_prompt": empty_prompt,
        "choices": [item["id"] for item in record["menus"][agent]],
        "original_record": deepcopy(record),
        "original_analysis": deepcopy(analysis), "original_formal": deepcopy(formal),
        "call_ids": {"analysis": analysis["call"], "formal": formal["call"]},
        "source_checks": checks,
    }


def load_history_cases(run_dir=None):
    """Return the fixed B/step2 and C/step9 cases, or raise on provenance mismatch.

    The only model-facing values are the two-message ``prompt`` (original
    history) and ``empty_history_prompt``. ``seed`` is the paired inference
    seed, while ``world_seed`` identifies the physical scene. Outputs include
    full original records for separate one-step counterfactual settlement.
    No source/result file is written and no inference is performed.
    """
    source_root = Path(__file__).resolve().parents[1] / "qwen_language_v3"
    run_dir = (Path(run_dir).resolve() if run_dir is not None else
               source_root / "results" / "20260915_152343" / "capability_full_information")
    hashes = {}
    manifest = _read(run_dir / "manifest.json", hashes)
    frozen = _read(run_dir / "core_source_frozen.json", hashes)
    source_checks = _check_sources(run_dir, source_root, manifest, frozen)
    _require(manifest["phase"] == "calibration"
             and manifest["conditions"] == ["full_information"]
             and manifest["groups"] == [17]
             and manifest["history_mode"] == "independent_episodes"
             and manifest["scenarios"][0] == {"episode": 1, "seed": 92015, "variant": 0},
             "Run does not match the preregistered S source scene")
    status = _read(run_dir / "status.json", hashes)
    _require(status["status"] == "completed", "Source run is not completed")
    checkpoint_path = run_dir / "checkpoint_calibration_full_information_17_1_before.json"
    checkpoint = _read(checkpoint_path, hashes)
    _require(checkpoint == {a: [] for a in "ABC"}, "Source episode does not start with fresh histories")
    all_rows = _read(run_dir / "steps.jsonl", hashes, jsonl=True)
    rows = [r for r in all_rows if r["phase"] == "calibration"
            and r["condition"] == "full_information" and r["group"] == 17 and r["episode"] == 1]
    _require(all(r["seed"] == 92015 for r in rows), "World seed differs within source episode")
    inferences = _read(run_dir / "inference.jsonl", hashes, jsonl=True)
    cases = [_reconstruct_case(spec, run_dir=run_dir, rows=rows, inferences=inferences,
                               checkpoint=checkpoint, manifest=manifest,
                               source_checks=source_checks, hashes=hashes)
             for spec in CASE_SPECS]
    for case in cases:
        case["source_checks"]["source_run_dir"] = str(run_dir)
        case["source_checks"]["data_file_sha256"] = deepcopy(hashes)
    # Check the public return contract without transforming prompt strings.
    json.dumps(cases, ensure_ascii=False, allow_nan=False)
    return cases
