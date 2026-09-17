"""Read-only G/J map-presentation cases anchored to the frozen M1 input.

Only case['prompt'] is model-facing.  States, witnesses, truths, answer keys
and the prior inference rows are researcher evidence.  Import performs no I/O;
construction reads frozen files and settles World copies without model calls
or file writes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCE_RUN = ROOT / "results/20260915_M_01"
MAP_SEED = 20260916
ORIGINAL_MAP = "营地、林地、河滩两两有道路，所有道路在本任务中永久双向开放，每次移动一步到达相邻地点。"
EXPLICIT_MAP = "道路如下：营地—林地、营地—河滩、林地—河滩。这三条道路永久双向开放，沿每条道路移动均占一个物理动作步。"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _state_sha(state):
    return _sha(json.dumps(state, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def build_map_cases(source_run=None):
    """Return states/checks, eight cases, the sole edit, and the M1 anchor.

    Every case uses the identical frozen S0 observation, literal rendering,
    empty history/broadcasts and affirm-first answers.  Questions only change
    the statement; each representation pair only changes one system sentence.
    """
    from qwen_language_v3.environment import AGENTS, RULES, World

    run_dir = Path(source_run).resolve() if source_run is not None else SOURCE_RUN
    prepared_raw = (run_dir / "prepared_cases.json").read_bytes()
    manifest_raw = (run_dir / "manifest.json").read_bytes()
    inference_raw = (run_dir / "inference.jsonl").read_bytes()
    prepared, manifest = json.loads(prepared_raw), json.loads(manifest_raw)
    _require(_sha(prepared_raw) == manifest["cases_sha256"], "Frozen M cases digest mismatch")
    environment_path = WORK / "qwen_language_v3/environment.py"
    environment_sha = _sha(environment_path.read_bytes())
    _require(environment_sha == manifest["source_sha256"]["qwen_language_v3/environment.py"]
             == _sha((run_dir / "code_snapshot/qwen_language_v3/environment.py").read_bytes()),
             "Frozen M environment code mismatch")

    matching = [c for c in prepared["cases"] if c["case_id"] == "M1_affirm_first"]
    _require(len(matching) == 1, "Expected exactly one frozen M1 affirm-first case")
    old_case = matching[0]
    base_prompt = deepcopy(old_case["prompt"])
    _require(old_case["seed"] == MAP_SEED and old_case["state_id"] == "S0"
             and old_case["order"] == "affirm_first" and old_case["choices"] == [0, 1],
             "Unexpected frozen M1 seed/state/answer order")
    _require(old_case["answers"] == ["陈述符合环境规则", "陈述不符合环境规则"],
             "Unexpected M1 answer wording")
    _require(base_prompt[0]["content"].count(ORIGINAL_MAP) == 1
             and RULES in base_prompt[0]["content"],
             "The exact old map sentence and complete v3 RULES must occur in the base prompt")
    base_data = json.loads(base_prompt[1]["content"])
    _require(base_data["你自己的历史"] == [] and base_data["当前已可见广播"] == []
             and "本次可选动作" not in base_data
             and base_data["待判断陈述"] == old_case["statement"],
             "The frozen base prompt has unexpected history, broadcasts, menu or statement")

    inferences = [json.loads(line) for line in inference_raw.splitlines() if line.strip()]
    old_rows = [r for r in inferences if r.get("label", {}).get("case_id") == old_case["case_id"]]
    _require(len(old_rows) == 2, "Expected exactly two recorded M1 anchor calls")
    analysis = next(r for r in old_rows if r["label"]["stage"] == "analysis")
    formal = next(r for r in old_rows if r["label"]["stage"] == "formal")
    _require(analysis["messages"][:-1] == base_prompt
             and formal["messages"][:-2] == base_prompt
             and formal["messages"][-2] == {"role": "assistant", "content": analysis["output"]}
             and analysis["seed"] == formal["seed"] == MAP_SEED,
             "M1 frozen base prompt does not match its recorded two-stage anchor")

    s0 = deepcopy(prepared["states"]["S0"])
    world = World.from_state_dict(s0)
    _require(world.full_information_observe("C") == base_data["当前私有观察"],
             "Frozen S0 differs from the model-facing full-information observation")
    _require(world.positions["A"] == world.positions["C"] == "林地"
             and all(world.inventory[a]["cargo"] is None for a in ("A", "C")),
             "S0 must place A/C together and empty-handed in the forest")

    def waits():
        return {a: {"kind": "wait"} for a in AGENTS}

    # G1 supplies a direct one-step counterexample for the necessity claim G2.
    solo_actions = waits()
    solo_actions["A"] = {"kind": "move", "destination": "营地"}
    _require(solo_actions["A"] in [m["action"] for m in world.action_menu("A")],
             "Direct forest-to-camp movement is absent from the legal menu")
    solo = World.from_state_dict(s0)
    solo_feedback = solo.step(deepcopy(solo_actions))
    solo_after = solo.state_dict()
    g1_facts = {
        "A_move_succeeded": solo_feedback["A"]["action_succeeded"],
        "A_at_camp": solo.positions["A"] == "营地",
        "elapsed_physical_steps": solo.t - world.t,
        "other_agents_unchanged": all(solo.positions[a] == world.positions[a] for a in ("B", "C")),
        "items_and_inventory_unchanged": all(solo_after[k] == s0[k] for k in ("items", "inventory")),
    }
    g1_truth = (g1_facts["A_move_succeeded"] and g1_facts["A_at_camp"]
                and g1_facts["elapsed_physical_steps"] == 1)
    _require(g1_truth and g1_facts["other_agents_unchanged"]
             and g1_facts["items_and_inventory_unchanged"], "G1 direct movement did not settle as expected")
    g2_facts = {"direct_one_step_counterexample_exists": g1_truth,
                "counterexample_path": ["林地", "营地"], "river_visit_required": False}
    g2_truth = not g2_facts["direct_one_step_counterexample_exists"]

    # Reuse the frozen M1 witness, but verify it independently with the engine.
    carry_actions = deepcopy(prepared["checks"]["M1"]["actions"])
    target_handle = carry_actions["C"]["item"]
    oid = next(oid for oid, handle in s0["handles"]["C"].items() if handle == target_handle)
    target = world.items[oid]
    _require(target["kind"] == "木材" and target["length"] == "长"
             and target["condition"] == "干" and target["location"] == "林地"
             and target["carriers"] == [] and target["processed"] and not target["delivered"],
             "J1 target is not the expected ground dry-long wood")
    for actor, partner in (("A", "C"), ("C", "A")):
        expected = {"kind": "carry_together", "item": world.handles[actor][oid],
                    "partner": partner, "destination": "营地"}
        _require(carry_actions[actor] == expected
                 and expected in [m["action"] for m in world.action_menu(actor)],
                 "M1 witness is not the expected reciprocal local action")
    _require(carry_actions["B"] == {"kind": "wait"}, "B must wait in J1")
    carry = World.from_state_dict(s0)
    carry_feedback = carry.step(deepcopy(carry_actions))
    carry_after = carry.state_dict()
    j1_facts = {
        "both_carry_actions_succeeded": all(carry_feedback[a]["action_succeeded"] for a in ("A", "C")),
        "both_positions_are_camp": all(carry.positions[a] == "营地" for a in ("A", "C")),
        "both_cargo_slots_hold_target": all(carry.inventory[a]["cargo"] == oid for a in ("A", "C")),
        "target_carriers_are_A_C": carry.items[oid]["carriers"] == ["A", "C"],
        "target_no_longer_on_ground": carry.items[oid]["location"] is None,
    }
    j1_truth = all(j1_facts.values())
    _require(j1_truth and carry_after == prepared["states"]["S1"],
             "J1 does not independently reproduce the frozen M1 settlement")

    # Both actions in J2 are individually legal at S0; only destinations differ.
    mismatch_actions = deepcopy(carry_actions)
    mismatch_actions["C"]["destination"] = "河滩"
    legal = {a: mismatch_actions[a] in [m["action"] for m in world.action_menu(a)]
             for a in ("A", "C")}
    _require(all(legal.values()), "A J2 proposal is not individually in its local menu")
    mismatch = World.from_state_dict(s0)
    mismatch_feedback = mismatch.step(deepcopy(mismatch_actions))
    mismatch_after = mismatch.state_dict()
    j2_facts = {
        "each_proposal_is_individually_legal": legal,
        "both_carry_actions_succeeded": all(mismatch_feedback[a]["action_succeeded"] for a in ("A", "C")),
        "both_carry_actions_failed": all(not mismatch_feedback[a]["action_succeeded"] for a in ("A", "C")),
        "both_positions_still_forest": all(mismatch.positions[a] == "林地" for a in ("A", "C")),
        "target_still_on_ground": mismatch.items[oid]["location"] == "林地" and mismatch.items[oid]["carriers"] == [],
        "physical_state_unchanged": all(mismatch_after[k] == s0[k] for k in ("positions", "items", "inventory")),
    }
    j2_truth = (j2_facts["both_carry_actions_succeeded"]
                and all(mismatch.positions[a] == "营地" for a in ("A", "C"))
                and mismatch.items[oid]["carriers"] == ["A", "C"])
    _require(not j2_truth and j2_facts["both_carry_actions_failed"]
             and j2_facts["physical_state_unchanged"], "J2 mismatched destinations unexpectedly moved the load")

    states = {"S0": s0, "G1_after": solo_after, "J1_after": carry_after, "J2_after": mismatch_after}
    checks = {
        "source": {"run_dir": str(run_dir), "manifest_sha256": _sha(manifest_raw),
                   "prepared_cases_sha256": _sha(prepared_raw), "inference_sha256": _sha(inference_raw),
                   "environment_sha256": environment_sha, "S0_sha256": _state_sha(s0),
                   "base_observation_matches_S0": True, "model_calls": 0},
    }
    for q, truth, after, actions, feedback, facts in (
            ("G1", g1_truth, "G1_after", solo_actions, solo_feedback, g1_facts),
            ("G2", g2_truth, "G1_after", solo_actions, solo_feedback, g2_facts),
            ("J1", j1_truth, "J1_after", carry_actions, carry_feedback, j1_facts),
            ("J2", j2_truth, "J2_after", mismatch_actions, mismatch_feedback, j2_facts)):
        checks[q] = {"truth": truth, "before_state_id": "S0", "after_state_id": after,
                     "actions": deepcopy(actions), "feedback": deepcopy(feedback),
                     "evaluated_facts": facts, "before_state_sha256": _state_sha(s0),
                     "after_state_sha256": _state_sha(states[after])}
    checks["G2"]["note"] = "A real direct one-step move refutes the claim that the river detour and two steps are necessary."

    statements = {
        "G1": "本步A从林地提交前往营地的移动动作，B和C等待。判断以下陈述：本步结算后，A在营地。",
        "G2": "按照当前地图，从林地前往营地必须先经过河滩，至少需要两个物理动作步。",
        "J1": old_case["statement"],
        "J2": (f"本步A与C分别对林地地面上的同一件干长木材（你看到的句柄为{target_handle}），"
               "用各自的句柄提交共同搬运，两人互为搭档，但A的目的地选营地、C的目的地选河滩，B等待。"
               "判断以下陈述：本步结算后，两人仍共同把这件木材搬到营地。"),
    }
    cases = []
    for question_id in ("G1", "G2", "J1", "J2"):
        original_prompt = deepcopy(base_prompt)
        if question_id != "J1":
            data = deepcopy(base_data)
            data["待判断陈述"] = statements[question_id]
            original_prompt[1]["content"] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        for representation in ("original", "explicit_edges"):
            prompt = deepcopy(original_prompt)
            if representation == "explicit_edges":
                prompt[0]["content"] = prompt[0]["content"].replace(ORIGINAL_MAP, EXPLICIT_MAP, 1)
            truth = checks[question_id]["truth"]
            cases.append({
                "case_id": f"{question_id}_{representation}", "question_id": question_id,
                "representation": representation, "order": "affirm_first", "state_id": "S0",
                "truth": truth, "answers": deepcopy(old_case["answers"]), "choices": [0, 1],
                "correct_choice": 0 if truth else 1, "prompt": prompt, "seed": MAP_SEED,
                "statement": statements[question_id],
            })
    _require(next(c for c in cases if c["case_id"] == "J1_original")["prompt"] == base_prompt,
             "J1 original prompt is not an exact M1 replay anchor")
    return {
        "states": states, "checks": checks, "cases": cases,
        "presentation_change": {"original": ORIGINAL_MAP, "explicit_edges": EXPLICIT_MAP,
                                "replacement_count": 1, "changed_field": "prompt[0].content",
                                "observation_map_changed": False},
        "anchor": {"source_run": str(run_dir), "source_case_id": old_case["case_id"],
                   "case_id": "J1_original", "prompt": deepcopy(base_prompt), "seed": MAP_SEED,
                   "original_analysis": deepcopy(analysis), "original_formal": deepcopy(formal),
                   "call_ids": [analysis["call"], formal["call"]]},
    }
