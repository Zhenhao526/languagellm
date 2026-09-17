"""Construct the fixed M1--M4 action-semantics questions without inference.

Importing this module performs no I/O and loads no inference runtime.  Calling
``build_semantic_cases`` only reads frozen records and settles deterministic
World copies.  It never writes a file or changes the original experiment.

Only each case's ``prompt`` is model-facing.  ``truth``, ``correct_choice``,
``states`` and ``checks`` are researcher evidence and must not enter a prompt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCE_RUN = WORK / "qwen_language_v3/results/20260915_152343/capability_full_information"
SEMANTIC_SEED = 20260916
AFFIRM = "陈述符合环境规则"
DENY = "陈述不符合环境规则"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _state_sha(state):
    return _sha(json.dumps(state, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def _load_s0(run_dir):
    """Read the real episode-1 step-9 state and verify the pure code we use."""
    from qwen_language_v3.environment import World

    manifest_raw = (run_dir / "manifest.json").read_bytes()
    frozen_raw = (run_dir / "core_source_frozen.json").read_bytes()
    manifest, frozen = json.loads(manifest_raw), json.loads(frozen_raw)
    checked = {}
    for name in ("environment.py", "agents.py"):
        current = _sha((WORK / "qwen_language_v3" / name).read_bytes())
        snapshot = _sha((run_dir / "code_snapshot" / name).read_bytes())
        expected = frozen["source_sha256"][name]
        _require(current == snapshot == expected == manifest["source_sha256"][name],
                 f"Frozen source mismatch: {name}")
        checked[name] = current

    steps_raw = (run_dir / "steps.jsonl").read_bytes()
    source_lines = [(line, json.loads(line)) for line in steps_raw.splitlines(keepends=True)
                    if line.strip()]
    chosen = [(line, row) for line, row in source_lines
              if row.get("episode") == 1 and row.get("step") == 9
              and row.get("seed") == 92015 and row.get("group") == 17
              and row.get("condition") == "full_information"
              and row.get("phase") == "calibration"]
    _require(len(chosen) == 1, "Expected exactly one fixed source episode-1 step-9 record")
    line, row = chosen[0]
    state = deepcopy(row["state_before"])
    world = World.from_state_dict(state)
    _require(state["seed"] == 92015 and state["variant"] == 0 and world.t == 8,
             "Source state identifiers changed")
    _require(not world.done, "Source state is already terminal")
    _require(world.positions["A"] == world.positions["C"] == "林地",
             "A and C must start together in the forest")
    _require(all(world.inventory[a]["cargo"] is None
                 and world.inventory[a]["tool"] is None for a in ("A", "C")),
             "A and C must have empty carrying slots")
    _require(world.full_information_observe("C") == row["observations"]["C"]
             and world.action_menu("C") == row["menus"]["C"],
             "Source C observation/menu does not match the source state")

    # Replay the original joint action solely to check the source record.  This
    # original failed attempt is never used as a demonstration in the questions.
    replay = World.from_state_dict(state)
    _require(replay.step(deepcopy(row["actions"])) == row["feedback"]
             and replay.state_dict() == row["state_after"],
             "Original source step does not replay exactly")
    return state, {
        "run_dir": str(run_dir), "episode": 1, "step": 9, "agent": "C",
        "seed": 92015, "variant": 0, "group": 17,
        "manifest_sha256": _sha(manifest_raw), "frozen_manifest_sha256": _sha(frozen_raw),
        "steps_sha256": _sha(steps_raw), "source_line_sha256": _sha(line),
        "source_state_sha256": _state_sha(state), "pure_source_sha256": checked,
        "source_observation_and_menu_match": True, "source_step_replays_exactly": True,
        "model_calls": 0,
    }


def build_semantic_cases(source_run=None):
    """Return researcher ``states``, truth ``checks`` and eight fixed ``cases``.

    ``source_run`` may specify a copy of the frozen source records.  No model
    call, write, random search, learning history or rule-card branch occurs.
    The default order is M1..M4, each affirm-first then deny-first, all with the
    same seed.  Each question only exposes C's current full observation and its
    literal Chinese rendering, the hypothetical statement and two answers.
    """
    from qwen_language_v3.agents import current_status
    from qwen_language_v3.environment import AGENTS, RULES, World

    run_dir = Path(source_run).resolve() if source_run is not None else SOURCE_RUN
    s0, source = _load_s0(run_dir)
    world = World.from_state_dict(s0)
    goals = [(i, g) for i, g in enumerate(world.goals)
             if g["kind"] == "木材" and g["length"] == "长"
             and g["condition"] == "干" and g["destination"] == "营地"
             and g["quantity"] == 1 and g["delivered"] == 0]
    _require(len(goals) == 1, "Expected one unfilled one-unit dry-long-wood goal at camp")
    goal_index, goal = goals[0]
    candidates = [oid for oid, item in world.items.items()
                  if all(item[k] == goal[k] for k in ("kind", "length", "condition"))
                  and item["location"] == "林地" and not item["carriers"]
                  and item["processed"] and not item["delivered"]]
    _require(len(candidates) == 1, "Expected one matching long wood item on the ground")
    oid = candidates[0]

    def waits():
        return {a: {"kind": "wait"} for a in AGENTS}

    def pair_actions(kind):
        actions = waits()
        for actor, partner in (("A", "C"), ("C", "A")):
            actions[actor] = {"kind": kind, "item": world.handles[actor][oid],
                              "partner": partner}
            if kind == "carry_together":
                actions[actor]["destination"] = "营地"
        return actions

    # M1: the full reciprocal submission starts with the item on the ground.
    carry_actions = pair_actions("carry_together")
    for actor in ("A", "C"):
        _require(carry_actions[actor] in [m["action"] for m in world.action_menu(actor)],
                 "M1 reciprocal carry action is absent from the legal local menu")
    carry_world = World.from_state_dict(s0)
    carry_feedback = carry_world.step(deepcopy(carry_actions))
    s1 = carry_world.state_dict()
    m1_facts = {
        "both_carry_actions_succeeded": all(carry_feedback[a]["action_succeeded"]
                                           for a in ("A", "C")),
        "both_positions_are_camp": all(carry_world.positions[a] == "营地" for a in ("A", "C")),
        "both_cargo_slots_hold_target": all(carry_world.inventory[a]["cargo"] == oid
                                           for a in ("A", "C")),
        "target_carriers_are_A_C": carry_world.items[oid]["carriers"] == ["A", "C"],
        "target_no_longer_on_ground": carry_world.items[oid]["location"] is None,
    }
    m1_truth = all(m1_facts.values())
    _require(m1_truth, "M1 claim was not established by actual reciprocal carry")

    # M2: a researcher submits the absent single-agent pickup deliberately.
    # World.step accepts action dictionaries but rejects actions outside the
    # actor's current legal menu.  No model is asked to execute this action.
    pickup_action = {"kind": "pickup", "item": world.handles["C"][oid]}
    pickup_in_menu = pickup_action in [m["action"] for m in world.action_menu("C")]
    pickup_actions = waits()
    pickup_actions["C"] = pickup_action
    pickup_world = World.from_state_dict(s0)
    pickup_feedback = pickup_world.step(deepcopy(pickup_actions))
    m2_after = pickup_world.state_dict()
    m2_facts = {
        "single_pickup_is_in_menu": pickup_in_menu,
        "single_pickup_succeeded": pickup_feedback["C"]["action_succeeded"],
        "C_cargo_holds_target": pickup_world.inventory["C"]["cargo"] == oid,
        "physical_positions_items_inventory_unchanged": all(
            m2_after[k] == s0[k] for k in ("positions", "items", "inventory")),
    }
    m2_truth = m2_facts["single_pickup_succeeded"] and m2_facts["C_cargo_holds_target"]
    _require(not pickup_in_menu and not m2_truth
             and m2_facts["physical_positions_items_inventory_unchanged"],
             "M2 ordinary long-wood pickup was not rejected without physical movement")

    # M3 reads S1.  Delivered=0 is legitimate *current* full-information data;
    # no proposed future settlement or answer field is put in its prompt.
    m3_facts = {
        "both_positions_are_camp": m1_facts["both_positions_are_camp"],
        "target_goal_delivered": carry_world.goals[goal_index]["delivered"],
        "target_goal_quantity": carry_world.goals[goal_index]["quantity"],
        "target_item_delivered": carry_world.items[oid]["delivered"],
        "score_unchanged_after_carry": carry_world.score == world.score,
    }
    m3_truth = (m3_facts["target_goal_delivered"] >= m3_facts["target_goal_quantity"])
    _require(not m3_truth and not m3_facts["target_item_delivered"]
             and m3_facts["target_goal_delivered"] == 0
             and m3_facts["score_unchanged_after_carry"],
             "M3 carry already counted as delivery")

    # M4 starts from the genuinely carried S1 load, not the ground state S0.
    delivery_actions = pair_actions("deliver_together")
    delivery_world = World.from_state_dict(s1)
    for actor in ("A", "C"):
        _require(delivery_actions[actor] in [m["action"] for m in delivery_world.action_menu(actor)],
                 "M4 reciprocal delivery action is absent from the legal local menu")
    delivery_feedback = delivery_world.step(deepcopy(delivery_actions))
    s2 = delivery_world.state_dict()
    m4_facts = {
        "both_delivery_actions_succeeded": all(delivery_feedback[a]["action_succeeded"]
                                              for a in ("A", "C")),
        "target_goal_units_added": (delivery_world.goals[goal_index]["delivered"]
                                    - carry_world.goals[goal_index]["delivered"]),
        "both_cargo_slots_released": all(delivery_world.inventory[a]["cargo"] is None
                                        for a in ("A", "C")),
        "target_item_delivered": delivery_world.items[oid]["delivered"],
        "target_carriers_empty": delivery_world.items[oid]["carriers"] == [],
    }
    m4_truth = (m4_facts["both_delivery_actions_succeeded"]
                and m4_facts["target_goal_units_added"] == 1
                and m4_facts["both_cargo_slots_released"]
                and m4_facts["target_item_delivered"] and m4_facts["target_carriers_empty"])
    _require(m4_truth, "M4 did not add exactly one delivered unit and release both slots")

    states = {"S0": s0, "S1": s1, "M2_after": m2_after, "S2": s2}
    checks = {
        "source": source,
        "M1": {"truth": m1_truth, "before_state_id": "S0", "after_state_id": "S1",
               "actions": carry_actions, "feedback": carry_feedback, "evaluated_facts": m1_facts},
        "M2": {"truth": m2_truth, "before_state_id": "S0", "after_state_id": "M2_after",
               "actions": pickup_actions, "feedback": pickup_feedback, "evaluated_facts": m2_facts,
               "note": "Researcher-only invalid submission; World rejects the out-of-menu action. No model executed it."},
        "M3": {"truth": m3_truth, "before_state_id": "S1", "after_state_id": "S1",
               "actions": None, "feedback": None, "evaluated_facts": m3_facts,
               "note": "Read-only claim about the actual current S1 delivery count, without a new step."},
        "M4": {"truth": m4_truth, "before_state_id": "S1", "after_state_id": "S2",
               "actions": delivery_actions, "feedback": delivery_feedback, "evaluated_facts": m4_facts},
    }
    for check_id in ("M1", "M2", "M3", "M4"):
        check = checks[check_id]
        check["before_state_sha256"] = _state_sha(states[check["before_state_id"]])
        check["after_state_sha256"] = _state_sha(states[check["after_state_id"]])

    handle = world.handles["C"][oid]
    item = f"干长木材（你看到的句柄为{handle}）"
    statements = {
        "M1": (f"本步A与C分别对林地地面上的同一件{item}，用各自的句柄提交共同搬运，"
               "两人互为搭档、目的地都选营地，B等待。判断以下陈述：本步结算后，"
               "A、C和这件木材都在营地，木材由A、C共同携带，并占双方各一个普通携带位。"),
        "M2": (f"对于林地地面上的这件{item}，C可以先用普通单人拿取，把它放进自己的普通携带位，"
               "再在下一步与A共同搬运。"),
        "M3": (f"当前A、C共同携带的这件{item}已经到达营地，因此对应的木材交付需求已经计为完成。"),
        "M4": (f"本步A与C对当前共同携带的同一件{item}，用各自的句柄提交共同交付，"
               "两人互为搭档，B等待。判断以下陈述：本步结算后，这件木材的交付被接收，"
               "对应木材交付量增加一件，A和C的普通携带位都释放。"),
    }
    cases = []
    for question_id in ("M1", "M2", "M3", "M4"):
        state_id = "S0" if question_id in ("M1", "M2") else "S1"
        observation = World.from_state_dict(states[state_id]).full_information_observe("C")
        truth = checks[question_id]["truth"]
        for order in ("affirm_first", "deny_first"):
            answers = [AFFIRM, DENY] if order == "affirm_first" else [DENY, AFFIRM]
            data = {
                "题目说明": "这是一次性动作语义判断，不执行物理动作。陈述中给定的伙伴动作是题目假设，不是实际通信或已发生的行动。",
                "你自己的历史": [], "当前私有观察": deepcopy(observation),
                "当前观察的直读": current_status(observation), "当前已可见广播": [],
                "待判断陈述": statements[question_id],
                "答案选项": [{"编号": i, "含义": answer} for i, answer in enumerate(answers)],
            }
            prompt = [
                {"role": "system", "content": "你是本题中的C。依据以下完整环境规则和当前状态判断陈述。\n" + RULES
                 + "\n本题是独立的完整信息判断：你可查看以C自己的句柄表示的各地点当前状态和需求；待判断动作仍受物理规则限制。"},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))},
            ]
            cases.append({
                "case_id": f"{question_id}_{order}", "question_id": question_id,
                "order": order, "truth": truth, "answers": answers, "choices": [0, 1],
                "correct_choice": answers.index(AFFIRM if truth else DENY),
                "prompt": prompt, "seed": SEMANTIC_SEED,
                "statement": statements[question_id], "state_id": state_id,
            })
    return {"states": states, "checks": checks, "cases": cases}
