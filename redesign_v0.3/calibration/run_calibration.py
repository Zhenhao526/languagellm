#!/usr/bin/env python3
"""Run one logged native Minecraft calibration episode, without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
VENDOR = BASE / "vendor" / "minerl"
sys.path.insert(0, str(VENDOR))


def configure_game_presentation():
    """Disable Minecraft's external tutorial instructions before this JVM starts."""
    from minerl.env.malmo import MinecraftInstance
    original = MinecraftInstance._launch_minecraft
    overrides = {'tutorialStep': 'none', 'chatVisibility': '2',
                 'heldItemTooltips': 'false', 'advancedItemTooltips': 'false'}
    def launch(instance, port, headless, minecraft_dir, replaceable=False):
        path = Path(minecraft_dir) / 'options.txt'
        options = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    options[key] = value
        options.update(overrides)
        path.write_text(''.join(f'{key}:{value}\n' for key, value in options.items()))
        return original(instance, port, headless, minecraft_dir, replaceable=replaceable)
    MinecraftInstance._launch_minecraft = launch
    return overrides


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_json(path, data):
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n")


def flatten_numeric(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, val in value.items():
            result.update(flatten_numeric(val, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (float, int)):
        result[prefix] = value
    return result


def nonzero(value):
    if isinstance(value, dict):
        return {k: nonzero(v) for k, v in value.items()
                if isinstance(v, dict) or v not in (0, 0., False, None)}
    return value


def scoring_state(obs, info=None):
    """Private evaluator data, excluded from the policy request by construction."""
    result = {k: jsonable(v) for k, v in obs.items() if k != "pov"}
    for key in ("inventory", "use_item", "drop", "pickup", "break_item", "craft_item",
                "mine_block", "damage_dealt", "entity_killed_by", "kill_entity", "full_stats"):
        if key in result:
            result[key] = nonzero(result[key])
    if info:
        result["env_info"] = jsonable({k: v for k, v in info.items() if k != "pov"})
    return result


def attach_evaluation_tap(env):
    """Read raw backend receipts before MineRL drops unregistered monitor keys.

    The original parser receives exactly the same bytes and produces exactly
    the same observation. Only the runner can access this extra evaluator data.
    """
    original = env._process_observation
    env.calibration_evaluator_info = {}

    def tapped(actor_name, pov, info):
        if info:
            raw = json.loads(info)
            env.calibration_evaluator_info = {
                key: raw[key] for key in ("calibration_init", "tick", "tickCounter", "env_tick")
                if key in raw}
        return original(actor_name, pov, info)
    env._process_observation = tapped


def evaluated_state(env, obs, info=None):
    state = scoring_state(obs, info)
    state.update(jsonable(env.calibration_evaluator_info))
    return state


def position(state):
    loc = state.get("location_stats", {})
    if not all(k in loc for k in ("xpos", "ypos", "zpos")):
        return None
    return [float(loc[k]) for k in ("xpos", "ypos", "zpos")]


def log_total(state, section):
    return sum(v for k, v in flatten_numeric(state.get(section, {})).items()
               if k.rsplit("/", 1)[-1].split(":")[-1].endswith("_log"))


def item_total(state, section, item):
    return sum(v for k, v in flatten_numeric(state.get(section, {})).items()
               if k.rsplit("/", 1)[-1].split(":")[-1] == item)


def tick_observation(state):
    matches = {k: v for k, v in flatten_numeric(state.get("full_stats", {})).items()
               if k.split("/")[-1] in ("play_one_minute", "play_time", "total_world_time")}
    return matches or None


def verify_reset(state, config):
    errors = []
    actual = position(state)
    expected = [config["placement"][k] for k in ("x", "y", "z")]
    if actual is None or any(abs(a - b) > .15 for a, b in zip(actual or [], expected)):
        errors.append(f"placement mismatch: expected {expected}, observed {actual}")
    life = state.get("life_stats", {})
    for key, expected_value in (("food", config["food"]), ("life", config["health"]),
                                ("saturation", config["saturation"])):
        if key not in life or abs(float(life[key]) - expected_value) > .1:
            errors.append(f"{key} mismatch: expected {expected_value}, observed {life.get(key)}")
    expected_inventory = {}
    for entry in config["inventory"].values():
        expected_inventory[entry["type"]] = expected_inventory.get(entry["type"], 0) + entry["quantity"]
    for item, count in expected_inventory.items():
        actual_count = item_total(state, "inventory", item)
        if actual_count != count:
            errors.append(f"inventory {item}: expected {count}, observed {actual_count}")
    actual_inventory = state.get("inventory", {})
    for path, count in flatten_numeric(actual_inventory).items():
        name = path.rsplit("/", 1)[-1].split(":")[-1]
        if count and name not in expected_inventory:
            errors.append(f"unexpected initial inventory: {name}={count}")
    expected_hand = config["inventory"].get(0, {}).get("type", "air")
    actual_hand = state.get("equipped_items", {}).get("mainhand", {}).get("type")
    if actual_hand is None or str(actual_hand).split(":")[-1] != expected_hand:
        errors.append(f"initial mainhand mismatch: expected {expected_hand}, observed {actual_hand}")
    # Mandatory native initialization receipt is added by the calibration Java
    # patch. Older backends can parse XML without applying geometry or hunger.
    receipt = state.get("env_info", {}).get("calibration_init")
    if receipt is None:
        receipt = state.get("calibration_init")
    if receipt is None:
        errors.append("Missing calibration_init receipt: native initialization was not verified")
    elif not isinstance(receipt, dict):
        errors.append("calibration_init receipt must be a JSON object")
    else:
        required = {"schema": "calibration_native_init_v1", "summary": config["name"],
                    "applied": True, "applied_once": True, "server_thread_verified": True,
                    "verification_failures": 0, "block_mismatches": 0, "selected_slot": 0,
                    "day_time": 6000, "do_daylight_cycle": False, "do_mob_spawning": False}
        for key, expected_value in required.items():
            if key not in receipt or receipt[key] != expected_value:
                errors.append(f"initialization receipt {key}: expected {expected_value}, observed {receipt.get(key)}")
        total = receipt.get("final_unique_positions", 0)
        if not isinstance(total, (int, float)) or total <= 0 or receipt.get("verified_block_types") != total:
            errors.append("Native block verification was incomplete")
        for key, expected_value in (("x", expected[0]), ("y", expected[1]), ("z", expected[2]),
                                    ("food", config["food"]), ("health", config["health"])):
            if key not in receipt or abs(float(receipt[key]) - expected_value) > .1:
                errors.append(f"initialization receipt {key} does not match scenario")
    return {"passed": not errors, "errors": errors, "initialization_receipt": receipt,
            "geometry_verified_from_native_receipt": bool(not errors),
            "geometry_note": "Receipt verifies block types at initialization; initial image and reset state independently record what the actor actually receives after native bootstrap ticks."}


def actual_action(env, raw):
    import numpy as np
    action = env.action_space.noop()
    unknown = set(raw) - set(action)
    if unknown:
        raise ValueError(f"Policy supplied keys outside native action space: {unknown}")
    action.update(raw)
    action["camera"] = np.asarray(action["camera"], dtype=np.float32).reshape(2)
    if not env.action_space.contains(action):
        raise ValueError(f"Action is outside native action space: {jsonable(action)}")
    return action


def _read_command(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT,
                                       text=True, timeout=15).strip()
    except Exception as exc:
        return {"unavailable": str(exc)}


def runtime_metadata():
    versions = {}
    for package in ("minerl", "gym", "numpy", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source import or unavailable"
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "java": _read_command(["java", "-version"]),
            "minecraft": "1.16.5 (pinned MCP-Reborn build; see build provenance)",
            "minerl_commit": _read_command(["git", "rev-parse", "HEAD"], VENDOR),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scenarios_sha256": hashlib.sha256((BASE / "scenarios.py").read_bytes()).hexdigest()}


def summarize(config, states, raw_actions, termination):
    if not states:
        return {"observations": 0, "termination": termination, "behavior_evidence": False}
    initial, final = states[0], states[-1]
    positions = [position(s) for s in states]
    positions = [p for p in positions if p is not None]
    displacement = (math.hypot(positions[-1][0] - positions[0][0],
                               positions[-1][2] - positions[0][2]) if positions else None)
    distance = sum(math.hypot(b[0] - a[0], b[2] - a[2])
                   for a, b in zip(positions, positions[1:]))
    cells = len({(math.floor(p[0]), math.floor(p[2])) for p in positions})
    mined = log_total(final, "mine_block") - log_total(initial, "mine_block")
    picked = log_total(final, "pickup") - log_total(initial, "pickup")
    log_net = log_total(final, "inventory") - log_total(initial, "inventory")
    bread_net = item_total(final, "inventory", "bread") - item_total(initial, "inventory", "bread")
    uses = item_total(final, "use_item", "bread") - item_total(initial, "use_item", "bread")
    hunger = [s.get("life_stats", {}).get("food") for s in states]
    hunger = [float(x) for x in hunger if x is not None]
    max_food_gain = max(hunger) - hunger[0] if hunger else None
    first_log = next((i for i, s in enumerate(states)
                      if log_total(s, "pickup") > log_total(initial, "pickup")), None)
    first_food = next((i for i, s in enumerate(states)
                      if item_total(s, "use_item", "bread") > item_total(initial, "use_item", "bread")), None)
    wood_chain = mined > 0 and picked > 0 and log_net > 0
    food_chain = uses > 0 and bread_net < 0 and max_food_gain is not None and max_food_gain > 0
    return {
        "observations": len(states), "completed_steps": len(states) - 1,
        "termination": termination,
        "horizontal_displacement_blocks": displacement, "horizontal_path_blocks": distance,
        "distinct_horizontal_grid_cells": cells,
        "log_blocks_mined_delta": mined, "logs_picked_up_delta": picked,
        "logs_inventory_net": log_net, "first_log_pickup_observation": first_log,
        "bread_inventory_net": bread_net, "bread_native_use_stat_delta": uses,
        "maximum_food_level_increase": max_food_gain,
        "first_bread_use_observation": first_food,
        "model_or_controller_use_action_count": sum(bool(a.get("use", 0)) for a in raw_actions),
        "initial_state": initial, "final_state": final,
        "native_wood_chain_observed": wood_chain, "native_food_chain_observed": food_chain,
        "wood_chain_scoring_scope": "Requires mined, picked-up, and positive log inventory net; subsequent complete crafting may produce a conservative false negative and must be audited separately.",
        "movement_observed": bool(distance > 1.0 and cells > 1),
        "interpretation": "Descriptive single episode only; script actions validate the interface and never count as model evidence. Resource consequences alone do not demonstrate consequence-sensitive choice.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, choices=("script", "noop", "vpt"))
    parser.add_argument("--scenario", required=True,
                        choices=("movement", "wood", "food_selected", "food_select"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--food", type=int, default=8, help="Initial native hunger level, 1..19 for food scenarios")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Write and XML-validate configuration without creating a game")
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.output_dir or BASE / "runs" / f"{stamp}_{args.controller}_{args.scenario}_s{args.seed}_f{args.food}"
    out.mkdir(parents=True, exist_ok=False)
    status = {"status": "starting", "args": vars(args), "started_utc": stamp,
              "controller_label": "script_positive_control" if args.controller == "script" else args.controller,
              "training": False, "policy_inputs": ["pov"] if args.controller == "vpt" else [],
              "model_evidence": False, "social_learning_started": False}
    status["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    dump_json(out / "status.json", status)
    env = client = video = None
    states, raw_actions = [], []
    termination = "exception"
    started = time.monotonic()
    try:
        import cv2
        import numpy as np
        from lxml import etree
        from scenarios import CalibrationSurvival, DEFAULT_STEPS, script_action
        random.seed(args.seed)
        np.random.seed(args.seed)
        spec = CalibrationSurvival(args.scenario, args.seed, args.food)
        config = spec.configuration
        config['presentation_overrides'] = configure_game_presentation()
        config['presentation_scope'] = 'Tutorial instructions, chat, and held-item name disabled; normal inventory GUI may still contain native labels.'
        steps = args.steps if args.steps is not None else DEFAULT_STEPS[args.scenario]
        if steps <= 0:
            raise ValueError("steps must be positive")
        config["requested_steps"] = steps
        dump_json(out / "scenario.json", config)
        xml = spec.to_xml()
        root = etree.fromstring(xml.encode())
        ns = {"m": "http://ProjectMalmo.microsoft.com"}
        if not root.xpath(".//m:StartingFood[@foodSaturation]", namespaces=ns):
            raise RuntimeError("Correct StartingFood XML was not generated")
        drawing_nodes = root.xpath(".//m:DrawingDecorator/*", namespaces=ns)
        if not drawing_nodes or not all(etree.QName(x).localname in ("DrawCuboid", "DrawBlock")
                                        for x in drawing_nodes):
            raise RuntimeError("Drawing XML must contain actual block elements, not escaped text")
        if args.scenario == "wood" and not root.xpath(".//m:DrawBlock[@type='oak_log']", namespaces=ns):
            raise RuntimeError("Wood calibration requires actual log drawing elements")
        (out / "mission.xml").write_text(xml)
        dump_json(out / "runtime.json", runtime_metadata())
        if args.dry_run:
            from types import SimpleNamespace
            placement = config["placement"]
            dry_state = {"location_stats": {"xpos": placement["x"], "ypos": placement["y"],
                                            "zpos": placement["z"], "pitch": placement["pitch"],
                                            "yaw": placement["yaw"]}}
            for tick in range(340):
                actual_action(SimpleNamespace(action_space=spec.action_space),
                              script_action(config, tick, dry_state))
            dump_json(out / "dry_run_validation.json", {
                "xml_well_formed": True, "drawing_elements": len(drawing_nodes),
                "correct_starting_food_element": True,
                "script_action_space_checks": 340, "game_started": False,
                "behavior_or_initialization_verified": False,
                "note": "Static configuration/interface checks only; actual position, resources and geometry remain unverified until native reset."})
            termination = "dry_run_no_game"
            status.update(status="configuration_validated", dry_run=True,
                          behavior_evidence=False, termination=termination)
            return 0

        if args.controller == "vpt":
            from policy_client import PolicyClient
            kwargs = {"device": args.device}
            if args.policy_python:
                kwargs["python"] = str(args.policy_python)
            client = PolicyClient(**kwargs)
            dump_json(out / "policy_metadata.json", client.metadata)

        env = spec.make()
        attach_evaluation_tap(env)
        env.seed(args.seed)
        # EnvSpec.reset regenerates handlers from the same private seeded
        # configuration. No policy can observe experimental state during setup.
        obs = env.reset()
        if client:
            client.reset(seed=args.seed)
        state = evaluated_state(env, obs)
        states.append(state)
        reset_check = verify_reset(state, config)
        dump_json(out / "reset_check.json", reset_check)
        cv2.imwrite(str(out / "initial.png"), cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))
        if not reset_check["passed"]:
            raise RuntimeError("Reset-state validation failed: " + "; ".join(reset_check["errors"]))
        if tuple(obs["pov"].shape) != (360, 640, 3):
            raise RuntimeError(f"Unexpected native image shape: {obs['pov'].shape}")
        video = cv2.VideoWriter(str(out / "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                20.0, (640, 360))
        if not video.isOpened():
            raise RuntimeError("Could not create required episode.mp4 recording")
        video.write(cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))
        status.update(status="running", reset_validated=True)
        dump_json(out / "status.json", status)
        with (out / "events.jsonl").open("w", buffering=1) as events:
            events.write(json.dumps({"event": "episode_reset", "seed": args.seed,
                                     "policy_reset": bool(client), "state": state,
                                     "monotonic_seconds": time.monotonic() - started}) + "\n")
            for step in range(steps):
                before = state
                if args.controller == "vpt":
                    raw = client.act(obs["pov"])
                elif args.controller == "script":
                    raw = script_action(config, step, state)
                else:
                    raw = {}
                action = actual_action(env, raw)
                raw_actions.append(jsonable(raw))
                events.write(json.dumps({"event": "action_requested", "step": step,
                                         "raw_model_action": jsonable(raw) if client else None,
                                         "raw_controller_action": jsonable(raw),
                                         "actual_action": jsonable(action)}, ensure_ascii=False) + "\n")
                action_started = time.monotonic()
                obs, reward, done, info = env.step(action)
                if "error" in info:
                    raise RuntimeError(f"Native environment reported an error: {info['error']}")
                state = evaluated_state(env, obs, info)
                states.append(state)
                video.write(cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))
                before_flat, after_flat = flatten_numeric(before), flatten_numeric(state)
                deltas = {k: after_flat.get(k, 0) - before_flat.get(k, 0)
                          for k in set(after_flat) | set(before_flat)
                          if after_flat.get(k, 0) != before_flat.get(k, 0)}
                record = {"event": "step", "step": step, "observation_index": step + 1,
                          "episode_reset": False, "controller": status["controller_label"],
                          "raw_model_action": jsonable(raw) if client else None,
                          "raw_controller_action": jsonable(raw), "actual_action": jsonable(action),
                          "action_conversion": "native-noop-fill-and-float32-camera-v1",
                          "reward": jsonable(reward), "done": bool(done), "state": state,
                          "numeric_state_deltas": deltas, "native_game_tick_stats": tick_observation(state),
                          "observation_wall_utc": datetime.now(timezone.utc).isoformat(),
                          "elapsed_wall_seconds": time.monotonic() - started,
                          "env_step_wall_seconds": time.monotonic() - action_started,
                          "policy_info": jsonable(client.last_info) if client else None}
                events.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (step + 1) % 100 == 0:
                    print(json.dumps({"completed_steps": step + 1, "output": str(out)}, ensure_ascii=False), flush=True)
                if done:
                    termination = "native_environment_done"
                    break
                if not state.get("life_stats", {}).get("is_alive", True):
                    termination = "death"
                    break
                p = position(state)
                if p and (abs(p[0]) > 19.5 or abs(p[2]) > 19.5 or p[1] < 62):
                    termination = "left_calibrated_arena"
                    break
            else:
                termination = "step_budget"
        dump_json(out / "summary.json", summarize(config, states, raw_actions, termination))
        cv2.imwrite(str(out / "final.png"), cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))
        status.update(status="completed", termination=termination, completed_steps=len(states) - 1,
                      model_evidence=bool(client and len(states) > 1))
        return 0
    except BaseException as exc:
        (out / "failure.txt").write_text(traceback.format_exc())
        status.update(status="failed", termination="exception", error=repr(exc),
                      completed_steps=max(0, len(states) - 1), behavior_evidence=False)
        if states:
            dump_json(out / "summary.json", summarize(locals().get("config", {}), states,
                                                       raw_actions, "exception"))
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        if video:
            video.release()
        if env is not None:
            for index, instance in enumerate(getattr(env, "instances", [])):
                game_dir = getattr(instance, "minecraft_dir", None)
                if game_dir:
                    native_log = Path(game_dir) / "logs/latest.log"
                    if native_log.is_file():
                        shutil.copy2(native_log, out / f"native_game_{index}.log")
                    options = Path(game_dir) / 'options.txt'
                    if options.is_file():
                        shutil.copy2(options, out / f"native_options_{index}.txt")
        for resource in (env, client):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    status.setdefault("close_errors", []).append(repr(exc))
        status["elapsed_wall_seconds"] = time.monotonic() - started
        dump_json(out / "status.json", status)
        print(json.dumps({"status": status["status"], "output": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
