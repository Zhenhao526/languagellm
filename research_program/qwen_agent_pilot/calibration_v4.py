"""Factorial information-allocation and feedback calibration for Qwen agents."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from .factorized_environment import (
    AGENTS, ATTRIBUTES, DESTINATIONS, MEANING_POOL, OBJECTS, PAYLOADS,
    TOTAL_EPISODES, make_episode, score_episode,
)
from .pilot import ALPHABET, MODEL_ID, SYSTEM, _post, _visible_json, valid_message

CALIBRATION_VERSION = "factorized_information_feedback_v4"
MODEL_REVISION = "16daa4818c54ce5f5436f929d52542eb65bbed9d"
DEFAULT_SEEDS = (20261002, 20261003, 20261004)
CONDITIONS = (
    "hidden_scalar",
    "hidden_component",
    "public_partner_scalar",
    "public_partner_component",
    "known_codebook",
    "oracle_decoded",
)
MIN_MESSAGE_LENGTH = 4
MAX_MESSAGE_LENGTH = 8
TEMPERATURE = 0.35
MAX_TOKENS = 160

# Fixed, opaque, six-character codewords; this is a positive communication control.
CODEWORDS = (
    "&?&#+#", "=#@%&%", "=?@&~=", "?%#%=@",
    "#~~@+=", "%&#+#?", "&@@=~@", "##~?&#",
    "%#&+&~", "@+?#%%", "%~~#&%", "~&#~@?",
    "?@+#~&", "&&%~~&", "=&+==#", "&@+@?%",
)
CODEBOOK = {meaning_id: CODEWORDS[meaning_id] for meaning_id in range(len(MEANING_POOL))}
SYSTEM_PROMPT = SYSTEM + (
    "\nEach request has exactly one designated helper. Only the designated helper should act; "
    "the other helper must wait. Use the task feedback to improve later messages and actions. "
    "Do not reveal chain-of-thought. Return only the JSON object requested for the current turn."
)


def _mi(rows: list[dict], x_key: str, y_key: str) -> float:
    n = len(rows)
    if not n:
        return 0.0
    joint = Counter((row[x_key], row[y_key]) for row in rows)
    xs = Counter(row[x_key] for row in rows)
    ys = Counter(row[y_key] for row in rows)
    return sum(
        count / n * math.log2(count * n / (xs[x] * ys[y]))
        for (x, y), count in joint.items()
    )


def _entropy(values: list[str]) -> float:
    n = len(values)
    if not n:
        return 0.0
    counts = Counter(values)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def _condition_feedback(condition: str) -> str:
    if condition.endswith("component"):
        return "component"
    return "scalar"


def _partner_public(condition: str) -> bool:
    return condition.startswith("public_partner")


def _payload(goal: dict) -> tuple[str, str, str]:
    return goal["object"], goal["attribute"], goal["destination"]


def _codebook_rows_for_owner(episode: dict) -> list[dict]:
    rows = []
    for meaning_id, (obj, attr, partner_position, destination) in enumerate(MEANING_POOL):
        rows.append({
            "object": obj,
            "attribute": attr,
            "responsible_helper_position": partner_position,
            "destination": destination,
            "message": CODEBOOK[meaning_id],
        })
    return rows


def _codebook_rows_for_helper(episode: dict) -> list[dict]:
    return _codebook_rows_for_owner(episode)


def _owner_prompt(episode: dict, condition: str) -> str:
    if condition == "oracle_decoded":
        return (
            f"Round {episode['episode']}. Your identity is {episode['owner']}; you are the requester.\n"
            f"Your private order is {json.dumps(episode['goal'], ensure_ascii=False, separators=(',', ':'))}.\n"
            "The communication channel is closed. Return exactly JSON: {\"message\":\"\"}."
        )
    goal = episode["goal"]
    private_order = (
        f"object={goal['object']}; attribute={goal['attribute']}; "
        f"responsible partner={goal['partner']} (helper position {goal['partner_position']}); "
        f"destination={goal['destination']}"
    )
    lead = (
        f"Round {episode['episode']}. Your identity is {episode['owner']}; you are the requester.\n"
        f"Private order: {private_order}.\n"
        f"The two helpers are position 0={episode['helpers'][0]} and position 1={episode['helpers'][1]}.\n"
    )
    if condition == "known_codebook":
        return (
            lead
            + "Both sides know this fixed mapping from complete orders to opaque messages. "
            "Choose the exact message for this order; do not alter or explain it.\n"
            + json.dumps(_codebook_rows_for_owner(episode), ensure_ascii=False, separators=(",", ":"))
            + "\nReturn JSON only: {\"message\":\"CODE\"}."
        )
    if _partner_public(condition):
        scope = (
            "The environment privately tells each helper whether it is assigned. "
            "Your broadcast should encode only object, attribute, and destination; "
            "do not encode the responsible partner."
        )
    else:
        scope = "Your broadcast should encode the full order, including which helper is responsible."
    return (
        lead + scope + " Send one opaque symbol string, not words or an explanation. "
        f"Use 4–8 characters from this alphabet only: {''.join(sorted(ALPHABET))}.\n"
        "Return JSON only: {\"message\":\"SYMBOLS\"}."
    )


def _helper_prompt(episode: dict, agent: str, message: str, condition: str) -> str:
    position = episode["helpers"].index(agent)
    public = _partner_public(condition)
    status = "assigned to act" if agent == episode["goal"]["partner"] else "not assigned to act"
    lines = [
        f"Round {episode['episode']}. Your identity is {agent}; you are a helper.",
        f"Requester {episode['owner']} broadcast the symbol string {json.dumps(message)}.",
        "Public collection board: " + json.dumps(
            {"candidates": episode["scene"], "destinations": episode["destinations"]},
            ensure_ascii=False, separators=(",", ":"),
        ),
    ]
    if public:
        lines.append(
            f"Environment role cue: you are {status}. This cue reveals no object, attribute, or destination."
        )
    if condition == "oracle_decoded":
        decoded = {
            "object": episode["goal"]["object"],
            "attribute": episode["goal"]["attribute"],
            "responsible_helper": episode["goal"]["partner"],
            "destination": episode["goal"]["destination"],
        }
        lines.append(
            "Environment-provided decoded request (task control, not a peer message): "
            + json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
        )
    if condition == "known_codebook":
        lines.append(
            "Both sides know the fixed mapping from complete orders to opaque messages. "
            f"Your position among this round's helpers is {position}; helper positions are "
            f"0={episode['helpers'][0]}, 1={episode['helpers'][1]}. Find the exact matching message row."
        )
        lines.append(json.dumps(_codebook_rows_for_helper(episode), ensure_ascii=False, separators=(",", ":")))
    else:
        lines.append(
            "You have no supplied message mapping. Use only this string and your own prior interaction "
            "history to infer a shared convention; do not assume that the string is English."
        )
    lines.extend([
        "Before deciding, record your best private interpretation of object, attribute, responsible helper, "
        "and destination. Use null for a field you cannot infer. This structured diagnostic is visible only "
        "to you and the environment; it is not shown to the requester or other helper and has no direct reward.",
        "If you are not responsible, wait. If you are responsible and can determine the target, select the "
        "board item whose object and attribute exactly match, and choose the destination. If uncertain, wait.",
        "Do not send a natural-language reply. Return JSON only with both fields: "
        "{\"interpretation\":{\"object\":null,\"attribute\":null,\"responsible_helper\":null,"
        "\"destination\":null},\"action\":null} or the same interpretation plus "
        "{\"action\":{\"item_id\":\"I0\",\"destination\":\"river camp\"}}.",
    ])
    return "\n".join(lines)


def _parse_json(raw: str) -> tuple[dict, bool]:
    cleaned = _visible_json(raw)
    try:
        data = json.loads(cleaned)
        return (data, True) if isinstance(data, dict) else ({}, False)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
            return (data, isinstance(data, dict))
        except (json.JSONDecodeError, TypeError):
            return {}, False


def _parse_owner(raw: str, condition: str, meaning_id: int) -> dict:
    data, json_valid = _parse_json(raw)
    supplied = data.get("message")
    if condition == "oracle_decoded":
        return {
            "json_valid": json_valid,
            "message_valid": supplied == "",
            "message": "",
            "codebook_correct": None,
            "canonical": json.dumps({"message": ""}, separators=(",", ":")),
        }
    message_valid = valid_message(supplied)
    message = supplied if message_valid else ""
    codebook_correct = message == CODEBOOK[meaning_id] if condition == "known_codebook" else None
    return {
        "json_valid": json_valid,
        "message_valid": bool(message_valid),
        "message": message,
        "codebook_correct": codebook_correct,
        "canonical": json.dumps({"message": message}, ensure_ascii=False, separators=(",", ":")),
    }


def _parse_helper(raw: str) -> dict:
    data, json_valid = _parse_json(raw)
    source = data.get("interpretation")
    allowed = {
        "object": set(OBJECTS),
        "attribute": set(ATTRIBUTES),
        "responsible_helper": set(AGENTS),
        "destination": set(DESTINATIONS),
    }
    interpretation = {}
    valid_fields = True
    for field, values in allowed.items():
        value = source.get(field) if isinstance(source, dict) else None
        if not isinstance(source, dict) or field not in source:
            valid_fields = False
        if value is None:
            interpretation[field] = None
        elif isinstance(value, str) and value in values:
            interpretation[field] = value
        else:
            interpretation[field] = None
            valid_fields = False
    interpretation_valid = isinstance(source, dict) and valid_fields

    action_source = data.get("action")
    action = None
    if action_source is None:
        action_valid = json_valid and "action" in data
    elif isinstance(action_source, dict):
        item_id, destination = action_source.get("item_id"), action_source.get("destination")
        if (isinstance(item_id, str) and re.fullmatch(r"I[0-3]", item_id)
                and destination in DESTINATIONS):
            action = {"item_id": item_id, "destination": destination}
            action_valid = True
        else:
            action_valid = False
    else:
        action_valid = False
    canonical = json.dumps(
        {"interpretation": interpretation, "action": action},
        ensure_ascii=False, separators=(",", ":"),
    )
    return {
        "json_valid": json_valid,
        "interpretation_valid": bool(interpretation_valid),
        "interpretation": interpretation,
        "action_valid": bool(action_valid),
        "action": action,
        "canonical": canonical,
    }


def _feedback_text(episode: dict, agent: str, condition: str, message: str,
                   actions: dict[str, dict | None], outcome: dict) -> str:
    other = next(helper for helper in episode["helpers"] if helper != agent)
    if agent == episode["owner"]:
        content = {
            "round": episode["episode"],
            "your_broadcast": message,
            "helper_actions": actions,
            "team_success": outcome["success"],
            "team_reward": outcome["reward"],
        }
    else:
        content = {
            "round": episode["episode"],
            "received_symbols": message,
            "your_action": actions[agent],
            "other_helper_action": actions[other],
            "team_success": outcome["success"],
            "team_reward": outcome["reward"],
        }
    if _condition_feedback(condition) == "component":
        content["public_component_feedback"] = {
            helper: {
                "responsibility_action_correct": values["responsibility_correct"],
                "item_matches": values["item_correct"],
                "destination_matches": values["destination_correct"],
            }
            for helper, values in outcome["helper_feedback"].items()
        }
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _semantic_key(record: dict, condition: str) -> tuple:
    goal = record["goal"]
    if _partner_public(condition):
        return goal["object"], goal["attribute"], goal["destination"]
    return (goal["object"], goal["attribute"], goal["partner"], goal["destination"])


def summarize(condition: str, seed: int, records: list[dict], wall_time: float,
              prompt_tokens: int, completion_tokens: int) -> dict:
    n = len(records)
    helper_rows = [
        {"episode": row["episode"], "agent": agent, "goal": row["goal"],
         "interpretation": output["interpretation"]}
        for row in records
        for agent, output in row["helper_outputs"].items()
    ]
    fields = ("object", "attribute", "responsible_helper", "destination")
    decode = {}
    for field in fields:
        correct = sum(
            row["interpretation"].get(field) == (
                row["goal"]["partner"] if field == "responsible_helper" else row["goal"][field]
            )
            for row in helper_rows
        )
        decode[field] = {"correct": correct, "total": len(helper_rows),
                         "accuracy": correct / len(helper_rows) if helper_rows else 0.0}
    exact = sum(
        all(
            row["interpretation"].get(field) == (
                row["goal"]["partner"] if field == "responsible_helper" else row["goal"][field]
            ) for field in fields
        ) for row in helper_rows
    )
    message_rows = [row for row in records if row["message"]]
    repeated: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        repeated[(row["owner"], _semantic_key(row, condition))].append(row)
    # Each owner sees every full meaning once per block; public-partner runs have two
    # partner variants of each payload, so agreement is tested over those variants too.
    final_groups: dict[tuple, list[str]] = defaultdict(list)
    for row in records:
        if row["block"] == 1 and row["message"]:
            final_groups[_semantic_key(row, condition)].append(row["message"])
    agreement = sum(len(set(messages)) == 1 for messages in final_groups.values())

    by_block = {}
    for block in sorted({row["block"] for row in records}):
        block_rows = [row for row in records if row["block"] == block]
        block_helpers = [item for item in helper_rows
                         if any(record["episode"] == item["episode"] and record["block"] == block
                                for record in records)]
        by_block[str(block)] = {
            "episodes": len(block_rows),
            "successes": sum(row["outcome"]["success"] for row in block_rows),
            "exact_helper_interpretations": sum(
                all(item["interpretation"].get(field) == (
                    item["goal"]["partner"] if field == "responsible_helper" else item["goal"][field]
                ) for field in fields) for item in block_helpers
            ),
            "helper_interpretations": len(block_helpers),
        }

    return {
        "condition": condition,
        "seed": seed,
        "episodes": n,
        "model_calls": 3 * n,
        "wall_time_seconds": wall_time,
        "prompt_tokens_reported_by_server": prompt_tokens,
        "completion_tokens_reported_by_server": completion_tokens,
        "joint_successes": sum(row["outcome"]["success"] for row in records),
        "designated_item_correct": sum(row["outcome"]["designated_item_correct"] for row in records),
        "designated_destination_correct": sum(row["outcome"]["designated_destination_correct"] for row in records),
        "unassigned_waited": sum(row["outcome"]["unassigned_helper_waited"] for row in records),
        "owner_json_valid": sum(row["owner_json_valid"] for row in records),
        "owner_message_valid": sum(row["message_valid"] for row in records),
        "helper_json_valid": sum(output["json_valid"] for row in records for output in row["helper_outputs"].values()),
        "helper_interpretation_valid": sum(output["interpretation_valid"] for row in records for output in row["helper_outputs"].values()),
        "helper_action_valid": sum(output["action_valid"] for row in records for output in row["helper_outputs"].values()),
        "exact_codebook_encodings": sum(row["codebook_correct"] is True for row in records),
        "helper_field_accuracy": decode,
        "exact_helper_interpretations": exact,
        "helper_interpretations": len(helper_rows),
        "distinct_nonempty_messages": len({row["message"] for row in message_rows}),
        "message_entropy_bits": _entropy([row["message"] for row in message_rows]),
        "message_meaning_mutual_information_bits": _mi(message_rows, "message", "meaning_id"),
        "message_payload_mutual_information_bits": _mi(message_rows, "message", "payload_id"),
        "message_partner_mutual_information_bits": _mi(message_rows, "message", "partner_position"),
        "message_requester_mutual_information_bits": _mi(message_rows, "message", "owner"),
        "within_owner_repeat_exact": sum(
            len({row["message"] for row in rows}) == 1
            for rows in repeated.values() if len(rows) > 1
        ),
        "within_owner_repeat_groups": sum(len(rows) > 1 for rows in repeated.values()),
        "final_block_shared_form_groups": agreement,
        "final_block_semantic_groups": len(final_groups),
        "by_block": by_block,
    }


def _condition_order(seed_index: int) -> list[str]:
    offset = seed_index % len(CONDITIONS)
    return list(CONDITIONS[offset:]) + list(CONDITIONS[:offset])


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_condition(base_url: str, model: str, seed: int, condition: str,
                  temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                  timeout: int = 600, progress: bool = False) -> dict:
    histories = {agent: [{"role": "system", "content": SYSTEM_PROMPT}] for agent in AGENTS}
    records = []
    prompt_tokens = completion_tokens = 0
    started = time.time()
    for index in range(TOTAL_EPISODES):
        episode = make_episode(index, seed)
        owner = episode["owner"]
        owner_history = histories[owner]
        owner_history.append({"role": "user", "content": _owner_prompt(episode, condition)})
        owner_raw, usage = _post(base_url, model, owner_history, seed + index * 11,
                                 temperature, max_tokens, timeout)
        owner_result = _parse_owner(owner_raw, condition, episode["meaning_id"])
        owner_history.append({"role": "assistant", "content": owner_result["canonical"]})
        message = owner_result["message"]
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)

        actions: dict[str, dict | None] = {}
        helper_outputs = {}
        for helper_index, agent in enumerate(episode["helpers"]):
            history = histories[agent]
            history.append({"role": "user", "content": _helper_prompt(episode, agent, message, condition)})
            raw, usage = _post(base_url, model, history, seed + index * 11 + helper_index + 1,
                               temperature, max_tokens, timeout)
            parsed = _parse_helper(raw)
            history.append({"role": "assistant", "content": parsed["canonical"]})
            actions[agent] = parsed["action"]
            helper_outputs[agent] = {
                key: parsed[key] for key in
                ("json_valid", "interpretation_valid", "interpretation", "action_valid")
            }
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)

        outcome = score_episode(episode, actions)
        feedback = {}
        for agent in AGENTS:
            content = _feedback_text(episode, agent, condition, message, actions, outcome)
            histories[agent].append({"role": "user", "content": content})
            feedback[agent] = json.loads(content)

        records.append({
            "episode": index,
            "block": episode["block"],
            "meaning_id": episode["meaning_id"],
            "payload_id": episode["payload_id"],
            "partner_position": episode["goal"]["partner_position"],
            "owner": owner,
            "helpers": list(episode["helpers"]),
            "goal": episode["goal"],
            "public_board": episode["scene"],
            "target_item_id": episode["target_item_id"],
            "destinations": episode["destinations"],
            "message": message,
            "message_valid": owner_result["message_valid"],
            "codebook_correct": owner_result["codebook_correct"],
            "owner_json_valid": owner_result["json_valid"],
            "helper_outputs": helper_outputs,
            "actions": actions,
            "outcome": outcome,
            "feedback": feedback,
        })
        if progress and (index + 1) % 12 == 0:
            print(f"Progress {condition}, seed {seed}: {index + 1}/{TOTAL_EPISODES} episodes",
                  flush=True)
    wall_time = time.time() - started
    return {
        "summary": summarize(condition, seed, records, wall_time, prompt_tokens, completion_tokens),
        "episodes": records,
    }


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def run_matrix(base_url: str, model: str, seeds: tuple[int, ...], out: Path,
               temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
               timeout: int = 600, resume: bool = False) -> dict:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("formal v4 screening requires at least two unique seeds")
    if resume and out.exists():
        matrix = json.loads(out.read_text())
        expected = (CALIBRATION_VERSION, model, list(seeds), temperature, max_tokens)
        actual = (matrix["calibration_version"], matrix["model"], matrix["seeds"],
                  matrix["temperature"], matrix["max_tokens"])
        if actual != expected:
            raise ValueError("existing checkpoint configuration does not match this run")
    else:
        matrix = {
            "calibration_version": CALIBRATION_VERSION,
            "protocol_source_commit": _git_commit(),
            "model": model,
            "model_revision": MODEL_REVISION,
            "seeds": list(seeds),
            "conditions": list(CONDITIONS),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "message_alphabet": "@#%&+=?~",
            "message_length": [MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH],
            "meaning_space": [list(item) for item in MEANING_POOL],
            "payload_space": [list(item) for item in PAYLOADS],
            "known_codebook": {str(key): value for key, value in CODEBOOK.items()},
            "episode_schedule": "paired 96-episode, two-block balanced schedule per seed",
            "feedback_conditions": {condition: _condition_feedback(condition) for condition in CONDITIONS},
            "partner_visibility_conditions": {
                condition: _partner_public(condition) for condition in CONDITIONS
            },
            "condition_order_by_seed": {
                str(seed): _condition_order(index) for index, seed in enumerate(seeds)
            },
            "diagnostic_interpretation_visible_to_other_agents": False,
            "raw_model_completions_retained": False,
            "chain_of_thought_retained": False,
            "runs": [],
        }
        _write_checkpoint(out, matrix)

    completed = {(run["summary"]["condition"], run["summary"]["seed"]) for run in matrix["runs"]}
    for seed_index, seed in enumerate(seeds):
        for condition in _condition_order(seed_index):
            if (condition, seed) in completed:
                continue
            print(f"Starting {condition}, seed {seed}", flush=True)
            result = run_condition(base_url, model, seed, condition, temperature,
                                   max_tokens, timeout, progress=True)
            matrix["runs"].append({"summary": result["summary"], "episodes": result["episodes"]})
            _write_checkpoint(out, matrix)
            completed.add((condition, seed))
            summary = result["summary"]
            print(
                f"Completed {condition}, seed {seed}: "
                f"success={summary['joint_successes']}/{summary['episodes']}, "
                f"exact_decode={summary['exact_helper_interpretations']}/"
                f"{summary['helper_interpretations']}, "
                f"time={summary['wall_time_seconds']:.1f}s",
                flush=True,
            )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--condition", choices=CONDITIONS,
                        help="run one development condition and save it as a standalone record")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    if args.condition:
        if len(seeds) != 1:
            parser.error("--condition requires exactly one --seed")
        result = run_condition(args.base_url, args.model, seeds[0], args.condition,
                               args.temperature, args.max_tokens, args.timeout,
                               progress=True)
        matrix = {
            "calibration_version": CALIBRATION_VERSION,
            "protocol_source_commit": _git_commit(),
            "model": args.model,
            "model_revision": MODEL_REVISION,
            "seed": seeds[0],
            "condition": args.condition,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "message_alphabet": "@#%&+=?~",
            "message_length": [MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH],
            "known_codebook": {str(key): value for key, value in CODEBOOK.items()},
            "raw_model_completions_retained": False,
            "chain_of_thought_retained": False,
            "summary": result["summary"],
            "episodes": result["episodes"],
        }
        _write_checkpoint(args.out, matrix)
        print(json.dumps({"out": str(args.out), "summary": result["summary"]}, ensure_ascii=False))
        return
    matrix = run_matrix(args.base_url, args.model, seeds, args.out,
                        args.temperature, args.max_tokens, args.timeout, args.resume)
    print(json.dumps({
        "out": str(args.out),
        "runs": len(matrix["runs"]),
        "episodes": sum(run["summary"]["episodes"] for run in matrix["runs"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
