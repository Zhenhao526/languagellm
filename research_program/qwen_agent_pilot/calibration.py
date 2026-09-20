"""Matched channel calibration: blank, known codebook, and free symbols."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import requests

from .environment import AGENTS, MEANING_POOL, PILOT_EPISODES, make_episode, score_episode
from .pilot import (
    ALPHABET, MAX_MESSAGE_LENGTH, MIN_MESSAGE_LENGTH, MODEL_ID, SYSTEM,
    _feedback_helper, _feedback_owner, _helper_prompt, _owner_prompt, _post,
    parse_output, valid_message,
)

CONDITIONS = ("blank", "known_codebook", "free_symbols")
CALIBRATION_VERSION = "matched_channel_calibration_v2_1_clear_codebook_action"
DEFAULT_SEEDS = (20260925, 20260926, 20260927)
CALIBRATION_SYSTEM = SYSTEM + (
    "\nEvery order has exactly one designated helper. Only that helper should act; "
    "the other helper must wait. If you cannot confidently identify yourself as the "
    "designated helper from the information available to you, wait rather than duplicate work."
)

# Six opaque, fixed-length codewords, chosen before collection and held constant
# across seeds. Each pair has Hamming distance >= 4 over the permitted alphabet.
CODEWORDS = ("+%~?=&", "~=@%+~", "?##&+?", "%?~~&=", "~=~&?&", "@%=~#+")
CODEBOOK = {meaning_id: CODEWORDS[meaning_id] for meaning_id in range(len(MEANING_POOL))}


def _mi(rows: list[dict], x_key: str, y_key: str) -> float:
    n = len(rows)
    joint = Counter((row[x_key], row[y_key]) for row in rows)
    xs = Counter(row[x_key] for row in rows)
    ys = Counter(row[y_key] for row in rows)
    return sum(count / n * math.log2(count * n / (xs[x] * ys[y]))
               for (x, y), count in joint.items()) if n else 0.0


def _owner_codebook(episode: dict) -> str:
    rows = []
    for meaning_id, (obj, attr, partner_position, destination) in enumerate(MEANING_POOL):
        rows.append({
            "object": obj,
            "attribute": attr,
            "responsible_helper_position": partner_position,
            "destination": destination,
            "message": CODEBOOK[meaning_id],
        })
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _helper_codebook(episode: dict) -> str:
    rows = []
    for meaning_id, (obj, attr, partner_position, destination) in enumerate(MEANING_POOL):
        rows.append({
            "message": CODEBOOK[meaning_id],
            "object": obj,
            "attribute": attr,
            "responsible_helper": episode["helpers"][partner_position],
            "destination": destination,
        })
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _condition_owner_prompt(episode: dict, condition: str) -> str:
    if condition == "free_symbols":
        return _owner_prompt(episode)
    if condition == "blank":
        goal = episode["goal"]
        return (
            f"Round {episode['episode']}. Your identity is {episode['owner']}; you are the requester.\n"
            f"Private order: object={goal['object']}; attribute={goal['attribute']}; "
            f"responsible partner={goal['partner']}; destination={goal['destination']}.\n"
            "The communication channel is closed. No symbols or other content can be sent to the helpers. "
            "Return exactly JSON: {\"message\":\"\"}."
        )
    if condition == "known_codebook":
        goal = episode["goal"]
        return (
            f"Round {episode['episode']}. Your identity is {episode['owner']}; you are the requester.\n"
            f"Private order: object={goal['object']}; attribute={goal['attribute']}; "
            f"responsible partner={goal['partner']} (helper position {goal['partner_position']}); "
            f"destination={goal['destination']}.\n"
            "Both sides already know this fixed mapping from complete orders to opaque messages. "
            "Choose the message for the exact matching order. Do not alter or explain the code.\n"
            f"Shared codebook: {_owner_codebook(episode)}\n"
            "Return JSON only: {\"message\":\"CODE\"}."
        )
    raise ValueError(f"unknown condition: {condition}")


def _condition_helper_prompt(episode: dict, agent: str, message: str, condition: str) -> str:
    prompt = _helper_prompt(episode, agent, message)
    prompt += (
        "\nAction-allocation rule: exactly one helper is responsible for this order. "
        "Only the responsible helper should select an item and destination; the other helper "
        "must wait by returning null. If you cannot confidently determine that you are the "
        "responsible helper, wait rather than duplicate the action."
    )
    if condition == "known_codebook":
        prompt = prompt.replace(
            "You do not know the private order.",
            "Use the exact codebook below to decode the private order.",
        )
        prompt += (
            "\nBoth sides already know this fixed mapping from opaque messages to complete orders. "
            "Follow this procedure exactly: find the row whose message equals the received string. "
            "If that row's responsible_helper equals your identity, act on its object, attribute and "
            "destination. If the row names a different helper, return null and wait. "
            "If no row matches, return null and wait. Do not abstain when the matching row names you.\n"
            f"Shared codebook: {_helper_codebook(episode)}"
        )
    elif condition == "blank":
        prompt += "\nThe communication channel is closed; the empty message contains no information."
    elif condition != "free_symbols":
        raise ValueError(f"unknown condition: {condition}")
    return prompt


def _summarize(condition: str, seed: int, records: list[dict], wall_time: float,
               prompt_tokens: int, completion_tokens: int) -> dict:
    n = len(records)
    blocks = sorted({row["block"] for row in records})
    meanings = sorted({row["meaning_id"] for row in records})
    by_block = {
        str(block): {
            "n": sum(row["block"] == block for row in records),
            "team_success": sum(row["outcome"]["success"] for row in records if row["block"] == block),
            "designated_item_correct": sum(row["outcome"]["designated_item_correct"] for row in records if row["block"] == block),
            "designated_destination_correct": sum(row["outcome"]["designated_destination_correct"] for row in records if row["block"] == block),
            "unassigned_helper_waited": sum(row["outcome"]["unassigned_helper_waited"] for row in records if row["block"] == block),
        } for block in blocks
    }
    pairs: dict[tuple[str, int], list[dict]] = {}
    for row in records:
        pairs.setdefault((row["owner"], row["meaning_id"]), []).append(row)
    repeated = [pair for pair in pairs.values() if len(pair) == 2]
    stable = sum(pair[0]["message"] == pair[1]["message"] for pair in repeated)
    final_by_meaning: dict[int, list[dict]] = {}
    for (owner, meaning_id), pair in pairs.items():
        if pair:
            final_by_meaning.setdefault(meaning_id, []).append(max(pair, key=lambda row: row["block"]))
    cross_sender = sum(
        len(final_by_meaning.get(meaning, [])) == len(AGENTS)
        and len({row["message"] for row in final_by_meaning[meaning]}) == 1
        for meaning in meanings
    )
    syntax_records = [r for r in records if r["message"]]
    return {
        "condition": condition,
        "seed": seed,
        "episodes": n,
        "wall_time_seconds": wall_time,
        "model_calls": 3 * n,
        "prompt_tokens_reported_by_server": prompt_tokens,
        "completion_tokens_reported_by_server": completion_tokens,
        "owner_json_valid_rate": sum(r["owner_json_valid"] for r in records) / n,
        "helper_json_valid_rate": (
            sum(result["json_valid"] for r in records for result in r["helper_results"].values())
            / (2 * n)
        ),
        "helper_action_valid_rate": (
            sum(result["action_valid"] for r in records for result in r["helper_results"].values())
            / (2 * n)
        ),
        "valid_message_rate_nonblank_channels": (
            sum(r["message_valid"] is True for r in records) / n
            if condition != "blank" else None
        ),
        "known_codebook_encoder_accuracy": (
            sum(r["encoder_correct"] for r in records) / n
            if condition == "known_codebook" else None
        ),
        "unique_messages": len({r["message"] for r in syntax_records}),
        "sender_meaning_exact_repeat_rate": (
            stable / len(repeated) if repeated and condition != "blank" else None
        ),
        "cross_sender_agreement_final_block": (
            cross_sender / len(meanings) if meanings and condition != "blank" else None
        ),
        "message_owner_mutual_information_bits": (
            _mi(records, "message", "owner") if condition != "blank" else None
        ),
        "message_meaning_mutual_information_bits": (
            _mi(records, "message", "meaning_id") if condition != "blank" else None
        ),
        "team_success_rate": sum(r["outcome"]["success"] for r in records) / n,
        "designated_item_accuracy": sum(r["outcome"]["designated_item_correct"] for r in records) / n,
        "designated_destination_accuracy": sum(r["outcome"]["designated_destination_correct"] for r in records) / n,
        "designated_helper_both_correct_rate": sum(r["outcome"]["designated_helper_correct"] for r in records) / n,
        "unassigned_wait_rate": sum(r["outcome"]["unassigned_helper_waited"] for r in records) / n,
        "team_success_by_block": {key: value["team_success"] / value["n"] for key, value in by_block.items()},
        "metrics_by_block": by_block,
        "records": records,
    }


def run_condition(base_url: str, model: str, seed: int, condition: str,
                  temperature: float = 0.35, max_tokens: int = 120,
                  timeout: int = 600) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    requests.get(base_url.rstrip("/") + "/models", timeout=timeout).raise_for_status()
    histories = {agent: [{"role": "system", "content": CALIBRATION_SYSTEM}] for agent in AGENTS}
    records = []
    prompt_tokens = completion_tokens = 0
    started = time.time()
    for index in range(PILOT_EPISODES):
        episode = make_episode(index, seed)
        owner = episode["owner"]
        owner_history = histories[owner]
        owner_history.append({"role": "user", "content": _condition_owner_prompt(episode, condition)})
        owner_raw, usage = _post(base_url, model, owner_history, seed + index * 11,
                                 temperature, max_tokens, timeout)
        owner_result = parse_output(owner_raw, "owner")
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        generated = owner_result["message"] or ""
        if condition == "blank":
            message = ""
            encoder_correct = None
            message_valid = None
            canonical_turn = json.dumps({"message": ""}, separators=(",", ":"))
        else:
            message = generated
            message_valid = valid_message(message)
            encoder_correct = (
                message == CODEBOOK[episode["meaning_id"]]
                if condition == "known_codebook" else None
            )
            canonical_turn = owner_result["canonical_assistant_turn"]
        owner_history.append({"role": "assistant", "content": canonical_turn})

        actions = {}
        helper_results = {}
        for j, agent in enumerate(episode["helpers"]):
            history = histories[agent]
            history.append({"role": "user", "content": _condition_helper_prompt(episode, agent, message, condition)})
            raw, usage = _post(base_url, model, history, seed + index * 11 + j + 1,
                               temperature, max_tokens, timeout)
            parsed = parse_output(raw, "helper")
            history.append({"role": "assistant", "content": parsed["canonical_assistant_turn"]})
            actions[agent] = parsed["action"]
            helper_results[agent] = {key: parsed[key] for key in ("json_valid", "action_valid")}
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)

        outcome = score_episode(episode, actions)
        owner_history.append({"role": "user", "content": _feedback_owner(episode, outcome, message, actions)})
        for agent in episode["helpers"]:
            other = next(a for a in episode["helpers"] if a != agent)
            histories[agent].append({"role": "user", "content": _feedback_helper(
                episode, agent, message, actions[agent], actions[other], outcome)})

        records.append({
            "episode": index,
            "block": episode["block"],
            "meaning_id": episode["meaning_id"],
            "owner": owner,
            "goal": episode["goal"],
            "public_board": episode["scene"],
            "target_item_id": episode["target_item_id"],
            "destinations": episode["destinations"],
            "message": message,
            "message_valid": message_valid,
            "encoder_correct": encoder_correct,
            "owner_json_valid": owner_result["json_valid"],
            "helper_results": helper_results,
            "actions": actions,
            "outcome": outcome,
        })

    return _summarize(condition, seed, records, time.time() - started,
                      prompt_tokens, completion_tokens)


def _condition_order(seed_index: int) -> list[str]:
    offset = seed_index % len(CONDITIONS)
    return list(CONDITIONS[offset:]) + list(CONDITIONS[:offset])


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_matrix(base_url: str, model: str, seeds: tuple[int, ...], out: Path,
               temperature: float = 0.35, max_tokens: int = 120,
               timeout: int = 600, resume: bool = False) -> dict:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("screening matrix requires at least two unique paired seeds")
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
            "model_revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
            "seeds": list(seeds),
            "conditions": list(CONDITIONS),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "message_alphabet": "@#%&+=?~",
            "message_length": [MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH],
            "known_codebook": {str(key): value for key, value in CODEBOOK.items()},
            "episode_schedule": "same balanced 36 episodes within each paired seed and condition",
            "condition_order_by_seed": {
                str(seed): _condition_order(index) for index, seed in enumerate(seeds)
            },
            "paired_decoding_seeds": True,
            "raw_model_completions_retained": False,
            "chain_of_thought_retained": False,
            "runs": [],
        }
        _write_checkpoint(out, matrix)

    completed = {(run["condition"], run["seed"]) for run in matrix["runs"]}
    for seed_index, seed in enumerate(seeds):
        for condition in _condition_order(seed_index):
            if (condition, seed) in completed:
                continue
            result = run_condition(base_url, model, seed, condition,
                                   temperature, max_tokens, timeout)
            matrix["runs"].append(result)
            _write_checkpoint(out, matrix)
    matrix["runs"].sort(key=lambda run: (seeds.index(run["seed"]), CONDITIONS.index(run["condition"])))
    matrix["paired_seed_contrasts"] = [
        {
            "seed": seed,
            "success_rates": {
                condition: next(run for run in matrix["runs"]
                               if run["seed"] == seed and run["condition"] == condition)["team_success_rate"]
                for condition in CONDITIONS
            },
            "free_minus_blank_success_rate": (
                next(run for run in matrix["runs"] if run["seed"] == seed and run["condition"] == "free_symbols")["team_success_rate"]
                - next(run for run in matrix["runs"] if run["seed"] == seed and run["condition"] == "blank")["team_success_rate"]
            ),
            "free_minus_codebook_success_rate": (
                next(run for run in matrix["runs"] if run["seed"] == seed and run["condition"] == "free_symbols")["team_success_rate"]
                - next(run for run in matrix["runs"] if run["seed"] == seed and run["condition"] == "known_codebook")["team_success_rate"]
            ),
        } for seed in seeds
    ] if len(matrix["runs"]) == len(seeds) * len(CONDITIONS) else []
    matrix["screening_only"] = True
    _write_checkpoint(out, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_matrix(args.base_url, args.model, tuple(args.seeds), Path(args.out),
                        args.temperature, args.max_tokens, args.timeout, args.resume)
    print(json.dumps({
        "status": "completed" if len(result["runs"]) == len(args.seeds) * len(CONDITIONS) else "checkpointed",
        "runs": len(result["runs"]),
        "paired_seed_contrasts": result.get("paired_seed_contrasts", []),
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
