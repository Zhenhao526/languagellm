"""Run the exploratory three-context pilot against a local mlx-vlm server."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

from .environment import AGENTS, PILOT_EPISODES, make_episode, score_episode

ALPHABET = set("@#%&+=?~")
MIN_MESSAGE_LENGTH = 4
MAX_MESSAGE_LENGTH = 8
MODEL_ID = "mlx-community/Qwen3.5-9B-8bit"

SYSTEM = """You are one member of a small collection group. Your private conversation history belongs only to you. Other agents never see your reasoning or natural-language text. Between agents, the only communication is the broadcast symbol string in the task. Do not send natural language to other agents. Use the task feedback to improve your future messages or actions. Do not reveal chain-of-thought. Return only one JSON object matching the requested schema."""


def valid_message(value: object) -> bool:
    return (isinstance(value, str)
            and MIN_MESSAGE_LENGTH <= len(value) <= MAX_MESSAGE_LENGTH
            and all(char in ALPHABET for char in value))


def _visible_json(raw: str) -> str:
    """Remove any hidden-thought section and simple Markdown fence before parse."""
    value = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
    return value


def parse_output(raw: str, role: str) -> dict:
    if role not in {"owner", "helper"}:
        raise ValueError(f"unknown role: {role}")
    cleaned = _visible_json(raw)
    try:
        data = json.loads(cleaned)
        json_valid = isinstance(data, dict)
        if not json_valid:
            data = {}
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
            json_valid = isinstance(data, dict)
            if not json_valid:
                data = {}
        except json.JSONDecodeError:
            data, json_valid = {}, False

    message = data.get("message") if role == "owner" else None
    message_ok = role != "owner" or valid_message(message)
    action = None
    action_ok = role == "owner"
    if role == "helper":
        candidate = data.get("action")
        if "action" in data and candidate is None:
            action_ok = json_valid
        elif isinstance(candidate, dict):
            item_id = candidate.get("item_id")
            destination = candidate.get("destination")
            if (isinstance(item_id, str) and re.fullmatch(r"I[0-3]", item_id)
                    and isinstance(destination, str)):
                action, action_ok = {"item_id": item_id, "destination": destination}, True
    # Invalid symbol strings never cross the inter-agent channel.
    safe_message = message if role == "owner" and message_ok else ("" if role == "owner" else None)
    return {
        "json_valid": bool(json_valid),
        "message": safe_message,
        "message_valid": bool(message_ok),
        "action": action,
        "action_valid": bool(action_ok),
        "canonical_assistant_turn": json.dumps(
            {"message": safe_message} if role == "owner" else {"action": action},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _scene_text(episode: dict) -> str:
    return json.dumps({"candidates": episode["scene"], "destinations": episode["destinations"]},
                      ensure_ascii=False, separators=(",", ":"))


def _owner_prompt(episode: dict) -> str:
    goal = episode["goal"]
    return (
        f"Round {episode['episode']}. Your identity is {episode['owner']}; you are the requester.\n"
        f"Private order: object={goal['object']}; attribute={goal['attribute']}; "
        f"responsible partner={goal['partner']}; destination={goal['destination']}.\n"
        f"The two helpers are {episode['helpers'][0]} and {episode['helpers'][1]}. "
        "Broadcast exactly one opaque message so they can coordinate. Do not write the order in words. "
        f"The message must contain 4–8 characters, all chosen from: {''.join(sorted(ALPHABET))}.\n"
        "Return JSON only: {\"message\":\"SYMBOLS\"}."
    )


def _helper_prompt(episode: dict, agent: str, message: str) -> str:
    return (
        f"Round {episode['episode']}. Your identity is {agent}; you are a helper.\n"
        f"Requester {episode['owner']} broadcast this symbol string: {json.dumps(message)}\n"
        f"Public collection board: {_scene_text(episode)}\n"
        "Choose one candidate item and one destination, or wait by returning null. "
        "Do not send a natural-language reply. You do not know the private order.\n"
        "Return JSON only: {\"action\":null} or "
        "{\"action\":{\"item_id\":\"I0\",\"destination\":\"river camp\"}}."
    )


def _post(base_url: str, model: str, messages: list[dict], seed: int,
          temperature: float, max_tokens: int, timeout: int) -> tuple[str, dict]:
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": messages, "temperature": temperature,
              "top_p": 0.95, "max_tokens": max_tokens, "seed": seed, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("server returned non-text message content")
    return content, payload.get("usage", {})


def _feedback_owner(episode: dict, outcome: dict, message: str, actions: dict) -> str:
    return json.dumps({"round": episode["episode"], "your_broadcast": message,
                       "helper_actions": actions, "team_success": outcome["success"],
                       "team_reward": outcome["reward"]}, ensure_ascii=False, separators=(",", ":"))


def _feedback_helper(episode: dict, agent: str, message: str, own_action: dict | None,
                     other_action: dict | None, outcome: dict) -> str:
    return json.dumps({"round": episode["episode"], "received_symbols": message,
                       "your_action": own_action, "other_helper_action": other_action,
                       "team_success": outcome["success"], "team_reward": outcome["reward"]},
                      ensure_ascii=False, separators=(",", ":"))


def run_pilot(base_url: str, model: str, seed: int, episodes: int = PILOT_EPISODES,
              temperature: float = 0.35, max_tokens: int = 120, timeout: int = 600) -> dict:
    if episodes != PILOT_EPISODES:
        raise ValueError(f"this pilot plan is frozen at {PILOT_EPISODES} balanced episodes")
    requests.get(base_url.rstrip("/") + "/models", timeout=timeout).raise_for_status()
    histories = {agent: [{"role": "system", "content": SYSTEM}] for agent in AGENTS}
    records = []
    prompt_tokens = completion_tokens = 0
    start = time.time()

    for index in range(episodes):
        episode = make_episode(index, seed)
        owner = episode["owner"]
        owner_history = histories[owner]
        owner_history.append({"role": "user", "content": _owner_prompt(episode)})
        owner_raw, usage = _post(base_url, model, owner_history, seed + index * 11,
                                 temperature, max_tokens, timeout)
        owner_result = parse_output(owner_raw, "owner")
        owner_history.append({"role": "assistant", "content": owner_result["canonical_assistant_turn"]})
        message = owner_result["message"] or ""
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)

        actions = {}
        helper_results = {}
        for j, agent in enumerate(episode["helpers"]):
            history = histories[agent]
            history.append({"role": "user", "content": _helper_prompt(episode, agent, message)})
            raw, usage = _post(base_url, model, history, seed + index * 11 + j + 1,
                               temperature, max_tokens, timeout)
            parsed = parse_output(raw, "helper")
            history.append({"role": "assistant", "content": parsed["canonical_assistant_turn"]})
            actions[agent] = parsed["action"]
            helper_results[agent] = {k: parsed[k] for k in ("json_valid", "action_valid")}
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
            "message": message,
            "message_valid": owner_result["message_valid"],
            "owner_json_valid": owner_result["json_valid"],
            "helper_results": helper_results,
            "actions": actions,
            "outcome": outcome,
        })

    repeat_pairs = {}
    for record in records:
        repeat_pairs.setdefault((record["owner"], record["meaning_id"]), []).append(record)
    pair_message_matches = [
        len(pair) == 2
        and all(record["message_valid"] for record in pair)
        and pair[0]["message"] == pair[1]["message"]
        for pair in repeat_pairs.values()
    ]
    both_valid_pairs = [
        pair for pair in repeat_pairs.values()
        if len(pair) == 2 and all(record["message_valid"] for record in pair)
    ]
    final_messages_by_meaning = {}
    for (owner, meaning_id), pair in repeat_pairs.items():
        last = max(pair, key=lambda record: record["block"])
        final_messages_by_meaning.setdefault(meaning_id, []).append(last)
    cross_sender_match = []
    cross_sender_valid = []
    for meaning_id in range(6):
        senders = final_messages_by_meaning.get(meaning_id, [])
        all_valid = len(senders) == len(AGENTS) and all(x["message_valid"] for x in senders)
        cross_sender_valid.append(all_valid)
        cross_sender_match.append(all_valid and len({x["message"] for x in senders}) == 1)

    return {
        "schema": "qwen_three_context_collection_pilot_v2",
        "task_version": "balanced_six_meaning_two_block_v2",
        "model": model,
        "model_revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
        "seed": seed,
        "episodes": episodes,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "message_alphabet": "@#%&+=?~",
        "message_length": [MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH],
        "independent_contexts": list(AGENTS),
        "wall_time_seconds": time.time() - start,
        "prompt_tokens_reported_by_server": prompt_tokens,
        "completion_tokens_reported_by_server": completion_tokens,
        "valid_message_rate": sum(r["message_valid"] for r in records) / episodes,
        "owner_meaning_repeat_pairs": len(repeat_pairs),
        "exact_message_repeat_rate_all_pairs": sum(pair_message_matches) / len(pair_message_matches),
        "exact_message_repeat_rate_conditional_on_valid": (
            sum(pair[0]["message"] == pair[1]["message"] for pair in both_valid_pairs)
            / len(both_valid_pairs) if both_valid_pairs else None
        ),
        "cross_sender_message_agreement_rate_all_meanings": sum(cross_sender_match) / 6,
        "cross_sender_message_agreement_rate_if_all_valid": (
            sum(cross_sender_match) / sum(cross_sender_valid) if any(cross_sender_valid) else None
        ),
        "unique_valid_message_count": len({r["message"] for r in records if r["message_valid"]}),
        "team_success_rate_by_block": {
            str(block): sum(r["outcome"]["success"] for r in records if r["block"] == block)
            / sum(r["block"] == block for r in records)
            for block in sorted({r["block"] for r in records})
        },
        "team_success_rate_by_requester_and_block": {
            f"{owner}_block_{block}": sum(
                r["outcome"]["success"] for r in records
                if r["owner"] == owner and r["block"] == block
            ) / sum(r["owner"] == owner and r["block"] == block for r in records)
            for owner in AGENTS for block in sorted({r["block"] for r in records})
        },
        "records": records,
        "raw_model_completions_retained": False,
        "chain_of_thought_retained": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--episodes", type=int, default=PILOT_EPISODES)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_pilot(args.base_url, args.model, args.seed, args.episodes,
                       args.temperature, args.max_tokens, args.timeout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "completed", "episodes": result["episodes"],
                      "valid_message_rate": result["valid_message_rate"],
                      "success_by_block": result["team_success_rate_by_block"],
                      "seconds": result["wall_time_seconds"]}, ensure_ascii=False))
