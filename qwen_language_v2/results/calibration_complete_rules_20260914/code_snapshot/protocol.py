"""Synchronous public communication for the v0.2 three-agent pilot.

This module has no model dependency.  A model adapter must constrain generation
before returning its string; this module rejects invalid output instead of
silently repairing it.  Research timing belongs in a separate inference log.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping


AGENTS = ("A", "B", "C")
ALPHABET = "@#%&*+=~"
WINDOWS = 4
MAX_MESSAGE_SYMBOLS = 32
PER_STEP_SYMBOLS = 64
CONDITIONS = ("immediate", "delayed")

Message = dict[str, Any]
Decide = Callable[[str, int, int, list[Message]], str]


class ProtocolError(ValueError):
    """An invalid configuration or channel output; no output is repaired."""


def message_limit(remaining: int) -> int:
    if type(remaining) is not int or not 0 <= remaining <= PER_STEP_SYMBOLS:
        raise ProtocolError("remaining must be an integer between 0 and 64")
    return min(MAX_MESSAGE_SYMBOLS, remaining)


def validate_message(text: str, remaining: int = PER_STEP_SYMBOLS) -> None:
    """Validate exact channel content, including an allowed empty message."""
    limit = message_limit(remaining)
    if not isinstance(text, str):
        raise ProtocolError("a message must be a string")
    if any(symbol not in ALPHABET for symbol in text):
        raise ProtocolError("message contains a symbol outside @#%&*+=~")
    if len(text) > limit:
        raise ProtocolError(f"message has {len(text)} symbols; allowed maximum is {limit}")


def _check_step(step: int | None) -> None:
    if step is not None and (type(step) is not int or step < 0):
        raise ProtocolError("step must be a nonnegative integer or None")


def _copy_transcript(transcript: list[Message], *, symbolic: bool = True) -> list[Message]:
    if not isinstance(transcript, list):
        raise ProtocolError("a transcript must be a list of public message records")
    for record in transcript:
        if not isinstance(record, dict):
            raise ProtocolError("each public message record must be a dictionary")
        keys = set(record)
        if keys not in ({"sender", "window", "text"}, {"sender", "window", "text", "step"}):
            raise ProtocolError("public records contain only sender, window, text and optional step")
        if record["sender"] not in AGENTS:
            raise ProtocolError("unknown sender")
        if type(record["window"]) is not int or not 1 <= record["window"] <= WINDOWS:
            raise ProtocolError("window must be an integer from 1 to 4")
        if symbolic:
            validate_message(record["text"])
        elif not isinstance(record["text"], str):
            raise ProtocolError("a natural-language message must be a string")
        if "step" in record:
            _check_step(record["step"])
    return deepcopy(transcript)


def run_communication(
    condition: str,
    decide: Decide,
    *,
    previous_transcripts: Mapping[str, list[Message]] | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Run one action step's four communication windows.

    ``decide(agent, window, own_remaining, visible_transcript)`` returns the exact
    formal message. Windows are 1-based. The callback receives a fresh copy of
    that agent's visible transcript and is invoked even when remaining is zero.
    It must then return the empty string. A callback should use
    ``message_limit(own_remaining)`` for its generation constraint.

    Calls happen in fixed A/B/C order for reproducibility. No same-window output
    is delivered until all three have returned. In the delayed condition, only
    an agent's own earlier current-step messages are visible during generation;
    all three agents receive the complete batch before physical action.

    Previous transcripts must already be public-to-the-respective-agent records;
    private observations and reasoning must not be supplied here. The original
    mapping is never mutated. The result has independent transcript copies.
    ``messages`` contains this step's researcher-visible public record;
    ``usage`` is a researcher-only budget audit, never passed to the callback.
    """
    if condition not in CONDITIONS:
        raise ProtocolError("condition must be immediate or delayed")
    if not callable(decide):
        raise ProtocolError("decide must be callable")
    _check_step(step)
    if previous_transcripts is None:
        previous_transcripts = {agent: [] for agent in AGENTS}
    if set(previous_transcripts) != set(AGENTS):
        raise ProtocolError("previous_transcripts must have exactly A, B and C")

    previous = {agent: _copy_transcript(previous_transcripts[agent]) for agent in AGENTS}
    visible = deepcopy(previous)
    remaining = {agent: PER_STEP_SYMBOLS for agent in AGENTS}
    messages: list[Message] = []
    usage: list[dict[str, Any]] = []

    for window in range(1, WINDOWS + 1):
        pending: list[Message] = []
        pending_usage: list[dict[str, Any]] = []
        for agent in AGENTS:
            before = remaining[agent]
            text = decide(agent, window, before, deepcopy(visible[agent]))
            try:
                validate_message(text, before)
            except ProtocolError as error:
                raise ProtocolError(f"agent {agent}, window {window}: {error}") from error
            record = {"sender": agent, "window": window, "text": text}
            if step is not None:
                record["step"] = step
            pending.append(record)
            pending_usage.append({
                "sender": agent,
                "window": window,
                "symbols": len(text),
                "allowed_symbols": message_limit(before),
                "remaining_before": before,
                "remaining_after": before - len(text),
                "at_message_limit": len(text) == MAX_MESSAGE_SYMBOLS,
                "at_current_limit": message_limit(before) > 0 and len(text) == message_limit(before),
            })

        # Barrier: nothing from this window becomes visible before this point.
        for record, audit in zip(pending, pending_usage):
            remaining[record["sender"]] = audit["remaining_after"]
        messages.extend(deepcopy(pending))
        usage.extend(pending_usage)
        for agent in AGENTS:
            delivered = pending if condition == "immediate" else [r for r in pending if r["sender"] == agent]
            visible[agent].extend(deepcopy(delivered))

    # Delayed own-message previews are replaced, not appended, so each message
    # appears exactly once in the final canonical window/sender order.
    transcripts = {agent: previous[agent] + deepcopy(messages) for agent in AGENTS}
    return {
        "condition": condition,
        "transcripts": transcripts,
        "messages": deepcopy(messages),
        "remaining": remaining,
        "usage": usage,
    }


def run_natural_communication(
    decide: Callable[[str, int, None, list[Message]], str],
    *,
    step: int | None = None,
    previous_transcripts: Mapping[str, list[Message]] | None = None,
) -> dict[str, Any]:
    """Four immediate public windows for an independent capability control.

    The callback receives ``remaining=None``. Its model adapter, not this
    pure-Python scheduler, must enforce the declared 96-token output limit.
    Natural language is not a bandwidth-matched emergence condition. It must use
    independent histories and must not enter ``run_communication`` transcripts.
    """
    if not callable(decide):
        raise ProtocolError("decide must be callable")
    _check_step(step)
    if previous_transcripts is None:
        previous_transcripts = {agent: [] for agent in AGENTS}
    if set(previous_transcripts) != set(AGENTS):
        raise ProtocolError("previous_transcripts must have exactly A, B and C")
    visible = {agent: _copy_transcript(previous_transcripts[agent], symbolic=False) for agent in AGENTS}
    messages: list[Message] = []
    usage: list[dict[str, Any]] = []
    for window in range(1, WINDOWS + 1):
        pending: list[Message] = []
        for agent in AGENTS:
            text = decide(agent, window, None, deepcopy(visible[agent]))
            if not isinstance(text, str):
                raise ProtocolError(f"agent {agent}, window {window}: natural-language message must be a string")
            record = {"sender": agent, "window": window, "text": text}
            if step is not None:
                record["step"] = step
            pending.append(record)
            usage.append({"sender": agent, "window": window, "characters": len(text)})
        # Same barrier as the symbolic condition: never expose earlier calls
        # inside the current window, even when callback execution is sequential.
        messages.extend(deepcopy(pending))
        for agent in AGENTS:
            visible[agent].extend(deepcopy(pending))
    return {
        "condition": "natural_language",
        "transcripts": visible,
        "messages": deepcopy(messages),
        "remaining": dict.fromkeys(AGENTS, None),
        "usage": usage,
    }


_PRIVATE_ANALYSIS_KEYS = frozenset({
    "analysis", "private_analysis", "reasoning", "thought", "thoughts",
    "rendered_prompt", "raw_prompt",
})


def _check_no_analysis(value: Any) -> None:
    if isinstance(value, dict):
        if set(value) & _PRIVATE_ANALYSIS_KEYS:
            raise ProtocolError("private analysis or raw prompts cannot enter persistent history")
        for item in value.values():
            _check_no_analysis(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_no_analysis(item)


class PrivateHistory:
    """Own observation/action/visible-result memory with explicit input fields.

    Environment privacy remains the environment adapter's responsibility: this
    class cannot infer which physical facts an observation should contain.
    Private analysis has no persistence field and known analysis keys are rejected.
    """

    def __init__(self, agent: str) -> None:
        if agent not in AGENTS:
            raise ProtocolError("unknown history owner")
        self.agent = agent
        self._steps: list[dict[str, Any]] = []

    def append_step(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        feedback: dict[str, Any],
        transcript: list[Message],
        episode: int | None = None,
    ) -> None:
        _check_step(step)
        if step is None:
            raise ProtocolError("history step cannot be None")
        _check_step(episode)
        if not all(isinstance(value, dict) for value in (observation, action, feedback)):
            raise ProtocolError("observation, action and feedback must be dictionaries")
        record = {
            "step": step,
            "observation": observation,
            "action": action,
            "feedback": feedback,
            "transcript": _copy_transcript(transcript),
        }
        if episode is not None:
            record["episode"] = episode
        _check_no_analysis(record)
        self._steps.append(deepcopy(record))

    def snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self._steps)

    def fork(self) -> "PrivateHistory":
        result = PrivateHistory(self.agent)
        result._steps = self.snapshot()
        return result


def new_private_histories() -> dict[str, PrivateHistory]:
    return {agent: PrivateHistory(agent) for agent in AGENTS}
