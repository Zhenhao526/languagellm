"""Finite token-sequence action choices; imports no model runtime until init/infer.

Numeric paths retain the original per-digit encoding. Semantic paths use the
frozen tokenizer's complete canonical encodings, never one-token-per-character.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time

from qwen_language_v3.backend import Backend


ANALYSIS_REQUEST = (
    "先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。"
    "他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。"
    "核对可选动作编号与动作描述；只依据自己的可见信息选择。"
)
FORMAL_INSTRUCTIONS = {
    "number": "只输出你选择的一个本地动作编号，不要解释。",
    "semantic": "只输出你选择的那一项的完整动作原文，与菜单中的动作文字完全一致；不要编号或解释。",
}


def allowed_next(prefix, sequences, eos):
    """Exact next-token set, including EOS only at a complete choice.

    Streaming may issue a discarded look-ahead after a legal EOS. Only that
    explicitly verified terminal extension is accepted; arbitrary invalid
    prefixes never fall back to EOS.
    """
    prefix, sequences, eos = tuple(prefix), tuple(tuple(s) for s in sequences), tuple(eos)
    if not sequences or any(not s or any(t in eos for t in s) for s in sequences):
        raise ValueError("Choices must be nonempty token sequences without EOS")
    if prefix and prefix[-1] in eos:
        if prefix[:-1] not in sequences:
            raise ValueError("EOS did not follow a complete choice")
        return sorted(set(eos))
    matches = [s for s in sequences if s[:len(prefix)] == prefix]
    if not matches:
        raise ValueError("Generated prefix is outside the complete choice set")
    next_ids = {s[len(prefix)] for s in matches if len(s) > len(prefix)}
    if prefix in sequences:
        next_ids.update(eos)
    return sorted(next_ids)


class TokenSequenceMask:
    def __init__(self, mx, sequences, eos):
        self.mx, self.sequences, self.eos = mx, tuple(tuple(s) for s in sequences), tuple(eos)
        self.calls = 0
        allowed_next((), self.sequences, self.eos)

    def __call__(self, tokens, logits):
        prefix = tuple(int(t) for t in tokens[-self.calls:].tolist()) if self.calls else ()
        allowed = allowed_next(prefix, self.sequences, self.eos)
        self.calls += 1
        mask = self.mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        mask[..., self.mx.array(allowed)] = 0
        return logits + mask


class InterfaceBackend(Backend):
    def __init__(self, model_dir, log_path, *, formal_max_tokens, tokenization):
        super().__init__(model_dir, log_path)
        self.formal_max_tokens = formal_max_tokens
        self.tokenization = deepcopy(tokenization)
        if sorted(self.eos) != sorted(tokenization["eos_token_ids"]):
            raise ValueError("Loaded tokenizer EOS differs from frozen preparation")
        for text, sequence in tokenization["semantic_sequences"].items():
            if self.tokenizer.encode(text, add_special_tokens=False) != sequence or self.tokenizer.decode(sequence) != text:
                raise ValueError("Loaded tokenizer semantic encoding differs")
        for text, sequence in tokenization["number_sequences"].items():
            if sequence != [self.char_ids[c] for c in text] or self.tokenizer.decode(sequence) != text:
                raise ValueError("Loaded tokenizer digit encoding differs")

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0):
        # Natural communication and analysis keep the old implementation exactly.
        if mode not in ("action_number", "action_semantic"):
            return super().infer(messages, mode=mode, label=label, seed=seed, limit=limit,
                                 choices=choices, temperature=temperature)
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        arm = mode.removeprefix("action_")
        key = "number_sequences" if arm == "number" else "semantic_sequences"
        choices = tuple(choices or ())
        if len(choices) != 17 or len(set(choices)) != 17:
            raise ValueError("All17 distinct physical choices are required")
        sequences = [self.tokenization[key][text] for text in choices]
        if self.formal_max_tokens < max(map(len, sequences)) + 1:
            raise ValueError("Frozen formal budget cannot hold the longest candidate and EOS")
        self.mx.random.seed(seed)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        parts, last = [], None
        for out in stream_generate(self.model, self.tokenizer, prompt, max_tokens=self.formal_max_tokens,
                sampler=make_sampler(temp=temperature), logits_processors=[TokenSequenceMask(self.mx, sequences, self.eos)],
                prompt_cache=None, prefill_step_size=512):
            parts.append(out.text)
            last = out
        if last is None:
            raise RuntimeError("Generation returned no record")
        result, elapsed = "".join(parts), time.perf_counter() - started
        valid = result in choices
        self.calls += 1
        self.elapsed += elapsed
        self.max_prompt = max(self.max_prompt, last.prompt_tokens)
        self.tokens[mode] = self.tokens.get(mode, 0) + last.generation_tokens
        entry = dict(call=self.calls, label=deepcopy(label), mode=mode, seed=seed, temperature=temperature,
            symbol_limit=None, messages=deepcopy(messages), choices=list(choices),
            allowed_token_sequences=sequences, max_generation_tokens=self.formal_max_tokens,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), output=result, valid=valid,
            started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(), seconds=elapsed,
            prompt_tokens=last.prompt_tokens, generation_tokens=last.generation_tokens,
            prompt_tps=last.prompt_tps, generation_tps=last.generation_tps,
            peak_memory_gb=last.peak_memory, finish_reason=last.finish_reason,
            fresh_cache=True, native_thinking=False)
        self.log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.log.flush()
        if not valid:
            raise RuntimeError(f"Invalid finite-sequence output: {result!r}")
        return result

    def decide_action(self, messages, *, arm, choices, label, seed):
        analysis = self.infer(deepcopy(messages) + [{"role": "user", "content": ANALYSIS_REQUEST}],
                              mode="private_analysis", label={**label, "stage": "analysis"}, seed=seed)
        return self.infer(deepcopy(messages) + [{"role": "assistant", "content": analysis},
            {"role": "user", "content": FORMAL_INSTRUCTIONS[arm]}], mode="action_" + arm,
            label={**label, "stage": "formal"}, seed=seed, choices=choices, temperature=0)
