"""Fixed local Qwen; per-call fresh cache; source-constrained messages/actions."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ALPHABET = '@#%&*+=~'


def allowed_choice_tokens(prefix, choices):
    """Return next characters and whether this prefix can terminate."""
    strings = tuple(str(x) for x in choices)
    return sorted({s[len(prefix)] for s in strings if s.startswith(prefix) and len(s) > len(prefix)}), prefix in strings


class SymbolMask:
    def __init__(self, mx, ids, eos, limit):
        self.mx, self.ids, self.eos, self.limit = mx, ids, eos, limit
        self.calls = 0

    def __call__(self, tokens, logits):
        allowed = self.eos if self.calls >= self.limit else self.ids + self.eos
        self.calls += 1
        mask = self.mx.full(logits.shape, -float('inf'), dtype=logits.dtype)
        mask[..., self.mx.array(allowed)] = 0
        return logits + mask


class ChoiceMask:
    def __init__(self, mx, ids, eos, choices):
        self.mx, self.ids, self.eos, self.choices = mx, ids, eos, tuple(choices)
        self.reverse = {v:k for k,v in ids.items()}
        self.calls = 0

    def __call__(self, tokens, logits):
        prefix = ''.join(self.reverse.get(int(t), '?') for t in tokens[-self.calls:].tolist()) if self.calls else ''
        chars, can_end = allowed_choice_tokens(prefix, self.choices)
        allowed = [self.ids[c] for c in chars] + (self.eos if can_end else [])
        # Streaming may make one discarded look-ahead call after EOS.
        if not allowed:
            allowed = self.eos
        self.calls += 1
        mask = self.mx.full(logits.shape, -float('inf'), dtype=logits.dtype)
        mask[..., self.mx.array(allowed)] = 0
        return logits + mask


class Backend:
    def __init__(self, model_dir: Path, log_path: Path):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.log = log_path.open('a')
        self.calls, self.elapsed, self.tokens, self.max_prompt = 0, 0., {}, 0
        config = json.loads((model_dir/'config.json').read_text())
        if config.get('model_file') or config.get('quantization', {}).get('bits') != 8:
            raise ValueError('Expected the audited 8-bit checkpoint without repository Python')
        started = time.perf_counter()
        self.model, self.tokenizer = load(str(model_dir), tokenizer_config={'trust_remote_code': False})
        self.load_seconds = time.perf_counter() - started
        self.char_ids = {}
        for c in ALPHABET + '0123456789':
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != c:
                raise ValueError(f'Not a single round-tripping token: {c}')
            self.char_ids[c] = ids[0]
        self.eos = list(self.tokenizer.eos_token_ids)
        print(json.dumps({'event':'model_loaded', 'seconds':self.load_seconds, 'symbols':self.char_ids}), flush=True)

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx.random.seed(seed)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        max_tokens, processors = 256, None
        if mode == 'message':
            max_tokens = limit + 1
            processors = [SymbolMask(self.mx, [self.char_ids[c] for c in ALPHABET], self.eos, limit)]
        elif mode == 'action':
            if not choices:
                raise ValueError('Action menu must not be empty')
            processors = [ChoiceMask(self.mx, {c:self.char_ids[c] for c in '0123456789'}, self.eos, choices)]
            max_tokens = max(len(str(c)) for c in choices) + 1
        elif mode == 'natural_message':
            max_tokens = 96
        elif mode != 'private_analysis':
            raise ValueError(mode)
        started = time.perf_counter()
        parts, last = [], None
        for out in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature), logits_processors=processors,
                prompt_cache=None, prefill_step_size=512):
            parts.append(out.text)
            last = out
        result = ''.join(parts)
        elapsed = time.perf_counter() - started
        valid = (len(result) <= limit and all(c in ALPHABET for c in result)) if mode == 'message' else result in tuple(map(str, choices)) if mode == 'action' else True
        self.calls += 1
        self.elapsed += elapsed
        self.max_prompt = max(self.max_prompt, last.prompt_tokens)
        self.tokens[mode] = self.tokens.get(mode, 0) + last.generation_tokens
        entry = dict(call=self.calls, label=label, mode=mode, seed=seed, temperature=temperature,
            symbol_limit=limit if mode=='message' else None, messages=messages,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), output=result, valid=valid,
            seconds=elapsed, prompt_tokens=last.prompt_tokens, generation_tokens=last.generation_tokens,
            prompt_tps=last.prompt_tps, generation_tps=last.generation_tps,
            peak_memory_gb=last.peak_memory, finish_reason=last.finish_reason,
            fresh_cache=True, native_thinking=False)
        self.log.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.log.flush()
        if not valid:
            raise RuntimeError(f'Invalid source-constrained {mode}: {result!r}')
        return result

    def decide(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0, final_instruction=None):
        analysis_request = ('先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。'
            '他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。'
            + ('核对可选动作编号与动作描述；只依据自己的可见信息选择。' if mode=='action' else '考虑此次要表达什么、是否需回应伙伴，或是否沉默。'))
        analysis = self.infer(messages + [{'role':'user', 'content':analysis_request}], mode='private_analysis', label={**label,'stage':'analysis'}, seed=seed)
        final = ('只输出你选择的一个本地动作编号，不要解释。' if mode=='action' else
            '现在输出正式广播消息；最多约50个汉字，也可以沉默。不输出私有分析。' if mode=='natural_message' else
            f'现在只输出正式广播：只能使用{ALPHABET}，长度0至{limit}个字符，可以为空。不含空格、汉字或解释。')
        if final_instruction is not None:
            final = final_instruction
        return self.infer(messages + [{'role':'assistant','content':analysis}, {'role':'user','content':final}],
            mode=mode, label={**label,'stage':'formal'}, seed=seed, limit=limit, choices=choices, temperature=temperature)

    def stats(self):
        return dict(calls=self.calls, load_seconds=self.load_seconds, inference_seconds=self.elapsed,
            generation_tokens_by_mode=self.tokens, max_prompt_tokens=self.max_prompt,
            peak_mlx_memory_gb=self.mx.get_peak_memory()/1e9)
