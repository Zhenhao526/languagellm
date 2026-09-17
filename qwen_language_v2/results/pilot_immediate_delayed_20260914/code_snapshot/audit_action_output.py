"""Bounded, independent audit of formal 0/1 output on two saved failed cases.

Does not edit the live backend or replace model answers with an expected label.
Each case receives six fresh-cache formal inferences; original analysis is held
fixed.  Expected labels are loaded for reporting only, never added to prompts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from .backend import Backend, ChoiceMask
from .run import DEFAULT_MODEL


SOURCE = Path(__file__).parent / "results/rule_check_20260914_194041"
CASES = ("ground_is_not_carried", "processing_preserves_attributes")
EXPLICIT = "请回答上面的真假判断题，不选择物理动作。只输出一个数字：0表示否，1表示是。不要解释。"


class InspectProcessor:
    def __init__(self, backend, constrained):
        self.backend = backend
        self.mask = ChoiceMask(backend.mx, {c: backend.char_ids[c] for c in "0123456789"},
                               backend.eos, [0, 1]) if constrained else None
        self.records = []

    def __call__(self, tokens, logits):
        mx, tokenizer = self.backend.mx, self.backend.tokenizer
        n = len(self.records)
        entry = {"processor_call": n + 1, "input_token_suffix": tokens[-4:].tolist()}
        if n == 0:
            mx.eval(logits)
            logprobs = logits[0] - mx.logsumexp(logits[0])
            top_ids = mx.argsort(logits[0])[-12:][::-1].tolist()
            entry["raw_top_tokens"] = [{"token_id": int(i), "text": tokenizer.decode([i]),
                                        "logit": float(logits[0, i]), "logprob": float(logprobs[i])}
                                       for i in top_ids]
            entry["raw_digits"] = {c: {"token_id": self.backend.char_ids[c],
                                       "logit": float(logits[0, self.backend.char_ids[c]]),
                                       "logprob": float(logprobs[self.backend.char_ids[c]])}
                                   for c in "01"}
        result = self.mask(tokens, logits) if self.mask else logits
        if self.mask:
            allowed = [i for i, value in enumerate(result[0].tolist()) if math.isfinite(value)]
            entry["allowed_token_ids"] = allowed
            entry["allowed_texts"] = [tokenizer.decode([i]) for i in allowed]
            entry["argmax_after_mask"] = int(mx.argmax(result[0]))
        self.records.append(entry)
        return result


def run_one(backend, messages, *, constrained, prefix, seed, label):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    backend.mx.random.seed(seed)
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
    rendered += prefix
    observer = InspectProcessor(backend, constrained)
    start = time.perf_counter()
    chunks, generated_ids = [], []
    last = None
    for output in stream_generate(backend.model, backend.tokenizer, rendered,
                                  max_tokens=2 if constrained else 96,
                                  sampler=make_sampler(temp=0), logits_processors=[observer],
                                  prompt_cache=None, prefill_step_size=512):
        chunks.append(output.text)
        generated_ids.append(output.token)
        last = output
    return {**label, "seed": seed, "messages": messages, "assistant_prefix": prefix,
            "constrained": constrained, "choices": [0, 1] if constrained else None,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "output": "".join(chunks), "generated_token_ids": generated_ids,
            "generation_tokens": last.generation_tokens, "prompt_tokens": last.prompt_tokens,
            "finish_reason": last.finish_reason, "seconds": time.perf_counter() - start,
            "fresh_cache": True, "native_thinking": False, "temperature": 0,
            "processor_trace": observer.records}


def main():
    out = Path(__file__).parent / "results" / ("action_output_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    source_rows = [json.loads(line) for line in (SOURCE / "inference.jsonl").read_text().splitlines() if line.strip()]
    formal = {row["label"]["case"]: row for row in source_rows if row["mode"] == "action"}
    expected = {row["case"]: row["expected"] for row in json.loads((SOURCE / "results.json").read_text())["cases"]}
    backend = Backend(DEFAULT_MODEL, out / "backend_initialization.jsonl")
    rows = []
    variants = (("original_masked", True, False, ""), ("original_raw", False, False, ""),
                ("explicit_masked", True, True, ""), ("explicit_raw", False, True, ""),
                ("prefix_masked", True, False, "答案："), ("prefix_raw", False, False, "答案："))
    for name in CASES:
        source = formal[name]
        for variant, constrained, explicit, prefix in variants:
            messages = deepcopy(source["messages"])
            if explicit:
                messages[-1]["content"] = EXPLICIT
            row = run_one(backend, messages, constrained=constrained, prefix=prefix, seed=source["seed"],
                          label={"case": name, "variant": variant, "source_call": source["call"],
                                 "source_prompt_sha256": source["prompt_sha256"]})
            # Labels are attached only after generation.
            row["expected_for_reporting_only"] = expected[name]
            rows.append(row)
            with (out / "inference.jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"case": name, "variant": variant, "output": row["output"],
                              "seconds": row["seconds"], "initial_digits": row["processor_trace"][0]["raw_digits"]},
                             ensure_ascii=False), flush=True)
    result = {"scope": "two selected failed declarative cases; exploratory interface audit, not a held-out benchmark",
              "source_run": str(SOURCE.resolve()), "infer_count": len(rows), "model": str(DEFAULT_MODEL.resolve()),
              "variants": [v[0] for v in variants], "cases": list(CASES),
              "generation_seconds": sum(row["seconds"] for row in rows), "load_seconds": backend.load_seconds,
              "no_expected_labels_in_prompts": True, "rows": rows}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"run_dir": str(out.resolve()), "infer_count": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
