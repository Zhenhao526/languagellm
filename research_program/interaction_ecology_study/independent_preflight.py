"""Independent input/grammar audit. Tokenizer and NumPy only; no model forward.

This exercises the real decide wrappers with infer replaced by an explicit
recorder, and independently checks their fully expanded message arrays.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research_program.interaction_ecology_study import action_interface as runner
from research_program.interaction_ecology_study import interface_backend as ib
from research_program.triadic_task import environment as env
from research_program.triadic_task import qwen_capability as old

HERE = Path(__file__).resolve().parent


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedBackend(ib.InterfaceBackend):
    def __init__(self, tokenizer):
        self.calls, self.records, self.tokenizer = 0, [], tokenizer

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0):
        self.calls += 1
        if mode == "private_analysis":
            # This is a fake fixed answer, not a model judgment or correct plan.
            output = "仅供无模型输入核验。"
        elif mode == "natural_message":
            output = f"假广播_{self.calls}"
        else:
            # Pick a preset arbitrary position, never inspect a world or witness.
            output = choices[(env.AGENTS.index(label["agent"]) + label["episode"]) % 17]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        record = {"call": self.calls, "messages": deepcopy(messages), "output": output,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "mode": mode,
                  "label": deepcopy(label), "seed": seed, "temperature": temperature,
                  "choices": deepcopy(choices), "valid": True, "fresh_cache": True,
                  "native_thinking": False, "finish_reason": "fake_no_generation",
                  "prompt_tokens": len(self.tokenizer.encode(prompt)), "generation_tokens": 0}
        self.records.append(record)
        return output


def independent_reward(state, actions):
    active = [a for a in env.AGENTS if actions[a]["kind"] == "transport"]
    if len(active) != 2:
        return 0.0
    a, b = active
    x, y = actions[a], actions[b]
    if not (x["partner"] == b and y["partner"] == a and x["site"] == y["site"] and x["destination"] == y["destination"]):
        return 0.0
    material = state["layout"][int(x["site"][1:])]
    destination = ("L", "R").index(x["destination"])
    score = 0
    for actor in active:
        need_id = state["needs"][env.AGENTS.index(actor)]
        resource, accepted_destination = divmod(need_id, 3)
        good_resource = ((0, 1), (2, 3), (0, 2), (1, 3))[resource]
        good_destination = ((0,), (1,), (0, 1))[accepted_destination]
        score += material in good_resource and destination in good_destination
    return score / 2


def audit_grammar(tokenizer, cases):
    tokenization = runner.build_tokenization(tokenizer, cases)
    eos = tokenization["eos_token_ids"]
    menus_checked, prefix_checks, mask_checks, numeric_legacy_checks = 0, 0, 0, 0
    for agent in env.AGENTS:
        menu = env.action_menu(agent, list(range(17)))
        for arm in runner.ARMS:
            mapping = tokenization["number_sequences" if arm == "number" else "semantic_sequences"]
            choices = runner.choice_strings(menu, arm)
            sequences = [mapping[text] for text in choices]
            # Independent adjacency table built by inserting complete token paths.
            trie = {}
            for seq in sequences:
                for i, token in enumerate(seq):
                    trie.setdefault(tuple(seq[:i]), set()).add(token)
                trie.setdefault(tuple(seq), set()).update(eos)
            for prefix, expected in trie.items():
                check(set(ib.allowed_next(prefix, sequences, eos)) == expected, "Prefix grammar differs from independent trie")
                prefix_checks += 1
            for seq in sequences:
                processor = ib.TokenSequenceMask(np, sequences, eos)
                for i in range(len(seq) + 1):
                    generated = seq[:i]
                    logits = np.zeros((1, max(max(s) for s in sequences) + max(eos) + 2), dtype=np.float32)
                    masked = processor(np.array([987654321] + generated), logits)
                    allowed = set(np.flatnonzero(np.isfinite(masked[0])).tolist())
                    check(allowed == trie[tuple(generated)], "Stateful mask differs or includes prompt tokens")
                    mask_checks += 1
                # One lookahead AFTER a legal EOS is the installed generator's behavior.
                check(set(ib.allowed_next(seq + [eos[0]], sequences, eos)) == set(eos), "Legal EOS lookahead rejected")
            if arm == "number":
                from qwen_language_v3.backend import allowed_choice_tokens
                digits = {int(v[0]): k for k, v in tokenization["number_sequences"].items() if len(k) == 1}
                for prefix, expected in trie.items():
                    chars = "".join(digits[t] for t in prefix)
                    old_next, old_can_end = allowed_choice_tokens(chars, range(17))
                    old_ids = {tokenization["number_sequences"][c][0] for c in old_next}
                    if old_can_end:
                        old_ids.update(eos)
                    check(old_ids == expected, "Numeric grammar changed from legacy per-digit path")
                    numeric_legacy_checks += 1
            for illegal in ([987654321], [eos[0]], [987654321, eos[0]]):
                try:
                    ib.allowed_next(illegal, sequences, eos)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid prefix accepted")
            menus_checked += 1
    return {"menus_by_agent_arm": menus_checked, "independent_trie_prefix_checks": prefix_checks,
            "stateful_numpy_mask_checks": mask_checks, "legacy_number_prefix_checks": numeric_legacy_checks,
            "semantic_unique_strings": len(tokenization["semantic_sequences"]),
            "semantic_token_length_min": min(map(len, tokenization["semantic_sequences"].values())),
            "semantic_token_length_max": max(map(len, tokenization["semantic_sequences"].values())),
            "formal_budget_both_arms": tokenization["formal_max_tokens"], "eos_ids": eos}


def run():
    tokenizer = runner.load_tokenizer_only(old.DEFAULT_MODEL)
    cases = runner.build_cases()
    original_cases = deepcopy(cases)
    check(len(cases["scenarios"]) == 4 and len(cases["trials"]) == 12, "Wrong scenario grid")
    grammar = audit_grammar(tokenizer, cases)
    backend = RecordedBackend(tokenizer)
    result = runner.run_cases(backend, cases)
    check(cases == original_cases, "Execution mutated prepared inputs")
    check(len(backend.records) == 288 and result["calls"] == 288, "Wrong full budget")
    check(len(result["snapshots"]) == 12 and len(result["trials"]) == 24, "Incomplete outcomes")
    base_seeds = set()
    full_inputs, menu_checks = 0, 0
    for case_i, (case, snapshot) in enumerate(zip(cases["trials"], result["snapshots"])):
        expected_transcript = []
        check(snapshot["private_histories_before"] == {a: [] for a in env.AGENTS}, "Nonempty private history")
        for window in (1, 2):
            new = []
            for agent_i, agent in enumerate(env.AGENTS):
                offset = case_i * 12 + (window - 1) * 6 + agent_i * 2
                analysis, formal = backend.records[offset:offset + 2]
                data = json.loads(analysis["messages"][1]["content"])
                check(data["你自己的历史"] == [], "Cross-trial history leaked")
                check(data["本轮已公开广播"] == expected_transcript, "Same-window broadcast or foreign trial leaked")
                check(data["本轮完整信息观察"] == old.literal_observation(case["observations"][agent]), "Wrong complete observation")
                check(set(data) == {"你自己的历史", "本轮完整信息观察", "本轮已公开广播", "当前决策"}, "Unexpected researcher fields")
                check(formal["messages"][:-2] == analysis["messages"][:-1], "Common stages changed base")
                check(formal["messages"][-2] == {"role": "assistant", "content": analysis["output"]}, "Foreign private analysis")
                check(analysis["temperature"] == 0 and formal["temperature"] == .7, "Common temperatures differ")
                check(analysis["seed"] == formal["seed"], "Two-stage seed differs")
                base_seeds.add(analysis["seed"])
                new.append({"agent": agent, "window": window, "text": formal["output"]})
                full_inputs += 2
            expected_transcript.extend(new)
        check(snapshot["messages"] == expected_transcript, "Snapshot broadcasts differ")
        for agent_i, agent in enumerate(env.AGENTS):
            menu = case["menus"][agent]
            check({json.dumps(m["action"], sort_keys=True) for m in menu} ==
                  {json.dumps(a, sort_keys=True) for a in env.all_actions(agent)}, "Filtered legal action")
            menu_checks += 1
            paired = []
            for arm_i, arm in enumerate(runner.ARMS):
                offset = 144 + arm_i * 72 + case_i * 6 + agent_i * 2
                analysis, formal = backend.records[offset:offset + 2]
                base = analysis["messages"][:-1]
                data = json.loads(base[1]["content"])
                check(data["你自己的历史"] == [] and data["本轮已公开广播"] == expected_transcript, "Action history/broadcast differs")
                check(set(data) == {"你自己的历史", "本轮完整信息观察", "本轮已公开广播", "本次可选动作", "当前决策"}, "Action/witness leaked into input")
                check(data["本次可选动作"] == [{"编号": m["id"], "动作": old.action_text(m["action"])} for m in menu], "Menu changed")
                check(formal["messages"][:-2] == base, "Formal base changed")
                check(formal["messages"][-2] == {"role": "assistant", "content": analysis["output"]}, "Wrong same-decision analysis")
                check(formal["messages"][-1] == {"role": "user", "content": ib.FORMAL_INSTRUCTIONS[arm]}, "Incorrect formal instruction")
                check(analysis["messages"][-1] == {"role": "user", "content": ib.ANALYSIS_REQUEST}, "Different analysis request")
                check(analysis["seed"] == formal["seed"] and analysis["temperature"] == formal["temperature"] == 0, "Action seed/temp changed")
                base_seeds.add(analysis["seed"])
                paired.append(analysis)
                full_inputs += 2
            check(paired[0]["messages"] == paired[1]["messages"] and paired[0]["seed"] == paired[1]["seed"], "Paired action analysis input differs")
    check(full_inputs == 288 and len(base_seeds) == 108, "Input/seed denominator differs")
    for trial in result["trials"]:
        check(trial["outcome"]["reward"] == independent_reward(trial["state"], trial["actions"]), "Physical reward differs")
    reproduction = runner.analysis_reproduction(backend.records)
    check(reproduction["exact"], "Equal fake analyses failed replay gate")
    mutated = deepcopy(backend.records)
    mutated[216]["output"] += "修改"
    check(not runner.analysis_reproduction(mutated)["exact"], "Mismatched analysis did not fail gate")
    source_files = [Path(runner.__file__), Path(ib.__file__), HERE / "plan.md", HERE / "tests/test_action_interface.py",
                    Path(env.__file__), Path(old.__file__), old.BACKEND, Path(__file__),
                    Path(old.DEFAULT_MODEL) / "tokenizer.json", Path(old.DEFAULT_MODEL) / "tokenizer_config.json",
                    Path(old.DEFAULT_MODEL) / "chat_template.jinja"]
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {str(p.resolve()): sha(p) for p in source_files},
            "actual_model_calls": 0, "model_weights_loaded": 0, "tokenizer_only_loaded": True,
            "fake_expanded_calls": full_inputs, "complete_private_menus": menu_checks,
            "new_semantic_draws": 4, "context_repetitions_per_draw": 3, "unique_decision_seeds": len(base_seeds),
            "fake_outcomes_independently_scored": 24, "private_analysis_pairs": 36,
            "analysis_mismatch_gate_rejects": True, "same_window_and_cross_trial_isolation_passed": True,
            "grammar": grammar,
            "limitations": ["Fake infer validates dataflow, not model ability or real cache behavior",
                            "Semantic canonical token paths and greedy branching differ; this is a whole output-interface contrast",
                            "Actual analysis equality, log hashes, stopping, budget and outcomes require post-execution audit",
                            "Four IID worlds with three repetitions are not twelve independent semantic draws"]}


if __name__ == "__main__":
    result = run()
    path = HERE / "独立预审.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))
