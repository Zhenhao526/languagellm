"""Offline checks: no model weights, MLX import, or neural forward calls."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from random import Random
import tempfile
import unittest
from unittest.mock import patch

from research_program.interaction_ecology_study import action_interface as study
from research_program.interaction_ecology_study.interface_backend import (
    InterfaceBackend, TokenSequenceMask, allowed_next, ANALYSIS_REQUEST, FORMAL_INSTRUCTIONS,
)
from research_program.triadic_task import environment as env
from research_program.triadic_task import qwen_capability as cap


class RecordingBackend(InterfaceBackend):
    """Exercise both real two-stage wrappers, replacing only neural infer."""
    def __init__(self, mismatch=False):
        self.calls, self.entries, self.mismatch = 0, [], mismatch

    def infer(self, messages, *, mode, label, seed=0, temperature=0, choices=None, **kwargs):
        data = json.loads(messages[1]["content"])
        if mode == "private_analysis":
            answer = "私有分析不进入广播或下一决定。"
            if self.mismatch and label["arm"] == "semantic":
                answer += "不同。"
        elif mode == "natural_message":
            answer = f"公开g{label['group']}e{label['episode']}w{label['window']}{label['agent']}"
        elif mode == "action_number":
            answer = str(next(m["编号"] for m in data["本次可选动作"] if m["动作"] == "等待"))
        elif mode == "action_semantic":
            answer = "等待"
        else:
            raise AssertionError(mode)
        self.calls += 1
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        self.entries.append(dict(call=self.calls, messages=deepcopy(messages), label=deepcopy(label), output=answer,
            mode=mode, choices=deepcopy(choices), seed=seed, temperature=temperature,
            prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(), prompt_tokens=len(rendered),
            generation_tokens=len(answer), finish_reason="stop", valid=True, fresh_cache=True, native_thinking=False))
        return answer


class PrefixTests(unittest.TestCase):
    def test_shared_prefix_complete_prefix_and_eos(self):
        seqs = [[1], [1, 2], [3, 4, 5]]
        self.assertEqual(allowed_next([], seqs, [99]), [1, 3])
        self.assertEqual(allowed_next([1], seqs, [99]), [2, 99])
        self.assertEqual(allowed_next([3, 4], seqs, [99]), [5])
        self.assertEqual(allowed_next([3, 4, 5], seqs, [99]), [99])
        self.assertEqual(allowed_next([3, 4, 5, 99], seqs, [99]), [99])

    def test_incomplete_or_illegal_prefix_never_ends(self):
        for prefix in ([3, 99], [7], [1, 2, 4], [99], [1, 99, 99]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                allowed_next(prefix, [[1], [1, 2], [3, 4]], [99])
        for seqs in ([], [[]], [[1, 99]]):
            with self.subTest(seqs=seqs), self.assertRaises(ValueError):
                allowed_next([], seqs, [99])

    def test_mask_uses_generated_suffix_not_prompt(self):
        import numpy as np
        mask = TokenSequenceMask(np, [[2, 3], [2, 4], [5]], [9])
        self.assertEqual(np.flatnonzero(np.isfinite(mask(np.array([42, 43]), np.zeros(10)))).tolist(), [2, 5])
        self.assertEqual(np.flatnonzero(np.isfinite(mask(np.array([42, 43, 2]), np.zeros(10)))).tolist(), [3, 4])
        self.assertEqual(np.flatnonzero(np.isfinite(mask(np.array([42, 43, 2, 3]), np.zeros(10)))).tolist(), [9])
        self.assertEqual(np.flatnonzero(np.isfinite(mask(np.array([42, 43, 2, 3, 9]), np.zeros(10)))).tolist(), [9])


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = study.build_cases()
        cls.before = deepcopy(cls.prepared)
        cls.backend = RecordingBackend()
        cls.result = study.run_cases(cls.backend, cls.prepared)

    def test_raw_iid_draws_not_selected_for_easy_solutions(self):
        self.assertEqual(len(self.prepared["scenarios"]), 4)
        for scene, seed in zip(self.prepared["scenarios"], study.SCENE_SEEDS):
            self.assertEqual(scene["state"], json.loads(json.dumps(asdict(env.draw_state(Random(seed), env.support())))))
        self.assertEqual(len(self.prepared["trials"]), 12)
        for episode in range(1, 5):
            subset = [x for x in self.prepared["trials"] if x["episode"] == episode]
            self.assertEqual([x["group"] for x in subset], list(study.GROUPS))
            self.assertTrue(all(x["state"] == subset[0]["state"] for x in subset))

    def test_all_seventeen_actions_unchanged_no_witness_input(self):
        for case in self.prepared["trials"]:
            for agent in env.AGENTS:
                menu = case["menus"][agent]
                self.assertEqual([x["id"] for x in menu], list(range(17)))
                self.assertCountEqual([x["action"] for x in menu], env.all_actions(agent))
                for arm in study.ARMS:
                    values = study.choice_strings(menu, arm)
                    self.assertEqual(len(set(values)), 17)
                prompt = study.build_prompt(agent, case["observations"][agent], [], menu=menu)
                data = json.loads(prompt[1]["content"])
                self.assertEqual(set(data), {"你自己的历史", "本轮完整信息观察", "本轮已公开广播", "本次可选动作", "当前决策"})
                self.assertEqual(len(data["本次可选动作"]), 17)
                self.assertNotIn("case_id", prompt[1]["content"])
                self.assertNotIn("scene_seed", prompt[1]["content"])
                self.assertNotIn("witness", prompt[1]["content"])

    def test_fixed_288_budget_even_if_every_action_waits(self):
        self.assertEqual(self.backend.calls, 288)
        self.assertEqual(len(self.result["decisions"]), 144)
        self.assertEqual(len(self.result["trials"]), 24)
        self.assertEqual([r["full_success_trials"] for r in self.result["arms"]], [0, 0])
        self.assertTrue(all(r["outcome"]["reward"] == 0 for r in self.result["trials"]))
        self.assertEqual(self.prepared, self.before)

    def test_synchronous_windows_and_fresh_independent_histories(self):
        snapshots = {s["case_id"]: s for s in self.result["snapshots"]}
        for call in self.backend.entries:
            data = json.loads(call["messages"][1]["content"])
            self.assertEqual(data["你自己的历史"], [])
            snapshot = snapshots[call["label"]["case_id"]]
            expected = snapshot["messages"][:0 if call["label"]["window"] == 1 else 3 if call["label"]["window"] == 2 else 6]
            self.assertEqual(data["本轮已公开广播"], expected)
            self.assertNotIn("私有分析不进入", json.dumps(data, ensure_ascii=False))
            self.assertEqual(data["本轮完整信息观察"]["你是"], call["label"]["agent"])

    def test_analysis_same_formal_only_instruction_and_grammar_change(self):
        self.assertTrue(study.analysis_reproduction(self.backend.entries)["exact"])
        grouped = {(c["label"]["arm"], c["label"]["case_id"], c["label"]["agent"], c["label"]["stage"]): c
                   for c in self.backend.entries if c["label"]["arm"] in study.ARMS}
        for case in self.prepared["trials"]:
            for agent in env.AGENTS:
                a, b = [grouped[(arm, case["case_id"], agent, "analysis")] for arm in study.ARMS]
                self.assertEqual(a["messages"], b["messages"])
                self.assertEqual(a["messages"][-1]["content"], ANALYSIS_REQUEST)
                n, s = [grouped[(arm, case["case_id"], agent, "formal")] for arm in study.ARMS]
                self.assertEqual(n["messages"][:-1], s["messages"][:-1])
                self.assertEqual(n["messages"][-1]["content"], FORMAL_INSTRUCTIONS["number"])
                self.assertEqual(s["messages"][-1]["content"], FORMAL_INSTRUCTIONS["semantic"])
                self.assertEqual(n["seed"], s["seed"])
                self.assertEqual(n["temperature"], s["temperature"])
                self.assertEqual(n["choices"], study.choice_strings(case["menus"][agent], "number"))
                self.assertEqual(s["choices"], study.choice_strings(case["menus"][agent], "semantic"))

    def test_seed_reuse_and_action_order(self):
        from collections import Counter
        counts = Counter(x["seed"] for x in self.backend.entries)
        self.assertEqual(len(counts), 108)
        self.assertEqual(Counter(counts.values()), {2: 72, 4: 36})
        self.assertEqual([x["label"]["arm"] for x in self.backend.entries], ["common"]*144+["number"]*72+["semantic"]*72)

    def test_analysis_mismatch_is_detected_without_extra_calls(self):
        backend = RecordingBackend(mismatch=True)
        study.run_cases(backend, self.prepared)
        self.assertEqual(backend.calls, 288)
        self.assertFalse(study.analysis_reproduction(backend.entries)["exact"])

    def test_callbacks_cannot_mutate_saved_inputs_or_other_agents(self):
        backend = RecordingBackend()
        def corrupt(row):
            row.clear()
        result = study.run_cases(backend, self.prepared, on_snapshot=corrupt, on_decision=corrupt, on_trial=corrupt)
        self.assertEqual(self.prepared, self.before)
        self.assertEqual(len(result["snapshots"]), 12)
        self.assertTrue(all(len(x["messages"]) == 6 for x in result["snapshots"]))
        self.assertTrue(study.analysis_reproduction(backend.entries)["exact"])

    def test_no_gate_or_next_scene_mutation_and_settlement_replay(self):
        for row in self.result["trials"]:
            self.assertEqual(row["outcome"], env.settle(env.State(**row["state"]), row["actions"], require_match=True))
        self.assertTrue(all(s["private_histories_before"] == {a: [] for a in env.AGENTS} for s in self.result["snapshots"]))


class LocalTokenizerAndPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = study.load_tokenizer_only(cap.DEFAULT_MODEL)
        cls.prepared = study.build_cases()
        cls.tokenization = study.build_tokenization(cls.tokenizer, cls.prepared)

    def test_real_tokenizer_all_paths_and_shared_budget(self):
        tokenization = self.tokenization
        for case in self.prepared["trials"]:
            for agent in env.AGENTS:
                for arm, key in (("number", "number_sequences"), ("semantic", "semantic_sequences")):
                    values = study.choice_strings(case["menus"][agent], arm)
                    seqs = [tokenization[key][s] for s in values]
                    for value, seq in zip(values, seqs):
                        self.assertEqual(self.tokenizer.decode(seq), value)
                        self.assertLess(len(seq), tokenization["formal_max_tokens"])
                        for n, next_id in enumerate(seq):
                            allowed = allowed_next(seq[:n], seqs, tokenization["eos_token_ids"])
                            self.assertIn(next_id, allowed)
                            self.assertEqual(bool(set(allowed) & set(tokenization["eos_token_ids"])), tuple(seq[:n]) in map(tuple, seqs))
                        self.assertTrue(set(tokenization["eos_token_ids"]) <= set(allowed_next(seq, seqs, tokenization["eos_token_ids"])))
        self.assertEqual(tokenization["number_sequences"]["10"], self.tokenizer.encode("1", add_special_tokens=False) + self.tokenizer.encode("0", add_special_tokens=False))
        self.assertEqual(tokenization["formal_max_tokens"], max(map(len, list(tokenization["semantic_sequences"].values()) + list(tokenization["number_sequences"].values()))) + 1)

    def test_prepare_verify_and_overwrite_refusal_without_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "load_tokenizer_only", return_value=self.tokenizer):
            output = Path(directory) / "pack"
            result = study.prepare(output)
            self.assertEqual(result["calls_planned"], 288)
            self.assertEqual(result["formal_max_tokens"], self.tokenization["formal_max_tokens"])
            self.assertEqual(study.verify(output)[1]["trials"], self.prepared["trials"])
            self.assertFalse((output / "execution").exists())
            with self.assertRaises(ValueError): study.prepare(output)

    def test_tampered_inputs_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "load_tokenizer_only", return_value=self.tokenizer):
            output = Path(directory) / "pack"
            study.prepare(output)
            original = (output / "prepared_cases.json").read_bytes()
            data = json.loads(original); data["trials"][0]["menus"]["A"].pop()
            (output / "prepared_cases.json").write_text(json.dumps(data))
            with patch("research_program.interaction_ecology_study.interface_backend.InterfaceBackend") as constructor:
                with self.assertRaises(ValueError): study.execute(output)
                constructor.assert_not_called()
            self.assertFalse((output / "execution").exists())
            (output / "prepared_cases.json").write_bytes(original)
            snapshot = output / "source_snapshot" / Path(study.__file__).resolve().relative_to(study.WORK)
            snapshot.write_text("changed snapshot")
            with self.assertRaises(ValueError): study.verify(output)


if __name__ == "__main__":
    unittest.main()
