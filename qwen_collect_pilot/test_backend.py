"""Real MLX mask tests and mocked inference, without downloading model weights."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx

import model_backend
from model_backend import ALPHABET, Backend, ChannelMask


class ChannelMaskTests(unittest.TestCase):
    @staticmethod
    def finite_ids(logits):
        return {index for index, value in enumerate(mx.isfinite(logits).tolist()) if value}

    def test_messages_allow_only_symbols_or_eos_and_stop_after_two_symbols(self):
        logits = mx.arange(12, dtype=mx.float32)
        original = logits.tolist()
        mask = ChannelMask([2, 3, 4, 5], [0, 1], max_chars=2, allow_empty=True)
        self.assertEqual(self.finite_ids(mask(mx.array([11]), logits)), {0, 1, 2, 3, 4, 5})
        self.assertEqual(self.finite_ids(mask(mx.array([11, 2]), logits)), {0, 1, 2, 3, 4, 5})
        self.assertEqual(self.finite_ids(mask(mx.array([11, 2, 3]), logits)), {0, 1})
        self.assertEqual(logits.tolist(), original)

    def test_actions_cannot_be_empty_or_contain_a_second_digit(self):
        logits = mx.zeros(12)
        mask = ChannelMask([6, 7], [0, 1], max_chars=1, allow_empty=False)
        self.assertEqual(self.finite_ids(mask(mx.array([11]), logits)), {6, 7})
        self.assertEqual(self.finite_ids(mask(mx.array([11, 6]), logits)), {0, 1})

    def test_mask_preserves_relative_allowed_logits_and_supports_batch_axis(self):
        logits = mx.array([[0.2, 0.4, 0.6, 0.8, 1.0]])
        result = ChannelMask([2, 4], [0], 1, False)(mx.array([1]), logits)
        self.assertEqual(self.finite_ids(result[0]), {2, 4})
        self.assertEqual(result[0, 2].item(), logits[0, 2].item())
        self.assertEqual(result[0, 4].item(), logits[0, 4].item())


class FakeTokenizer:
    ids = {char: index for index, char in enumerate(ALPHABET + "01", start=2)}
    eos_token_ids = {0, 1}

    def encode(self, text, add_special_tokens=False):
        return [self.ids[text]]

    def decode(self, ids):
        inverse = {token: char for char, token in self.ids.items()}
        return "".join(inverse[token] for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.last_template_kwargs = kwargs
        return json.dumps(messages, ensure_ascii=False)


def response(text):
    return SimpleNamespace(text=text, prompt_tokens=25, generation_tokens=2,
                           prompt_tps=100.0, generation_tps=30.0, peak_memory=0.1,
                           finish_reason="stop")


class BackendBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}))
        self.tokenizer = FakeTokenizer()
        with patch.object(model_backend, "load", return_value=(object(), self.tokenizer)) as loader:
            with patch("sys.stdout", new_callable=StringIO):
                self.backend = Backend(self.root, self.root / "calls.jsonl")
        self.assertEqual(loader.call_args.kwargs, {"tokenizer_config": {"trust_remote_code": False}})

    def tearDown(self):
        self.backend.log.close()
        self.directory.cleanup()

    def test_each_call_uses_only_given_messages_and_a_fresh_cache_and_mask(self):
        private_a = [{"role": "user", "content": "A_PRIVATE_SENTINEL"}]
        private_b = [{"role": "user", "content": "B_PRIVATE_SENTINEL"}]
        with patch.object(model_backend, "stream_generate", side_effect=[iter([response("@#")]), iter([response("1")])]) as generate:
            self.assertEqual(self.backend.infer(private_a, mode="message", seed=10), "@#")
            self.assertEqual(self.backend.infer(private_b, mode="action", seed=11), "1")
        first, second = generate.call_args_list
        self.assertIn("A_PRIVATE_SENTINEL", first.args[2])
        self.assertNotIn("B_PRIVATE_SENTINEL", first.args[2])
        self.assertIn("B_PRIVATE_SENTINEL", second.args[2])
        self.assertNotIn("A_PRIVATE_SENTINEL", second.args[2])
        for call in (first, second):
            self.assertIn("prompt_cache", call.kwargs)
            self.assertIsNone(call.kwargs["prompt_cache"])
        first_mask = first.kwargs["logits_processors"][0]
        second_mask = second.kwargs["logits_processors"][0]
        self.assertIsNot(first_mask, second_mask)
        self.assertEqual(set(first_mask.token_ids), {self.tokenizer.ids[c] for c in ALPHABET})
        self.assertEqual(set(second_mask.token_ids), {self.tokenizer.ids[c] for c in "01"})
        self.assertEqual((first.kwargs["max_tokens"], second.kwargs["max_tokens"]), (3, 2))
        self.assertIs(self.tokenizer.last_template_kwargs["enable_thinking"], False)
        logs = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertNotEqual(logs[0]["prompt_sha256"], logs[1]["prompt_sha256"])
        self.assertTrue(all(row["fresh_cache"] and not row["thinking"] for row in logs))

    def test_invalid_channel_output_is_logged_then_rejected(self):
        for mode, invalid in (("message", "@@@"), ("message", "@说明"), ("action", ""), ("action", "01")):
            with patch.object(model_backend, "stream_generate", return_value=iter([response(invalid)])):
                with self.assertRaises(RuntimeError):
                    self.backend.infer([{"role": "user", "content": "test"}], mode=mode)
            last_log = json.loads((self.root / "calls.jsonl").read_text().splitlines()[-1])
            self.assertFalse(last_log["valid"])
            self.assertEqual(last_log["output"], invalid)

    def test_empty_message_is_valid_and_distinct_from_empty_action(self):
        with patch.object(model_backend, "stream_generate", return_value=iter([response("")])):
            self.assertEqual(self.backend.infer([], mode="message"), "")

    def test_private_reasoning_is_local_bounded_and_final_output_remains_masked(self):
        self.backend.private_reasoning = True
        original_a = [
            {"role": "system", "content": "你是A。只输出0或1。不要解释。"},
            {"role": "user", "content": "A_OBSERVATION_SENTINEL"},
        ]
        original_b = [
            {"role": "system", "content": "你是B。只输出0或1。不要解释。"},
            {"role": "user", "content": "B_OBSERVATION_SENTINEL"},
        ]
        before_a, before_b = deepcopy(original_a), deepcopy(original_b)
        label_a = {"role": "A", "phase": "test"}
        streams = [iter([response("A_PRIVATE_REASONING_SENTINEL")]), iter([response("1")]),
                   iter([response("B_PRIVATE_REASONING_SENTINEL")]), iter([response("0")])]
        with patch.object(model_backend, "stream_generate", side_effect=streams) as generate:
            self.assertEqual(self.backend.decide(original_a, mode="action", temperature=0.7,
                                                 seed=10, label=label_a), "1")
            self.assertEqual(self.backend.decide(original_b, mode="action", seed=11,
                                                 label={"role": "B"}), "0")
        self.assertEqual(original_a, before_a)
        self.assertEqual(original_b, before_b)
        self.assertEqual(label_a, {"role": "A", "phase": "test"})
        a_analysis, a_final, b_analysis, b_final = generate.call_args_list
        self.assertIn("A_PRIVATE_REASONING_SENTINEL", a_final.args[2])
        for call in (b_analysis, b_final):
            self.assertNotIn("A_PRIVATE_REASONING_SENTINEL", call.args[2])
            self.assertNotIn("A_OBSERVATION_SENTINEL", call.args[2])
        for call in (a_analysis, a_final):
            self.assertNotIn("B_PRIVATE_REASONING_SENTINEL", call.args[2])
        for call in (a_analysis, b_analysis):
            self.assertEqual(call.kwargs["max_tokens"], 256)
            self.assertIsNone(call.kwargs["logits_processors"])
        for call in (a_final, b_final):
            self.assertEqual(call.kwargs["max_tokens"], 2)
            mask = call.kwargs["logits_processors"][0]
            self.assertEqual(set(mask.token_ids), {self.tokenizer.ids[c] for c in "01"})
            self.assertFalse(mask.allow_empty)
            self.assertEqual(mask.max_chars, 1)
        for call in generate.call_args_list:
            self.assertIsNone(call.kwargs["prompt_cache"])
        logs = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([row["temperature"] for row in logs], [0, 0.7, 0, 0])
        self.assertEqual([row["label"]["stage"] for row in logs],
                         ["private_reasoning", "final_output", "private_reasoning", "final_output"])

    def test_private_sender_analysis_does_not_bypass_symbol_mask(self):
        self.backend.private_reasoning = True
        with patch.object(model_backend, "stream_generate", side_effect=[
            iter([response("私有分析中可以写自然语言，但广播只能使用特殊符号。")]),
            iter([response("#%")]),
        ]) as generate:
            self.assertEqual(self.backend.decide([], mode="message", temperature=0.7), "#%")
        final = generate.call_args_list[1]
        mask = final.kwargs["logits_processors"][0]
        self.assertEqual(set(mask.token_ids), {self.tokenizer.ids[c] for c in ALPHABET})
        self.assertEqual(mask.max_chars, 2)
        self.assertTrue(mask.allow_empty)
        self.assertEqual(final.kwargs["max_tokens"], 3)
        # The final output validator must also remain active after analysis.
        with patch.object(model_backend, "stream_generate", side_effect=[
            iter([response("PRIVATE")]), iter([response("这里是自然语言")]),
        ]):
            with self.assertRaises(RuntimeError):
                self.backend.decide([], mode="message")

    def test_calibration_and_private_analysis_do_not_enter_free_episodic_history(self):
        from agents import build_messages, empty_histories, remember
        from env import World

        self.backend.private_reasoning = True
        world = World(0, 0, 0, 0)
        histories = empty_histories()
        calibration = build_messages("A", world.private_observation("A"), [],
                                     received="@@", known_protocol=True)
        stream_texts = ["CALIBRATION_ANALYSIS_SENTINEL", "0",
                        "C_ANALYSIS_SENTINEL", "@#",
                        "A_ANALYSIS_SENTINEL", "0",
                        "B_ANALYSIS_SENTINEL", "1"]
        streams = [iter([response(text)]) for text in stream_texts]
        with patch.object(model_backend, "stream_generate", side_effect=streams) as generate:
            self.backend.decide(calibration, mode="action", label={"phase": "known_protocol"})
            self.assertEqual(histories, empty_histories())
            message = self.backend.decide(
                build_messages("C", world.private_observation("C"), histories["C"]), mode="message")
            actions = {}
            for role in ("A", "B"):
                actions[role] = int(self.backend.decide(
                    build_messages(role, world.private_observation(role), histories[role], received=message),
                    mode="action"))
        for call in generate.call_args_list[2:]:
            self.assertNotIn("CALIBRATION_ANALYSIS_SENTINEL", call.args[2])
        for call in generate.call_args_list[4:]:
            self.assertNotIn("C_ANALYSIS_SENTINEL", call.args[2])
        for call in generate.call_args_list[6:]:
            self.assertNotIn("A_ANALYSIS_SENTINEL", call.args[2])
        remember(histories, world, message, actions, world.score(actions))
        serialized = json.dumps(histories, ensure_ascii=False)
        self.assertNotIn("ANALYSIS_SENTINEL", serialized)
        self.assertEqual(histories["C"][0]["发送"], "@#")
        self.assertEqual(histories["A"][0]["动作"], 0)
        self.assertEqual(histories["B"][0]["动作"], 1)


if __name__ == "__main__":
    unittest.main()
