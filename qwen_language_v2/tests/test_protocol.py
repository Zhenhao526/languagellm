"""Channel privacy and budget tests; no model or network needed."""

import copy
import unittest

from qwen_language_v2.protocol import (
    AGENTS,
    ALPHABET,
    PrivateHistory,
    ProtocolError,
    message_limit,
    new_private_histories,
    run_communication,
    run_natural_communication,
    validate_message,
)


class CommunicationTests(unittest.TestCase):
    def test_immediate_same_window_is_never_visible(self):
        seen = {}

        def decide(agent, window, remaining, transcript):
            seen[agent, window] = transcript
            self.assertEqual(len(transcript), 3 * (window - 1))
            self.assertTrue(all(record["window"] < window for record in transcript))
            self.assertEqual({record["sender"] for record in transcript}, set(AGENTS) if window > 1 else set())
            return ALPHABET[AGENTS.index(agent)] * window

        result = run_communication("immediate", decide, step=1)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(result["messages"]), 12)
        for window in range(1, 5):
            self.assertEqual(seen["A", window], seen["C", window])

    def test_delayed_sees_own_earlier_messages_and_public_past(self):
        prior = {"sender": "B", "window": 1, "text": "#", "step": 1}
        previous = {agent: [prior] for agent in AGENTS}

        def decide(agent, window, remaining, transcript):
            self.assertEqual(transcript[0], prior)
            current = transcript[1:]
            self.assertEqual(len(current), window - 1)
            self.assertTrue(all(record["sender"] == agent for record in current))
            self.assertTrue(all(record["window"] < window for record in current))
            return "@"

        result = run_communication("delayed", decide, previous_transcripts=previous, step=2)
        for agent in AGENTS:
            self.assertEqual(result["transcripts"][agent], [prior] + result["messages"])
            self.assertEqual(len(result["transcripts"][agent]), 13)

    def test_equal_final_delivery_with_nonresponsive_sender(self):
        def decide(agent, window, remaining, transcript):
            return ALPHABET[AGENTS.index(agent)] * window

        immediate = run_communication("immediate", decide)
        delayed = run_communication("delayed", decide)
        self.assertEqual(immediate["messages"], delayed["messages"])
        self.assertEqual(immediate["transcripts"], delayed["transcripts"])
        self.assertEqual(immediate["usage"], delayed["usage"])

    def test_callback_and_returned_mutations_do_not_cross_contexts(self):
        previous = {agent: [{"sender": "A", "window": 1, "text": "@"}] for agent in AGENTS}
        original = copy.deepcopy(previous)

        def decide(agent, window, remaining, transcript):
            self.assertEqual(transcript[0]["text"], "@")
            transcript[0]["text"] = "corrupted private copy"
            transcript.append({"private_analysis": "must not be retained"})
            return "#"

        result = run_communication("immediate", decide, previous_transcripts=previous)
        self.assertEqual(previous, original)
        result["transcripts"]["A"][0]["text"] = "changed"
        self.assertEqual(result["transcripts"]["B"][0]["text"], "@")
        result["transcripts"]["A"][-1]["text"] = "changed"
        self.assertEqual(result["transcripts"]["C"][-1]["text"], "#")
        self.assertEqual(result["messages"][-1]["text"], "#")

    def test_zero_budget_still_has_four_windows_and_twelve_calls(self):
        calls = []

        def decide(agent, window, remaining, transcript):
            calls.append((agent, window, remaining))
            return "@" * message_limit(remaining)

        for condition in ("immediate", "delayed"):
            calls.clear()
            result = run_communication(condition, decide)
            self.assertEqual(len(calls), 12)
            self.assertEqual(result["remaining"], dict.fromkeys(AGENTS, 0))
            for agent in AGENTS:
                self.assertEqual([r for a, w, r in calls if a == agent], [64, 32, 0, 0])
                self.assertEqual([r["text"] for r in result["messages"] if r["sender"] == agent], ["@" * 32, "@" * 32, "", ""])

    def test_remaining_budget_reduces_generation_limit(self):
        def decide(agent, window, remaining, transcript):
            lengths = (32, 31, 1, 0)
            self.assertEqual(message_limit(remaining), (32, 32, 1, 0)[window - 1])
            return "#" * lengths[window - 1]

        result = run_communication("immediate", decide)
        self.assertEqual(result["remaining"]["A"], 0)
        self.assertEqual(result["usage"][6]["allowed_symbols"], 1)

    def test_invalid_output_is_rejected_not_truncated_or_cleaned(self):
        for text in ("@" * 33, "@ ", "@\n", "￥", None, ["@"]):
            with self.subTest(text=text), self.assertRaises(ProtocolError):
                run_communication("immediate", lambda *args: text)

    def test_exhausted_budget_rejects_nonempty_message(self):
        with self.assertRaisesRegex(ProtocolError, "allowed maximum is 0"):
            run_communication("immediate", lambda *args: "@" * 32)

    def test_partial_budget_rejects_overrun(self):
        def decide(agent, window, remaining, transcript):
            return "@" * (32, 31, 2, 0)[window - 1]

        with self.assertRaisesRegex(ProtocolError, "allowed maximum is 1"):
            run_communication("delayed", decide)

    def test_no_call_time_or_order_metadata_enters_public_channel(self):
        result = run_communication("immediate", lambda *args: "", step=3)
        for transcript in result["transcripts"].values():
            self.assertTrue(all(set(record) == {"sender", "window", "text", "step"} for record in transcript))
        for forbidden in ("elapsed_seconds", "call_id", "private_analysis", "recipient"):
            bad = {"sender": "A", "window": 1, "text": "@", forbidden: "hidden"}
            with self.subTest(forbidden=forbidden), self.assertRaises(ProtocolError):
                run_communication("immediate", lambda *args: "", previous_transcripts={agent: [bad] for agent in AGENTS})

    def test_invalid_protocol_configuration_fails_before_calling(self):
        def never(*args):
            self.fail("invalid configuration reached model callback")

        with self.assertRaises(ProtocolError):
            run_communication("unknown", never)
        with self.assertRaises(ProtocolError):
            run_communication("immediate", never, previous_transcripts={"A": []})
        with self.assertRaises(ProtocolError):
            run_communication("immediate", never, step=True)

    def test_full_alphabet_and_empty_are_valid(self):
        validate_message(ALPHABET * 4)
        validate_message("", 0)
        with self.assertRaises(ProtocolError):
            message_limit(True)

    def test_natural_control_preserves_synchronous_isolation(self):
        def decide(agent, window, remaining, transcript):
            self.assertIsNone(remaining)
            self.assertEqual(len(transcript), 3 * (window - 1))
            self.assertTrue(all(record["window"] < window for record in transcript))
            return f"我是{agent}，请先把工具带到林地。"

        result = run_natural_communication(decide, step=1)
        self.assertEqual(len(result["messages"]), 12)
        self.assertEqual(result["transcripts"]["A"], result["transcripts"]["C"])
        result["transcripts"]["A"][0]["text"] = "changed"
        self.assertNotEqual(result["transcripts"]["A"], result["transcripts"]["B"])
        self.assertIn("我是A", result["messages"][0]["text"])
        # A natural-language control must not weaken the symbolic channel.
        with self.assertRaises(ProtocolError):
            run_communication("immediate", lambda *args: "请帮忙")


class PrivateHistoryTests(unittest.TestCase):
    def _append(self, history, **overrides):
        kwargs = {
            "step": 1,
            "episode": 1,
            "observation": {"location": "camp", "items": ["own view"]},
            "action": {"type": "wait"},
            "feedback": {"moved": False},
            "transcript": [{"sender": "A", "window": 1, "text": "@"}],
        }
        kwargs.update(overrides)
        history.append_step(**kwargs)
        return kwargs

    def test_distinct_owners_groups_and_snapshots(self):
        histories = new_private_histories()
        other_group = new_private_histories()
        inputs = self._append(histories["A"])
        inputs["observation"]["items"].append("later mutation")
        snapshot = histories["A"].snapshot()
        self.assertEqual(snapshot[0]["observation"]["items"], ["own view"])
        snapshot[0]["transcript"][0]["text"] = "#"
        self.assertEqual(histories["A"].snapshot()[0]["transcript"][0]["text"], "@")
        self.assertEqual(histories["B"].snapshot(), [])
        self.assertEqual(other_group["A"].snapshot(), [])

    def test_evaluation_fork_does_not_update_training_history(self):
        training = PrivateHistory("A")
        self._append(training)
        evaluation = training.fork()
        self._append(evaluation, step=2)
        self.assertEqual(len(evaluation.snapshot()), 2)
        self.assertEqual(len(training.snapshot()), 1)

    def test_no_private_analysis_persistence_field_or_nested_keys(self):
        for field in ("private_analysis", "analysis", "reasoning", "raw_prompt"):
            history = PrivateHistory("A")
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                self._append(history, observation={"location": "camp", "nested": [{field: "private thought"}]})
            self.assertEqual(history.snapshot(), [])
        with self.assertRaises(TypeError):
            self._append(PrivateHistory("A"), private_analysis="not an accepted field")


if __name__ == "__main__":
    unittest.main()
