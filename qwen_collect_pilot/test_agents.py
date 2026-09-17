"""Test actual private prompt payloads and feedback boundaries without a model."""
from copy import deepcopy
from io import StringIO
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agents import build_messages, empty_histories, remember
from env import World


def payload(messages):
    return json.loads(messages[1]["content"])


class PrivateContextTests(unittest.TestCase):
    def test_current_collector_prompts_are_invariant_to_hidden_state(self):
        for role in ("A", "B"):
            for own_position in (0, 1):
                prompts = []
                for d_a, d_b, other_position in product((0, 1), repeat=3):
                    world = (World(d_a, d_b, own_position, other_position)
                             if role == "A"
                             else World(d_a, d_b, other_position, own_position))
                    prompts.append(build_messages(role, world.private_observation(role), [], received="@%"))
                self.assertTrue(all(prompt == prompts[0] for prompt in prompts))
                self.assertEqual(
                    set(payload(prompts[0])),
                    {"你的过去互动", "本轮私有观察", "本轮收到的消息", "本轮可选动作"},
                )

    def test_coordinator_prompt_contains_no_slot_order_or_incoming_channel(self):
        prompts = [
            build_messages("C", World(0, 1, p_a, p_b).private_observation("C"), [])
            for p_a, p_b in product((0, 1), repeat=2)
        ]
        self.assertTrue(all(prompt == prompts[0] for prompt in prompts))
        self.assertEqual(set(payload(prompts[0])), {"你的过去互动", "本轮私有观察"})
        with self.assertRaises(ValueError):
            build_messages("C", World(0, 0, 0, 0).private_observation("C"), [], received="@")

    def test_feedback_contains_only_own_result_and_team_result(self):
        world = World(0, 1, 0, 0)
        actions = {"A": 0, "B": 0}
        histories = empty_histories()
        scores = world.score(actions)
        remember(histories, world, "@&", actions, scores)
        self.assertEqual(histories["A"][0], {
            "观察": world.private_observation("A"), "收到": "@&", "动作": 0,
            "反馈": {"自己匹配": True, "群体成功": False},
        })
        self.assertEqual(histories["B"][0], {
            "观察": world.private_observation("B"), "收到": "@&", "动作": 0,
            "反馈": {"自己匹配": False, "群体成功": False},
        })
        self.assertEqual(histories["C"][0], {
            "观察": world.private_observation("C"), "发送": "@&",
            "反馈": {"群体成功": False},
        })
        # Mutating experimenter-side results must not rewrite past feedback.
        actions["A"] = 1
        scores["A"] = False
        self.assertEqual(histories["A"][0]["动作"], 0)
        self.assertIs(histories["A"][0]["反馈"]["自己匹配"], True)

    def test_individual_and_experimental_group_histories_are_separate(self):
        group_one, group_two = empty_histories(), empty_histories()
        self.assertEqual(len({id(history) for history in group_one.values()}), 3)
        world = World(1, 0, 1, 0)
        actions = world.targetslots()
        remember(group_one, world, "#%", actions, world.score(actions))
        self.assertEqual(group_two, {"A": [], "B": [], "C": []})
        snapshot_b, snapshot_c = deepcopy(group_one["B"]), deepcopy(group_one["C"])
        group_one["A"][0]["观察"]["slots"][0]["resource"] = "PRIVATE_A_MARKER"
        self.assertEqual(group_one["B"], snapshot_b)
        self.assertEqual(group_one["C"], snapshot_c)
        for role in ("B", "C"):
            kwargs = {} if role == "C" else {"received": "#"}
            messages = build_messages(role, world.private_observation(role), group_one[role], **kwargs)
            self.assertNotIn("PRIVATE_A_MARKER", str(messages))

    def test_payload_serialization_does_not_mutate_or_alias_history(self):
        histories = empty_histories()
        world = World(0, 0, 1, 0)
        remember(histories, world, "%", world.targetslots(), world.score(world.targetslots()))
        before = deepcopy(histories)
        messages = build_messages("A", world.private_observation("A"), histories["A"], received="&")
        self.assertEqual(histories, before)
        histories["A"].clear()
        self.assertEqual(len(payload(messages)["你的过去互动"]), 1)

    def test_known_protocol_is_confined_to_requested_control(self):
        world = World(0, 0, 0, 0)
        normal = build_messages("A", world.private_observation("A"), [], received="@@")
        control = build_messages("A", world.private_observation("A"), [], received="@@", known_protocol=True)
        normal_again = build_messages("A", world.private_observation("A"), [], received="@@")
        self.assertNotIn("@@表示F0", normal[0]["content"])
        self.assertIn("@@表示F0", control[0]["content"])
        self.assertNotIn("不存在预先给定的词典", control[0]["content"])
        self.assertEqual(normal, normal_again)

    def test_channel_and_role_violations_are_rejected(self):
        world = World(0, 0, 0, 0)
        observation = world.private_observation("A")
        for invalid in (None, "@@@", "@ #", "你好", "@\n", "＠", "0", "￥"):
            with self.assertRaises(ValueError, msg=repr(invalid)):
                build_messages("A", observation, [], received=invalid)
        for valid in ("", "@", "#%", "&&"):
            self.assertEqual(payload(build_messages("A", observation, [], received=valid))["本轮收到的消息"], valid)
        with self.assertRaises(ValueError):
            build_messages("A", world.private_observation("B"), [], received="@")
        leaked = dict(observation, demand={"fiber": "F0"})
        with self.assertRaises(ValueError):
            build_messages("A", leaked, [], received="@")


class ProtocolBackend:
    """A deterministic known encoder/decoder used only to test experiment wiring."""
    def __init__(self):
        self.calls = 0
        self.seen = []

    def infer(self, messages, *, mode, **kwargs):
        self.calls += 1
        self.seen.append(deepcopy(messages))
        data = payload(messages)
        observation = data["本轮私有观察"]
        if mode == "message":
            demand = observation["demand"]
            return "@#"[int(demand["fiber"][-1])] + "@#"[int(demand["fuel"][-1])]
        message = data["本轮收到的消息"]
        if not message:
            return "0"
        index = 0 if observation["role"] == "A" else 1
        target = "@#".index(message[index])
        return str(next(item["slot"] for item in observation["slots"]
                        if int(item["resource"][-1]) == target))

    decide = infer


class FrozenEvaluationTests(unittest.TestCase):
    def test_interventions_and_memoization_preserve_information_boundaries(self):
        from run_pilot import Experiment

        with TemporaryDirectory() as directory:
            backend = ProtocolBackend()
            experiment = Experiment(backend, Path(directory))
            histories = empty_histories()
            before = deepcopy(histories)
            try:
                with patch("sys.stdout", new_callable=StringIO):
                    result = experiment.frozen_eval(histories, "test_group_one")
                self.assertEqual(histories, before)
                self.assertEqual(result["conditions"]["original"]["overall"], 16)
                self.assertEqual(result["conditions"]["empty"]["overall"], 4)
                self.assertEqual(result["conditions"]["permuted"]["overall"], 0)
                self.assertEqual(result["distinct_messages"], 4)
                self.assertEqual(result["permutation_changed_message_fraction"], 1.0)
                # Four encoder inputs, 16 collector message/position inputs,
                # and four empty-message collector inputs exhaust private prompts.
                self.assertEqual(backend.calls, 24)
                for messages in backend.seen:
                    self.assertNotIn("test_group_one", str(messages))
                    self.assertEqual(payload(messages)["你的过去互动"], [])
                with patch("sys.stdout", new_callable=StringIO):
                    experiment.frozen_eval(empty_histories(), "test_group_two")
                # A second frozen group must build its own memoization cache.
                self.assertEqual(backend.calls, 48)
                rows = [json.loads(line) for line in
                        (Path(directory) / "trials.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 96)
                self.assertTrue(all(row["feedback_returned"] is False for row in rows))
            finally:
                experiment.trace.close()


if __name__ == "__main__":
    unittest.main()
