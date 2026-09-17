"""Frozen probe integrity checks using a fake backend, never the local model."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from qwen_language_v2.environment import AGENTS, World
from qwen_language_v2.probe import replay_action_step


def make_case():
    world = World(6101, variant=0)
    witness = deepcopy(world.witness)
    for actions in witness[:-1]:
        world.step(actions)
    return {
        "condition": "immediate", "group": 17, "episode": 1,
        "step": world.t + 1,
        "state_before": world.state_dict(),
        "observations": {agent: world.observe(agent) for agent in AGENTS},
        "menus": {agent: world.action_menu(agent) for agent in AGENTS},
        "private_histories_before_step": {
            agent: [{"only_owner_memory": f"secret_for_{agent}"}] for agent in AGENTS
        },
        "messages": [
            {"sender": agent, "window": window, "text": "@#", "step": world.t + 1}
            for window in range(1, 5) for agent in AGENTS
        ],
        "actions": witness[-1],
    }


class FakeBackend:
    def __init__(self, case, *, wait_for=()):
        self.case = case
        self.wait_for = set(wait_for)
        self.calls = []

    def decide(self, messages, **kwargs):
        agent = kwargs["label"]["agent"]
        self.calls.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        action = {"kind": "wait"} if agent in self.wait_for else self.case["actions"][agent]
        return str(next(entry["id"] for entry in self.case["menus"][agent] if entry["action"] == action))


class ProbeTests(unittest.TestCase):
    def test_both_arms_leave_world_input_and_training_memories_unchanged(self):
        case = make_case()
        before = deepcopy(case)
        replay_action_step(FakeBackend(case), case, "original")
        self.assertEqual(case, before)
        result = replay_action_step(FakeBackend(case, wait_for=AGENTS), case, "empty_text")
        self.assertEqual(case, before)
        self.assertEqual(result["score_delta"], 0)
        self.assertEqual(len(case["private_histories_before_step"]["A"]), 1)

    def test_original_replays_saved_actions_and_simultaneous_transition(self):
        case = make_case()
        backend = FakeBackend(case)
        actual_step = World.step
        state_at_commit = []

        def checked_step(world, actions):
            self.assertEqual(len(backend.calls), 3)
            self.assertEqual(set(actions), set(AGENTS))
            self.assertEqual(world.state_dict(), case["state_before"])
            state_at_commit.append(world.state_dict())
            return actual_step(world, actions)

        with patch.object(World, "step", checked_step):
            result = replay_action_step(backend, case, "original")
        self.assertEqual(len(state_at_commit), 1)
        self.assertEqual(result["actions"], case["actions"])
        self.assertEqual(result["changed_from_recorded_agents"], [])
        self.assertEqual(result["score_after_one_step"], 1)

    def test_each_decision_uses_only_own_history_and_no_earlier_selected_action(self):
        case = make_case()
        first = FakeBackend(case)
        second = FakeBackend(case, wait_for=("A",))
        replay_action_step(first, case, "original")
        replay_action_step(second, case, "original")
        for left, right in zip(first.calls, second.calls):
            # Changing the already-selected action for A does not enter B or C's prompt.
            self.assertEqual(left["messages"], right["messages"])
            agent = left["label"]["agent"]
            data = json.loads(left["messages"][1]["content"])
            self.assertEqual(data["你自己的历史"], [{"only_owner_memory": f"secret_for_{agent}"}])
            for other in AGENTS:
                if other != agent:
                    self.assertNotIn(f"secret_for_{other}", left["messages"][1]["content"])
            self.assertNotIn("state_before", data)
            self.assertNotIn("selections", data)
            self.assertNotIn("witness", left["messages"][1]["content"])

    def test_empty_arm_changes_only_current_message_text_in_decision_input(self):
        case = make_case()
        original_backend, empty_backend = FakeBackend(case), FakeBackend(case)
        replay_action_step(original_backend, case, "original")
        replay_action_step(empty_backend, case, "empty_text")
        for original, empty in zip(original_backend.calls, empty_backend.calls):
            original_data = json.loads(original["messages"][1]["content"])
            empty_data = json.loads(empty["messages"][1]["content"])
            expected = deepcopy(original_data)
            for record in expected["当前已可见广播"]:
                record["text"] = ""
            self.assertEqual(empty_data, expected)
            self.assertEqual(original["messages"][0], empty["messages"][0])
            self.assertEqual(original["choices"], empty["choices"])
            self.assertEqual(original["seed"], empty["seed"])
            self.assertEqual(len(empty_data["当前已可见广播"]), 12)
        self.assertTrue(all(record["text"] == "@#" for record in case["messages"]))

    def test_returned_result_mutation_does_not_change_frozen_case(self):
        case = make_case()
        before = deepcopy(case)
        result = replay_action_step(FakeBackend(case), case, "original")
        result["actions"]["A"]["kind"] = "corrupted"
        result["selections"]["B"]["action"]["kind"] = "corrupted"
        result["feedback"]["C"]["result"] = "corrupted"
        self.assertEqual(case, before)

    def test_unknown_arm_does_not_call_backend(self):
        case = make_case()
        backend = FakeBackend(case)
        with self.assertRaises(ValueError):
            replay_action_step(backend, case, "not_a_registered_arm")
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
