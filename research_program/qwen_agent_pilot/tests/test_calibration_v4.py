from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_program.qwen_agent_pilot import calibration_v4 as v4
from research_program.qwen_agent_pilot.audit_calibration_v4 import audit
from research_program.qwen_agent_pilot.factorized_environment import (
    AGENTS, MEANING_POOL, TOTAL_EPISODES, make_episode, score_episode,
)


class FactorizedEnvironmentTests(unittest.TestCase):
    def test_balanced_two_block_schedule(self):
        episodes = [make_episode(index, 31415) for index in range(TOTAL_EPISODES)]
        self.assertEqual(len(MEANING_POOL), 16)
        self.assertEqual(len({tuple(item) for item in MEANING_POOL}), 16)
        for block in (0, 1):
            rows = [episode for episode in episodes if episode["block"] == block]
            self.assertEqual(len(rows), 48)
            for owner in AGENTS:
                owned = [row["meaning_id"] for row in rows if row["owner"] == owner]
                self.assertEqual(sorted(owned), list(range(16)))

    def test_task_score_requires_target_action_and_other_helper_wait(self):
        episode = make_episode(0, 712)
        designated = episode["goal"]["partner"]
        other = next(agent for agent in episode["helpers"] if agent != designated)
        right = {
            "item_id": episode["target_item_id"],
            "destination": episode["goal"]["destination"],
        }
        self.assertTrue(score_episode(episode, {designated: right, other: None})["success"])
        self.assertFalse(score_episode(episode, {designated: right, other: right})["success"])
        self.assertFalse(score_episode(episode, {designated: None, other: None})["success"])


class CalibrationV4Tests(unittest.TestCase):
    def test_public_partner_condition_discloses_only_assignment_status(self):
        episode = make_episode(0, 11)
        selected = episode["goal"]["partner"]
        helper_prompt = v4._helper_prompt(episode, selected, "@#%&", "public_partner_scalar")
        self.assertIn("Environment role cue", helper_prompt)
        self.assertIn("you are assigned to act", helper_prompt)
        self.assertNotIn("Private order:", helper_prompt)
        self.assertNotIn("responsible partner=", helper_prompt)

    def test_component_feedback_has_slotwise_outcomes_but_scalar_does_not(self):
        episode = make_episode(0, 19)
        designated = episode["goal"]["partner"]
        other = next(agent for agent in episode["helpers"] if agent != designated)
        actions = {
            designated: {"item_id": episode["target_item_id"],
                         "destination": episode["goal"]["destination"]},
            other: None,
        }
        outcome = score_episode(episode, actions)
        scalar = json.loads(v4._feedback_text(
            episode, designated, "hidden_scalar", "@#%&", actions, outcome))
        component = json.loads(v4._feedback_text(
            episode, designated, "hidden_component", "@#%&", actions, outcome))
        self.assertNotIn("public_component_feedback", scalar)
        self.assertIn("public_component_feedback", component)
        self.assertNotIn("should_act", json.dumps(component["public_component_feedback"]))

    def test_parser_suppresses_malformed_action_and_interpretation(self):
        parsed = v4._parse_helper(
            '{"interpretation":{"object":[],"attribute":"purple"},"action":{"item_id":"I9"}}'
        )
        self.assertFalse(parsed["interpretation_valid"])
        self.assertFalse(parsed["action_valid"])
        self.assertIsNone(parsed["action"])

    def test_mock_run_records_private_interpretations_without_raw_completion(self):
        calls = 0

        def fake_post(_url, _model, messages, _seed, _temperature, _max_tokens, _timeout):
            nonlocal calls
            calls += 1
            prompt = messages[-1]["content"]
            if "you are the requester" in prompt:
                return '{"message":"@@##"}', {"prompt_tokens": 1, "completion_tokens": 1}
            return (
                '{"interpretation":{"object":null,"attribute":null,'
                '"responsible_helper":null,"destination":null},"action":null}',
                {"prompt_tokens": 1, "completion_tokens": 1},
            )

        with patch.object(v4, "_post", side_effect=fake_post):
            result = v4.run_condition("http://unused", "test-model", 99, "hidden_scalar")
        self.assertEqual(calls, 3 * TOTAL_EPISODES)
        self.assertEqual(len(result["episodes"]), TOTAL_EPISODES)
        self.assertIn("interpretation", result["episodes"][0]["helper_outputs"][
            result["episodes"][0]["helpers"][0]])
        self.assertFalse(any("raw" in key for key in result["episodes"][0]))
        self.assertEqual(result["summary"]["owner_message_valid"], TOTAL_EPISODES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "run.json"
            audit_path = root / "audit.json"
            results_path.write_text(json.dumps({
                "calibration_version": v4.CALIBRATION_VERSION,
                "model_revision": v4.MODEL_REVISION,
                "summary": result["summary"],
                "episodes": result["episodes"],
            }))
            audited = audit(results_path, audit_path)
            self.assertEqual(audited["status"], "passed", audited["errors"])


if __name__ == "__main__":
    unittest.main()
