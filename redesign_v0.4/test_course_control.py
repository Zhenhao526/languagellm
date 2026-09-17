"""Matched-design checks; these tests never start experiment training."""
import copy
import unittest

import torch

from run_course_control import (REFERENCE, build_config, read_json, same_states,
                                validate_design)


class CourseControlDesignTests(unittest.TestCase):
    def setUp(self):
        self.reference = read_json(REFERENCE / "config.json")
        self.config = build_config()

    def test_full_task_and_matched_exploration_at_all_1200_updates(self):
        audit = validate_design(self.config, self.reference)
        self.assertTrue(audit["all_control_updates_use_full_task"])
        self.assertTrue(audit["exploration_schedule_matches_every_update"])
        self.assertTrue(audit["both_auxiliary_weights_zero_every_update"])
        self.assertEqual(audit["joint_samples_per_run"], 1228800)

    def test_unintended_optimizer_change_rejected(self):
        self.config["learning_rate"] *= 2
        with self.assertRaises(AssertionError):
            validate_design(self.config, self.reference)

    def test_missing_or_shortened_control_schedule_rejected(self):
        for key, value in (("course_updates", 1), ("transition_updates", 600),
                           ("pure_reward_updates", 0)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(AssertionError):
                validate_design(config, self.reference)

    def test_tensor_comparison_detects_parameter_changes(self):
        states = [{"layer": torch.tensor([1., 2.])}, {"layer": torch.tensor([3., 4.])}]
        self.assertTrue(same_states(states, copy.deepcopy(states)))
        changed = copy.deepcopy(states)
        changed[1]["layer"][0] += 1
        self.assertFalse(same_states(states, changed))


if __name__ == "__main__":
    unittest.main()
