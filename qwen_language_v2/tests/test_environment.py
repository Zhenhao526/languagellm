from copy import deepcopy
import json
import unittest

from qwen_language_v2.environment import AGENTS, LOCATIONS, World


def waits():
    return {a: {"kind": "wait"} for a in AGENTS}


def act(world, agent, kind, oid=None, **kwargs):
    action = {"kind": kind, **kwargs}
    if oid:
        action["item"] = world.handles[agent][oid]
    return action


def ready_joint():
    w = World(17, 0)
    a, b = w.roles["wood"], w.roles["tool"]
    oid = next(k for k, x in w.items.items() if x["kind"] == "木材" and x["target_for_witness"])
    w.items[oid]["processed"] = True
    w.positions[b] = w.positions[a]
    return w, a, b, oid


class EnvironmentTests(unittest.TestCase):
    def test_witness_family_replays_within_twelve_steps(self):
        for seed in (0, 17, 6101, 6148):
            for variant in range(48):
                with self.subTest(seed=seed, variant=variant):
                    w = World(seed, variant)
                    self.assertEqual(len(w.items), 9)
                    self.assertLessEqual(len(w.witness), 12)
                    for actions in w.witness:
                        feedback = w.step(actions)
                        self.assertTrue(all(f["action_succeeded"] for f in feedback.values()))
                    self.assertEqual(w.status, "success")
                    self.assertEqual(w.score, 1)

    def test_determinism_and_serialized_continuation(self):
        a, b = World(6101, 0), World(6101, 0)
        self.assertEqual(a.state_dict(), b.state_dict())
        a.step(a.witness[0])
        c = World.from_state_dict(json.loads(json.dumps(a.state_dict())))
        for joint in a.witness[1:]:
            self.assertEqual(a.step(joint), c.step(joint))
        self.assertEqual(a.state_dict(), c.state_dict())

    def test_attribute_quantity_destination_and_roles_vary(self):
        worlds = [World(6101 + v, v) for v in range(48)]
        for kind in ("wood", "fiber"):
            self.assertEqual({w.scenario_parameters[f"{kind}_attributes"][0] for w in worlds}, {"长", "短"})
            self.assertEqual({w.scenario_parameters[f"{kind}_attributes"][1] for w in worlds}, {"干", "湿"})
        self.assertEqual({w.goals[1]["quantity"] for w in worlds}, {1, 2})
        self.assertEqual({w.goals[0]["destination"] for w in worlds}, set(LOCATIONS))
        self.assertEqual({w.roles["collector"] for w in worlds}, set(AGENTS))

    def test_remote_goals_and_items_do_not_change_local_view_or_menu(self):
        w = World(17, 0)
        observer = w.roles["wood"]
        initial_obs, initial_menu = w.observe(observer), w.action_menu(observer)
        for goal in w.goals:
            goal["quantity"] += 1
            goal["condition"] = "湿"
        for x in w.items.values():
            if x["location"] != w.positions[observer]:
                x["condition"] = "different remote value"
        self.assertEqual(w.observe(observer), initial_obs)
        self.assertEqual(w.action_menu(observer), initial_menu)

    def test_static_board_does_not_broadcast_remote_deliveries(self):
        w = World(6148, 7)
        observer = w.roles["collector"]
        before = w.observe(observer)
        w.goals[1]["delivered"] = 1
        self.assertEqual(w.observe(observer), before)
        self.assertNotIn("delivered", before["task_board"][0])
        self.assertIn("delivered", w.full_information_observe(observer)["task_board"][0])

    def test_local_resource_frequencies_do_not_encode_the_hidden_target(self):
        for variant in range(16):
            w = World(6101, variant)
            wood = [x for x in w.items.values() if x["kind"] == "木材"]
            self.assertEqual({(x["length"], x["condition"]) for x in wood},
                             {("长", "干"), ("长", "湿"), ("短", "干"), ("短", "湿")})
            counts = {}
            for x in w.items.values():
                if x["kind"] == "纤维":
                    key = (x["length"], x["condition"])
                    counts[key] = counts.get(key, 0) + 1
            self.assertEqual(sorted(counts.values()), [2, 2])
            for observer in (w.roles["wood"], w.roles["tool"]):
                before_obs, before_menu = w.observe(observer), w.action_menu(observer)
                for goal in w.goals:
                    goal["length"] = "短" if goal["length"] == "长" else "长"
                    goal["condition"] = "湿" if goal["condition"] == "干" else "干"
                self.assertEqual(w.observe(observer), before_obs)
                self.assertEqual(w.action_menu(observer), before_menu)

    def test_private_handles_and_no_unique_public_object_identifiers(self):
        w = World(17, 1)
        all_handles = [set(w.handles[a].values()) for a in AGENTS]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(all_handles[i] & all_handles[j])
        for a in AGENTS:
            encoded = json.dumps(w.observe(a), ensure_ascii=False)
            for hidden in ("target_for_witness", "witness", "seed", "variant", "marking"):
                self.assertNotIn(hidden, encoded)
            self.assertTrue(all(f'"{oid}"' not in encoded for oid in w.items))

    def test_equivalent_observation_field_and_object_order_is_private(self):
        w = World(17, 0)
        for a in AGENTS:
            w.positions[a] = w.layout["wood_site"]
        orders = [list(w.observe(a)) for a in AGENTS]
        self.assertGreater(len({tuple(x) for x in orders}), 1)
        for a in AGENTS:
            self.assertEqual(w.observe(a), w.observe(a))

    def test_road_event_is_local_and_not_predicted_by_menu(self):
        w = World(17, 1)
        observer = w.roles["wood"]
        before_obs, before_menu = w.observe(observer), w.action_menu(observer)
        w.event["occurred"] = True
        self.assertEqual(w.observe(observer), before_obs)
        self.assertEqual(w.action_menu(observer), before_menu)
        camper = w.roles["collector"]
        self.assertEqual(w.observe(camper)["known_closed_roads"], [w.event["edge"]])
        actions = waits()
        actions[observer] = {"kind": "move", "destination": "营地"}
        feedback = w.step(actions)
        self.assertFalse(feedback[observer]["action_succeeded"])
        self.assertEqual(w.observe(observer)["known_closed_roads"], [w.event["edge"]])

    def test_full_information_control_has_current_state_without_future(self):
        w = World(17, 1)
        obs = w.full_information_observe("A")
        self.assertEqual(sum(len(s["items"]) for s in obs["all_current_sites"]), 9)
        self.assertEqual(obs["all_current_closed_roads"], [])
        encoded = json.dumps(obs, ensure_ascii=False)
        for hidden in ("at_step", "target_for_witness", "witness", "seed", "variant", "observer_site", "enabled"):
            self.assertNotIn(hidden, encoded)
        for site in obs["all_current_sites"]:
            for item in site["items"]:
                self.assertIn(item["handle"], w.handles["A"].values())

    def test_joint_carry_requires_reciprocal_matching(self):
        w, a, b, oid = ready_joint()
        origin = w.positions[a]
        actions = waits()
        actions[a] = act(w, a, "carry_together", oid, partner=b, destination="营地")
        feedback = w.step(actions)
        self.assertFalse(feedback[a]["action_succeeded"])
        self.assertEqual(w.positions[a], origin)
        actions[b] = act(w, b, "carry_together", oid, partner=a, destination="营地")
        feedback = w.step(actions)
        self.assertTrue(feedback[a]["action_succeeded"] and feedback[b]["action_succeeded"])
        self.assertEqual(w.positions[a], "营地")
        self.assertEqual(w.inventory[a]["cargo"], oid)
        self.assertEqual(w.inventory[b]["cargo"], oid)
        self.assertFalse(any(e["action"]["kind"] == "move" for e in w.action_menu(a)))

    def test_joint_failure_does_not_report_partner_intention(self):
        w, a, b, oid = ready_joint()
        acts = waits()
        acts[a] = act(w, a, "carry_together", oid, partner=b, destination="营地")
        destination = next(x for x in LOCATIONS if x not in ("营地", w.positions[b]))
        acts[b] = act(w, b, "carry_together", oid, partner=a, destination=destination)
        before = deepcopy(w.positions)
        feedback = w.step(acts)
        self.assertEqual(w.positions, before)
        self.assertEqual(feedback[a]["result"], "共同动作没有发生。")
        self.assertNotIn(destination, feedback[a]["result"])

    def test_simultaneous_actions_ignore_input_dictionary_order(self):
        w, a, b, oid = ready_joint()
        acts = waits()
        acts[a] = act(w, a, "carry_together", oid, partner=b, destination="营地")
        acts[b] = act(w, b, "carry_together", oid, partner=a, destination="营地")
        clone = World.from_state_dict(w.state_dict())
        self.assertEqual(w.step(acts), clone.step(dict(reversed(list(acts.items())))))
        self.assertEqual(w.state_dict(), clone.state_dict())

    def test_pickup_conflict_fails_for_every_contender(self):
        w = World(17, 0)
        axe = next(oid for oid, x in w.items.items() if x["kind"] == "斧具")
        for a in ("A", "B"):
            w.positions[a] = w.items[axe]["location"]
        actions = waits()
        for a in ("A", "B"):
            actions[a] = act(w, a, "pickup", axe)
        feedback = w.step(actions)
        self.assertFalse(feedback["A"]["action_succeeded"])
        self.assertFalse(feedback["B"]["action_succeeded"])
        self.assertEqual(w.items[axe]["carriers"], [])

    def test_give_conflict_fails_for_both_without_priority(self):
        w = World(17, 0)
        for a in AGENTS:
            w.positions[a] = "营地"
        fibers = [oid for oid, x in w.items.items() if x["kind"] == "纤维"][:2]
        for a, oid in zip(("A", "B"), fibers):
            w.inventory[a]["cargo"] = oid
            w.items[oid]["location"] = None
            w.items[oid]["carriers"] = [a]
        actions = waits()
        for a, oid in zip(("A", "B"), fibers):
            actions[a] = act(w, a, "give", oid, partner="C")
        feedback = w.step(actions)
        self.assertFalse(feedback["A"]["action_succeeded"])
        self.assertFalse(feedback["B"]["action_succeeded"])
        self.assertIsNone(w.inventory["C"]["cargo"])

    def test_cannot_pick_up_and_use_remote_axe_in_one_tick(self):
        w = World(17, 0)
        axe = next(oid for oid, x in w.items.items() if x["kind"] == "斧具")
        wood = next(oid for oid, x in w.items.items() if x["kind"] == "木材")
        agent = w.roles["tool"]
        self.assertFalse(any(e["action"]["kind"] == "cut" for e in w.action_menu(agent)))
        actions = waits()
        actions[agent] = act(w, agent, "cut", wood)
        feedback = w.step(actions)
        self.assertFalse(feedback[agent]["action_succeeded"])
        self.assertIsNone(w.inventory[agent]["tool"])
        self.assertFalse(w.items[wood]["processed"])

    def test_indices_and_direct_actions_have_same_effect(self):
        w = World(6101, 0)
        clone = World.from_state_dict(w.state_dict())
        actions = w.witness[0]
        indices = {a: next(e["id"] for e in w.action_menu(a) if e["action"] == actions[a]) for a in AGENTS}
        self.assertEqual(w.step(indices), clone.step(actions))

    def test_terminal_feedback_and_no_repeated_delivery_reward(self):
        w = World(6101, 0)
        for i, joint in enumerate(w.witness):
            feedback = w.step(joint)
            if i < len(w.witness) - 1:
                self.assertTrue(all("team_score" not in f for f in feedback.values()))
        self.assertTrue(all(f["team_score"] == 1 for f in feedback.values()))
        with self.assertRaises(RuntimeError):
            w.step(waits())

    def test_simultaneous_excess_delivery_accepts_both_but_caps_score(self):
        w = World(6101, 0)
        goal = w.goals[1]
        self.assertEqual(goal["quantity"], 1)
        fibers = [oid for oid, x in w.items.items()
                  if x["kind"] == "纤维" and x["target_for_witness"]]
        for a, oid in zip(("A", "B"), fibers):
            w.positions[a] = goal["destination"]
            w.inventory[a]["cargo"] = oid
            w.items[oid]["location"] = None
            w.items[oid]["carriers"] = [a]
        before_remote = w.observe("C")
        actions = waits()
        for a, oid in zip(("A", "B"), fibers):
            actions[a] = act(w, a, "deliver", oid)
        clone = World.from_state_dict(w.state_dict())
        feedback = w.step(actions)
        self.assertEqual(feedback, clone.step(dict(reversed(list(actions.items())))))
        self.assertEqual(w.state_dict(), clone.state_dict())
        self.assertTrue(feedback["A"]["action_succeeded"])
        self.assertTrue(feedback["B"]["action_succeeded"])
        self.assertEqual(w.goals[1]["delivered"], 1)
        self.assertTrue(all(w.items[oid]["delivered"] for oid in fibers))
        self.assertTrue(all(w.inventory[a]["cargo"] is None for a in ("A", "B")))
        self.assertNotIn("team_score", feedback["C"])
        if "task_board" in before_remote:
            self.assertEqual(w.observe("C")["task_board"], before_remote["task_board"])
        # An otherwise matching delivery on a later tick is rejected.
        extra = next(oid for oid, x in w.items.items()
                     if x["kind"] == "纤维" and not x["target_for_witness"])
        w.items[extra].update(length=goal["length"], condition=goal["condition"],
                              location=None, carriers=["C"])
        w.positions["C"] = goal["destination"]
        w.inventory["C"]["cargo"] = extra
        actions = waits()
        actions["C"] = act(w, "C", "deliver", extra)
        feedback = w.step(actions)
        self.assertFalse(feedback["C"]["action_succeeded"])
        self.assertEqual(w.goals[1]["delivered"], 1)
        self.assertFalse(w.items[extra]["delivered"])
        # Re-submitting an already delivered object is not a legal new action.
        actions = waits()
        actions["A"] = act(w, "A", "deliver", fibers[0])
        self.assertFalse(w.step(actions)["A"]["action_succeeded"])
        self.assertEqual(w.goals[1]["delivered"], 1)


if __name__ == "__main__":
    unittest.main()
