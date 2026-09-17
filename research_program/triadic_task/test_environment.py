"""Behavioral and observation-boundary checks, without model calls."""
from copy import deepcopy
from itertools import permutations, product
import unittest

from .environment import (AGENTS, State, action_menu, all_actions, observe, settle,
                          sufficient_information_witness, support)


def action(site='S0', destination='L', partner='B'):
    return {'kind': 'transport', 'site': site, 'destination': destination, 'partner': partner}


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.state = State((0, 6, 9), (0, 1, 2, 3))
        self.pair = {'A': action(), 'B': action(partner='A'), 'C': {'kind': 'wait'}}

    def test_matching_and_individual_partial_reward(self):
        self.assertEqual(settle(self.state, self.pair, require_match=True)['reward'], 1)
        # A wants wood, B wants short: long wood still executes both transports,
        # but only A's need is fulfilled, with team reward one half.
        wrong = deepcopy(self.pair)
        wrong['A']['site'] = wrong['B']['site'] = 'S1'
        result = settle(self.state, wrong, require_match=True)
        self.assertEqual(result['reward'], .5)
        self.assertTrue(result['individual_feedback']['B']['executed'])
        self.assertFalse(result['individual_feedback']['B']['own_need_satisfied'])

    def test_all_matching_fields_are_required(self):
        for field, value in [('partner', 'C'), ('site', 'S1'), ('destination', 'R')]:
            with self.subTest(field=field):
                wrong = deepcopy(self.pair)
                wrong['B'][field] = value
                self.assertEqual(settle(self.state, wrong, require_match=True)['reward'], 0)

    def test_overload_wait_and_single_transport(self):
        overloaded = deepcopy(self.pair)
        overloaded['C'] = action(partner='A')
        for matching in (False, True):
            self.assertEqual(settle(self.state, overloaded, require_match=matching)['reward'], 0)
        single = deepcopy(self.pair)
        single['B'] = {'kind': 'wait'}
        self.assertEqual(settle(self.state, single, require_match=True)['reward'], 0)
        self.assertEqual(settle(self.state, single, require_match=False)['reward'], .5)

    def test_d0_can_deliver_different_objects_and_ignore_partner(self):
        separate = {'A': action(site='S1', partner='C'), 'B': action(site='S2', partner='C'), 'C': {'kind': 'wait'}}
        self.assertEqual(settle(self.state, separate, require_match=False)['reward'], 1)
        self.assertEqual(settle(self.state, separate, require_match=True)['reward'], 0)

    def test_hidden_state_cannot_leak_into_observation(self):
        changed = State((0, 2, 4), (0, 1, 3, 2))
        self.assertEqual(observe(self.state, 'A', shared_needs=False), observe(changed, 'A', shared_needs=False))
        self.assertNotEqual(observe(self.state, 'A', shared_needs=True), observe(changed, 'A', shared_needs=True))
        same_needs = State(self.state.needs, changed.layout)
        self.assertEqual(observe(self.state, 'A', shared_needs=True), observe(same_needs, 'A', shared_needs=True))

    def test_full_information_is_explicit_and_distinct(self):
        local = observe(self.state, 'B', shared_needs=False)
        full = observe(self.state, 'B', shared_needs=False, full_information=True)
        self.assertEqual(len(local['visible_materials']), 2)
        self.assertNotIn('shared_needs', local)
        self.assertEqual(len(full['visible_materials']), 4)
        self.assertEqual(set(full['shared_needs']), set(AGENTS))

    def test_menu_is_complete_and_state_independent(self):
        for agent in AGENTS:
            normal = action_menu(agent, list(range(17)))
            reverse = action_menu(agent, list(reversed(range(17))))
            self.assertEqual([m['action'] for m in normal], list(reversed([m['action'] for m in reverse])))
            self.assertEqual(len(all_actions(agent)), 17)
        with self.assertRaises(ValueError):
            action_menu('A', [0]*17)
        with self.assertRaises(ValueError):
            settle(self.state, {'A': {'kind': 'wait'}}, require_match=True)

    def test_all_representative_states_have_valid_witness(self):
        demand_support = support()
        self.assertEqual(len(demand_support), 996)
        checked = 0
        for needs, layout in product(demand_support, permutations(range(4))):
            state = State(needs, layout)
            witness = sufficient_information_witness(state)
            for matching in (False, True):
                self.assertEqual(settle(state, witness, require_match=matching)['reward'], 1)
                checked += 1
        self.assertEqual(checked, 47808)

    def test_state_copies_mutable_inputs_and_rejects_noninteger_ids(self):
        needs, layout, private = [0, 6, 9], [0, 1, 2, 3], [1, 2, 3]
        state = State(needs, layout, private)
        needs[0], layout[0], private[0] = 11, 3, 3
        self.assertEqual(state, self.state)
        for value in (False, 0.0, '0'):
            with self.assertRaises(ValueError):
                State((value, 6, 9), (0, 1, 2, 3))

    def test_unknown_identity_is_not_a_valid_policy_interface(self):
        with self.assertRaises(ValueError):
            all_actions('D')
        with self.assertRaises(ValueError):
            action_menu('D', list(range(17)))


if __name__ == '__main__':
    unittest.main()
