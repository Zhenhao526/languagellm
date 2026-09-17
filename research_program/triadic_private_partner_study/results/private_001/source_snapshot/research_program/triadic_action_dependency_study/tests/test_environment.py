import copy
from itertools import permutations, product
import unittest
import numpy as np

from research_program.triadic_action_dependency_study import environment as e
from research_program.triadic_action_dependency_study import dataset as d
from research_program.triadic_task import environment as old


class EnvironmentTests(unittest.TestCase):
    def test_resource_conjunctions_and_original_needs(self):
        for n, m, dest in product(range(24), range(4), range(2)):
            view = e.need_view(n)
            expected = (e.MATERIALS[m][0] in view['kinds'] and e.MATERIALS[m][1] in view['lengths']
                        and e.DESTINATIONS[dest] in view['destinations'])
            self.assertEqual(e.accepts(n, m, dest), expected)
            if n < 12:
                self.assertEqual(e.accepts(n, m, dest), old.accepts(n, m, dest))

    def test_state_defensively_copies_integer_inputs(self):
        inputs = [[0, 1, 6], [0, 1, 2, 3], [1, 2, 3]]
        state = e.State(*inputs); inputs[0][0] = 23
        self.assertEqual(state.needs, (0, 1, 6))
        self.assertEqual(e.State(np.array([0, 1, 6]), [0, 1, 2, 3]).needs, (0, 1, 6))
        for bad in ([True, 1, 6], [0., 1, 6], [24, 1, 6]):
            with self.assertRaises(ValueError): e.State(bad, [0, 1, 2, 3])
        with self.assertRaises(ValueError): e.State([0, 1, 6], [0, 0, 2, 3])

    def test_complete_menus_unchanged_and_independent(self):
        for who in e.AGENTS:
            self.assertEqual(e.all_actions(who), old.all_actions(who))
            first = e.all_actions(who); first[0]['kind'] = 'tampered'
            self.assertEqual(e.all_actions(who)[0], {'kind': 'wait'})
            self.assertEqual(len(e.action_menu(who, list(reversed(range(17))))), 17)
        with self.assertRaises(ValueError): e.all_actions('D')

    def test_information_boundaries_all_owners(self):
        for owner in permutations((1, 2, 3)):
            s = e.State((0, 3, 14), (0, 1, 2, 3), owner)
            for agent in e.AGENTS:
                who = e.AGENTS.index(agent)
                for mode in ('PL', 'LL'):
                    v = e.observe(s, agent, information=mode)
                    self.assertNotIn('shared_needs', v)
                    self.assertNotIn('information_control', v)
                    self.assertEqual(len(v['visible_materials']), 4 if mode == 'PL' else 2)
                    n = list(s.needs); n[(who + 1) % 3] = (n[(who + 1) % 3] + 1) % 24
                    self.assertEqual(v, e.observe(e.State(n, s.layout, owner), agent, information=mode))
                v = e.observe(s, agent, information='FI')
                self.assertEqual(set(v['shared_needs']), set(e.AGENTS))
                hidden = [site for site in range(1, 4) if site != owner[who]]
                other = list(s.layout); other[hidden[0]], other[hidden[1]] = other[hidden[1]], other[hidden[0]]
                s2 = e.State(s.needs, other, owner)
                self.assertEqual(e.observe(s, agent), e.observe(s2, agent))
                self.assertNotEqual(e.observe(s, agent, information='PL'), e.observe(s2, agent, information='PL'))

    def test_full_support_witnesses_and_layout_equivariance(self):
        support = e.support()
        self.assertEqual(len(support), 5376)
        for i, n in enumerate(support):
            layout = tuple(permutations(range(4)))[i % 24]
            s = e.State(n, layout, tuple(permutations((1, 2, 3)))[i % 6])
            self.assertEqual(len(e.full_success_plans(n, layout)), 1)
            self.assertEqual(e.settle(s, e.sufficient_information_witness(s))['reward'], 1.)

    def test_all_4913_joint_actions_native_score(self):
        s = e.State((0, 3, 14), (2, 0, 3, 1))
        successful = 0
        for ids in product(range(17), repeat=3):
            actions = {a: e.all_actions(a)[ids[i]] for i, a in enumerate(e.AGENTS)}
            original = copy.deepcopy(actions)
            result = e.settle(s, actions)
            active = [i for i, v in enumerate(ids) if v]
            reward = 0.
            if len(active) == 2:
                i, j = active; ai, aj = actions[e.AGENTS[i]], actions[e.AGENTS[j]]
                matched = (ai['partner'] == e.AGENTS[j] and aj['partner'] == e.AGENTS[i]
                           and ai['site'] == aj['site'] and ai['destination'] == aj['destination'])
                if matched:
                    material = s.layout[e.SITES.index(ai['site'])]; dest = e.DESTINATIONS.index(ai['destination'])
                    reward = sum(e.accepts(s.needs[x], material, dest) for x in active) / 2
            self.assertEqual(result['reward'], reward)
            self.assertEqual(actions, original)
            self.assertEqual(result, e.settle(s, dict(reversed(list(actions.items())))))
            successful += reward == 1.
        self.assertEqual(successful, 1)


if __name__ == '__main__': unittest.main()
