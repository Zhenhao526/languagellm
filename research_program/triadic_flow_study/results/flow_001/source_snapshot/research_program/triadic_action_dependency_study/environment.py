"""Triadic D1 task with partial or exact two-attribute resource requirements.

No policy, vocabulary, correct-plan label, or hidden demand enters observation.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from numbers import Integral

AGENTS = ('A', 'B', 'C')
SITES = ('S0', 'S1', 'S2', 'S3')
DESTINATIONS = ('L', 'R')
MATERIALS = (('wood', 'short'), ('wood', 'long'), ('fiber', 'short'), ('fiber', 'long'))
RESOURCE_ACCEPTANCE = ((0, 1), (2, 3), (0, 2), (1, 3), (0,), (1,), (2,), (3,))
DESTINATION_ACCEPTANCE = ((0,), (1,), (0, 1))
NEEDS = tuple(product(range(8), range(3)))
INFORMATION = ('FI', 'PL', 'LL')


def need_view(need_id):
    resource, destination = NEEDS[need_id]
    materials = [MATERIALS[m] for m in RESOURCE_ACCEPTANCE[resource]]
    return {'kinds': [k for k in ('wood', 'fiber') if any(m[0] == k for m in materials)],
            'lengths': [l for l in ('short', 'long') if any(m[1] == l for m in materials)],
            'destinations': [DESTINATIONS[d] for d in DESTINATION_ACCEPTANCE[destination]]}


def accepts(need_id, material, destination):
    resource, destinations = NEEDS[need_id]
    return material in RESOURCE_ACCEPTANCE[resource] and destination in DESTINATION_ACCEPTANCE[destinations]


def compatible_pairs(needs):
    return tuple((i, j) for i, j in combinations(range(3), 2)
                 if any(accepts(needs[i], m, d) and accepts(needs[j], m, d)
                        for m, d in product(range(4), range(2))))


def full_success_plans(needs, layout=(0, 1, 2, 3)):
    """Researcher only: all full-success (agent_i,agent_j,site,destination)."""
    return tuple((i, j, site, d) for i, j in combinations(range(3), 2)
                 for site, d in product(range(4), range(2))
                 if accepts(needs[i], layout[site], d) and accepts(needs[j], layout[site], d))


@lru_cache(maxsize=1)
def support():
    return tuple(n for n in product(range(24), repeat=3) if len(full_success_plans(n)) == 1)


@dataclass(frozen=True)
class State:
    needs: tuple[int, int, int]
    layout: tuple[int, int, int, int]
    private_sites: tuple[int, int, int] = (1, 2, 3)

    def __post_init__(self):
        for name in ('needs', 'layout', 'private_sites'):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, Integral) or isinstance(v, bool) for v in values):
                raise ValueError('State fields require integer ids')
            object.__setattr__(self, name, tuple(int(v) for v in values))
        if len(self.needs) != 3 or any(n not in range(24) for n in self.needs):
            raise ValueError('Three valid need ids required')
        if tuple(sorted(self.layout)) != (0, 1, 2, 3):
            raise ValueError('Each material must occur once')
        if tuple(sorted(self.private_sites)) != (1, 2, 3):
            raise ValueError('Private sites must partition the nonpublic sites')


def observe(state, agent, *, information='LL'):
    if information not in INFORMATION:
        raise ValueError('Unknown information condition')
    who = AGENTS.index(agent)
    visible = range(4) if information in ('FI', 'PL') else (0, state.private_sites[who])
    view = {'self': agent, 'public_site': 'S0',
            'private_view_owners': {SITES[site]: AGENTS[i] for i, site in enumerate(state.private_sites)},
            'own_need': need_view(state.needs[who]),
            'visible_materials': [{'site': SITES[site], 'kind': MATERIALS[state.layout[site]][0],
                                   'length': MATERIALS[state.layout[site]][1]} for site in visible]}
    if information == 'FI':
        view['shared_needs'] = {AGENTS[i]: need_view(n) for i, n in enumerate(state.needs)}
        view['information_control'] = 'full_information'
    return view


def all_actions(agent):
    if agent not in AGENTS:
        raise ValueError('Unknown agent')
    return [{'kind': 'wait'}] + [
        {'kind': 'transport', 'site': site, 'destination': destination, 'partner': partner}
        for site, destination, partner in product(SITES, DESTINATIONS, [a for a in AGENTS if a != agent])]


def action_menu(agent, order):
    if sorted(order) != list(range(17)):
        raise ValueError('Menu must permute all 17 actions')
    menu = all_actions(agent)
    return [{'id': i, 'action': menu[index]} for i, index in enumerate(order)]


def settle(state, actions, *, require_match=True):
    if set(actions) != set(AGENTS):
        raise ValueError('Exactly one action per agent required')
    for agent, action in actions.items():
        if action not in all_actions(agent):
            raise ValueError('Action outside complete menu for ' + agent)
    active = [a for a in AGENTS if actions[a]['kind'] == 'transport']
    overload = len(active) > 2
    execution, success = {}, {}
    for agent in AGENTS:
        action = actions[agent]
        if overload or action['kind'] == 'wait':
            execution[agent] = success[agent] = False
            continue
        partner = action['partner']; other = actions[partner]
        matched = (other.get('kind') == 'transport' and other.get('partner') == agent
                   and other.get('site') == action['site'] and other.get('destination') == action['destination'])
        execution[agent] = bool(matched or not require_match)
        success[agent] = execution[agent] and accepts(state.needs[AGENTS.index(agent)],
            state.layout[SITES.index(action['site'])], DESTINATIONS.index(action['destination']))
    units = sum(success.values())
    assert units <= 2
    feedback = {a: {'executed': execution[a], 'own_need_satisfied': success[a],
                    'team_reward': units / 2} for a in AGENTS}
    return {'reward': units / 2, 'satisfied_units': units, 'full_success': units == 2,
            'individual_feedback': feedback,
            'researcher': {'overload': overload, 'active_agents': active, 'matching_required': bool(require_match)}}


def sufficient_information_witness(state):
    full = full_success_plans(state.needs, state.layout)
    if not full:
        raise ValueError('No full-success plan')
    i, j, site, destination = full[0]
    result = {a: {'kind': 'wait'} for a in AGENTS}
    for who, partner in ((i, j), (j, i)):
        result[AGENTS[who]] = {'kind': 'transport', 'site': SITES[site],
            'destination': DESTINATIONS[destination], 'partner': AGENTS[partner]}
    return result
