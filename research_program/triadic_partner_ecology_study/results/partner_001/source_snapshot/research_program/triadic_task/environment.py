"""One-round triadic collection task with explicit observation boundaries.

No policies, trained models, dictionary, or message encoding are supplied here.
The current development support is the previously audited 996 demand tables.
"""
from dataclasses import dataclass
from itertools import combinations, permutations, product
from numbers import Integral
from random import Random

AGENTS = ('A', 'B', 'C')
SITES = ('S0', 'S1', 'S2', 'S3')
DESTINATIONS = ('L', 'R')
MATERIALS = (('wood', 'short'), ('wood', 'long'), ('fiber', 'short'), ('fiber', 'long'))
# Researcher-side ids, never exposed as a global scene/target identifier.
RESOURCE_ACCEPTANCE = ((0, 1), (2, 3), (0, 2), (1, 3))
DESTINATION_ACCEPTANCE = ((0,), (1,), (0, 1))
NEEDS = tuple(product(range(4), range(3)))


def _need_view(need_id):
    resource, destination = NEEDS[need_id]
    return {'factor': 'kind' if resource < 2 else 'length',
            'value': ('wood', 'fiber', 'short', 'long')[resource],
            'destinations': [DESTINATIONS[d] for d in DESTINATION_ACCEPTANCE[destination]]}


def accepts(need_id, material, destination):
    resource, destinations = NEEDS[need_id]
    return material in RESOURCE_ACCEPTANCE[resource] and destination in DESTINATION_ACCEPTANCE[destinations]


def compatible_pairs(needs):
    return tuple((i, j) for i, j in combinations(range(3), 2)
                 if any(accepts(needs[i], material, destination) and accepts(needs[j], material, destination)
                        for material, destination in product(range(4), range(2))))


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
        if len(self.needs) != 3 or any(n not in range(12) for n in self.needs):
            raise ValueError('Three valid need ids required')
        if tuple(sorted(self.layout)) != (0, 1, 2, 3):
            raise ValueError('Each material must occur once')
        if tuple(sorted(self.private_sites)) != (1, 2, 3):
            raise ValueError('Private sites must partition the nonpublic sites')


def observe(state, agent, *, shared_needs, full_information=False):
    """Full information is a separately named capability control, never implicit."""
    who = AGENTS.index(agent)
    visible = list(range(4)) if full_information else [0, state.private_sites[who]]
    view = {
        'self': agent, 'public_site': 'S0',
        'private_view_owners': {SITES[site]: AGENTS[i] for i, site in enumerate(state.private_sites)},
        'own_need': _need_view(state.needs[who]),
        'visible_materials': [{'site': SITES[site], 'kind': MATERIALS[state.layout[site]][0],
                               'length': MATERIALS[state.layout[site]][1]} for site in visible],
    }
    if shared_needs or full_information:
        view['shared_needs'] = {AGENTS[i]: _need_view(n) for i, n in enumerate(state.needs)}
    if full_information:
        view['information_control'] = 'full_information'
    return view


def all_actions(agent):
    if agent not in AGENTS:
        raise ValueError('Unknown agent')
    return [{'kind': 'wait'}] + [
        {'kind': 'transport', 'site': site, 'destination': destination, 'partner': partner}
        for site, destination, partner in product(SITES, DESTINATIONS, [a for a in AGENTS if a != agent])]


def action_menu(agent, order):
    """The caller fixes a private random permutation independently of answers."""
    if sorted(order) != list(range(17)):
        raise ValueError('Menu order must be a permutation of all 17 actions')
    actions = all_actions(agent)
    return [{'id': i, 'action': actions[index]} for i, index in enumerate(order)]


def settle(state, actions, *, require_match):
    """Atomically score the three submitted actions, with no within-step update."""
    if set(actions) != set(AGENTS):
        raise ValueError('Exactly one submitted action per agent required')
    for agent, action in actions.items():
        if action not in all_actions(agent):
            raise ValueError(f'Action is not in the complete legal menu for {agent}')
    active = [a for a in AGENTS if actions[a]['kind'] == 'transport']
    overload = len(active) > 2
    execution = {}
    success = {}
    for agent in AGENTS:
        action = actions[agent]
        if overload or action['kind'] == 'wait':
            execution[agent] = success[agent] = False
            continue
        partner = action['partner']
        other = actions[partner]
        matched = (other.get('kind') == 'transport' and other.get('partner') == agent
                   and other.get('site') == action['site'] and other.get('destination') == action['destination'])
        execution[agent] = bool(matched or not require_match)
        success[agent] = execution[agent] and accepts(state.needs[AGENTS.index(agent)],
            state.layout[SITES.index(action['site'])], DESTINATIONS.index(action['destination']))
    units = sum(success.values())
    assert units <= 2
    # Partner's need, unseen item, full state and counterfactual correctness are
    # not revealed through individual feedback. Team reward is a declared channel.
    feedback = {a: {'executed': execution[a], 'own_need_satisfied': success[a],
                    'team_reward': units / 2} for a in AGENTS}
    return {'reward': units / 2, 'satisfied_units': units, 'full_success': units == 2,
            'individual_feedback': feedback,
            'researcher': {'overload': overload, 'active_agents': active,
                           'matching_required': bool(require_match)}}


def support():
    return tuple(t for t in product(range(12), repeat=3) if len(compatible_pairs(t)) >= 2)


def draw_state(rng: Random, demand_support):
    """IID semantic worlds; the random seed/state itself is researcher-only."""
    layout, private_sites = list(range(4)), [1, 2, 3]
    rng.shuffle(layout)
    rng.shuffle(private_sites)
    return State(tuple(rng.choice(demand_support)), tuple(layout), tuple(private_sites))


def sufficient_information_witness(state):
    """Researcher verifier only. Never passed to a policy or its memory."""
    for i, j in compatible_pairs(state.needs):
        for site, destination in product(range(4), range(2)):
            if all(accepts(state.needs[a], state.layout[site], destination) for a in (i, j)):
                actions = {a: {'kind': 'wait'} for a in AGENTS}
                actions[AGENTS[i]] = {'kind': 'transport', 'site': SITES[site],
                    'destination': DESTINATIONS[destination], 'partner': AGENTS[j]}
                actions[AGENTS[j]] = {'kind': 'transport', 'site': SITES[site],
                    'destination': DESTINATIONS[destination], 'partner': AGENTS[i]}
                return actions
    raise ValueError('State has no full-success matching pair')
