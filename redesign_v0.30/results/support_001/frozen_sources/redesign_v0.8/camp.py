"""Small visual scouting/collection game. Labels stay inside the simulator."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OLD = Path(__file__).resolve().parents[1] / 'redesign_v0.4'
sys.path.insert(0, str(OLD))
from agents import draw
from run_pilot import ImageBank, make_agents, individual_practice, write_json

SITES = 6
HISTORY = 2 * (SITES + 3)
MAPS = np.asarray(list(itertools.permutations(range(SITES), 2)), dtype=np.int64)
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)),
             2: ((0, 2), (1, 4), (3, 5)),
             3: ((0, 3), (1, 5), (2, 4))}


def split_maps(split):
    if split == 0:
        return np.arange(len(MAPS)), np.asarray([], dtype=np.int64)
    pairs = {p for a, b in MATCHINGS[split] for p in ((a, b), (b, a))}
    held = np.asarray([i for i, p in enumerate(MAPS) if tuple(p) in pairs], dtype=np.int64)
    return np.setdiff1d(np.arange(len(MAPS)), held), held


def initial_world(rng, n, map_pool=None):
    pool = np.arange(len(MAPS)) if map_pool is None else np.asarray(map_pool, dtype=np.int64)
    assert len(pool), 'empty training support'
    return MAPS[rng.choice(pool, n)].copy(), np.zeros((n, 2), np.int64)


def collect(positions, inventory, goals, places, refill_uniform, *, replenish):
    """Gather, cap at two, consume private demand, then replenish taken item.

    Positions are (food, water), one unit each. Replenishment chooses uniformly
    among the five sites other than the untouched resource, including old site.
    Returned feedback exposes only the collector's actual gathered resources.
    """
    n = len(positions)
    assert positions.shape == (n, 2) and inventory.shape == (n, 2)
    assert np.all(positions[:, 0] != positions[:, 1])
    assert np.all((places >= 0) & (places < SITES))
    gathered = (positions == places[:, None]).astype(np.int64)
    available = inventory + gathered
    overflow = np.maximum(available - 2, 0)
    stored = np.minimum(available, 2)
    reward = (stored[np.arange(n), goals] > 0).astype(np.float32)
    consumed = np.eye(2, dtype=np.int64)[goals] * reward[:, None].astype(np.int64)
    after = stored - consumed
    nxt = positions.copy()
    if replenish:
        for kind in range(2):
            rows = np.flatnonzero(gathered[:, kind])
            other = positions[rows, 1 - kind]
            # Increasing list of all legal sites; uniform draws are exogenous.
            choices = np.asarray([[p for p in range(SITES) if p != int(o)] for o in other], dtype=np.int64).reshape(-1, SITES - 1)
            pick = np.minimum((refill_uniform[rows] * (SITES - 1)).astype(np.int64), SITES - 2)
            nxt[rows, kind] = choices[np.arange(len(rows)), pick]
    assert np.array_equal(inventory + gathered, after + consumed + overflow)
    return nxt, after, reward, gathered, overflow


def binding_matrix(seed, who, representation):
    """Private fixed raw-input transform; local RNG never changes model/world RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(who), 4707]))
    if representation == 'identity':
        matrix = np.eye(SITES)
    elif representation == 'permute':
        order = rng.permutation(SITES)
        permutation = np.empty(SITES, dtype=np.int64)
        permutation[order] = np.roll(order, 1)
        matrix = np.eye(SITES)[permutation]
    elif representation == 'orthogonal':
        q, r = np.linalg.qr(rng.normal(size=(SITES, SITES)))
        matrix = q * np.where(np.diag(r) >= 0, 1., -1.)
    else:
        raise ValueError(representation)
    return torch.from_numpy(matrix.astype(np.float32))


class CampAgent(nn.Module):
    """One personal interface; no other agent's differentiable state is accepted."""
    def __init__(self, prepared_project, vocab=7, length=2, width=96, transform=None):
        super().__init__()
        self.vocab, self.length, self.width = vocab, length, width
        self.project = copy.deepcopy(prepared_project)
        self.project.requires_grad_(False)
        # Six location-bound visual vectors + six existence bits + view bit.
        self.memory = nn.GRUCell(SITES * 64 + SITES + 1, width)
        self.send_context = nn.Sequential(nn.Linear(width + 4, width), nn.Tanh())
        self.send_embedding = nn.Embedding(vocab, 16)
        self.send_recur = nn.GRUCell(16, width)
        self.send_out = nn.Linear(width, vocab)
        self.receive_embedding = nn.Embedding(vocab, 16)
        # Only received integers, own need/inventory, and actual collection history.
        context = length * 16 + 2 + 2 + HISTORY
        # A nonlinguistic goal-conditioned action interface. Each private need
        # selects a location policy, but neither message position has a set meaning.
        self.actor = nn.Sequential(nn.Linear(context - 2, width), nn.Tanh(), nn.Linear(width, 2 * SITES))
        self.send_value = nn.Sequential(nn.Linear(width + 4, width), nn.Tanh(), nn.Linear(width, 1))
        self.receive_value = nn.Sequential(nn.Linear(context, width), nn.Tanh(), nn.Linear(width, 1))
        # Shared local nonlinearity; no output slot is assigned a resource meaning.
        # Instantiate after the existing modules to retain their initialization stream.
        self.slot_phi = nn.Sequential(nn.Linear(65, 65), nn.Tanh())
        self.register_buffer('input_transform', torch.eye(SITES) if transform is None else transform.clone())

    def encode_slots(self, visual):
        assert visual.shape[1] == SITES * 65
        # Source layout is [6*64 visual values, 6 existence bits], not 6*65 interleaved.
        slots = torch.cat((visual[:, :SITES * 64].reshape(-1, SITES, 64),
                           visual[:, SITES * 64:].unsqueeze(-1)), -1)
        mixed = torch.einsum('ij,bjd->bid', self.input_transform, slots)
        return self.slot_phi(mixed).flatten(1)

    def observe(self, visual, delay=0, memory_mode='retain'):
        n = len(visual)
        zero = visual.new_zeros(n, self.width)
        encoded = self.encode_slots(visual)
        seen = torch.cat((encoded, visual.new_ones(n, 1)), -1)
        h = self.memory(seen, zero)
        for _ in range(delay):
            h = self.memory(torch.zeros_like(seen), h)
        if memory_mode == 'reset':
            # Complete removal of the private scene; explicitly an information-loss control.
            h = torch.zeros_like(h)
        elif memory_mode == 'replay':
            # External re-presentation of the same legally observed scene.
            h = self.memory(seen, zero)
        elif memory_mode != 'retain':
            raise ValueError(memory_mode)
        return h

    def send(self, h, visible_goal, inventory, rng, greedy):
        local = torch.cat((h, visible_goal, inventory / 2), -1)
        state = self.send_context(local)
        tokens, logps, ents = [], [], []
        for t in range(self.length):
            token, logp, ent = draw(self.send_out(state), rng, greedy)
            tokens.append(token.detach()); logps.append(logp); ents.append(ent)
            if t + 1 < self.length:
                state = self.send_recur(self.send_embedding(token.detach()), state)
        return torch.stack(tokens, 1), sum(logps), sum(ents), self.send_value(local).squeeze(-1)

    def receive(self, messages, own_goal, inventory, history, menu):
        assert messages.dtype == torch.int64 and not messages.requires_grad
        context = torch.cat((self.receive_embedding(messages).flatten(1), own_goal,
                             inventory / 2, history), -1)
        action_context = torch.cat((self.receive_embedding(messages).flatten(1), inventory / 2, history), -1)
        location_logits = self.actor(action_context).reshape(-1, 2, SITES)
        logits = location_logits[torch.arange(len(messages)), own_goal.argmax(-1)].gather(1, menu)
        return logits, self.receive_value(context).squeeze(-1)


@torch.no_grad()
def projected_banks(agents, bank):
    return [a.project(bank.features).detach() for a in agents]


def scene_visual(positions, ids, projected):
    """Only call for the current scout; collector never receives this tensor."""
    n = len(positions)
    options = projected.new_zeros(n, SITES, 64)
    exists = projected.new_zeros(n, SITES)
    rows = torch.arange(n)
    for kind in range(2):
        sites = torch.from_numpy(positions[:, kind])
        options[rows, sites] = projected[torch.from_numpy(ids[:, kind])]
        exists[rows, sites] = 1
    return torch.cat((options.flatten(1), exists), -1)


def photo_ids(bank, rng, n, split):
    return np.column_stack([rng.choice(bank.pools[split, k], n) for k in range(2)])


def remake_agents(seed, prepared, vocab=7, length=2, representation='identity'):
    agents = []
    for i in range(2):
        torch.manual_seed(seed * 1000 + 701 + i)
        base = make_agents(seed)[i]
        base.load_state_dict(prepared[i])
        # make_agents changes Torch RNG, so restore independent interface seed.
        torch.manual_seed(seed * 1000 + 701 + i)
        agents.append(CampAgent(base.project, vocab, length, transform=binding_matrix(seed, i, representation)))
    pointers = [p.data_ptr() for a in agents for p in a.parameters()]
    assert len(pointers) == len(set(pointers))
    return agents


@torch.no_grad()
def evaluate_prepared_six_sites(prepared_agents, bank, seed, n=2048):
    rng = np.random.default_rng(seed)
    positions, _ = initial_world(rng, n)
    ids = photo_ids(bank, rng, n, 'test')
    goals = rng.integers(2, size=n)
    results = []
    for agent in prepared_agents:
        options = torch.zeros(n, SITES, 64)
        exists = torch.zeros(n, SITES, dtype=torch.bool)
        for kind in range(2):
            p = torch.from_numpy(positions[:, kind])
            options[torch.arange(n), p] = agent.project(bank.features[torch.from_numpy(ids[:, kind])])
            exists[torch.arange(n), p] = True
        inv = 1 - np.eye(2, dtype=np.float32)[goals]
        public = torch.from_numpy(np.column_stack((inv, np.ones(n, np.float32))))
        local = torch.cat((options.sum(1) / 2, public), -1)
        signal = F.one_hot(torch.zeros(n, dtype=torch.int64), 5).float()
        context = torch.cat((local, signal), -1)[:, None].expand(-1, SITES, -1)
        scores = agent.actor(torch.cat((options, context), -1)).squeeze(-1).masked_fill(~exists, -1e9)
        place = scores.argmax(-1).numpy()
        results.append(float((place == positions[np.arange(n), goals]).mean()))
    return results
