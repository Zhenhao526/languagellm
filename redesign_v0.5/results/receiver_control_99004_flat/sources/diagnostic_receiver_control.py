"""External-code positive controls for the new receiver; not language formation.

Both controls start with exactly the same fresh 25-token receiver interface.
Only receive_embedding, actor and receive_value learn. No sender is run, and
no diagnostic weights are loaded into the social experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from camp import (MAPS, ImageBank, make_agents, individual_practice,
                  evaluate_prepared_four_sites, remake_agents, initial_world,
                  photo_ids, collect, draw, write_json)

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('oracle_map_code', 'oracle_target_location_code')
RECEIVER_PREFIXES = ('receive_embedding.', 'actor.', 'receive_value.')
MAP_CODES = np.full((4, 4), -1, np.int64)
for _i, (_f, _w) in enumerate(MAPS):
    MAP_CODES[_f, _w] = _i


def digest_state(agents):
    h = hashlib.sha256()
    for i, agent in enumerate(agents):
        for key, value in sorted(agent.state_dict().items()):
            h.update(f'{i}/{key}'.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fixture(bank, seed, n, split):
    """Same one-round physical sampling and reward rules as stage A."""
    rng = np.random.default_rng(seed)
    positions, inventory = initial_world(rng, n)
    ids = photo_ids(bank, rng, n, split)
    goals = rng.integers(2, size=n)
    menu = np.argsort(rng.random((n, 4)), axis=1)
    refill = rng.random(n)
    new_ids = photo_ids(bank, rng, n, split)  # Same unused refill-photo draw as A.
    return dict(positions=positions, inventory=inventory, photo_ids=ids,
                goals=goals, menu=menu, refill_uniform=refill, new_photo_ids=new_ids)


def oracle_messages(world, condition):
    p, g = world['positions'], world['goals']
    if condition == 'oracle_map_code':
        tokens = MAP_CODES[p[:, 0], p[:, 1]]
    elif condition == 'oracle_target_location_code':
        tokens = p[np.arange(len(p)), g]
    else:
        raise ValueError(condition)
    assert (tokens >= 0).all()
    return tokens[:, None].copy()


def attempt(agent, world, messages, policy_seed, greedy):
    n = len(messages)
    goal = torch.from_numpy(np.eye(2, dtype=np.float32)[world['goals']])
    inventory = torch.from_numpy(world['inventory'].astype(np.float32))
    # No photographs, map truth, correct action, or provider state enter receive.
    logits, value = agent.receive(torch.from_numpy(messages), goal, inventory,
        torch.zeros(n, 14), torch.from_numpy(world['menu']))
    action, logp, entropy = draw(logits, np.random.default_rng(policy_seed), greedy)
    place = world['menu'][np.arange(n), action.detach().numpy()]
    _, after, reward, gathered, overflow = collect(world['positions'], world['inventory'],
        world['goals'], place, world['refill_uniform'], replenish=False)
    return logp, entropy, value, reward, dict(action=action.detach().numpy(), place=place,
        reward=reward, gathered=gathered, overflow=overflow, next_inventory=after)


@torch.no_grad()
def evaluate(agents, bank, seed, n, condition, out=None):
    result = {}
    for mode in ('normal', 'shuffle', 'blank', 'stochastic'):
        accuracy, traces = [], []
        for who, agent in enumerate(agents):
            ws = seed + who * 100000
            world = fixture(bank, ws, n, 'test')
            messages = oracle_messages(world, condition)
            delivered = messages.copy()
            if mode == 'blank':
                delivered[:] = 0
            elif mode == 'shuffle':
                rng = np.random.default_rng(ws + 20000)
                for goal in (0, 1):
                    rows = np.flatnonzero(world['goals'] == goal)
                    delivered[rows] = messages[rng.permutation(rows)]
            _, _, _, reward, record = attempt(agent, world, delivered, ws + 30000,
                                               greedy=mode != 'stochastic')
            accuracy.append(float(reward.mean()))
            traces.append(dict(**world, **record, agent=np.full(n, who),
                               oracle_message=messages, delivered=delivered))
        result[mode] = dict(per_agent=accuracy, mean_reward=float(np.mean(accuracy)),
                            episodes_per_agent=n, split='old_test_16',
                            interpretation='external-code diagnostic, not emergent communication')
        if out is not None:
            np.savez_compressed(out / f'final_{mode}.npz',
                **{k: np.concatenate([t[k] for t in traces]) for k in traces[0]})
    return result


def run_condition(prepared, bank, args, condition, expected_initial=None):
    out = args.out / condition
    out.mkdir()
    agents = remake_agents(args.seed, prepared, vocab=25, length=1)
    for agent in agents:
        for name, p in agent.named_parameters():
            p.requires_grad_(name.startswith(RECEIVER_PREFIXES))
    initial = digest_state(agents)
    if expected_initial is not None:
        assert initial == expected_initial, 'diagnostic controls must have identical starts'
    untouched = [{k: v.clone() for k, v in a.state_dict().items()
                  if not k.startswith(RECEIVER_PREFIXES)} for a in agents]
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    optimizers = [torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=.0007)
                  for a in agents]
    checkpoints = sorted(set([0, min(100, args.updates), min(300, args.updates),
                              args.updates // 2, args.updates]))
    curve, start = [], time.monotonic()
    with (out / 'training.jsonl').open('w') as log:
        for update in range(args.updates + 1):
            if update in checkpoints:
                scores = evaluate(agents, bank, args.seed + 9100000, min(args.eval_n, 1024), condition)
                curve.append(dict(update=update, scores=scores))
                write_json(out / 'curve.json', curve)
                print(json.dumps(dict(condition=condition, update=update,
                    normal=scores['normal']['per_agent'])), flush=True)
            if update == args.updates:
                break
            rows = []
            for who, (agent, optimizer) in enumerate(zip(agents, optimizers)):
                ws = args.seed * 1000000 + update * 10 + who
                world = fixture(bank, ws, args.batch, 'train')
                messages = oracle_messages(world, condition)
                lp, ent, value, reward, _ = attempt(agent, world, messages, ws + 200000, False)
                target = torch.from_numpy(reward - 1)
                coefficient = .02 if update < int(args.updates * .8) else 0.
                loss = -(lp * (target - value).detach()).mean() + .5 * F.mse_loss(value, target)
                loss = loss - coefficient * ent.mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward()
                assert all(p.grad is None for name, p in agent.named_parameters()
                           if not name.startswith(RECEIVER_PREFIXES))
                norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 2.)
                optimizer.step()
                rows.append(dict(agent=who, reward=float(reward.mean()),
                                 loss=float(loss.detach()), gradient_norm=float(norm)))
            log.write(json.dumps(dict(update=update + 1, agents=rows)) + '\n')
            if (update + 1) % 100 == 0:
                log.flush()
    for agent, before in zip(agents, untouched):
        for key, value in before.items():
            assert torch.equal(agent.state_dict()[key], value), key
    scores = evaluate(agents, bank, args.seed + 9200000, args.eval_n, condition, out)
    torch.save([a.state_dict() for a in agents], out / 'final.pt')
    result = dict(condition=condition, seed=args.seed, updates_per_agent=args.updates,
        batch_per_agent=args.batch, train_choices_per_agent=args.updates * args.batch,
        initial_sha256=initial, final_sha256=digest_state(agents),
        frozen_nonreceiver_verified=True, scores=scores, seconds=time.monotonic()-start)
    write_json(out / 'result.json', result)
    return initial, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/receiver_control_99004')
    p.add_argument('--seed', type=int, default=99004)
    p.add_argument('--updates', type=int, default=1200)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--eval-n', type=int, default=8192)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    bank = ImageBank()
    sources = [Path(__file__), ROOT / 'camp.py', ROOT.parent / 'redesign_v0.4/agents.py',
               ROOT.parent / 'redesign_v0.4/run_pilot.py']
    (args.out / 'sources').mkdir()
    for source in sources:
        shutil.copy2(source, args.out / 'sources' / source.name)
    config = dict(seed=args.seed, conditions=CONDITIONS, updates_per_agent=args.updates,
        batch_per_agent=args.batch, eval_n_per_agent=args.eval_n, vocab=25, length=1,
        learning_rate=.0007, entropy=.02, entropy_off_after=int(args.updates * .8),
        purpose='positive engineering control of fresh receiver with externally assigned complete code',
        prohibited_inference='not language emergence, not a pretrained receiver capability, not main-study evidence',
        no_sender=True, trainable_prefixes=RECEIVER_PREFIXES, world_rule='stage A, uniform 12 maps, one collection',
        world_knowledge_route='oracle integer code deliberately reveals map or goal location only in this diagnostic',
        data_scope='existing 44 train / 16 development test photos; receiver does not see photos',
        map_code_table=MAPS.tolist(), source_hashes={str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
    write_json(args.out / 'config.json', config)
    base = make_agents(args.seed)
    practice = individual_practice(base, bank, args.seed)
    four = evaluate_prepared_four_sites(base, bank, args.seed + 7000000)
    write_json(args.out / 'preparation.json', dict(two_sites=practice, old_actor_four_sites=four,
        scope='visual resource preparation and old action head only; fresh receiver not yet trained'))
    if min(four) < .9 or min(x['heldout_need_sensitive_choice'] for x in practice) < .9:
        raise RuntimeError('preparation gate failed; no replacement seed')
    prepared = [a.state_dict() for a in base]
    torch.save(prepared, args.out / f'prepared_{args.seed}.pt')
    first, results = None, []
    for condition in CONDITIONS:
        first, result = run_condition(prepared, bank, args, condition, first)
        results.append(result)
    write_json(args.out / 'summary.json', dict(status='completed', diagnostic_only=True,
        matched_initialization=True, independent_seed_count=1, receivers=2, results=results))
    print('COMPLETED receiver positive controls', flush=True)


if __name__ == '__main__':
    main()
