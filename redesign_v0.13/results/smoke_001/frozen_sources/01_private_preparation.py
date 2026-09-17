"""Paired nonlinguistic spatial-action preparation, independent of social training.

Only the memory/slot interface transfers; the personal policy head is discarded.
Evaluation never selects sources or changes the fixed training budget.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
V8 = ROOT.parent / 'redesign_v0.8'
sys.path.insert(0, str(V8))
import run_experiment as v8
from camp import MAPS, MATCHINGS, ImageBank, remake_agents, projected_banks, scene_visual, photo_ids, write_json

SEEDS = (31101, 31102, 31103, 31104)
ARMS = ('control', 'equivariant')
CHECKPOINTS = (0, 100, 300, 600, 1200, 1800, 2400)
PREFIXES = ('memory.', 'slot_phi.')
PERMUTATIONS = np.asarray(list(itertools.permutations(range(6))), dtype=np.int64)
WORLD_KEYS = ('triple_id', 'source_map', 'target_map', 'permutation', 'positions', 'photo_ids', 'goals')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_sha(arrays, keys=None):
    h = hashlib.sha256()
    for key in sorted(arrays) if keys is None else keys:
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def module_sha(module, include=None):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if include is not None and not include(key): continue
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def seed_for(seed, partition, who, purpose, update=0):
    return int(np.random.SeedSequence([13013, seed, partition, who, purpose, update]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def partition_maps(partition):
    assert partition in (1, 2, 3)
    def matching(k):
        pairs = {p for f, w in MATCHINGS[k] for p in ((f, w), (w, f))}
        return np.asarray([i for i, pair in enumerate(MAPS) if tuple(pair) in pairs], dtype=np.int64)
    added, sealed = matching(partition), matching(partition % 3 + 1)
    old = np.setdiff1d(np.arange(30), np.union1d(added, sealed))
    return dict(old=old, added=added, sealed=sealed, all=np.arange(30))


def map_index(positions):
    positions = np.asarray(positions)
    return positions[..., 0] * 5 + positions[..., 1] - (positions[..., 1] > positions[..., 0])


@lru_cache(None)
def legal_triples(partition):
    """Columns source_map, permutation_index, target_map. Includes stabilizers."""
    old = partition_maps(partition)['old']
    rows = []
    for source in old:
        targets = map_index(PERMUTATIONS[:, MAPS[source]])
        for permutation in np.flatnonzero(np.isin(targets, old)):
            rows.append((int(source), int(permutation), int(targets[permutation])))
    triples = np.asarray(rows, dtype=np.int64)
    assert triples.shape == (7776, 3)
    for column in (0, 2):
        assert np.array_equal(np.bincount(triples[:, column], minlength=30)[old], np.full(18, 432))
    assert np.all(np.bincount(triples[:, 0] * 30 + triples[:, 2], minlength=900).reshape(30, 30)[np.ix_(old, old)] == 24)
    triples.setflags(write=False)
    return triples


def fixture(bank, seed, partition, who, update, batch):
    rng = np.random.default_rng(seed_for(seed, partition, who, 1, update))
    choices = rng.integers(len(legal_triples(partition)), size=batch)
    rows = legal_triples(partition)[choices]
    permutations = PERMUTATIONS[rows[:, 1]]
    positions = np.stack((MAPS[rows[:, 0]], MAPS[rows[:, 2]]), axis=1)
    ids = photo_ids(bank, rng, batch, 'train')
    goals = rng.integers(2, size=batch)
    assert np.array_equal(permutations[np.arange(batch)[:, None], positions[:, 0]], positions[:, 1])
    return dict(triple_id=choices, source_map=rows[:, 0], target_map=rows[:, 2],
                permutation=permutations, positions=positions, photo_ids=ids, goals=goals)


def jsd_terms(source_log_probs, target_log_probs, permutation):
    """g maps old sites to new sites; output at a new site uses g inverse."""
    inverse = torch.argsort(permutation, dim=-1)
    moved = source_log_probs.gather(-1, inverse[:, None, :].expand(-1, 2, -1))
    middle = torch.logaddexp(moved, target_log_probs) - np.log(2.)
    return .5 * ((moved.exp() * (moved - middle)).sum(-1)
                 + (target_log_probs.exp() * (target_log_probs - middle)).sum(-1))


def policy_logits(agent, head, projected, positions, ids):
    visual = scene_visual(positions, ids, projected)
    return head(agent.observe(visual)).reshape(len(positions), 2, 6)


def loss_terms(agent, head, projected, world, action_seed, weight, entropy_coefficient):
    """Actual sampled goal controls reward; both potential goals enter JSD only."""
    batch = len(world['goals'])
    positions = np.concatenate((world['positions'][:, 0], world['positions'][:, 1]))
    ids = np.tile(world['photo_ids'], (2, 1)); goals = np.tile(world['goals'], 2)
    logits = policy_logits(agent, head, projected, positions, ids)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs[torch.arange(2 * batch), torch.from_numpy(goals)]
    uniform = np.random.default_rng(action_seed).random((2 * batch, 1)).astype(np.float32)
    action = (selected.detach().exp().cumsum(-1) < torch.from_numpy(uniform)).sum(-1).clamp(max=5)
    places = action.detach().numpy()
    reward = (places == positions[np.arange(2 * batch), goals]).astype(np.float32)
    logp = selected.gather(1, action[:, None]).squeeze(1)
    entropy = -(selected.exp() * selected).sum(-1).mean()
    policy = -(logp * torch.from_numpy(reward - .5)).mean()
    jsd = jsd_terms(log_probs[:batch], log_probs[batch:], torch.from_numpy(world['permutation'])).mean()
    loss = policy - entropy_coefficient * entropy + weight * jsd
    assert torch.isfinite(loss)
    trace = dict(**world, logits=logits.detach().numpy(), probabilities=log_probs.detach().exp().numpy(),
                 action_uniform=uniform, action=places, reward=reward)
    components = dict(loss=float(loss.detach()), policy_loss=float(policy.detach()),
                      entropy=float(entropy.detach()), jsd=float(jsd.detach()), weight=weight,
                      entropy_coefficient=entropy_coefficient, mean_reward=float(reward.mean()))
    return loss, components, trace


def evaluation_worlds(bank):
    ids = [np.asarray(bank.pools['test', kind][:4], dtype=np.int64) for kind in (0, 1)]
    assert all(len(x) == 4 for x in ids)
    pairs = np.asarray(list(itertools.product(*ids)), dtype=np.int64)
    mids = np.repeat(np.arange(30), 16)
    return dict(map_ids=mids, positions=MAPS[mids].copy(), photo_ids=np.tile(pairs, (30, 1)))


def log_probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    centered = logits - logits.max(-1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(-1, keepdims=True))


def equivariance_statistics(logits, partition):
    """Exact all30 x all720 permutations x fixed16 photos, two goal policies."""
    lp = log_probabilities(logits).reshape(30, 16, 2, 6)
    inverse = np.argsort(PERMUTATIONS, axis=-1)
    rows = []
    for source in range(30):
        targets = map_index(PERMUTATIONS[:, MAPS[source]])
        moved = np.take_along_axis(np.broadcast_to(lp[source], (720, 16, 2, 6)), inverse[:, None, None, :], axis=-1)
        target = lp[targets]
        middle = np.logaddexp(moved, target) - np.log(2.)
        jsd = .5 * ((np.exp(moved) * (moved - middle)).sum(-1) + (np.exp(target) * (target - middle)).sum(-1))
        moved_actions = PERMUTATIONS[:, lp[source].argmax(-1)]
        correct = target.argmax(-1) == moved_actions
        rows.append(dict(source_map=source, jsd=float(jsd.mean()), food_jsd=float(jsd[:, :, 0].mean()),
                         water_jsd=float(jsd[:, :, 1].mean()), greedy_single_consistency=float(correct.mean()),
                         greedy_both_consistency=float(correct.all(-1).mean())))
    groups = {g: {key: float(np.mean([rows[i][key] for i in pool])) for key in rows[0] if key != 'source_map'}
              for g, pool in partition_maps(partition).items()}
    return dict(groups=groups, per_source_map=rows, source_maps=30, permutations=720,
                fixed_photo_pairs=16, paired_worlds=30 * 720 * 16, goal_comparisons=30 * 720 * 16 * 2,
                scope='Group labels refer to source map; every target map under all720 permutations is allowed at evaluation.')


def capability_statistics(logits, world, partition):
    lp = log_probabilities(logits); prob = np.exp(lp); actions = np.asarray(logits).argmax(-1)
    success = actions == world['positions']
    correct_prob = np.take_along_axis(prob, world['positions'][:, :, None], axis=-1).squeeze(-1)
    entropy = -(prob * lp).sum(-1)
    result = {}
    for group, pool in partition_maps(partition).items():
        mask = np.isin(world['map_ids'], pool); good = success[mask]
        result[group] = dict(worlds=int(mask.sum()), correct_both=int(good.all(-1).sum()),
            correct_goals=int(good.sum()), J=float(good.all(-1).mean()), single=float(good.mean()),
            food=float(good[:, 0].mean()), water=float(good[:, 1].mean()),
            stochastic_J=float(correct_prob[mask].prod(-1).mean()), stochastic_single=float(correct_prob[mask].mean()),
            policy_entropy=float(entropy[mask].mean()), food_entropy=float(entropy[mask, 0].mean()), water_entropy=float(entropy[mask, 1].mean()))
    return result


@torch.no_grad()
def evaluate(agents, heads, banks, bank, partition, out=None, update=None):
    world = evaluation_worlds(bank); logits = []
    persons = []
    for who, (agent, head, projected) in enumerate(zip(agents, heads, banks)):
        raw = policy_logits(agent, head, projected, world['positions'], world['photo_ids']).numpy()
        capability = capability_statistics(raw, world, partition)
        persons.append(dict(person=who, groups=capability, equivariance=equivariance_statistics(raw, partition),
                            old_single_at_least_090=capability['old']['single'] >= .90))
        logits.append(raw)
    if out is not None:
        np.savez_compressed(Path(out)/f'evaluation_{update:04d}.npz', **world, logits=np.stack(logits))
    return dict(persons=persons, pair_both_old_gate=all(p['old_single_at_least_090'] for p in persons),
                evaluation_worlds_per_person=480, capability_gate_selects_no_sources=True,
                head_policy_equivariance_is_not_latent_equivariance=True)


def make_heads(seed):
    heads = []
    for who in range(2):
        torch.manual_seed(seed * 1000 + 881 + who)
        heads.append(nn.Sequential(nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 12)))
    return heads


def snapshot(agents, heads):
    return dict(agents=[copy.deepcopy(a.state_dict()) for a in agents], heads=[copy.deepcopy(h.state_dict()) for h in heads])


def run(seed, partition, arm, prepared, bank, out, updates=2400, batch=512, source_hashes=None):
    assert arm in ARMS and updates > 0 and batch > 0
    out = Path(out).resolve(); dest = out/f'private_s{seed}_p{partition}_{arm}'
    assert not dest.exists(), f'Refuse to overwrite an existing private run: {dest}'
    dest.mkdir(parents=True)
    weight = 0. if arm == 'control' else .1
    agents = remake_agents(seed, prepared, 7, 2, 'identity'); heads = make_heads(seed)
    frozen = [module_sha(a, lambda k: not k.startswith(PREFIXES)) for a in agents]
    initial = snapshot(agents, heads)
    for a in agents:
        for name, parameter in a.named_parameters(): parameter.requires_grad_(name.startswith(PREFIXES))
    parameters = [list(head.parameters()) + [p for p in a.parameters() if p.requires_grad] for a, head in zip(agents, heads)]
    optimizers = [torch.optim.Adam(params, lr=.0007) for params in parameters]
    banks = projected_banks(agents, bank)
    times = sorted({0, updates, *(t for t in CHECKPOINTS if t < updates)})
    hashes = {str(Path(k).resolve()): v for k, v in (source_hashes or {}).items()}
    hashes.setdefault(str(Path(__file__).resolve()), sha(__file__))
    assert all(sha(k) == v for k, v in hashes.items())
    cfg = dict(seed=seed, partition=partition, arm=arm, updates=updates, batch_pairs_per_person=batch,
        scenarios_per_person_per_update=2*batch, single_goal_actions_per_person_per_update=2*batch,
        coefficient=weight, learning_rate=.0007, gradient_clip=2., entropy_coefficient=.02, entropy_off_after=2100,
        checkpoints=times, fitted_prefixes=list(PREFIXES), head='Linear96x96,Tanh,Linear96x12; private demand chooses branch',
        head_initialization_seeds=[seed*1000+881+who for who in (0, 1)],
        map_groups={k: v.tolist() for k, v in partition_maps(partition).items()},
        legal_triples=7776, triples_sha256=arrays_sha({'triples': legal_triples(partition)}),
        permutation_convention='g[old_site]=new_site; probability action at new site uses inverse(g)',
        goal_pairing='one Bernoulli private goal per pair, same goal on both world sides; both goals only in JSD',
        world_pairing='same food/water photos; each side old18; all24 empty-site stabilizers retained',
        action_policy='two side actions sampled independently from learned policy, physical site order0..5',
        reward='current goal correct?1:0; constant baseline0.5; average1024 single-goal policy terms',
        jsd='mean over512 pairs and both demand branches; both sides differentiable',
        source_hashes=hashes, prepared_source={'path':str(out/f'prepared_{seed}.pt'), 'sha256':sha(out/f'prepared_{seed}.pt')},
        common_initial_sha256=v8.state_sha(agents), initial_head_sha256=[module_sha(h) for h in heads],
        frozen_sha256=frozen, trainable_parameters=[sum(p.numel() for p in ps) for ps in parameters],
        trainable_agent_names=[[n for n, p in a.named_parameters() if p.requires_grad] for a in agents],
        test_photo_ids=[bank.pools['test', k][:4].tolist() for k in (0, 1)], gate=.90,
        capability_results_never_select_sources=True, social_transfer='Only memory and slot_phi are altered; no private head enters communication',
        rng_schema='SeedSequence([13013,seed,partition,person,purpose,update]), uint64>>1; purpose1world,2action; arm excluded')
    write_json(dest/'config.json', cfg)
    torch.save(initial, dest/'initial.pt'); torch.save([o.state_dict() for o in optimizers], dest/'initial_optimizer.pt')
    np.savez_compressed(dest/'legal_triples.npz', triples=legal_triples(partition), permutations=PERMUTATIONS)
    curve = []; started = time.monotonic()
    with (dest/'training.jsonl').open('w') as stream:
        for update in range(updates+1):
            if update in times:
                scores = evaluate(agents, heads, banks, bank, partition, dest, update)
                curve.append(dict(update=update, scores=scores, agent_sha256=v8.state_sha(agents),
                                  head_sha256=[module_sha(h) for h in heads]))
                write_json(dest/'curve.json', curve)
                torch.save(snapshot(agents, heads), dest/f'checkpoint_{update:04d}.pt')
                torch.save([o.state_dict() for o in optimizers], dest/f'optimizer_{update:04d}.pt')
                # Do not display capability scores or use them to change training.
                print(json.dumps(dict(private=dest.name, update=update, capability_saved=True)), flush=True)
            if update == updates: break
            if update == 2100:
                torch.save(snapshot(agents, heads), dest/'forensic_2100.pt')
                torch.save([o.state_dict() for o in optimizers], dest/'forensic_optimizer_2100.pt')
            records = []
            for who, (agent, head, opt, params) in enumerate(zip(agents, heads, optimizers, parameters)):
                step = update+1; world = fixture(bank, seed, partition, who, step, batch)
                action_seed = seed_for(seed, partition, who, 2, step)
                loss, components, trace = loss_terms(agent, head, banks[who], world, action_seed, weight, .02 if update < 2100 else 0.)
                opt.zero_grad(set_to_none=True); loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(params, 2.)); opt.step()
                if step in (1, 2101):
                    np.savez_compressed(dest/f'train_{step:04d}_person{who}.npz', **trace)
                records.append(dict(person=who, world_seed=seed_for(seed, partition, who, 1, step), action_seed=action_seed,
                    world_sha256=arrays_sha(world, WORLD_KEYS), policy_sha256=arrays_sha({'logits':trace['logits']}),
                    probability_sha256=arrays_sha({'probabilities':trace['probabilities']}),
                    action_sha256=arrays_sha({'action':trace['action']}), action_uniform_sha256=arrays_sha({'uniform':trace['action_uniform']}),
                    map_exposure={'old':2*batch, 'added':0, 'sealed':0}, components=components, gradient_norm=norm))
            stream.write(json.dumps(dict(update=step, persons=records))+'\n')
            if step == 1:
                torch.save(snapshot(agents, heads), dest/'after_first_update.pt')
                torch.save([o.state_dict() for o in optimizers], dest/'after_first_optimizer.pt')
            if step % 100 == 0: stream.flush()
    verified = all(module_sha(a, lambda k: not k.startswith(PREFIXES)) == digest for a, digest in zip(agents, frozen))
    assert verified and all(sha(k) == v for k, v in hashes.items())
    final = snapshot(agents, heads)
    transfer = copy.deepcopy(initial['agents'])
    for who in (0, 1):
        for name, value in final['agents'][who].items():
            if name.startswith(PREFIXES): transfer[who][name] = value.clone()
        assert all(torch.equal(value, transfer[who][name]) for name, value in final['agents'][who].items())
    torch.save(final, dest/'final.pt'); torch.save([o.state_dict() for o in optimizers], dest/'final_optimizer.pt')
    torch.save(transfer, dest/'transferred.pt')
    result = dict(complete=True, seed=seed, partition=partition, arm=arm, updates=updates, batch_pairs_per_person=batch,
        scores=curve[-1]['scores'], frozen_verified=verified, transfer_agent_sha256=v8.state_sha(agents),
        final_sha256=sha(dest/'final.pt'), transferred_sha256=sha(dest/'transferred.pt'), seconds=time.monotonic()-started,
        evaluation_uses_previous_development_photos=True, all_persons_proceed_to_social_training=True)
    write_json(dest/'result.json', result)
    return result


def analyze_private(out, seeds=SEEDS, partitions=(1,2,3)):
    """Read only after the complete declared private batch, never a gate."""
    out = Path(out).resolve()
    expected = [out/f'private_s{s}_p{p}_{a}' for s,p,a in itertools.product(seeds, partitions, ARMS)]
    missing = [str(p) for p in expected if not (p/'result.json').is_file()]
    if missing: return dict(status='pending', missing=missing, result_content_read=False)
    rows = []
    for path in expected:
        result = json.loads((path/'result.json').read_text()); assert result['complete']
        for person in result['scores']['persons']:
            metrics = {f'{g}.{k}': v for g,r in person['groups'].items() for k,v in r.items()}
            metrics.update({f'equivariance.{g}.{k}':v for g,r in person['equivariance']['groups'].items() for k,v in r.items()})
            rows.append(dict(seed=result['seed'], partition=result['partition'], arm=result['arm'], person=person['person'],
                metrics=metrics, old_gate=person['old_single_at_least_090'], path=str(path), result_sha256=sha(path/'result.json')))
    cells=[]
    for seed, arm in itertools.product(seeds, ARMS):
        chosen=[r for r in rows if r['seed']==seed and r['arm']==arm]; assert len(chosen)==2*len(partitions)
        cells.append(dict(seed=seed,arm=arm,persons_averaged=len(chosen),old_gate_count=sum(r['old_gate'] for r in chosen),
            metrics={k:float(np.mean([r['metrics'][k] for r in chosen])) for k in rows[0]['metrics']}))
    lookup={(r['seed'],r['arm']):r for r in cells}
    differences={k:[lookup[s,'equivariant']['metrics'][k]-lookup[s,'control']['metrics'][k] for s in seeds] for k in rows[0]['metrics']}
    summary=dict(status='complete',seed_count=len(seeds),private_pair_runs=len(expected),persons=len(rows),rows=rows,seed_cells=cells,
        aggregate={a:{k:float(np.mean([r['metrics'][k] for r in cells if r['arm']==a])) for k in rows[0]['metrics']} for a in ARMS},
        seed_differences=differences,mean_differences={k:float(np.mean(v)) for k,v in differences.items()},
        source_hashes={str(p/'result.json'):sha(p/'result.json') for p in expected},
        scope='Four new source pairs; average three partitions and two persons within seed. Capability outcomes never select social sources; head-policy equivariance is not latent equivariance.')
    write_json(out/'private_capability_analysis.json',summary)
    lines=['# 私人空间行动准备能力','',summary['scope'],'','| 条件 | 旧图单目标 | 新增图双目标J | 封存图双目标J | 全30图JSD | 全30图策略熵 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,m in summary['aggregate'].items():
        lines.append(f"| {arm} | {m['old.single']:.4f} | {m['added.J']:.4f} | {m['sealed.J']:.4f} | {m['equivariance.all.jsd']:.6f} | {m['all.policy_entropy']:.6f} |")
    lines += ['', '该结果只评价私人行动头连同接口的策略。私人行动头不进入社会通信，低JSD可由高熵、低效的常量策略得到，必须同时检查正确率。0.90门槛逐主体完整记录，不据此删除或替换主体；新增与封存地图均未进入私人训练。', '', '| 指标：equivariant−control | 各来源种子差 | 均值 |','| --- | --- | ---: |']
    for key in ('old.single','added.J','sealed.J','equivariance.all.jsd','all.policy_entropy'):
        lines.append(f"| {key} | "+'、'.join(f'{v:+.6f}' for v in differences[key])+f" | {summary['mean_differences'][key]:+.6f} |")
    (out/'私人空间行动能力.md').write_text('\n'.join(lines)+'\n')
    return summary


def self_test():
    for p in (1,2,3):
        triples=legal_triples(p);old=partition_maps(p)['old'];assert len(old)==18
        assert np.array_equal(map_index(PERMUTATIONS[triples[:,1]][np.arange(7776)[:,None],MAPS[triples[:,0]]]),triples[:,2])
    generator=torch.Generator().manual_seed(913)
    raw=torch.randn(7,2,6,generator=generator,requires_grad=True);lp=F.log_softmax(raw,-1)
    perm=torch.tensor(PERMUTATIONS[[1,10,33,77,120,301,719]])
    moved=lp.gather(-1,torch.argsort(perm)[:,None,:].expand(-1,2,-1))
    assert float(jsd_terms(lp,moved,perm).abs().max().detach())<1e-7
    target=torch.log_softmax(torch.randn(7,2,6,generator=generator),-1)
    js=jsd_terms(lp,target,perm);assert float(js.min().detach())>=-1e-7 and float(js.max().detach())<=np.log(2)+1e-7
    js.mean().backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
    # Ideal physical-location policy is perfectly equivariant under every permutation.
    world=dict(map_ids=np.repeat(np.arange(30),16), positions=np.repeat(MAPS,16,axis=0))
    logits=np.full((480,2,6),-10.)
    for goal in (0,1):logits[np.arange(480),goal,world['positions'][:,goal]]=10.
    eq=equivariance_statistics(logits,1);cap=capability_statistics(logits,world,1)
    assert eq['groups']['all']['jsd']<1e-12 and eq['groups']['all']['greedy_both_consistency']==1
    assert cap['all']['J']==1
    uniform=np.zeros_like(logits);eq0=equivariance_statistics(uniform,1)
    assert abs(eq0['groups']['all']['jsd'])<1e-12
    assert capability_statistics(uniform,world,1)['all']['J']==0
    assert np.isclose(capability_statistics(uniform,world,1)['all']['stochastic_J'],1/36)
    return dict(passed=True,checks=['7776 legal triples in all3 partitions','Both old-map marginals432; each source-target has24 stabilizers',
        'Inverse permutation of probability and differentiable two-goal JSD','All720 exact evaluation verifies ideal equivariant policy',
        'Uniform policies have zeroJSD without task ability','Arm is absent from all sampling and initialization RNG'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--analyze',type=Path);args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if args.analyze:
        result=analyze_private(args.analyze);print(json.dumps({k:v for k,v in result.items() if k in ('status','seed_count','private_pair_runs','persons','missing')}));return
    parser.error('Use --self-test or --analyze; root orchestrator calls run() for training.')


if __name__=='__main__':main()
