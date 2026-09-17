"""Finite-support reward-score variance diagnostic; no fitting or optimizer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'redesign_v0.15'))
import run_scaled as v15
from camp import MAPS, scene_visual
sys.path.insert(0, str(ROOT))
import gradient_moments as gm

FORMAL_SOURCE = PROJECT / 'redesign_v0.15/results/scaled_001'
DEV_SOURCE = PROJECT / 'redesign_v0.15/results/smoke_002'
TIMES = [0, 100, 600]
SEEDS = [31101, 31102, 31103, 31104]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def state_sha(a):
    digest = hashlib.sha256()
    for key, value in a.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes(source):
    inherited = read(source / 'invocation.json')['source_hashes']
    for p, digest in inherited.items():
        assert sha(p) == digest, p
    own = [ROOT / x for x in ('run_gradient.py', 'gradient_moments.py', '固定执行方案.md', '前置审查.md')]
    return {**inherited, **{str(p): sha(p) for p in own}}


def input_receipt(source, seeds, partitions, formal):
    paths = [source / name for name in ('invocation.json', 'training_complete.json', 'calibration_receipt.json')]
    assert read(source / 'training_complete.json')['status'] == 'complete'
    for seed in seeds:
        paths.append(source / f'prepared_{seed}.pt')
        for p in partitions:
            folder = source / f'social_s{seed}_p{p}_reset_scaled'
            paths.extend([folder / 'config.json'] + [folder / f'checkpoint_{t:04d}.pt' for t in TIMES])
    if formal:
        old_manifest = read(source / 'completion_manifest.json')
        assert old_manifest['status'] == 'complete'
        for p in paths:
            assert sha(p) == old_manifest['artifacts'][str(p.relative_to(PROJECT))], p
        paths.append(source / 'completion_manifest.json')
    paths += [PROJECT / 'redesign_v0.4/data' / n for n in ('manifest.json', 'features.npz', 'encoder_report.json')]
    encoder = read(paths[-1])
    assert encoder['manifest_sha256'] == sha(paths[-3])
    assert encoder['features_sha256'] == sha(paths[-2])
    return {str(p): sha(p) for p in paths}


@torch.no_grad()
def partner_logits(partner):
    messages = torch.tensor(list(itertools.product(range(7), repeat=2)), dtype=torch.int64)
    logits = []
    for goal_id in (0, 1):
        goals = torch.zeros(49, 2)
        goals[:, goal_id] = 1
        out, _ = partner.receive(messages, goals, torch.zeros(49, 2), torch.zeros(49, 18),
                                 torch.arange(6).repeat(49, 1))
        logits.append(out.numpy())
    return np.stack(logits, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dev', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    source = DEV_SOURCE if args.dev else FORMAL_SOURCE
    seeds = [99513] if args.dev else SEEDS
    partitions = [1] if args.dev else [1, 2, 3]
    hashes = source_hashes(source)
    receipt = input_receipt(source, seeds, partitions, not args.dev)
    gate = None
    if not args.dev:
        gate_path = ROOT / 'preflight_qa.json'
        gate = read(gate_path)
        assert gate['passed'] and gate['source_hashes'] == hashes
        gate = dict(path=str(gate_path), sha256=sha(gate_path))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out / 'frozen_sources'
    snapshots.mkdir()
    for p in hashes:
        target = snapshots / Path(p).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    bank = v15.ImageBank()
    photo_pair = [int(np.min(bank.pools['train', k])) for k in (0, 1)]
    assert all(bank.entries[row]['split'] == 'train' for row in photo_pair)
    photo_manifest = [dict(feature_row=row, **bank.entries[row]) for row in photo_pair]
    invocation = dict(started_utc=datetime.now(timezone.utc).isoformat(), formal=not args.dev,
        source=str(source), source_hashes=hashes, inputs=receipt, preflight=gate,
        seeds=seeds, partitions=partitions, times=TIMES, directions=[0, 1],
        photos=photo_manifest, photo_rule='minimum feature row in each original training category',
        support='18 sorted old maps per partition; one fixed training photo pair; one independent draw per world',
        training_updates=0, optimizer_steps=0, new_dino_inferences=0,
        device='cpu', threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        torch_version=str(torch.__version__), numpy_version=np.__version__, platform=platform.platform(),
        numerical_definition='float32 original modules; float64 log_softmax; float32 backprop; float64 moments',
        inference_scope='mathematical categorical policies, not bitwise finite-PRNG binning')
    write(out / 'invocation.json', invocation)
    all_rows = []
    started = time.monotonic()
    for seed in seeds:
        prepared = torch.load(source / f'prepared_{seed}.pt', weights_only=True)
        for partition in partitions:
            maps = np.sort(v15.v13.private.partition_maps(partition)['old'])
            assert len(maps) == 18
            for checkpoint in TIMES:
                checkpoint_path = source / f'social_s{seed}_p{partition}_reset_scaled/checkpoint_{checkpoint:04d}.pt'
                states = torch.load(checkpoint_path, weights_only=True)
                agents = v15.scaled.remake_scaled_agents(seed, prepared, states)
                for agent in agents:
                    agent.requires_grad_(False)
                    for name, parameter in agent.named_parameters():
                        parameter.requires_grad_(name.startswith(('send_context.', 'send_embedding.', 'send_recur.', 'send_out.')))
                    agent.train()
                before = [state_sha(a) for a in agents]
                projected = v15.projected_banks(agents, bank)
                for who in (0, 1):
                    name = f's{seed}_p{partition}_d{who}_t{checkpoint:04d}'
                    folder = out / name
                    folder.mkdir()
                    logits = partner_logits(agents[1-who])
                    summaries = []
                    for world_index, map_id in enumerate(maps):
                        position = MAPS[map_id].copy()
                        with torch.no_grad():
                            visual = scene_visual(position[None, :], np.asarray(photo_pair)[None, :], projected[who])
                            h = agents[who].observe(visual).detach()
                        result = gm.measure_world(agents[who], h, logits, position)
                        arrays = dict(result['arrays'])
                        assert not any(k in arrays for k in ('scores', 'score_matrix', 'mu', 'mean_score'))
                        arrays.update(h=h.numpy(), receiver_logits=logits, map_id=np.asarray(map_id),
                                      photo_ids=np.asarray(photo_pair), world_index=np.asarray(world_index))
                        path = folder / f'world_{world_index:02d}.npz'
                        np.savez_compressed(path, **arrays)
                        summaries.append(dict(world_index=world_index, map_id=int(map_id),
                            file=str(path.relative_to(out)), sha256=sha(path), **result['summary']))
                    result = dict(seed=seed, partition=partition, direction=who, checkpoint=checkpoint,
                        source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=sha(checkpoint_path),
                        sender_state_sha256=before[who], partner_state_sha256=before[1-who],
                        world_summaries=summaries, aggregate=gm.aggregate_support(summaries))
                    write(folder / 'result.json', result)
                    all_rows.append(dict(seed=seed, partition=partition, direction=who, checkpoint=checkpoint,
                        file=str((folder / 'result.json').relative_to(out)), sha256=sha(folder / 'result.json'),
                        **result['aggregate']))
                    print(json.dumps(dict(policy=name, worlds=18, seconds=round(time.monotonic()-started, 2))), flush=True)
                assert [state_sha(a) for a in agents] == before, 'diagnostic changed checkpoint tensors'
    for p, digest in {**hashes, **receipt}.items():
        assert sha(p) == digest, p
    write(out / 'policy_results.json', all_rows)
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write(out / 'measurement_complete.json', dict(status='complete', formal=not args.dev,
        completed_utc=datetime.now(timezone.utc).isoformat(), policies=len(all_rows), worlds=18*len(all_rows),
        message_scores=18*49*len(all_rows), seconds=time.monotonic()-started,
        max_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        training_updates=0, source_hashes=hashes, input_hashes=receipt, files=files))
    print(json.dumps(dict(status='complete', policies=len(all_rows), seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
