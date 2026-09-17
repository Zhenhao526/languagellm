"""Train the v0.34 complementary protocol with a partner-topology probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "redesign_v0.8"))
from camp import remake_agents
sys.path.insert(0, str(PROJECT / "redesign_v0.21"))
import social_model as communication
sys.path.insert(0, str(PROJECT / "redesign_v0.28"))
import world
sys.path.insert(0, str(ROOT))
import dual
import metrics
import support
import views


SEEDS = [34101, 34102, 34103, 34104]
PARTITIONS = [1, 2, 3]
AGENTS = 4
PRIVATE_TYPES = (0, 1, 0, 1)
CHECKPOINTS = (0, 100, 600, 1200, 2100, 2400)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_sha(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_sha(state):
    return arrays_sha({key: value.detach().numpy() for key, value in state.items()})


def checkpoint_times(updates):
    return sorted({0, updates, *[t for t in CHECKPOINTS if t < updates]})


def source_hashes():
    files = [ROOT / name for name in ("run_support.py", "support.py", "views.py", "dual.py", "metrics.py", "code_relabelings.npy", "support_design.json", "固定执行方案.md")]
    files += [PROJECT / name for name in ("redesign_v0.28/world.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.21/social_model.py", "redesign_v0.8/camp.py", "redesign_v0.4/run_pilot.py", "redesign_v0.4/agents.py", "redesign_v0.4/resource_env.py")]
    return {str(path): sha(path) for path in files}


def input_hashes(source, seeds, parts):
    files = [source / name for name in ("training_complete.json", "train_worlds.npz", "test_worlds.npz", "private_applicability.json")]
    files += [PROJECT / "redesign_v0.28/data" / name for name in ("feature_cache.pt", "selection.json", "encoder_receipt.json")]
    for seed in seeds:
        files.append(source / f"prepared_{seed}.pt")
        for part in parts:
            files.append(source / "cache" / f"s{seed}_p{part}_all" / "manifest.json")
            for d in (0, 1):
                files += [source / "cache" / f"s{seed}_p{part}_all" / f"{split}_d{d}.npy" for split in ("train", "test")]
                files += [source / "private" / f"s{seed}_p{part}_d{d}_all" / name for name in ("final.pt", "result.json")]
    return {str(path): sha(path) for path in files}


def restore_population(seed, part, prepared, source):
    templates = remake_agents(seed, prepared, 7, 2, "identity")
    agents = []
    resets = []
    for i, private_type in enumerate(PRIVATE_TYPES):
        agent = copy.deepcopy(templates[private_type])
        blob = torch.load(source / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
        agent.load_state_dict(blob["agent"])
        resets.append(communication.reset_communication(agent, support.seed_value(seed, part, i, 1)))
        agents.append(agent)
    return agents, resets


def _encode_chunks(agent, projected, worlds, view, chunk=240):
    values = []
    for start in range(0, len(worlds["map_id"]), chunk):
        stop = min(len(worlds["map_id"]), start + chunk)
        subset = {key: value[start:stop] for key, value in worlds.items()}
        values.append(views.encode(agent, projected, subset, view).numpy())
    return np.concatenate(values, axis=0)


def build_view_cache(seed, part, prepared, source, tables, out):
    """Prepare full and masked h tables, checking full-view identity."""
    bank = world.NonSquareImageBank(PROJECT / "redesign_v0.28/data/feature_cache.pt", PROJECT / "redesign_v0.28/data/selection.json")
    templates = remake_agents(seed, prepared, 7, 2, "identity")
    folder = out / "cache" / f"s{seed}_p{part}"
    folder.mkdir(parents=True)
    cache = {}
    full_errors = {}
    for private_type in (0, 1):
        agent = copy.deepcopy(templates[private_type])
        blob = torch.load(source / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
        agent.load_state_dict(blob["agent"])
        agent.requires_grad_(False)
        with torch.no_grad():
            projected = agent.project(bank.features).detach()
        if projected.shape != (12, 64) or not bool(torch.isfinite(projected).all()):
            raise ValueError("private projection must be finite [12,64]")
        for split, worlds in tables.items():
            for view in views.VIEWS:
                encoded = _encode_chunks(agent, projected, worlds, view)
                if encoded.shape != (len(worlds["map_id"]), 96) or not np.isfinite(encoded).all():
                    raise ValueError("invalid encoded view")
                np.save(folder / f"{split}_{view}_d{private_type}.npy", encoded)
                cache[split, view, private_type] = torch.from_numpy(encoded)
                if view == "full":
                    inherited = np.load(source / "cache" / f"s{seed}_p{part}_all" / f"{split}_d{private_type}.npy")
                    error = float(np.max(np.abs(encoded - inherited)))
                    if not np.allclose(encoded, inherited, atol=3e-5, rtol=3e-5):
                        raise AssertionError(f"full view mismatch: {seed}/{part}/{split}/{private_type}: {error}")
                    full_errors[f"{split}_d{private_type}"] = error
    write(folder / "view_manifest.json", dict(seed=seed, partition=part, views=list(views.VIEWS), full_view_errors=full_errors, source_cache=str((source / "cache" / f"s{seed}_p{part}_all").resolve()), files={str(path.name): sha(path) for path in sorted(folder.glob("*.npy"))}))
    return cache, full_errors


@torch.no_grad()
def social_eval(agents, cache, worlds, part, condition, folder, step, endpoint, schedule="A", file_prefix="protocol", save_agreement=True):
    scores = {}
    for slot, team in enumerate(support.team_slots(condition, schedule=schedule)):
        receiver = int(team[0])
        if condition == "single_full":
            sender = int(team[1])
            h = cache["test", "full", PRIVATE_TYPES[sender]]
            raw = dict(worlds, sender_log_probs=communication.message_log_probs(agents[sender], h).numpy(), tokens=communication.greedy_messages(agents[sender], h).numpy(), receiver_logits=communication.receiver_logits(agents[receiver]).numpy())
        else:
            sender_food, sender_water = int(team[1]), int(team[2])
            view_food = "food_only" if condition == "dual_complementary" else "full"
            view_water = "water_only" if condition == "dual_complementary" else "full"
            h_food = cache["test", view_food, PRIVATE_TYPES[sender_food]]
            h_water = cache["test", view_water, PRIVATE_TYPES[sender_water]]
            log_food = dual.first_token_log_probs(agents[sender_food], h_food)
            log_water = dual.first_token_log_probs(agents[sender_water], h_water)
            raw = dict(worlds, sender_log_probs=dual.joint_sender_tables(log_food, log_water).numpy(), tokens=torch.stack((log_food.argmax(-1), log_water.argmax(-1)), dim=1).numpy(), receiver_logits=communication.receiver_logits(agents[receiver]).numpy(), sender_log_probs_food=log_food.numpy(), sender_log_probs_water=log_water.numpy())
        key = f"team{slot}"
        np.savez_compressed(folder / f"{file_prefix}_{step:04d}_{key}.npz", **raw)
        scores[key] = metrics.social(raw, part, condition)
        if step == endpoint and file_prefix == "protocol":
            metrics.save_null(raw, part, condition, folder / f"recombination_null_{key}.npz")
    # Paired same-private-type evaluations are saved separately from team
    # protocols. They use identical frozen test worlds and make agreement a
    # transparent auxiliary measure rather than a by-product of team pairing.
    if save_agreement:
        agreement = {}
        for view, getter in (("full", communication.greedy_messages), ("food_only", dual.first_token_greedy), ("water_only", dual.first_token_greedy)):
            for private_type, (a, b) in ((0, (0, 2)), (1, (1, 3))):
                ha = cache["test", view, private_type]
                agreement[f"{view}_type{private_type}_a"] = getter(agents[a], ha).numpy()
                agreement[f"{view}_type{private_type}_b"] = getter(agents[b], ha).numpy()
        np.savez_compressed(folder / f"agreement_{step:04d}.npz", **agreement)
    return scores


def update_population(agents, opts, cache, tables, seed, part, step, condition):
    losses = [[] for _ in agents]
    worlds = {}
    traces = {}
    entropy_weight = 0.02 if step < 2100 else 0.0
    for slot, team in enumerate(support.team_slots(condition, step=step)):
        fixture = support.fixture(seed, part, slot, step, condition, tables["train"])
        idx = fixture["indices"]
        if condition == "single_full":
            receiver, sender = map(int, team)
            sender_loss, receiver_loss, trace = communication.direction_loss(agents[sender], agents[receiver], cache["train", "full", PRIVATE_TYPES[sender]][idx], fixture["uniforms"], tables["train"]["positions"][idx], entropy_weight)
            losses[sender].append(sender_loss); losses[receiver].append(receiver_loss)
            traces.update({f"slot{slot}__r{receiver}_s{sender}__{key}": np.asarray(value) for key, value in trace.items()})
        else:
            receiver, sender_food, sender_water = map(int, team)
            view_food = "food_only" if condition == "dual_complementary" else "full"
            view_water = "water_only" if condition == "dual_complementary" else "full"
            sender_food_loss, sender_water_loss, receiver_loss, trace = dual.dual_loss(agents[sender_food], agents[sender_water], agents[receiver], cache["train", view_food, PRIVATE_TYPES[sender_food]][idx], cache["train", view_water, PRIVATE_TYPES[sender_water]][idx], fixture["uniforms"], tables["train"]["positions"][idx], entropy_weight)
            losses[sender_food].append(sender_food_loss); losses[sender_water].append(sender_water_loss); losses[receiver].append(receiver_loss)
            traces.update({f"slot{slot}__r{receiver}_sf{sender_food}_sw{sender_water}__{key}": np.asarray(value) for key, value in trace.items()})
        worlds.update({f"slot{slot}__{key}": value for key, value in fixture.items()})
    logs = []
    for i, (agent, opt) in enumerate(zip(agents, opts)):
        if not losses[i]:
            raise ValueError("every agent must receive a role loss")
        loss = sum(losses[i]) / len(losses[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms = {role: float(torch.nn.utils.clip_grad_norm_(params, 2.0)) for role, params in communication.trainable_groups(agent).items()}
        if not all(np.isfinite(value) for value in norms.values()) or not all(param.grad is None for param in agent.parameters() if not param.requires_grad):
            raise ValueError("invalid gradient state")
        logs.append(dict(loss=float(loss.detach()), norms=norms, role_loss_count=len(losses[i])))
    for opt in opts:
        opt.step()
    return worlds, traces, logs


def fit_population(seed, part, condition, prepared, cache, tables, source, out, args):
    folder = out / "social" / f"s{seed}_p{part}_{condition}"
    folder.mkdir(parents=True)
    agents, resets = restore_population(seed, part, prepared, source)
    opts = [torch.optim.Adam([param for group in communication.trainable_groups(agent).values() for param in group], lr=0.0007) for agent in agents]
    prefixes = communication.SENDER_MODULES + communication.RECEIVER_MODULES
    frozen = lambda agent: state_sha({key: value for key, value in agent.state_dict().items() if not key.startswith(prefixes)})
    frozen_hashes = [frozen(agent) for agent in agents]
    save = lambda: [copy.deepcopy(agent.state_dict()) for agent in agents]
    torch.save(save(), folder / "initial.pt")
    torch.save([opt.state_dict() for opt in opts], folder / "initial_optimizer.pt")
    if condition == "single_full":
        messages_per_update = 4 * 240
        token_instances_per_update = 8 * 240
    else:
        messages_per_update = 8 * 240
        token_instances_per_update = messages_per_update
    write(folder / "config.json", dict(seed=seed, partition=part, condition=condition, updates=args.updates, checkpoints=checkpoint_times(args.updates), reset=resets, frozen_hashes=frozen_hashes, cache=str((out / "cache" / f"s{seed}_p{part}").resolve()), population_agents=AGENTS, private_types=list(PRIVATE_TYPES), teams=[list(team) for team in support.team_slots(condition, schedule="A")], training_schedule=("A" if condition != "dual_complementary" else "A_even_B_odd"), evaluation_schedules=["A", "B", "C"], views=list(views.VIEWS), learning_rate=0.0007, clip=2.0, messages_per_population_update=messages_per_update, token_instances_per_population_update=token_instances_per_update, actions_per_population_update=8 * 240, payoff=support._DESIGN["payoff"]))
    curve = []
    started = time.monotonic()
    with (folder / "training.jsonl").open("w") as log:
        for step in range(args.updates + 1):
            if step in checkpoint_times(args.updates):
                scores = social_eval(agents, cache, tables["test"], part, condition, folder, step, args.updates, schedule="A", file_prefix="protocol", save_agreement=True)
                if step == args.updates and condition == "dual_complementary":
                    for transfer_schedule in ("B", "C"):
                        social_eval(agents, cache, tables["test"], part, condition, folder, step, args.updates, schedule=transfer_schedule, file_prefix=f"transfer_{transfer_schedule}", save_agreement=False)
                curve.append(dict(update=step, scores=scores))
                write(folder / "curve.json", curve)
                torch.save(save(), folder / f"checkpoint_{step:04d}.pt")
                torch.save([opt.state_dict() for opt in opts], folder / f"optimizer_{step:04d}.pt")
                print(json.dumps(dict(phase="team_social", run=folder.name, update=step, seconds=round(time.monotonic() - started, 2))), flush=True)
            if step == args.updates:
                break
            worlds, trace, people = update_population(agents, opts, cache, tables, seed, part, step, condition)
            if step in (0, 2100, args.updates - 1):
                np.savez_compressed(folder / f"train_{step + 1:04d}.npz", **{f"world__{key}": value for key, value in worlds.items()}, **{f"trace__{key}": value for key, value in trace.items()})
                torch.save(save(), folder / f"after_{step + 1:04d}.pt")
                torch.save([opt.state_dict() for opt in opts], folder / f"after_{step + 1:04d}_optimizer.pt")
            log.write(json.dumps(dict(update=step + 1, world_sha256=arrays_sha(worlds), trace_sha256=arrays_sha(trace), people=people)) + "\n")
            if (step + 1) % 100 == 0:
                log.flush()
    if frozen_hashes != [frozen(agent) for agent in agents]:
        raise AssertionError("frozen frontend changed")
    torch.save(save(), folder / "final.pt")
    write(folder / "result.json", dict(status="complete", condition=condition, scores=curve[-1]["scores"], updates=args.updates, seconds=time.monotonic() - started, messages=args.updates * messages_per_update, token_instances=args.updates * token_instances_per_update, actions=args.updates * 8 * 240, frozen_verified=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--updates", type=int, default=2400)
    args = ap.parse_args()
    torch.set_num_threads(1)
    seeds = [99528] if args.dev else SEEDS
    parts = [1] if args.dev else PARTITIONS
    source = PROJECT / "redesign_v0.28/results" / ("smoke_001" if args.dev else "formation_001")
    hashes = source_hashes()
    inputs = input_hashes(source, seeds, parts)
    preflight = None
    if args.dev:
        if args.updates != 40:
            raise ValueError("development run must use 40 updates")
    else:
        gate = read(ROOT / "preflight_qa.json")
        if not (gate["passed"] and gate["source_hashes"] == hashes and gate["input_hashes"] == inputs and args.updates == 2400):
            raise AssertionError("formal preflight gate failed")
        preflight = dict(path=str((ROOT / "preflight_qa.json").resolve()), sha256=sha(ROOT / "preflight_qa.json"))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(out / "invocation.json", dict(formal=not args.dev, seeds=seeds, partitions=parts, updates=args.updates, source=str(source.resolve()), source_hashes=hashes, input_hashes=inputs, preflight=preflight, torch_version=str(torch.__version__), numpy_version=np.__version__, platform=platform.platform(), threads=1, new_dino_inferences=0, conditions=list(support.CONDITIONS), population_agents=AGENTS, private_types=list(PRIVATE_TYPES), team_slots=4, study_scope="complementary_two_sender_one_receiver_team_protocol", new_private_fits=0, test_worlds=180, train_worlds=720))
    tables = {split: dict(np.load(source / f"{split}_worlds.npz")) for split in ("train", "test")}
    for split, worlds in tables.items():
        np.savez_compressed(out / f"{split}_worlds.npz", **worlds)
    started = time.monotonic()
    run_count = 0
    for seed, part in itertools.product(seeds, parts):
        prepared = torch.load(source / f"prepared_{seed}.pt", weights_only=True)
        cache, errors = build_view_cache(seed, part, prepared, source, tables, out)
        for condition in support.CONDITIONS:
            fit_population(seed, part, condition, prepared, cache, tables, source, out, args)
            run_count += 1
    for path, digest in {**hashes, **inputs}.items():
        if sha(path) != digest:
            raise AssertionError(f"bound file changed: {path}")
    write(out / "training_complete.json", dict(status="complete", formal=not args.dev, social_runs=run_count, pair_updates=run_count * args.updates, messages=sum(args.updates * (4 * 240 if c == "single_full" else 8 * 240) for _ in seeds for _ in parts for c in support.CONDITIONS), token_instances=sum(args.updates * (8 * 240 if c == "single_full" else 8 * 240) for _ in seeds for _ in parts for c in support.CONDITIONS), actions=run_count * args.updates * 8 * 240, seconds=time.monotonic() - started, source_hashes=hashes, input_hashes=inputs, new_private_fits=0, new_dino_inferences=0, files={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}))
    print(json.dumps(dict(status="complete", social_runs=run_count, seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    main()
