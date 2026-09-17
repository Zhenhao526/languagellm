"""Post-hoc single-slot payload and symbol probe for the FI-001 policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, metrics, runner
from research_program.triadic_factorized_neutral_altpartner_information_control_study import recode_mask_probe as base_probe
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core


SLOT_TRANSFORMS = tuple(f"{family}_{slot}" for family in ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4") for slot in range(4))
FAMILIES = {"slot_payload_zero": [f"slot_payload_zero_{i}" for i in range(4)], "slot_symbol_shift": [f"slot_symbol_shift_{i}" for i in range(4)], "slot_symbol_xor4": [f"slot_symbol_xor4_{i}" for i in range(4)]}
METRICS = ("q_rate", "conditional_q_rate", "target_pair_legal_rate", "proposal_legal_rate", "physical_execution_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); require(not path.exists(), "Refuse to overwrite " + str(path)); path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def slot_spec(transform):
    family, slot_text = transform.rsplit("_", 1); require(family in FAMILIES and slot_text in {str(i) for i in range(4)}, "Invalid slot transform"); return family, int(slot_text)


def route(tokens, mode, sender=None, transform=None):
    tokens = np.asarray(tokens); require(mode in ("live", "own"), "Unknown route mode"); require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and ((tokens >= 0) & (tokens < 8)).all(), "Invalid token array"); require((sender is None) == (transform is None), "sender/transform must be paired")
    visibility = np.ones((3, 3), dtype=np.float64) if mode == "live" else np.eye(3, dtype=np.float64); onehot = np.eye(8, dtype=np.float64)[tokens]; visible = onehot[:, None] * visibility[None, :, :, None, None]; bits = np.broadcast_to(visibility[None], (len(tokens), 3, 3)).copy()
    if sender is not None and mode == "live":
        family, slot = slot_spec(transform); viewers = [v for v in range(3) if v != int(sender)]
        if family == "slot_payload_zero":
            for viewer in viewers: visible[:, viewer, int(sender), slot, :] = 0.0
        else:
            changed = tokens[:, int(sender), slot].copy()
            changed = (changed + 1) % 8 if family == "slot_symbol_shift" else (changed ^ 4)
            changed_onehot = np.eye(8, dtype=np.float64)[changed]
            for viewer in viewers: visible[:, viewer, int(sender), slot, :] = changed_onehot
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def action_logits_after_first(networks, x, first, mode, sender, transform):
    routed_first = route(first, mode, sender, transform); second_input = np.concatenate((x, routed_first), axis=-1); second_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor]); second_logits.append(z.reshape(len(x), 4, 8))
    p, _ = core.base.policy_distribution(np.stack(second_logits, axis=1)); second = core.categorical_tokens(p); action_input = np.concatenate((x, routed_first, route(second, mode)), axis=-1)
    action_logits = [core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])[0] for actor in range(3)]
    return np.stack(action_logits, axis=1)


def first_messages(networks, observations):
    x = np.asarray(observations, dtype=np.float64); logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor], x[:, actor]); logits.append(z.reshape(len(x), 4, 8))
    p, _ = core.base.policy_distribution(np.stack(logits, axis=1)); return core.categorical_tokens(p)


def evaluate(networks, arrays, spec, first, sender, transform):
    n = int(spec["world_count"]); c = base_probe._empty_counter(); ids = np.arange(n, dtype=np.int64)
    for start in range(0, n, runner.CONFIG["evaluation_batch_size"]):
        ix = ids[start:min(start + runner.CONFIG["evaluation_batch_size"], n)]; logits = action_logits_after_first(networks, arrays["x_FI"][ix], first[ix], "live", sender, transform); base_probe._accumulate(c, logits, arrays, ix)
    return base_probe.finalize(c, n, spec, "live", sender, transform)


def run_policy(source, static, arrays, seed, schedule, channel):
    condition = f"{schedule}_altpair_factorized_strict_FI_{'live' if channel == 'live' else 'silent'}"; path = source / "execution" / f"seed_{seed}_{condition}"; checkpoint = path / "checkpoint_6000.npz"; result_path = path / "result.json"; require(checkpoint.is_file() and result_path.is_file(), "Missing source run")
    source_result = json.loads(result_path.read_text()); natural = source_result["final"]["new_layouts"]; mode = "live" if channel == "live" else "own"; networks = runner.load_networks(checkpoint); first = first_messages(networks, arrays["x_FI"]) if mode == "live" else None
    rows = []
    for sender in range(3):
        for transform in SLOT_TRANSFORMS:
            if mode == "live": intervened = evaluate(networks, arrays, static["partitions"]["new_layouts"], first, sender, transform); alias = False; forwards = 6 * int(static["partitions"]["new_layouts"]["world_count"])
            else: intervened = dict(natural); alias = True; forwards = 0
            rows.append(dict(seed=seed, schedule=schedule, trained_channel=channel, condition=condition, sender=sender, transform=transform, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), natural=natural, intervened=intervened, natural_minus_intervened={key: float(natural[key] - intervened[key]) for key in METRICS}, alias_of_natural=alias, neural_forward_samples=forwards, optimizer_updates=0, intervention="selected first-window slot transformed for cross-viewers only; W2 and action recomputed greedily"))
    return dict(seed=seed, schedule=schedule, trained_channel=channel, condition=condition, natural_mode=mode, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), natural=natural, first_messages_sha256=None if first is None else core.base.array_sha(first), rows=rows)


def worker(payload):
    source, seed, schedule, channel = payload; source = Path(source); static = json.loads((source / "prepared.json").read_text()); arrays = runner.make_arrays(static["partitions"]["new_layouts"]); return run_policy(source, static, arrays, int(seed), schedule, channel)


def stat(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (16,), "Sixteen seed values required"); return metrics.stats(x)


def summaries(policies):
    by = {(int(p["seed"]), p["schedule"], p["trained_channel"]): p for p in policies}; require(len(by) == 64, "Expected 64 policies"); out = {}
    for schedule in design.SCHEDULES:
        for channel in ("live", "own"):
            for family, transforms in FAMILIES.items():
                for slot in range(4):
                    tr = f"{family}_{slot}"; label = f"{schedule}_{channel}_{tr}"; block = {"schedule": schedule, "trained_channel": channel, "family": family, "slot": slot, "transform": tr}
                    for key in METRICS:
                        vals = [float(np.mean([r["natural_minus_intervened"][key] for r in by[seed, schedule, channel]["rows"] if r["transform"] == tr])) for seed in design.SEEDS]; block[key] = stat(vals); block[key]["by_seed"] = vals
                    out[label] = block
                label = f"{schedule}_{channel}_{family}"; block = {"schedule": schedule, "trained_channel": channel, "family": family, "transforms": transforms}
                for key in METRICS:
                    vals = [float(np.mean([r["natural_minus_intervened"][key] for r in by[seed, schedule, channel]["rows"] if r["transform"] in transforms])) for seed in design.SEEDS]; block[key] = stat(vals); block[key]["by_seed"] = vals
                out[label] = block
    return out


def prepare(out, source):
    out = Path(out).resolve(); source = Path(source).resolve(); require(not out.exists(), "Never overwrite preparation"); out.mkdir(parents=True); manifest = dict(schema="triadic_fi_single_slot_manifest_v1", transforms=list(SLOT_TRANSFORMS), families=FAMILIES, definitions={"slot_payload_zero":"zero one selected sender payload position for cross-viewers", "slot_symbol_shift":"replace one selected token t with (t+1) mod 8 for cross-viewers", "slot_symbol_xor4":"replace one selected token t with t xor 4 for cross-viewers"}, scope="first-window selected sender slot only; self-view retained; W2/action recomputed greedily", controls="FI-silent rows are route-invariant aliases because cross-agent route is absent", no_training=True); write_new(out / "transformation_manifest.json", manifest); prep = dict(schema="triadic_fi_single_slot_probe_v1", source=str(source), source_prepared_sha256=sha(source / "prepared.json"), source_freeze_sha256=sha(source / "freeze.json"), source_execution_status_sha256=sha(source / "execution/status.json"), manifest_sha256=sha(out / "transformation_manifest.json"), transforms=list(SLOT_TRANSFORMS), policy_grid="16 seeds × 2 schedules × 2 trained channels × 3 senders × 12 slot transforms", endpoint_worlds=56160, no_training=True, no_optimizer_updates=True, no_external_model=True); write_new(out / "prepared.json", prep); plan = dict(status="prepared_without_policy_reads", created_at=core.base.now(), runtime=dict(python=platform.python_version(), numpy=np.__version__), source_sha256={"prepared.json": prep["source_prepared_sha256"], "freeze.json": prep["source_freeze_sha256"], "execution/status.json": prep["source_execution_status_sha256"]}, manifest_sha256=prep["manifest_sha256"], prepared_sha256=sha(out / "prepared.json"), transforms=list(SLOT_TRANSFORMS), no_training=True, no_optimizer_updates=True); write_new(out / "plan.json", plan); write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), manifest_sha256=sha(out / "transformation_manifest.json"))); write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), manifest_sha256=sha(out / "transformation_manifest.json"), neural_forward_samples=0, optimizer_updates=0)); return plan


def verify(out):
    out = Path(out).resolve(); plan = json.loads((out / "plan.json").read_text()); prep = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text()); require(sha(out / "plan.json") == freeze["plan_sha256"] and sha(out / "prepared.json") == freeze["prepared_sha256"] and sha(out / "transformation_manifest.json") == freeze["manifest_sha256"] == prep["manifest_sha256"] and plan["prepared_sha256"] == sha(out / "prepared.json"), "Probe preparation hash chain"); return plan, prep


def execute(out, source, workers=4):
    out = Path(out).resolve(); source = Path(source).resolve(); _plan, prep = verify(out); require(prep["source"] == str(source), "Prepared source differs"); execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir(); tasks = [(str(source), seed, schedule, channel) for seed in design.SEEDS for schedule in design.SCHEDULES for channel in ("live", "own")]; started = time.perf_counter();
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool: policies = pool.map(worker, tasks)
    require(len(policies) == 64 and all(len(p["rows"]) == 36 for p in policies), "Complete slot grid required"); rows = [r for p in policies for r in p["rows"]]; forward = int(sum(r["neural_forward_samples"] for r in rows) + sum(3 * prep["endpoint_worlds"] for p in policies if p["natural_mode"] == "live")); result = dict(status="completed_json_only_single_slot_probe", completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started, source=str(source), source_prepared_sha256=prep["source_prepared_sha256"], source_freeze_sha256=prep["source_freeze_sha256"], manifest_sha256=prep["manifest_sha256"], transforms=list(SLOT_TRANSFORMS), policy_blocks=64, intervention_rows=len(rows), live_intervention_rows=sum(not r["alias_of_natural"] for r in rows), alias_rows=sum(r["alias_of_natural"] for r in rows), endpoint_worlds=prep["endpoint_worlds"], model_forward_samples=forward, optimizer_updates=0, no_new_training=True, summaries=summaries(policies), policies=policies, interpretation_boundary="Post-hoc single-slot sensitivity of a fixed task-coordination policy; not evidence of lexical meaning or language origin."); write_new(execution / "results.json", result); write_new(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json"), completed_at=core.base.now(), elapsed_seconds=result["elapsed_seconds"])); receipt = dict(status="passed", policy_blocks=64, intervention_rows=len(rows), live_intervention_rows=result["live_intervention_rows"], alias_rows=result["alias_rows"], endpoint_worlds=prep["endpoint_worlds"], model_forward_samples=forward, optimizer_updates=0, results_sha256=sha(execution / "results.json")); write_new(execution / "receipt.json", receipt); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--source", required=True); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); value = prepare(args.out, args.source) if args.command == "prepare" else (dict(status="verified", plan_sha256=verify(args.out)[0]["prepared_sha256"]) if args.command == "verify" else execute(args.out, args.source, args.workers)); print(json.dumps(value, ensure_ascii=False))
