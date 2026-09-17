"""Independent replay and pairing audit for the sender-alignment study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from . import design, runner, policy, environment


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require_close(a, b, label, state):
    delta = abs(float(a) - float(b))
    state["max_error"] = max(state["max_error"], delta)
    design.require(delta < 1e-12, f"replay mismatch {label}: {a} != {b}")


def compare_eval(expected, fresh, state, prefix):
    for goal_kind in ("all", "seen", "heldout"):
        for mode in ("natural", "permuted", "silent", "raw_recombined"):
            for key in ("episodes", "team_return_mean", "team_return_sd", "positive_episode_rate", "oracle_team_return_mean"):
                require_close(expected[goal_kind][mode][key], fresh[goal_kind][mode][key], f"{prefix}/{goal_kind}/{mode}/{key}", state)
    design.require(expected["sender_codebook"] == fresh["sender_codebook"], f"codebook mismatch {prefix}")
    require_close(expected["sender_consistency"], fresh["sender_consistency"], f"{prefix}/sender_consistency", state)
    design.require(expected["pairwise_min_hamming"] == fresh["pairwise_min_hamming"], f"hamming mismatch {prefix}")
    for i, (e, f) in enumerate(zip(expected["all_by_partner"], fresh["all_by_partner"])):
        for key in ("episodes", "team_return_mean", "team_return_sd", "positive_episode_rate", "oracle_team_return_mean"):
            require_close(e[key], f[key], f"{prefix}/all_by_partner/{i}/{key}", state)


def stream_ok(log_row, ep):
    names = {
        "site_semantic_sha256": ep["site_semantic"],
        "site_surface_sha256": ep["site_surface"],
        "goal_sha256": ep["goal"],
        "partner_sha256": ep["partner_id"],
        "message_uniform_sha256": ep["message_uniforms"],
        "action_uniform_sha256": ep["action_uniforms"],
    }
    for key, value in names.items():
        design.require(log_row[key] == design.array_sha(value), f"stream mismatch {key}")


def checkpoint_ok(path, params, log_row, state):
    design.require(Path(path).is_file(), f"missing checkpoint {path}")
    design.require(log_row.get("checkpoint_sha256") == runner.sha(path), f"checkpoint hash mismatch {path}")
    with np.load(path, allow_pickle=False) as data:
        for name in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
            require_close(np.max(np.abs(np.asarray(data[name]) - params[name])), 0.0, f"checkpoint/{path}/{name}", state)


def audit(prepared, execution, out):
    runner.verify(prepared)
    execution = Path(execution)
    payload = json.loads((execution / "results.json").read_text())
    state = {"max_error": 0.0}
    parents = {}
    children = {}
    parent_logs = child_logs = parent_checkpoints = child_checkpoints = 0

    for result in payload["parents"]:
        key = (int(result["seed"]), result["population_mode"])
        design.require(key not in parents, f"duplicate parent {key}")
        parents[key] = result
        seed, mode = key
        params = policy.make_policy(seed)
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"parent initial hash mismatch {key}")
        run = execution / "parents" / f"seed_{seed}_{mode}"
        for log_row in rows(run / "training.jsonl"):
            update = int(log_row["update"])
            ep = design.episode_stream(seed, design.BATCH_SIZE, support="full", population_mode=mode, update=update)
            tr = environment.rollout(params, ep, "hidden", channel="live", sample=True)
            grad = runner.gradient(params, ep, tr, "hidden", train_sender=True, train_workers=True)
            norm, scale = runner.update_params(policy.clone(params), grad, visibility="hidden", train_sender=True, train_workers=True)
            # update_params above was applied to a clone only to obtain deterministic norm/scale;
            # use the same update once on the live parameters below.
            require_close(norm, log_row["gradient_norm"], f"parent/{key}/{update}/norm", state)
            require_close(scale, log_row["gradient_clip_scale"], f"parent/{key}/{update}/scale", state)
            require_close(float(tr["team_return"].mean()), log_row["return_mean"], f"parent/{key}/{update}/return", state)
            stream_ok(log_row, ep)
            runner.update_params(params, grad, visibility="hidden", train_sender=True, train_workers=True)
            design.require(log_row["parameter_sha256"] == policy.parameter_hash(params), f"parent parameter mismatch {key}/{update}")
            if update in result["checkpoints"]:
                checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, log_row, state)
                parent_checkpoints += 1
            parent_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"parent final hash mismatch {key}")
        fresh = runner.evaluate(params, seed, mode, "hidden", design.heldout_goal(seed))
        compare_eval(result["final"], fresh, state, f"parent/{key}/eval")

    for result in payload["children"]:
        key = (int(result["seed"]), result["condition"])
        design.require(key not in children, f"duplicate child {key}")
        children[key] = result
        seed, condition = key
        mode, visibility, adaptation, support = design.parse_child_condition(condition)
        parent = runner.load_checkpoint(result["parent_checkpoint"])
        params = runner.child_initial_params(seed, parent)
        design.require(result["parent_parameter_sha256"] == policy.parameter_hash(parent), f"child parent hash mismatch {key}")
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"child initial hash mismatch {key}")
        run = execution / "children" / f"seed_{seed}_{condition}"
        for log_row in rows(run / "training.jsonl"):
            update = int(log_row["update"])
            heldout = design.heldout_goal(seed)
            ep = design.episode_stream(seed, design.BATCH_SIZE, support=support, heldout=heldout, population_mode=mode, update=update)
            if support == "leave_one_out":
                design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {key}/{update}")
            tr = environment.rollout(params, ep, visibility, channel="live", sample=True)
            train_workers = adaptation == "coadapt"
            grad = runner.gradient(params, ep, tr, visibility, train_sender=True, train_workers=train_workers)
            norm, scale = runner.update_params(policy.clone(params), grad, visibility=visibility, train_sender=True, train_workers=train_workers)
            require_close(norm, log_row["gradient_norm"], f"child/{key}/{update}/norm", state)
            require_close(scale, log_row["gradient_clip_scale"], f"child/{key}/{update}/scale", state)
            require_close(float(tr["team_return"].mean()), log_row["return_mean"], f"child/{key}/{update}/return", state)
            stream_ok(log_row, ep)
            runner.update_params(params, grad, visibility=visibility, train_sender=True, train_workers=train_workers)
            design.require(log_row["parameter_sha256"] == policy.parameter_hash(params), f"child parameter mismatch {key}/{update}")
            if update in result["checkpoints"]:
                checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, log_row, state)
                child_checkpoints += 1
            child_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"child final hash mismatch {key}")
        fresh = runner.evaluate(params, seed, mode, visibility, design.heldout_goal(seed))
        compare_eval(result["final"], fresh, state, f"child/{key}/eval")

    expected_parents = {(seed, result["population_mode"]) for (seed, _), result in children.items()}
    design.require(set(parents) == expected_parents, "parent accounting mismatch")

    paired_visibility = paired_adaptation = paired_population = paired_support = 0
    stream_keys = ("site_semantic_sha256", "site_surface_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256")
    for seed in sorted({k[0] for k in children}):
        for mode in design.POPULATION_MODES:
            # visibility, adaptation and support pairs share the full environment stream.
            for adaptation in design.ADAPTATIONS:
                for support in design.SUPPORTS:
                    hidden = children.get((seed, f"{mode}_hidden_{adaptation}_{support}"))
                    visible = children.get((seed, f"{mode}_visible_{adaptation}_{support}"))
                    if hidden and visible:
                        left = rows(execution / "children" / f"seed_{seed}_{hidden['condition']}" / "training.jsonl")
                        right = rows(execution / "children" / f"seed_{seed}_{visible['condition']}" / "training.jsonl")
                        for a, b in zip(left, right):
                            for name in stream_keys + ("goal_sha256",):
                                design.require(a[name] == b[name], f"visibility pair mismatch {name}")
                            paired_visibility += 1
                for visibility in design.VISIBILITIES:
                    for support in design.SUPPORTS:
                        sender = children.get((seed, f"{mode}_{visibility}_sender_only_{support}"))
                        coadapt = children.get((seed, f"{mode}_{visibility}_coadapt_{support}"))
                        if sender and coadapt:
                            left = rows(execution / "children" / f"seed_{seed}_{sender['condition']}" / "training.jsonl")
                            right = rows(execution / "children" / f"seed_{seed}_{coadapt['condition']}" / "training.jsonl")
                            for a, b in zip(left, right):
                                for name in stream_keys + ("goal_sha256",):
                                    design.require(a[name] == b[name], f"adaptation pair mismatch {name}")
                                paired_adaptation += 1
            for visibility in design.VISIBILITIES:
                for adaptation in design.ADAPTATIONS:
                    for support in design.SUPPORTS:
                        hom = children.get((seed, f"homogeneous_{visibility}_{adaptation}_{support}"))
                        het = children.get((seed, f"heterogeneous_{visibility}_{adaptation}_{support}"))
                        if hom and het:
                            left = rows(execution / "children" / f"seed_{seed}_{hom['condition']}" / "training.jsonl")
                            right = rows(execution / "children" / f"seed_{seed}_{het['condition']}" / "training.jsonl")
                            changed = False
                            for a, b in zip(left, right):
                                for name in ("site_semantic_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                                    design.require(a[name] == b[name], f"population pair mismatch {name}")
                                changed = changed or a["site_surface_sha256"] != b["site_surface_sha256"]
                                paired_population += 1
                            design.require(changed, "population surface intervention is a no-op")
            for visibility in design.VISIBILITIES:
                for adaptation in design.ADAPTATIONS:
                    full = children.get((seed, f"homogeneous_{visibility}_{adaptation}_full"))
                    loo = children.get((seed, f"homogeneous_{visibility}_{adaptation}_leave_one_out"))
                    if full and loo:
                        left = rows(execution / "children" / f"seed_{seed}_{full['condition']}" / "training.jsonl")
                        right = rows(execution / "children" / f"seed_{seed}_{loo['condition']}" / "training.jsonl")
                        for a, b in zip(left, right):
                            for name in stream_keys:
                                design.require(a[name] == b[name], f"support pair mismatch {name}")
                            paired_support += 1

    design.require(state["max_error"] < 1e-12, f"replay error {state['max_error']}")
    report = {
        "schema": "partner_exposure_audit_v1",
        "status": "passed",
        "parent_runs": len(parents),
        "child_runs": len(children),
        "parent_training_log_rows": parent_logs,
        "child_training_log_rows": child_logs,
        "parent_checkpoints": parent_checkpoints,
        "child_checkpoints": child_checkpoints,
        "paired_visibility_rows": paired_visibility,
        "paired_adaptation_rows": paired_adaptation,
        "paired_population_rows": paired_population,
        "paired_support_rows": paired_support,
        "max_abs_replay_error": state["max_error"],
    }
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.execution, args.out), ensure_ascii=False))
