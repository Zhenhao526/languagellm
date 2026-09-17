"""Post-hoc four-choice content transfer probe for public/private codebooks.

The probe never updates parameters.  A host world keeps its observations and
the listener's candidate actions.  Only the first-window packet in the host
sender slot is replaced by a packet produced by the same sender, the host
listener, or the third agent in a need-permuted donor world.  W2 and the final
action are recomputed causally with the host policy.  A public wire map should
make the cross-sender replacements more interpretable than sender-private maps
if a shared convention has actually formed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import hashlib

import numpy as np

from research_program.triadic_action_dependency_study import dataset as task_dataset
from research_program.triadic_action_dependency_study import environment as task_environment
from research_program.triadic_content_response_study import dataset as content_dataset
from research_program.triadic_content_response_study import metrics as content_metrics
from research_program.triadic_message_study import runner as core


SEEDS = (68101, 68102, 68103, 68104, 68105, 68106, 68107, 68108)
TARGET = "new_needs_and_layouts"
GLOBAL_MAP = (3, 7, 1, 6, 0, 4, 2, 5)
PRIVATE_MAPS = (
    (6, 7, 1, 0, 4, 5, 3, 2),
    (4, 1, 3, 2, 6, 5, 7, 0),
    (4, 3, 2, 6, 1, 5, 7, 0),
)
MODES = ("within_sender", "cross_listener", "cross_third")
CONDITIONS = ("identity_live", "public_live", "private_live")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_condition(condition):
    if condition == "identity_live":
        return "identity", True, (tuple(range(8)),) * 3
    if condition == "public_live":
        return "public", True, (GLOBAL_MAP,) * 3
    if condition == "private_live":
        return "private", True, PRIVATE_MAPS
    raise ValueError(condition)


def recode(tokens, maps):
    tokens = np.asarray(tokens)
    output = np.empty_like(tokens)
    for sender, mapping in enumerate(maps):
        output[:, sender] = np.asarray(mapping, dtype=np.int8)[tokens[:, sender]]
    return output


def observations(states):
    states = np.asarray(states)
    require(states.ndim == 2 and states.shape[1:] == (10,), "Invalid packed states")
    worlds = [task_environment.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in states]
    return task_dataset.encode_observations([
        {actor: task_environment.observe(world, actor, information="PL")
         for actor in task_environment.AGENTS}
        for world in worlds
    ])


def greedy_rollout(networks, x, live, maps):
    """Greedy rollout independent of the training wire wrapper."""
    x = np.asarray(x, dtype=np.float64); n = len(x)
    inputs = x; messages = []; wire_messages = []
    for window in range(2):
        logits = []
        for actor in range(3):
            z, _ = core.base.actor_forward(networks[3 * actor + window], inputs[:, actor])
            logits.append(z.reshape(n, 4, 8))
        z = np.stack(logits, axis=1); probabilities, _ = core.base.policy_distribution(z)
        internal = np.argmax(probabilities, axis=-1).astype(np.int8)
        on_wire = recode(internal, maps)
        messages.append(internal); wire_messages.append(on_wire)
        if window == 0:
            inputs = np.concatenate((x, core.routed_window(on_wire, bool(live))), axis=-1)
    return np.stack(messages, axis=1), np.stack(wire_messages, axis=1)


def replace_sender_slot(route, sender, packet):
    packet = np.asarray(packet, dtype=np.int8)
    encoded = np.eye(8, dtype=np.float64)[packet].reshape(len(packet), 32)
    for viewer in range(3):
        rows = np.arange(len(packet))[sender != viewer]
        slots = 32 * sender[rows, None] + np.arange(32)
        route[rows[:, None], viewer, slots] = encoded[rows]
    return route


def intervene(networks, host_x, host_wire_first, host_sender, donor_wire_first, host_maps):
    """Replace W1 in a host sender slot, then recompute W2 and action."""
    route1 = core.routed_window(host_wire_first, True).copy()
    route1 = replace_sender_slot(route1, host_sender, donor_wire_first)
    second_input = np.concatenate((host_x, route1), axis=-1)
    second_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])
        second_logits.append(z.reshape(len(host_x), 4, 8))
    second_probabilities, _ = core.base.policy_distribution(np.stack(second_logits, axis=1))
    second_internal = np.argmax(second_probabilities, axis=-1).astype(np.int8)
    second_wire = recode(second_internal, host_maps)
    route2 = core.routed_window(second_wire, True)
    action_input = np.concatenate((host_x, route1, route2), axis=-1)
    action_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])
        action_logits.append(z)
    probabilities, _ = core.base.policy_distribution(np.stack(action_logits, axis=1))
    return probabilities


def pack_states(part, indices):
    return content_dataset.pack_states(part, np.asarray(indices, dtype=np.int64))


def endpoint_path(run_dir, run, part=TARGET):
    return Path(run["final"][part]["path"])


def compact_metrics(probabilities, rows):
    n = len(probabilities); listeners = rows["listeners"]
    candidate = rows["candidate_receiver_actions"]
    ix = np.arange(n); cp = probabilities[ix[:, None], listeners[:, None], candidate]
    target = cp[ix, rows["donor_endpoint"]]
    other = cp.copy(); other[ix, rows["donor_endpoint"]] = -np.inf
    margin = target - np.max(other, axis=1)
    off = rows["host_endpoint"] != rows["donor_endpoint"]
    hit4 = np.argmax(cp, axis=1) == rows["donor_endpoint"]
    hit17 = np.argmax(probabilities[ix, listeners], axis=-1) == rows["target_receiver_actions"]
    probe = content_metrics.probe_metrics_with_layers(
        probabilities, rows["group_index"], rows["background_index"], rows["host_endpoint"], rows["donor_endpoint"],
        listeners, candidate, rows["group_layer_index"]
    )
    return dict(M=float(probe["M"]), off_diagonal_margin=float(margin[off].mean()),
                target_probability=float(target[off].mean()), hit4=float(hit4[off].mean()), hit17=float(hit17[off].mean()),
                all_margin=float(margin.mean()), all_target_probability=float(target.mean()),
                all_hit4=float(hit4.mean()), all_hit17=float(hit17.mean()))


def run_policy(run, static, arrays, execution):
    seed, condition = run["seed"], run["condition"]
    if condition not in CONDITIONS:
        return None
    run_dir = Path(execution) / f"seed_{seed}_{condition}"
    checkpoint = run_dir / "checkpoint_6000.npz"
    networks = core.load_networks(checkpoint)
    map_name, live, maps = parse_condition(condition)
    require(live and map_name in ("identity", "public", "private"), "Probe requires a live codebook arm")
    meta, content_arrays = content_dataset.make_static()
    rows = content_dataset.flatten_rows(content_arrays)
    rows["group_layer_index"] = content_arrays["group_layer_index"]
    row_count = len(rows["host_indices"])
    part = static["partitions"][TARGET]
    host_states = pack_states(part, rows["host_indices"])
    host_x = observations(host_states)
    host_endpoint_npz = np.load(endpoint_path(run_dir, run), allow_pickle=False)
    host_internal = host_endpoint_npz["messages"][rows["host_indices"]]
    host_wire = host_endpoint_npz["wire_messages"][rows["host_indices"]]
    host_endpoint_npz.close()
    require(np.array_equal(host_internal[:, 0], host_internal[:, 0]), "Host message load failed")
    sender = rows["senders"].astype(np.int8)
    alternate = {
        "within_sender": sender,
        "cross_listener": rows["listeners"].astype(np.int8),
        "cross_third": (3 - sender - rows["listeners"]).astype(np.int8),
    }
    outputs = {}
    packet_hashes = {}
    for mode in MODES:
        alt = alternate[mode]
        donor_states = host_states.copy()
        if mode != "within_sender":
            # Packed states store the three needs in columns 0..2, not as a
            # separate actor axis.  Swap the two need columns row by row.
            for row_index in range(len(donor_states)):
                left, right = int(sender[row_index]), int(alt[row_index])
                donor_states[row_index, left], donor_states[row_index, right] = (
                    host_states[row_index, right], host_states[row_index, left]
                )
        donor_x = observations(donor_states)
        _donor_internal, donor_wire = greedy_rollout(networks, donor_x, True, maps)
        donor_packet = donor_wire[np.arange(len(donor_wire)), 0, alt, :]
        probabilities = intervene(networks, host_x, host_wire[:, 0], sender, donor_packet, maps)
        outputs[mode] = compact_metrics(probabilities, rows)
        packet_hashes[mode] = core.array_sha(donor_packet)
        if mode == "within_sender":
            # Replacing a packet with the same sender's own codebook is a
            # within-agent content control; it should not depend on cross-agent
            # codebook alignment.
            require(np.isfinite(probabilities).all(), "Non-finite within-sender probe")
    return dict(seed=seed, condition=condition, map_name=map_name, live=live,
                checkpoint_sha256=sha(checkpoint), groups=48, backgrounds=36, rows=row_count,
                outputs=outputs, donor_packet_sha256=packet_hashes,
                control_definition="within_sender uses the same agent's wire map; cross_listener swaps sender/listener need slots; cross_third swaps sender/third need slots and pastes the alternate agent's packet into the host sender slot.")


def stats(values):
    values = np.asarray(values, dtype=np.float64); mean = float(values.mean()); sd = float(values.std(ddof=1)); half = 2.3646242510102993 * sd / np.sqrt(len(values))
    return dict(n=len(values), mean=mean, sample_sd=sd, ci95_lower=mean-half, ci95_upper=mean+half)


def run_probe(out):
    out = Path(out).resolve(); execution = out / "execution"; result = read(execution / "results.json"); static = read(out / "prepared.json")
    meta, content_arrays = content_dataset.make_static(); rows = content_dataset.flatten_rows(content_arrays)
    outputs = []
    started = time.perf_counter()
    for run in result["runs"]:
        item = run_policy(run, static, content_arrays, execution)
        if item is not None:
            outputs.append(item)
            print(json.dumps(dict(seed=run["seed"], condition=run["condition"], elapsed_seconds=time.perf_counter()-started), ensure_ascii=False), flush=True)
    require(len(outputs) == len(SEEDS) * len(CONDITIONS), "Incomplete live probe grid")
    by = {(row["seed"], row["condition"]): row for row in outputs}
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for condition in CONDITIONS:
            for metric in ("M", "off_diagonal_margin", "target_probability", "hit4", "hit17"):
                summary[mode].setdefault(metric, {})[condition] = stats([by[(seed, condition)]["outputs"][mode][metric] for seed in SEEDS])
    contrasts = {}
    for mode in ("cross_listener", "cross_third"):
        for metric in ("M", "off_diagonal_margin", "target_probability", "hit4", "hit17"):
            public = np.asarray([by[(seed, "public_live")]["outputs"][mode][metric] for seed in SEEDS])
            private = np.asarray([by[(seed, "private_live")]["outputs"][mode][metric] for seed in SEEDS])
            identity = np.asarray([by[(seed, "identity_live")]["outputs"][mode][metric] for seed in SEEDS])
            contrasts[f"{mode}:{metric}"] = dict(public_minus_private=stats(public-private), public_minus_identity=stats(public-identity), private_minus_identity=stats(private-identity), paired_values=dict(public_minus_private=(public-private).tolist()))
    report = dict(schema="triadic_public_codebook_cross_sender_probe_v1", source_plan_sha256=result["plan_sha256"],
                  dataset_schema=meta["schema"], groups=48, backgrounds=36, modes=list(MODES), conditions=list(CONDITIONS),
                  rows_per_policy=outputs[0]["rows"], outputs=outputs, summary=summary, contrasts=contrasts,
                  elapsed_seconds=time.perf_counter()-started,
                  claim_boundary="This probe measures four-choice action probability after a causal packet replacement. Cross-sender gains are evidence about codebook compatibility under this task; they are not lexical meaning, compositional syntax, or human-language origin.")
    path = execution / "cross_sender_probe.json"; write(path, report)
    write(execution / "cross_sender_probe_receipt.json", dict(status="completed", report_sha256=sha(path), policies=len(outputs), model_updates=0, elapsed_seconds=report["elapsed_seconds"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    args = parser.parse_args(); report = run_probe(args.out)
    print(json.dumps({"schema": report["schema"], "policies": len(report["outputs"]), "contrasts": report["contrasts"]}, ensure_ascii=False))
