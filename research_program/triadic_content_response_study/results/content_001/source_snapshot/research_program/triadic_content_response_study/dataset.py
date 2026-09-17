"""Static four-choice content cases.

Cases are selected from the already frozen held-out need partition.  No policy
record, checkpoint, reward output, or learned message is read here.  A case
holds the listener's need fixed, varies the sender's single-valued need across
four values, and fixes the third actor's need.  Each endpoint has one physical
full-success plan and the four listener actions are distinct.
"""
from __future__ import annotations

from itertools import product, permutations
from pathlib import Path
from hashlib import sha256
import json
import platform
import shutil

import numpy as np

from research_program.triadic_action_dependency_study import environment as env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_action_dependency_study/results/context_001"
PARTITION = "new_needs_and_layouts"
AGENTS = (0, 1, 2)
OWNERS = tuple(permutations((1, 2, 3)))
RANK_PREFIX = "triadic_content_response_third_rank_v1"
SCHEMA = "triadic_content_response_dataset_v1"

# Four semantic layers.  The two receiver anchors accept both destinations and
# two materials; sender values are single-valued and determine four actions.
FAMILIES = (
    dict(family="kind", anchor_label="short", receiver_need=8,
         donor_needs=(12, 13, 18, 19)),
    dict(family="kind", anchor_label="long", receiver_need=11,
         donor_needs=(15, 16, 21, 22)),
    dict(family="length", anchor_label="wood", receiver_need=2,
         donor_needs=(12, 13, 15, 16)),
    dict(family="length", anchor_label="fiber", receiver_need=5,
         donor_needs=(18, 19, 21, 22)),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(compact(value) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def plan_action_indices(plan):
    i, j, site, destination = map(int, plan)
    result = [0, 0, 0]
    for who, partner in ((i, j), (j, i)):
        result[who] = 1 + site * 4 + destination * 2 + [a for a in AGENTS if a != who].index(partner)
    return np.asarray(result, dtype=np.int16)


def static_input_hashes():
    paths = (SOURCE / "prepared.json", SOURCE / "plan.json", SOURCE / "freeze.json",
             HERE / "__init__.py", HERE / "dataset.py", HERE / "intervene.py",
             HERE / "metrics.py", HERE / "runner.py", HERE / "audit.py")
    return {str(p.resolve()): sha(p) for p in paths}


def _source():
    prepared = read_json(SOURCE / "prepared.json")
    part = prepared["partitions"][PARTITION]
    require(part["state_order"] == "need-major, then layout, then owner", "Unexpected source order")
    needs = [tuple(map(int, row)) for row in part["needs"]]
    layouts = [tuple(map(int, row)) for row in part["layouts"]]
    owners = [tuple(map(int, row)) for row in part["private_sites"]]
    require(len(needs) == 1488 and len(layouts) == 6 and owners == list(OWNERS), "Held-out source shape changed")
    require(all(tuple(sorted(x)) == (0, 1, 2, 3) for x in layouts), "Invalid layouts")
    return prepared, part, needs, layouts, owners


def _eligible_thirds(lookup, family):
    anchor = family["receiver_need"]
    donors = family["donor_needs"]
    eligible = []
    for third in range(24):
        canonical = (anchor, *donors[:1], third)
        if canonical not in lookup:
            continue
        if not all((anchor, donor, third) in lookup for donor in donors):
            continue
        plans = [env.full_success_plans((anchor, donor, third)) for donor in donors]
        if not all(len(p) == 1 and tuple(sorted(p[0][:2])) == (0, 1) for p in plans):
            continue
        listener_actions = [int(plan_action_indices(p[0])[0]) for p in plans]
        if len(set(listener_actions)) != 4:
            continue
        rank = sha256((RANK_PREFIX + "|" + family["family"] + "|" +
                       family["anchor_label"] + "|" + str(third)).encode()).hexdigest()
        eligible.append((rank, third))
    eligible.sort()
    return [third for _, third in eligible]


def make_static():
    prepared, part, needs, layouts, owners = _source()
    lookup = {need: i for i, need in enumerate(needs)}
    groups = []
    family_records = []
    for family_index, family in enumerate(FAMILIES):
        eligible = _eligible_thirds(lookup, family)
        require(len(eligible) >= 2, "Fewer than two static third-needs for " + family["family"])
        selected = eligible[:2]
        family_records.append(dict(family_index=family_index, **family,
                                   all_static_eligible_thirds=eligible,
                                   selected_thirds=selected))
        for sender, listener in product(AGENTS, repeat=2):
            if sender == listener:
                continue
            third_actor = next(a for a in AGENTS if a not in (sender, listener))
            for third in selected:
                endpoint_tuples = []
                endpoint_need_indices = []
                for donor in family["donor_needs"]:
                    row = [None, None, None]
                    row[listener] = family["receiver_need"]
                    row[sender] = donor
                    row[third_actor] = third
                    need_tuple = tuple(row)
                    require(need_tuple in lookup, "Endpoint need tuple missing from frozen partition")
                    endpoint_tuples.append(list(need_tuple))
                    endpoint_need_indices.append(lookup[need_tuple])
                pair_index = sum(1 for a in AGENTS for b in AGENTS if a != b and (a, b) < (sender, listener))
                groups.append(dict(group_index=len(groups), layer_index=(pair_index * 2 + (0 if family["family"] == "kind" else 1)),
                                   family_index=family_index, family=family["family"],
                                   anchor_label=family["anchor_label"], sender=sender,
                                   listener=listener, third_actor=third_actor,
                                   receiver_need=family["receiver_need"], donor_needs=list(family["donor_needs"]),
                                   third_need=third, endpoint_needs=endpoint_tuples,
                                   endpoint_need_indices=endpoint_need_indices,
                                   endpoint_order="donor_needs order; four values, then layout, owner"))
    require(len(groups) == 48 and len(family_records) == 4, "Expected exactly 48 groups")

    nl, no = len(layouts), len(owners)
    world_indices = np.empty((len(groups), nl * no, 4), dtype=np.int64)
    candidate_actions = np.empty((len(groups), nl * no, 4, 3), dtype=np.int16)
    for g, group in enumerate(groups):
        for li, layout in enumerate(layouts):
            for oi, _owner in enumerate(owners):
                b = li * no + oi
                for endpoint, (need_tuple, need_index) in enumerate(zip(group["endpoint_needs"], group["endpoint_need_indices"])):
                    world_indices[g, b, endpoint] = (need_index * nl + li) * no + oi
                    plans = env.full_success_plans(tuple(need_tuple), layout)
                    require(len(plans) == 1, "Endpoint is not a unique full-success world")
                    plan = plans[0]
                    require(tuple(sorted(plan[:2])) == tuple(sorted((group["sender"], group["listener"]))),
                            "Third actor entered the full-success pair")
                    candidate_actions[g, b, endpoint] = plan_action_indices(plan)
                receiver = candidate_actions[g, b, :, group["listener"]]
                require(len(set(map(int, receiver))) == 4, "Four candidate receiver actions are not distinct")

    arrays = dict(
        world_indices=world_indices,
        candidate_actions=candidate_actions,
        candidate_receiver_actions=np.asarray(
            [candidate_actions[g, :, :, groups[g]["listener"]] for g in range(len(groups))], dtype=np.int16),
        group_sender=np.asarray([g["sender"] for g in groups], dtype=np.int8),
        group_listener=np.asarray([g["listener"] for g in groups], dtype=np.int8),
        group_third_actor=np.asarray([g["third_actor"] for g in groups], dtype=np.int8),
        group_family_index=np.asarray([g["family_index"] for g in groups], dtype=np.int8),
        group_layer_index=np.asarray([g["layer_index"] for g in groups], dtype=np.int8),
        endpoint_need_indices=np.asarray([g["endpoint_need_indices"] for g in groups], dtype=np.int64),
        background_layout_index=np.repeat(np.arange(nl, dtype=np.int16), no),
        background_owner_index=np.tile(np.arange(no, dtype=np.int16), nl),
    )
    metadata = dict(schema=SCHEMA, source_partition=PARTITION,
                    source_prepared_sha256=sha(SOURCE / "prepared.json"),
                    source_plan_sha256=sha(SOURCE / "plan.json"),
                    source_freeze_sha256=sha(SOURCE / "freeze.json"),
                    source_world_count=int(part["world_count"]),
                    layouts=[list(x) for x in layouts], private_sites=[list(x) for x in owners],
                    world_count=int(part["world_count"]), group_count=len(groups), background_count=nl * no,
                    endpoint_count=4, host_donor_cell_count=16, off_diagonal_cell_count=12,
                    family_records=family_records, groups=groups,
                    selection="static SHA-ranked eligible third-needs; no policy output or checkpoint read",
                    candidate_definition="receiver action for donor endpoint minus max of the other three receiver actions; all17 action probabilities retained",
                    array_shapes={k: list(v.shape) for k, v in arrays.items()},
                    source_schema=prepared["schema"])
    # JSON is the frozen interchange format; normalize tuples before hashing
    # so a freshly rebuilt static design compares byte-for-byte after reload.
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return metadata, arrays


def pack_states(part, indices=None):
    needs = np.asarray(part["needs"], dtype=np.int16)
    layouts = np.asarray(part["layouts"], dtype=np.int16)
    owners = np.asarray(part["private_sites"], dtype=np.int16)
    nl, no = len(layouts), len(owners)
    if indices is None:
        indices = np.arange(int(part["world_count"]), dtype=np.int64)
    ids = np.asarray(indices, dtype=np.int64)
    require(ids.ndim == 1 and np.all((ids >= 0) & (ids < len(needs) * nl * no)), "Invalid state indices")
    return np.concatenate((needs[ids // (nl * no)], layouts[(ids // no) % nl], owners[ids % no]), axis=1)


def flatten_rows(arrays):
    """Canonical rows in group, background, host endpoint, donor endpoint order."""
    g_count, b_count = arrays["world_indices"].shape[:2]
    g = np.repeat(np.arange(g_count, dtype=np.int64), b_count * 16)
    b = np.tile(np.repeat(np.arange(b_count, dtype=np.int64), 16), g_count)
    host = np.tile(np.repeat(np.arange(4, dtype=np.int8), 4), g_count * b_count)
    donor = np.tile(np.arange(4, dtype=np.int8), g_count * b_count * 4)
    require(len(g) == g_count * b_count * 16, "Row expansion shape")
    return dict(group_index=g, background_index=b, host_endpoint=host, donor_endpoint=donor,
                host_indices=arrays["world_indices"][g, b, host], donor_indices=arrays["world_indices"][g, b, donor],
                senders=arrays["group_sender"][g], listeners=arrays["group_listener"][g],
                candidate_receiver_actions=arrays["candidate_receiver_actions"][g, b],
                target_receiver_actions=arrays["candidate_receiver_actions"][g, b, donor])


def save_static(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite static preparation")
    metadata, arrays = make_static()
    out.mkdir(parents=True)
    npz_path = out / "cases.npz"
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["arrays_sha256"] = sha(npz_path)
    write_json(out / "prepared.json", metadata)
    source_hashes = static_input_hashes()
    for path in source_hashes:
        source = Path(path)
        target = out / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    inputs = {str(p.resolve()): sha(p) for p in (SOURCE / "prepared.json", SOURCE / "plan.json", SOURCE / "freeze.json")}
    config = dict(schema=SCHEMA, partition=PARTITION, groups=48, backgrounds=36, endpoints=4,
                  host_donor_cells=16, off_diagonal_cells=12, selected_thirds_per_anchor=2,
                  no_training=True, first_window_only=True, information="PL",
                  source_formation_plan_sha256="b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f",
                  execution_steps=[0, 100, 500, 1500, 3000, 6000],
                  primary="endpoint_four_choice_content_margin_rule_by_communication_interaction")
    plan = dict(status="prepared_without_policy_reads", created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                config=config, runtime=dict(python=platform.python_version(), numpy=np.__version__),
                source_sha256=source_hashes, input_sha256=inputs,
                prepared_sha256=sha(out / "prepared.json"), cases_sha256=sha(npz_path),
                neural_forward_samples=0, training_updates=0)
    write_json(out / "plan.json", plan)
    write_json(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), cases_sha256=sha(npz_path)))
    verify(out)
    receipt = dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"),
                   cases_sha256=sha(npz_path), groups=48, backgrounds=36, neural_forward_samples=0, training_updates=0)
    write_json(out / "receipt.json", receipt)
    return receipt


def load_cases(path):
    with np.load(Path(path) / "cases.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays


def verify(out):
    out = Path(out).resolve()
    plan = read_json(out / "plan.json")
    prepared = read_json(out / "prepared.json")
    freeze = read_json(out / "freeze.json")
    require(plan["status"] == "prepared_without_policy_reads", "Static plan status changed")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Static plan hash chain")
    require(sha(out / "prepared.json") == plan["prepared_sha256"] == freeze["prepared_sha256"], "Static prepared hash chain")
    require(sha(out / "cases.npz") == plan["cases_sha256"] == freeze["cases_sha256"], "Static array hash chain")
    require(plan["source_sha256"] == static_input_hashes(), "Static source changed")
    for path, digest in plan["source_sha256"].items():
        snap = out / "source_snapshot" / Path(path).relative_to(ROOT)
        require(sha(path) == digest and sha(snap) == digest, "Static source snapshot changed")
    for path, digest in plan["input_sha256"].items():
        require(sha(path) == digest, "Static input changed")
    expected, arrays = make_static()
    require({k: list(v.shape) for k, v in arrays.items()} == prepared["array_shapes"], "Static array shape metadata")
    require(prepared == dict(expected, arrays_sha256=prepared["arrays_sha256"]), "Static metadata changed")
    with np.load(out / "cases.npz", allow_pickle=False) as z:
        require(set(z.files) == set(arrays), "Static array keys changed")
        for key, value in arrays.items():
            require(np.array_equal(z[key], value), "Static array changed: " + key)
    return plan, prepared, arrays


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = save_static(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=verify(args.out)[0]["prepared_sha256"])
    print(json.dumps(result, ensure_ascii=False))
