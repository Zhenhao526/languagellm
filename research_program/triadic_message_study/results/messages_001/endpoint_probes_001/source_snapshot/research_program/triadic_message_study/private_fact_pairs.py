"""Outcome-independent, researcher-only PI counterfactual pairs. No model calls.

Each pair exchanges exactly the other two people's privately observed materials.
All official needs, site ownership, public material, and listener observation
remain unchanged. Never supply this file's witnesses/sets/ids to a policy.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

from research_program.triadic_task import environment as env

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENV_SHA = "b2d49ab829edc92e77df527f5b470602c289cdf3ad7728d5e71f411c53582ef6"
BASE_SHA = "1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def pack(state):
    return state.needs + state.layout + state.private_sites


def unpack(packed):
    require(len(packed) == 10, "Packed state must have ten integers")
    return env.State(tuple(packed[:3]), tuple(packed[3:7]), tuple(packed[7:]))


def joint_profiles():
    """Independent construction of all 24 nonzero-execution configurations."""
    menus = [env.all_actions(a) for a in env.AGENTS]
    result = []
    for i, j in combinations(range(3), 2):
        for site, destination in product(env.SITES, env.DESTINATIONS):
            indices = [0, 0, 0]
            for actor, partner in ((i, j), (j, i)):
                indices[actor] = menus[actor].index(dict(kind="transport", site=site,
                    destination=destination, partner=env.AGENTS[partner]))
            result.append(tuple(indices))
    require(len(set(result)) == 24, "Incomplete structural support")
    return tuple(result)


PROFILES = joint_profiles()
MENUS = tuple(env.all_actions(a) for a in env.AGENTS)


def independent_accepts(need, material, destination):
    """Semantic definition, not env.accepts or the training reward tensor."""
    factor, places = divmod(need, 3)
    compatible_materials = ((0, 1), (2, 3), (0, 2), (1, 3))[factor]
    return material in compatible_materials and (places == 2 or places == destination)


@lru_cache(maxsize=None)
def fullsuccess_projections(needs, layout):
    """Action0 (wait) is included if some R1 solution lets that actor wait."""
    projected = [set(), set(), set()]
    count = 0
    for profile in PROFILES:
        satisfied = 0
        for actor, index in enumerate(profile):
            if index:
                action = MENUS[actor][index]
                material = layout[env.SITES.index(action["site"])]
                destination = env.DESTINATIONS.index(action["destination"])
                satisfied += independent_accepts(needs[actor], material, destination)
        if satisfied == 2:
            count += 1
            for actor, index in enumerate(profile):
                projected[actor].add(index)
    require(count > 0, "No full-success configuration in official support")
    return tuple(tuple(sorted(x)) for x in projected)


def swapped_private_state(state, listener):
    require(listener in range(3), "Invalid listener")
    others = [a for a in range(3) if a != listener]
    sites = [state.private_sites[a] for a in others]
    layout = list(state.layout)
    layout[sites[0]], layout[sites[1]] = layout[sites[1]], layout[sites[0]]
    return env.State(state.needs, tuple(layout), state.private_sites)


def build_pairs(prepared=None):
    """Return full partition definitions, rows, counts. No IO or neural forward.

    Caller may supply an equivalent official prepared dictionary for testing.
    The default reads no run results; base.make_prepared() is purely static.
    """
    if prepared is None:
        from research_program.triadic_learning_baseline import runner as base
        require(sha(env.__file__) == ENV_SHA and sha(base.__file__) == BASE_SHA,
                "Official source differs from designated immutable task")
        prepared = base.make_prepared()
    rows, counts = [], {}
    for name, spec in prepared["partitions"].items():
        packed = [tuple(n) + tuple(l) + tuple(p)
                  for n, l, p in product(spec["needs"], spec["layouts"], spec["private_sites"])]
        require(len(packed) == spec["world_count"] == len(set(packed)), "Invalid partition")
        lookup = {state: index for index, state in enumerate(packed)}
        counts[name] = {}
        for listener, agent in enumerate(env.AGENTS):
            c = dict(worlds=len(packed), swapped_partner_outside_partition=0,
                     within_partition_unordered_pairs=0, overlapping_action_sets=0,
                     retained_pairs=0)
            for i, current in enumerate(packed):
                state = unpack(current)
                other = swapped_private_state(state, listener)
                alternate = pack(other)
                if alternate not in lookup:
                    c["swapped_partner_outside_partition"] += 1
                    continue
                if current > alternate:
                    continue
                require(current < alternate, "Swap did not change world")
                c["within_partition_unordered_pairs"] += 1
                low_set = fullsuccess_projections(state.needs, state.layout)[listener]
                high_set = fullsuccess_projections(other.needs, other.layout)[listener]
                if set(low_set).intersection(high_set):
                    c["overlapping_action_sets"] += 1
                    continue
                low_view = env.observe(state, agent, shared_needs=False)
                high_view = env.observe(other, agent, shared_needs=False)
                require(low_view == high_view, "Counterfactual leaked into listener observation")
                require(state.needs == other.needs and state.private_sites == other.private_sites
                        and state.layout[0] == other.layout[0]
                        and state.layout[state.private_sites[listener]] == other.layout[other.private_sites[listener]],
                        "A supposedly fixed private-fact component changed")
                require(0 not in low_set and 0 not in high_set,
                        "Invariant needs should preclude a disjoint pair with a waiting solution")
                rows.append(dict(partition=name, listener=agent, listener_index=listener,
                    row_index_low=i, row_index_high=lookup[alternate],
                    state_low=list(current), state_high=list(alternate),
                    listener_PI_observation_sha256=object_sha(low_view),
                    fullsuccess_actions_low=list(low_set), fullsuccess_actions_high=list(high_set)))
                c["retained_pairs"] += 1
            require(2*c["within_partition_unordered_pairs"] + c["swapped_partner_outside_partition"]
                    == len(packed), "Swap involution denominator mismatch")
            require(c["retained_pairs"] + c["overlapping_action_sets"]
                    == c["within_partition_unordered_pairs"], "Unclassified candidate pair")
            counts[name][agent] = c
    rows.sort(key=lambda r: (list(prepared["partitions"]).index(r["partition"]),
                            r["listener_index"], r["state_low"], r["state_high"]))
    for i, row in enumerate(rows):
        row["pair_id"] = f"private_swap_{i:06d}"
    return dict(schema="triadic_private_fact_pairs_v1", partitions=prepared["partitions"],
                profiles=[list(p) for p in PROFILES], rows=rows, counts=counts,
                total_pairs=len(rows), selection_uses_model_or_results=False,
                scientific_scope="PI identical-listener-observation local-action necessity; two other private facts co-vary")


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite private-fact manifest")
    result = build_pairs()
    output.mkdir(parents=True)
    for name, value in (("pairs.json", result["rows"]), ("counts.json", result["counts"]),
                        ("partitions.json", result["partitions"])):
        with (output/name).open("xb") as stream:
            stream.write(json_bytes(value))
    manifest = {k: v for k, v in result.items() if k not in ("rows", "counts", "partitions")}
    source_paths = [Path(__file__), Path(env.__file__),
                    ROOT/"research_program/triadic_learning_baseline/runner.py"]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    manifest["files_sha256"] = {name: sha(output/name) for name in ("pairs.json", "counts.json", "partitions.json")}
    manifest["neural_forward_calls"] = 0
    with (output/"manifest.json").open("xb") as stream:
        stream.write(json_bytes(manifest))
    return dict(output=str(output), total_pairs=len(result["rows"]), counts=result["counts"],
                manifest_sha256=sha(output/"manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), ensure_ascii=False, indent=2))
