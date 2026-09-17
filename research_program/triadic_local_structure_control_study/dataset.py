"""Static identity maps and deterministic equality-pattern metadata."""
from __future__ import annotations

import json
import platform
import shutil
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

from research_program.triadic_content_response_study import dataset as content

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"
FORMATION = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
SCHEMA = "triadic_local_structure_control_v1"
MODES = ("single_slot_cycle", "rank_canonical", "equality_pattern_relabel")
CONTENT_PLAN_SHA = "9fe83d350a20e186657014688d5289bbaf2fefb394b6247ccd877927e0e22119"
CONTENT_PREPARED_SHA = "8bdae8428341ab0681c3e6b13fe69161192651c85b38193b6340e1e2aec45805"
CONTENT_CASES_SHA = "72926a1d115f59002c8f0c99da999ffa92b541ababbac9556e0ddd91425d781d"
FORMATION_PLAN_SHA = "b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f"
FORMATION_PREPARED_SHA = "92acde1b3e535990fdb120bb9dc19b1c5c2f6c7177d504182d2f6f687ffbe885"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path); require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(compact(value) + "\n", encoding="utf8")


def source_arrays():
    plan = read(CONTENT / "plan.json"); prepared = read(CONTENT / "prepared.json"); freeze = read(CONTENT / "freeze.json")
    require(sha(CONTENT / "plan.json") == CONTENT_PLAN_SHA == freeze["plan_sha256"], "Content plan changed")
    require(sha(CONTENT / "prepared.json") == CONTENT_PREPARED_SHA == freeze["prepared_sha256"], "Content preparation changed")
    require(sha(CONTENT / "cases.npz") == CONTENT_CASES_SHA == freeze["cases_sha256"], "Content cases changed")
    with np.load(CONTENT / "cases.npz", allow_pickle=False) as z: arrays = {key: z[key].copy() for key in z.files}
    return plan, prepared, arrays


def equality_signature(packet):
    labels = {}; next_label = 0; signature = []
    for value in map(int, packet):
        if value not in labels: labels[value] = next_label; next_label += 1
        signature.append(labels[value])
    return tuple(signature)


def all_signatures():
    seen = set(); out = []
    for raw in product(range(4), repeat=4):
        sig = equality_signature(raw)
        if sig not in seen: seen.add(sig); out.append(sig)
    return tuple(sorted(out))


def pattern_mappings():
    result = {}
    for signature in all_signatures():
        k = max(signature) + 1
        ranked = sorted(range(8), key=lambda i: sha256(("triadic_local_pattern_v1|" + ",".join(map(str, signature)) + "|" + str(i)).encode()).hexdigest())
        result[signature] = tuple(ranked[:k])
    return result


PATTERN_MAPPINGS = pattern_mappings()


def make_maps():
    _plan, _prepared, arrays = source_arrays(); rows = content.flatten_rows(arrays)
    groups = rows["group_index"]; backgrounds = rows["background_index"]; donor = rows["donor_endpoint"]
    donor_indices = arrays["world_indices"][groups, backgrounds, donor]
    maps = {mode: dict(source_group_index=groups.copy(), source_packet_endpoint=donor.copy(), donor_indices=donor_indices.copy()) for mode in MODES}
    metadata = dict(schema=SCHEMA, modes=list(MODES), rows_per_policy_time=len(groups), groups=48, backgrounds=36, endpoints=4,
                    host_donor_cells=16, off_diagonal_rows=int(np.sum(rows["host_endpoint"] != donor)),
                    mode_definition={
                        "single_slot_cycle": "replace slot 0 by (x0+1) mod 8; other three tokens unchanged",
                        "rank_canonical": "map sorted distinct source symbols to 0,1,... while preserving equality and numeric rank",
                        "equality_pattern_relabel": "map first-occurrence equality classes by a fixed pattern-specific random label set; source identities and numeric rank removed",
                    },
                    equality_pattern_count=len(PATTERN_MAPPINGS),
                    pattern_mappings={",".join(map(str, key)): list(value) for key, value in PATTERN_MAPPINGS.items()},
                    source_content_plan_sha256=sha(CONTENT / "plan.json"), source_content_prepared_sha256=sha(CONTENT / "prepared.json"),
                    source_content_cases_sha256=sha(CONTENT / "cases.npz"),
                    selection="deterministic static identity maps; no policy output, checkpoint, or result is read")
    return metadata, maps


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), "Never overwrite local-control preparation")
    metadata, maps = make_maps(); out.mkdir(parents=True)
    payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
    with (out / "maps.npz").open("xb") as stream: np.savez_compressed(stream, **payload)
    metadata["map_shapes"] = {key: list(value.shape) for key, value in payload.items()}; metadata["maps_sha256"] = sha(out / "maps.npz")
    write(out / "prepared.json", metadata)
    source_paths = tuple(HERE / name for name in ("__init__.py", "dataset.py", "intervene.py", "metrics.py", "runner.py", "audit.py"))
    sources = {str(p.resolve()): sha(p) for p in source_paths}
    for path in sources:
        source = Path(path); target = out / "source_snapshot" / source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    inputs = {str(p.resolve()): sha(p) for p in (CONTENT / "plan.json", CONTENT / "prepared.json", CONTENT / "freeze.json", CONTENT / "cases.npz", CONTENT / "summary_001" / "summary.json", FORMATION / "plan.json", FORMATION / "prepared.json", FORMATION / "freeze.json")}
    config = dict(schema=SCHEMA, modes=list(MODES), control_count=len(MODES), groups=48, backgrounds=36, endpoints=4, host_donor_cells=16,
                  first_window_only=True, information="PL", no_training=True, execution_steps=[0, 100, 500, 1500, 3000, 6000],
                  primary="same_endpoint_live_M_minus_local_control_live_M_ensemble_and_by_mode", recoding="local packet controls only; message length and positions unchanged")
    write(out / "plan.json", dict(status="prepared_without_policy_reads", created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                                  config=config, runtime=dict(python=platform.python_version(), numpy=np.__version__), source_sha256=sources, input_sha256=inputs,
                                  prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz"), neural_forward_samples=0, training_updates=0))
    write(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz")))
    verify(out)
    receipt = dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz"), neural_forward_samples=0, training_updates=0)
    write(out / "receipt.json", receipt); return receipt


def load_maps(path):
    with np.load(Path(path) / "maps.npz", allow_pickle=False) as z: return {key: z[key].copy() for key in z.files}


def verify(out):
    out = Path(out).resolve(); plan = read(out / "plan.json"); prepared = read(out / "prepared.json"); freeze = read(out / "freeze.json")
    require(plan["status"] == "prepared_without_policy_reads", "Local-control plan status")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Local-control plan hash chain")
    require(sha(out / "prepared.json") == plan["prepared_sha256"] == freeze["prepared_sha256"], "Local-control prepared hash chain")
    require(sha(out / "maps.npz") == plan["maps_sha256"] == freeze["maps_sha256"] == prepared["maps_sha256"], "Local-control map hash chain")
    for path, digest in plan["source_sha256"].items():
        snap = out / "source_snapshot" / Path(path).relative_to(ROOT); require(sha(path) == digest and sha(snap) == digest, "Local-control source snapshot changed")
    for path, digest in plan["input_sha256"].items(): require(sha(path) == digest, "Local-control input changed: " + path)
    expected, maps = make_maps(); require(prepared == dict(expected, map_shapes=prepared["map_shapes"], maps_sha256=prepared["maps_sha256"]), "Local-control metadata changed")
    actual = load_maps(out); expected_payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
    require(set(actual) == set(expected_payload), "Local-control map keys")
    for key, value in expected_payload.items(): require(np.array_equal(actual[key], value), "Local-control map changed: " + key)
    return plan, prepared, actual


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify")); parser.add_argument("--out", required=True); args = parser.parse_args()
    value = prepare(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=sha(Path(args.out).resolve() / "plan.json"))
    print(json.dumps(value, ensure_ascii=False))
