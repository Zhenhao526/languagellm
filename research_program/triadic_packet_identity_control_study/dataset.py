"""Static maps for endpoint and cross-group packet identity controls."""
from __future__ import annotations

import json
import platform
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np

from research_program.triadic_content_response_study import dataset as content

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"
SCHEMA = "triadic_packet_identity_control_v1"
MODES = ("endpoint_cycle", "cross_group_cycle")


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


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path); require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(compact(value) + "\n", encoding="utf8")


def source_arrays():
    plan = read(CONTENT / "plan.json"); prepared = read(CONTENT / "prepared.json")
    freeze = read(CONTENT / "freeze.json")
    require(sha(CONTENT / "plan.json") == freeze["plan_sha256"] == "9fe83d350a20e186657014688d5289bbaf2fefb394b6247ccd877927e0e22119", "Content probe plan changed")
    require(sha(CONTENT / "prepared.json") == freeze["prepared_sha256"] == "8bdae8428341ab0681c3e6b13fe69161192651c85b38193b6340e1e2aec45805", "Content probe cases changed")
    with np.load(CONTENT / "cases.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    require(sha(CONTENT / "cases.npz") == plan["cases_sha256"] == "72926a1d115f59002c8f0c99da999ffa92b541ababbac9556e0ddd91425d781d", "Content case array hash")
    return plan, prepared, arrays


def make_maps():
    _plan, prepared, arrays = source_arrays()
    G, B = arrays["world_indices"].shape[:2]
    row = content.flatten_rows(arrays)
    partner = np.empty(G, dtype=np.int16)
    for layer in range(12):
        groups = np.flatnonzero(arrays["group_layer_index"] == layer)
        require(len(groups) == 4, "Each layer must contain four groups")
        for i, g in enumerate(groups):
            partner[g] = groups[(i + 1) % len(groups)]
    maps = {}
    for mode in MODES:
        if mode == "endpoint_cycle":
            source_group = row["group_index"].copy()
            source_endpoint = ((row["donor_endpoint"].astype(np.int16) + 1) % 4)
        else:
            source_group = partner[row["group_index"]]
            source_endpoint = row["donor_endpoint"].copy()
        donor_indices = arrays["world_indices"][source_group, row["background_index"], source_endpoint]
        maps[mode] = dict(source_group_index=source_group, source_packet_endpoint=source_endpoint,
                           donor_indices=donor_indices)
    metadata = dict(schema=SCHEMA, source_content_plan_sha256=sha(CONTENT / "plan.json"),
                    source_content_prepared_sha256=sha(CONTENT / "prepared.json"),
                    source_content_cases_sha256=sha(CONTENT / "cases.npz"), modes=list(MODES),
                    rows_per_policy_time=len(row["group_index"]), groups=48, backgrounds=36,
                    host_donor_cells=16, off_diagonal_rows=int(np.sum(row["host_endpoint"] != row["donor_endpoint"])),
                    mode_definition={
                        "endpoint_cycle": "donor packet endpoint d is replaced by d+1 within the same group/background",
                        "cross_group_cycle": "donor packet comes from the next group in the same sender-listener×attribute layer, same endpoint/background",
                    },
                    group_partner=partner.tolist(), group_layer_index=arrays["group_layer_index"].tolist(),
                    control_target="host group candidate action for endpoint d remains the target; only the packet source changes",
                    selection="deterministic static cyclic maps; no policy output, checkpoint, or result is read")
    return metadata, maps


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), "Never overwrite control preparation")
    metadata, maps = make_maps(); out.mkdir(parents=True)
    with (out / "maps.npz").open("xb") as stream:
        payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
        np.savez_compressed(stream, **payload)
    metadata["map_shapes"] = {k: list(v.shape) for k, v in payload.items()}
    metadata["maps_sha256"] = sha(out / "maps.npz")
    write(out / "prepared.json", metadata)
    sources = {str(p.resolve()): sha(p) for p in (HERE / "__init__.py", HERE / "dataset.py", HERE / "intervene.py", HERE / "metrics.py", HERE / "runner.py", HERE / "audit.py")}
    inputs = {str(p.resolve()): sha(p) for p in (CONTENT / "plan.json", CONTENT / "prepared.json", CONTENT / "freeze.json", CONTENT / "cases.npz")}
    for path in sources:
        source = Path(path); target = out / "source_snapshot" / source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    config = dict(schema=SCHEMA, modes=list(MODES), groups=48, backgrounds=36, endpoints=4, host_donor_cells=16,
                  first_window_only=True, information="PL", no_training=True,
                  execution_steps=[0, 100, 500, 1500, 3000, 6000],
                  primary="endpoint_cycle_same_minus_permuted_content_margin")
    write(out / "plan.json", dict(status="prepared_without_policy_reads", created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                                  config=config, runtime=dict(python=platform.python_version(), numpy=np.__version__), source_sha256=sources,
                                  input_sha256=inputs, prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz"),
                                  neural_forward_samples=0, training_updates=0))
    write(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz")))
    verify(out)
    receipt = dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), maps_sha256=sha(out / "maps.npz"), neural_forward_samples=0, training_updates=0)
    write(out / "receipt.json", receipt)
    return receipt


def load_maps(path):
    with np.load(Path(path) / "maps.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def verify(out):
    out = Path(out).resolve(); plan = read(out / "plan.json"); prepared = read(out / "prepared.json"); freeze = read(out / "freeze.json")
    require(plan["status"] == "prepared_without_policy_reads", "Control plan status")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Control plan hash chain")
    require(sha(out / "prepared.json") == plan["prepared_sha256"] == freeze["prepared_sha256"], "Control prepared hash chain")
    require(sha(out / "maps.npz") == plan["maps_sha256"] == freeze["maps_sha256"] == prepared["maps_sha256"], "Control map hash chain")
    for path, digest in plan["source_sha256"].items():
        snap = out / "source_snapshot" / Path(path).relative_to(ROOT); require(sha(path) == digest and sha(snap) == digest, "Control source snapshot changed")
    for path, digest in plan["input_sha256"].items(): require(sha(path) == digest, "Control input changed")
    expected, maps = make_maps(); require(prepared == dict(expected, map_shapes=prepared["map_shapes"], maps_sha256=prepared["maps_sha256"]), "Control static metadata changed")
    actual = load_maps(out); expected_payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
    require(set(actual) == set(expected_payload), "Control map keys")
    for key, value in expected_payload.items(): require(np.array_equal(actual[key], value), "Control map changed: " + key)
    return plan, prepared, actual


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify")); parser.add_argument("--out", required=True); args = parser.parse_args()
    value = prepare(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=sha(Path(args.out).resolve() / "plan.json"))
    print(json.dumps(value, ensure_ascii=False))
