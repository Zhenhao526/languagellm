"""Static transformations for symbol-identity and position controls."""
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
SCHEMA = "triadic_symbol_recode_v1"
MODES = ("global_symbol_permutation", "position_rotation", "position_reverse")
SYMBOL_PERMUTATION = (3, 7, 1, 6, 0, 4, 2, 5)
POSITION_ORDERS = {
    "global_symbol_permutation": (0, 1, 2, 3),
    "position_rotation": (1, 2, 3, 0),
    "position_reverse": (3, 2, 1, 0),
}


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
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(compact(value) + "\n", encoding="utf8")


def source_arrays():
    plan = read(CONTENT / "plan.json")
    prepared = read(CONTENT / "prepared.json")
    freeze = read(CONTENT / "freeze.json")
    require(sha(CONTENT / "plan.json") == freeze["plan_sha256"] == "9fe83d350a20e186657014688d5289bbaf2fefb394b6247ccd877927e0e22119", "Content plan changed")
    require(sha(CONTENT / "prepared.json") == freeze["prepared_sha256"] == "8bdae8428341ab0681c3e6b13fe69161192651c85b38193b6340e1e2aec45805", "Content preparation changed")
    with np.load(CONTENT / "cases.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    require(sha(CONTENT / "cases.npz") == plan["cases_sha256"] == "72926a1d115f59002c8f0c99da999ffa92b541ababbac9556e0ddd91425d781d", "Content cases changed")
    return plan, prepared, arrays


def make_maps():
    _plan, _prepared, arrays = source_arrays()
    rows = content.flatten_rows(arrays)
    groups = rows["group_index"]; backgrounds = rows["background_index"]; donor = rows["donor_endpoint"]
    donor_indices = arrays["world_indices"][groups, backgrounds, donor]
    maps = {}
    for mode in MODES:
        maps[mode] = dict(source_group_index=groups.copy(), source_packet_endpoint=donor.copy(), donor_indices=donor_indices.copy())
    metadata = dict(schema=SCHEMA, source_content_plan_sha256=sha(CONTENT / "plan.json"),
                    source_content_prepared_sha256=sha(CONTENT / "prepared.json"),
                    source_content_cases_sha256=sha(CONTENT / "cases.npz"), modes=list(MODES),
                    rows_per_policy_time=len(groups), groups=48, backgrounds=36, endpoints=4,
                    host_donor_cells=16, off_diagonal_rows=int(np.sum(rows["host_endpoint"] != donor)),
                    symbol_permutation=list(SYMBOL_PERMUTATION),
                    position_orders={mode: list(POSITION_ORDERS[mode]) for mode in MODES},
                    mode_definition={
                        "global_symbol_permutation": "apply one fixed bijection to every token in the donor packet",
                        "position_rotation": "replace packet [x0,x1,x2,x3] by [x1,x2,x3,x0]",
                        "position_reverse": "replace packet [x0,x1,x2,x3] by [x3,x2,x1,x0]",
                    },
                    selection="deterministic static recodings; no policy output, checkpoint, or result is read")
    return metadata, maps


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite recoding preparation")
    metadata, maps = make_maps(); out.mkdir(parents=True)
    payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
    with (out / "maps.npz").open("xb") as stream:
        np.savez_compressed(stream, **payload)
    metadata["map_shapes"] = {key: list(value.shape) for key, value in payload.items()}
    metadata["maps_sha256"] = sha(out / "maps.npz")
    write(out / "prepared.json", metadata)
    sources = {str(p.resolve()): sha(p) for p in (HERE / "__init__.py", HERE / "dataset.py", HERE / "intervene.py", HERE / "metrics.py", HERE / "runner.py", HERE / "audit.py")}
    inputs = {str(p.resolve()): sha(p) for p in (CONTENT / "plan.json", CONTENT / "prepared.json", CONTENT / "freeze.json", CONTENT / "cases.npz", CONTENT / "summary_001" / "summary.json")}
    for path in sources:
        source = Path(path); target = out / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    config = dict(schema=SCHEMA, modes=list(MODES), groups=48, backgrounds=36, endpoints=4,
                  host_donor_cells=16, first_window_only=True, information="PL", no_training=True,
                  execution_steps=[0, 100, 500, 1500, 3000, 6000],
                  primary="same_endpoint_live_M_minus_recoded_live_M_by_mode")
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
        return {key: z[key].copy() for key in z.files}


def verify(out):
    out = Path(out).resolve(); plan = read(out / "plan.json"); prepared = read(out / "prepared.json"); freeze = read(out / "freeze.json")
    require(plan["status"] == "prepared_without_policy_reads", "Recoding plan status")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Recoding plan hash chain")
    require(sha(out / "prepared.json") == plan["prepared_sha256"] == freeze["prepared_sha256"], "Recoding prepared hash chain")
    require(sha(out / "maps.npz") == plan["maps_sha256"] == freeze["maps_sha256"] == prepared["maps_sha256"], "Recoding map hash chain")
    for path, digest in plan["source_sha256"].items():
        snap = out / "source_snapshot" / Path(path).relative_to(ROOT)
        require(sha(path) == digest and sha(snap) == digest, "Recoding source snapshot changed")
    for path, digest in plan["input_sha256"].items():
        require(sha(path) == digest, "Recoding input changed")
    expected, maps = make_maps()
    require(prepared == dict(expected, map_shapes=prepared["map_shapes"], maps_sha256=prepared["maps_sha256"]), "Recoding metadata changed")
    actual = load_maps(out); expected_payload = {f"{mode}_{field}": value for mode, block in maps.items() for field, value in block.items()}
    require(set(actual) == set(expected_payload), "Recoding map keys")
    for key, value in expected_payload.items():
        require(np.array_equal(actual[key], value), "Recoding map changed: " + key)
    return plan, prepared, actual


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify")); parser.add_argument("--out", required=True); args = parser.parse_args()
    value = prepare(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=sha(Path(args.out).resolve() / "plan.json"))
    print(json.dumps(value, ensure_ascii=False))
