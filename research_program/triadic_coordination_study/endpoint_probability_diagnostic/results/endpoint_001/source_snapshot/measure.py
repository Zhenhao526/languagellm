"""Frozen post-hoc endpoint probability diagnostics; no parameter loading or forward."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "research_program/triadic_learning_baseline/results/learning_001"
SEEDS = (43101, 43102, 43103, 43104)
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
PAIRS = ((0, 1), (0, 2), (1, 2))
QUANTILES = (0, .25, .5, .75, .9, .99, 1)
TOL = 2e-12


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def payload(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def write(path, value):
    Path(path).write_bytes(payload(value))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def action(who, partner, site, destination):
    return 1+4*site+2*destination+[i for i in range(3) if i != who].index(partner)


def structure():
    rows = []
    for i, j in PAIRS:
        for site, dest in product(range(4), range(2)):
            a = [0, 0, 0]
            a[i], a[j] = action(i, j, site, dest), action(j, i, site, dest)
            rows.append(a)
    result = np.asarray(rows, dtype=np.int64)
    for i, j in PAIRS:
        require(len(np.unique(result[:, [i, j]], axis=0)) == 24, "Pair sparse indices must be unique")
    return result


JOINT = structure()
PAIR_COORDS = np.stack([JOINT[:, [i, j]] for i, j in PAIRS])


def native_rewards(states):
    n = len(states)
    result = np.zeros((n, 24), dtype=np.float64)
    resource = np.asarray([[m in g for m in range(4)] for g in ((0, 1), (2, 3), (0, 2), (1, 3))])
    destination = np.asarray([[d in g for d in range(2)] for g in ((0,), (1,), (0, 1))])
    t = 0
    for i, j in PAIRS:
        for site, dest in product(range(4), range(2)):
            material = states[:, 3+site]
            for who in (i, j):
                need = states[:, who]
                result[:, t] += .5*(resource[need//3, material] & destination[need % 3, dest])
            t += 1
    return result


def responses(probabilities, reward):
    """All17 single values; all289 pair values losslessly represented by24 entries."""
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 3 and p.shape[1:] == (3, 17), "Probability shape")
    require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(-1), 1., atol=TOL, rtol=TOL), "Invalid probability")
    require(reward.shape == (len(p), 24) and np.isin(reward, [0., .5, 1.]).all(), "Native reward shape/values")
    selected = np.stack([p[:, i, JOINT[:, i]] for i in range(3)], axis=1)
    baseline = (reward*selected.prod(1)).sum(1)
    singles = np.zeros((len(p), 3, 17), dtype=np.float64)
    pairs = np.zeros((len(p), 3, 24), dtype=np.float64)
    for i in range(3):
        other = [j for j in range(3) if j != i]
        term = reward*selected[:, other[0]]*selected[:, other[1]]
        for t, index in enumerate(JOINT[:, i]):
            singles[:, i, index] += term[:, t]
    single_best = singles.max(-1)
    pair_best, pair_first, pair_ties = [], [], []
    for pair_id, (i, j) in enumerate(PAIRS):
        k = 3-i-j
        pairs[:, pair_id] = reward*selected[:, k]
        full = np.zeros((len(p), 289), dtype=np.float64)
        flat_ids = JOINT[:, i]*17+JOINT[:, j]
        full[:, flat_ids] = pairs[:, pair_id]
        pair_best.append(full.max(1)); pair_first.append(full.argmax(1))
        pair_ties.append((full == full.max(1, keepdims=True)).sum(1))
        reconstructed = (pairs[:, pair_id]*selected[:, i]*selected[:, j]).sum(1)
        require(np.allclose(reconstructed, baseline, atol=TOL, rtol=TOL), "Pair policy average fails to recover baseline")
    pair_best = np.stack(pair_best, axis=1)
    require(np.allclose((singles*p).sum(-1), baseline[:, None], atol=TOL, rtol=TOL), "Single policy average fails to recover baseline")
    require((single_best >= baseline[:, None]-TOL).all() and (pair_best >= baseline[:, None]-TOL).all(), "Negative best-response gain")
    for pair_id, (i, j) in enumerate(PAIRS):
        require((pair_best[:, pair_id] >= single_best[:, [i, j]].max(1)-TOL).all(), "Pair response below contained single response")
    oracle = reward.max(1)
    require((pair_best <= oracle[:, None]+TOL).all(), "Pair response exceeds joint oracle")
    return {"baseline": baseline, "single_action_values": singles, "pair_action_values_sparse": pairs,
            "single_best": single_best, "pair_best": pair_best, "single_delta": single_best-baseline[:, None],
            "pair_delta": pair_best-baseline[:, None], "single_first_max_index": singles.argmax(-1),
            "single_max_tie_count": (singles == single_best[:, :, None]).sum(-1),
            "pair_first_max_flat_index": np.stack(pair_first, axis=1), "pair_max_tie_count": np.stack(pair_ties, axis=1),
            "joint_oracle": oracle, "single_envelope": single_best.max(1), "pair_envelope": pair_best.max(1),
            "single_envelope_delta": single_best.max(1)-baseline, "pair_envelope_delta": pair_best.max(1)-baseline,
            "pair_minus_single_envelope": pair_best.max(1)-single_best.max(1), "joint_minus_pair_envelope": oracle-pair_best.max(1)}


def vector_summary(v, improvement=False):
    r = {"n": int(v.size), "mean": float(v.mean()), "minimum": float(v.min()), "maximum": float(v.max()),
         "quantile_probabilities": list(QUANTILES), "quantiles": np.quantile(v, QUANTILES).tolist()}
    if improvement:
        r.update(positive_above_1e_minus12=int((v > 1e-12).sum()), numerical_near_zero=int((np.abs(v) <= 1e-12).sum()))
    return r


def response_summary(values):
    scalar_names = ("baseline", "joint_oracle", "single_envelope", "pair_envelope", "single_envelope_delta", "pair_envelope_delta",
                    "pair_minus_single_envelope", "joint_minus_pair_envelope")
    result = {name: vector_summary(values[name], "delta" in name or "minus" in name) for name in scalar_names}
    for prefix, labels in (("single", ("A", "B", "C")), ("pair", ("AB", "AC", "BC"))):
        for label_id, label in enumerate(labels):
            for suffix in ("best", "delta"):
                result[f"{prefix}_{label}_{suffix}"] = vector_summary(values[f"{prefix}_{suffix}"][:, label_id], suffix == "delta")
    return result


def group_measure(values):
    # values: group-member, actor, action; no pair subsampling.
    n = len(values); require(n >= 2, "Need at least two valid group members")
    coordinate_range = values.max(0)-values.min(0)
    # Shift by an actual group member before centering; exact constant groups
    # then have exactly zero squared distance even when repeated sums round.
    delta = values-values[:1]
    square_distance_mean = 2/(n-1)*((delta-delta.mean(0))**2).sum(axis=(0, 2))
    return {"identical": (coordinate_range == 0).all(1), "maximum_coordinate_range": coordinate_range.max(1),
            "mean_pair_squared_l2": square_distance_mean, "unordered_state_pairs": n*(n-1)//2}


def probability_changes(states, probabilities):
    axes = {"needs": (3, 4, 5, 6, 7, 8, 9), "layout": (0, 1, 2, 7, 8, 9), "owner": (0, 1, 2, 3, 4, 5, 6)}
    result, summary = {}, {}
    for axis, columns in axes.items():
        keys, inverse, counts = np.unique(states[:, columns], axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        edges = np.r_[0, np.cumsum(counts)]
        identical = np.empty((len(keys), 3), dtype=bool)
        amplitude = np.empty((len(keys), 3)); squared = np.empty((len(keys), 3))
        pair_counts = np.empty(len(keys), dtype=np.int64)
        for g in range(len(keys)):
            record = group_measure(probabilities[order[edges[g]:edges[g+1]]])
            identical[g], amplitude[g], squared[g], pair_counts[g] = (
                record["identical"], record["maximum_coordinate_range"], record["mean_pair_squared_l2"], record["unordered_state_pairs"])
        result.update({axis+"_group_keys": keys, axis+"_member_counts": counts, axis+"_unordered_pair_counts": pair_counts,
                       axis+"_identical": identical, axis+"_max_coordinate_range": amplitude, axis+"_mean_pair_squared_l2": squared})
        summary[axis] = {}
        for who, label in enumerate("ABC"):
            summary[axis][label] = {"groups": len(keys), "states": int(counts.sum()), "group_member_counts": sorted(set(map(int, counts))),
                "unordered_state_pairs": int(pair_counts.sum()), "identical_groups": int(identical[:, who].sum()),
                "changed_groups": int((~identical[:, who]).sum()), "changed_group_fraction": float((~identical[:, who]).mean()),
                "maximum_coordinate_range_mean": float(amplitude[:, who].mean()), "maximum_coordinate_range_maximum": float(amplitude[:, who].max()),
                "pair_rms_l2": float(np.sqrt(np.dot(squared[:, who], pair_counts)/pair_counts.sum())),
                "groups_above_threshold": {str(t): int((amplitude[:, who] > t).sum()) for t in (1e-12, 1e-9, 1e-6)}}
    return result, summary


def synthetic_tests():
    states = np.asarray([[2, 2, 2, 0, 1, 2, 3, 1, 2, 3], [2, 8, 11, 3, 1, 2, 0, 3, 2, 1],
                         [5, 8, 11, 0, 2, 3, 1, 2, 1, 3]], dtype=np.int16)
    rng = np.random.default_rng(20260920001)
    p = rng.random((3, 3, 17)); p /= p.sum(-1, keepdims=True)
    r = native_rewards(states)
    sparse = responses(p, r)
    dense = np.zeros((len(p), 17, 17, 17))
    for t, a in enumerate(JOINT):
        dense[:, a[0], a[1], a[2]] = r[:, t]
    v = np.einsum("nabc,na,nb,nc->n", dense, p[:, 0], p[:, 1], p[:, 2])
    require(np.allclose(v, sparse["baseline"], atol=1e-14), "Dense baseline synthetic")
    single_refs = (np.einsum("nabc,nb,nc->na", dense, p[:, 1], p[:, 2]),
                   np.einsum("nabc,na,nc->nb", dense, p[:, 0], p[:, 2]),
                   np.einsum("nabc,na,nb->nc", dense, p[:, 0], p[:, 1]))
    pair_refs = (np.einsum("nabc,nc->nab", dense, p[:, 2]), np.einsum("nabc,nb->nac", dense, p[:, 1]), np.einsum("nabc,na->nbc", dense, p[:, 0]))
    for i in range(3):
        require(np.allclose(single_refs[i], sparse["single_action_values"][:, i], atol=1e-14), "All17 dense single comparison")
        full = np.zeros((3, 17, 17)); coords = PAIR_COORDS[i]
        full[:, coords[:, 0], coords[:, 1]] = sparse["pair_action_values_sparse"][:, i]
        require(np.array_equal(full, pair_refs[i]), "All289 dense pair comparison")
    pure = np.zeros((1, 3, 17)); pure[:, :, 0] = 1
    pure[:, 2] = 0; pure[:, 2, action(2, 1, 0, 0)] = 1
    values = responses(pure, native_rewards(states[:1]))
    expected_pair = action(1, 2, 0, 0)
    require(values["pair_best"][0, 0] == 1 and values["pair_first_max_flat_index"][0, 0] == expected_pair,
            "Pair optimization must allow modified A to wait and B to match unchanged C")
    constant = np.repeat(p[:1], 5, axis=0)
    require(group_measure(constant)["identical"].all() and not group_measure(constant)["maximum_coordinate_range"].any()
            and not group_measure(constant)["mean_pair_squared_l2"].any(), "Constant policy group")
    perturbed = constant.copy(); perturbed[0, 1, 0] += .01; perturbed[0, 1, 1] -= .01
    measured = group_measure(perturbed)
    explicit = np.asarray([((perturbed[i]-perturbed[j])**2).sum(-1) for i, j in combinations(range(5), 2)]).mean(0)
    require(np.allclose(measured["mean_pair_squared_l2"], explicit, atol=1e-15) and measured["identical"].tolist() == [True, False, True], "Pair distance moment formula")
    require(np.isclose(measured["maximum_coordinate_range"][1], .01), "Coordinate amplitude")
    for bad in (np.full((1, 3, 17), np.nan), np.zeros((1, 3, 17))):
        try:
            responses(bad, native_rewards(states[:1]))
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid probabilities accepted")
    return {"status": "passed", "synthetic_worlds": 3, "dense_joint_terms_per_world": 4913,
            "single_action_values_compared": 153, "pair_action_values_compared": 2601,
            "dense_baseline_max_abs_error": float(np.max(np.abs(v-sparse["baseline"]))), "model_forward_calls": 0, "training_updates": 0}


def prepare(out):
    require(not out.exists(), "Do not overwrite prepared diagnostic")
    tests = synthetic_tests()
    original = read(SOURCE/"execution/results.json")
    require(original["status"] == "completed", "Source must be complete")
    metadata = [SOURCE/name for name in ("plan.json", "prepared.json", "freeze.json", "execution/results.json",
                "audit_execution_001/verification.json", "behavior_001/behavior.json")]
    files = [HERE/"plan.md", Path(__file__).resolve()]+metadata
    for seed in SEEDS:
        for part in PARTITIONS:
            files.append(SOURCE/f"execution/seed_{seed}/final_{part}.npz")
    digests = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for seed, row in zip(SEEDS, original["seeds"]):
        require(seed == row["seed"], "Source seed order")
        for part in PARTITIONS:
            name = str((SOURCE/f"execution/seed_{seed}/final_{part}.npz").relative_to(ROOT))
            require(digests[name] == row["final"][part]["data_sha256"], "Source final hash")
    require(read(SOURCE/"audit_execution_001/verification.json")["status"] == "passed", "Source execution audit required")
    out.mkdir(parents=True)
    for p in files[:2]:
        dest = out/"source_snapshot"/p.name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(p, dest)
    manifest = {"created_at": now(), "status": "prepared", "source_files_sha256": digests,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "seeds": list(SEEDS),
                "partitions": list(PARTITIONS), "source_run": str(SOURCE), "synthetic_tests": tests,
                "expected_states_per_seed": 143424, "model_forward_calls": 0, "training_updates": 0,
                "pair_sparse_default": 0., "pair_dense_action_count": 289, "pair_saved_structural_terms": 24}
    write(out/"manifest.json", manifest)
    write(out/"freeze.json", {"manifest_sha256": sha(out/"manifest.json")})
    return manifest


def verify(out):
    manifest = read(out/"manifest.json")
    require(sha(out/"manifest.json") == read(out/"freeze.json")["manifest_sha256"], "Frozen manifest changed")
    for name, digest in manifest["source_files_sha256"].items():
        require(sha(ROOT/name) == digest, "Source SHA changed: "+name)
    for path in (HERE/"plan.md", Path(__file__).resolve()):
        require(sha(out/"source_snapshot"/path.name) == sha(path), "Frozen source copy changed")
    require(manifest["runtime"] == {"python": platform.python_version(), "numpy": np.__version__}, "Numerical runtime changed")
    return manifest


def execute(out):
    manifest = verify(out); directory = out/"execution"
    require(not directory.exists(), "No overwrite or automatic retry")
    directory.mkdir(); started = time.perf_counter()
    write(directory/"started.json", {"started_at": now(), "manifest_sha256": sha(out/"manifest.json")})
    try:
        specs = read(SOURCE/"prepared.json")["partitions"]
        records = []; total_states = total_groups = 0; max_baseline_error = 0.
        for seed in SEEDS:
            for part in PARTITIONS:
                source_path = SOURCE/f"execution/seed_{seed}/final_{part}.npz"
                with np.load(source_path, allow_pickle=False) as z:
                    states, p, saved_v = z["states"], z["action_probabilities"], z["exact_expected_reward"]
                spec = specs[part]
                expected_states = np.asarray([tuple(n)+tuple(l)+tuple(o) for n, l, o in product(spec["needs"], spec["layouts"], spec["private_sites"])], dtype=np.int16)
                require(np.array_equal(states, expected_states) and len(states) == spec["world_count"], "Missing/duplicate/reordered source states")
                values = responses(p, native_rewards(states))
                require(np.all(values["joint_oracle"] == 1), "Native joint oracle not1")
                error = float(np.max(np.abs(values["baseline"]-saved_v)))
                require(np.allclose(values["baseline"], saved_v, atol=TOL, rtol=TOL), "Source expected reward mismatch")
                max_baseline_error = max(max_baseline_error, error)
                changes, change_summary = probability_changes(states, p)
                response_file = directory/f"seed_{seed}_{part}_responses.npz"
                change_file = directory/f"seed_{seed}_{part}_input_changes.npz"
                np.savez_compressed(response_file, states=states, pair_sparse_action_coordinates=PAIR_COORDS, **values)
                np.savez_compressed(change_file, **changes)
                records.append({"seed": seed, "partition": part, "worlds": len(states), "source_sha256": sha(source_path),
                    "responses_file": response_file.name, "responses_sha256": sha(response_file), "input_changes_file": change_file.name,
                    "input_changes_sha256": sha(change_file), "response_summary": response_summary(values), "input_change_summary": change_summary})
                total_states += len(states)
                total_groups += sum(v["A"]["groups"] for v in change_summary.values())*3
        require(len(records) == 16 and total_states == 573696, "Incomplete diagnostic grid")
        means, change_means = [], []
        for part in PARTITIONS:
            rows = [v for v in records if v["partition"] == part]
            require([v["seed"] for v in rows] == list(SEEDS), "Summary seed order")
            for metric in rows[0]["response_summary"]:
                values = [v["response_summary"][metric]["mean"] for v in rows]
                means.append({"partition": part, "metric": metric, "seeds": list(SEEDS), "seed_means": values,
                              "mean": float(np.mean(values)), "minimum_seed_mean": min(values), "maximum_seed_mean": max(values)})
            for axis in ("needs", "layout", "owner"):
                for who in "ABC":
                    for metric in ("changed_group_fraction", "maximum_coordinate_range_mean", "maximum_coordinate_range_maximum", "pair_rms_l2"):
                        values = [v["input_change_summary"][axis][who][metric] for v in rows]
                        change_means.append({"partition": part, "axis": axis, "actor": who, "metric": metric,
                            "seeds": list(SEEDS), "seed_values": values, "mean": float(np.mean(values)),
                            "minimum_seed_value": min(values), "maximum_seed_value": max(values)})
        verify(out)
        result = {"status": "complete", "completed_at": now(), "elapsed_seconds": time.perf_counter()-started,
            "manifest_sha256": sha(out/"manifest.json"), "records": records, "response_seed_aggregates": means,
            "input_change_seed_aggregates": change_means,
            "counts": {"independent_training_seeds": 4, "seed_partitions": 16, "worlds": total_states,
                "single_response_action_values": total_states*3*17, "pair_dense_equivalent_action_values": total_states*3*289,
                "pair_saved_structural_action_values": total_states*3*24, "input_change_actor_groups": total_groups,
                "model_forward_calls": 0, "training_updates": 0}, "max_baseline_abs_error": max_baseline_error,
            "scope": "Post-hoc statewise researcher best-response oracle and saved probability input sensitivity; not learned/deployed responses or training causality"}
        write(directory/"results.json", result)
        return result
    except BaseException as e:
        write(directory/"failure.json", {"status": "failed", "failed_at": now(), "error": str(e), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "test":
        result = synthetic_tests()
    else:
        require(args.out is not None, "Output directory required")
        result = {"prepare": prepare, "verify": verify, "execute": execute}[args.command](args.out.resolve())
    print(json.dumps({k: result[k] for k in ("status", "counts", "max_baseline_abs_error", "elapsed_seconds") if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
