"""Independent NumPy-only statistics for v0.33 team protocols."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MAPS = np.asarray(list(itertools.permutations(range(6), 2)), np.int64)
CONDITIONS = ("single_full", "dual_same_full", "dual_complementary")
MASKS = ("pooled", "food_only", "water_only")
PERMS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))
TRAIN = ((0, 2), (0, 5), (1, 0), (1, 3), (2, 1), (2, 4), (3, 1), (3, 5), (4, 2), (4, 3), (5, 0), (5, 4))
WORLD_KEYS = ("map_id", "photo_ids", "positions", "shown")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def arrays_sha(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


class Checks:
    def __init__(self):
        self.count = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = Counter()

    def require(self, ok, label):
        self.count += 1
        if not bool(ok):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        if isinstance(expected, dict):
            self.require(set(actual) == set(expected), label + " keys")
            for key in expected: self.exact(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, (tuple, list)):
            self.require(len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)): self.exact(a, b, label + "/" + str(i))
        else:
            self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, label):
        if isinstance(expected, dict):
            self.require(set(actual) == set(expected), label + " keys")
            for key in expected: self.close(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, (tuple, list)):
            self.require(len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)): self.close(a, b, label + "/" + str(i))
        elif isinstance(expected, (bool, str)) or expected is None:
            self.exact(actual, expected, label)
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            self.require(a.shape == b.shape, label + " shape")
            self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
            self.comparisons += int(a.size)
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
            self.max_error = max(self.max_error, error)
            self.require(np.allclose(a, b, rtol=1e-9, atol=1e-10), label + " numerical")


def groups(panel, condition):
    if panel not in (1, 2, 3) or condition not in CONDITIONS:
        raise ValueError("invalid panel or condition")
    q = PERMS[panel - 1]; index = {tuple(x): i for i, x in enumerate(MAPS)}
    target = {index[(q[i], q[j])] for i, j in TARGET}; train = {index[(q[i], q[j])] for i, j in TRAIN}
    return {key: np.asarray(sorted(value), np.int64) for key, value in dict(train12=train, target12=target, held18=set(range(30)) - train, common30=set(range(30))).items()}


def softmax(logits):
    z = np.asarray(logits, np.float64); e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def donor_indices(raw, panel, condition):
    train = np.isin(raw["map_id"], groups(panel, condition)["train12"])
    targets = np.flatnonzero(np.isin(raw["map_id"], groups(panel, condition)["target12"]))
    codes = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]; food = []; water = []
    for i in targets:
        allowed = [j for j in range(len(codes)) if train[j] and raw["shown"][j] == raw["shown"][i] and np.array_equal(raw["photo_ids"][i], raw["photo_ids"][j])]
        f = [j for j in allowed if raw["positions"][j, 0] == raw["positions"][i, 0]]; w = [j for j in allowed if raw["positions"][j, 1] == raw["positions"][i, 1]]
        if len(f) != 2 or len(w) != 2: raise ValueError("nonunique donor support")
        food.append(np.repeat(f, 2)); water.append(np.tile(w, 2))
    return targets, np.asarray(food), np.asarray(water)


def recombination(raw, panel, condition, permutation=None):
    ix, fd, wd = donor_indices(raw, panel, condition); decoder = np.argmax(raw["receiver_logits"], axis=-1)
    oldcodes = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]
    permutations = np.arange(49, dtype=np.int64)[None, :] if permutation is None else np.asarray(permutation)
    if permutations.ndim != 2 or permutations.shape[1] != 49 or not (np.sort(permutations, axis=1) == np.arange(49)).all():
        raise ValueError("not a 49-code bijection table")
    remapped = permutations[:, oldcodes]; inverse = np.argsort(permutations, axis=1); ff = remapped[:, fd]; ww = remapped[:, wd]
    codes = {"FW": 7 * (ff // 7) + (ww % 7), "WF": 7 * (ww // 7) + (ff % 7)}
    result = {}
    for name, code in codes.items():
        original = np.take_along_axis(inverse, code.reshape(len(permutations), -1), axis=1).reshape(code.shape)
        decoded = decoder[original]
        result[name] = np.all(decoded == raw["positions"][None, ix, None, :], axis=-1).mean(axis=-1)
    return dict(target_rows=ix, **result)


def partner_pairs(raw, panel, condition):
    target = np.flatnonzero(np.isin(raw["map_id"], groups(panel, condition)["target12"]))
    code = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]; action = np.argmax(raw["receiver_logits"], axis=-1)[code]; correct = np.all(action == raw["positions"], axis=-1); out = {}
    for kind, name in ((0, "food_partner_pair_J"), (1, "water_partner_pair_J")):
        values = []
        for location in range(6):
            rows0 = target[raw["positions"][target, kind] == location]
            for photo in np.unique(raw["photo_ids"][rows0], axis=0):
                for shown in (0, 1):
                    rows = rows0[(raw["shown"][rows0] == shown) & (raw["photo_ids"][rows0] == photo).all(-1)]
                    if len(rows) != 2: raise ValueError("partner group is not two")
                    values.append(correct[rows].all())
        out[name] = float(np.mean(values))
    return out


def component_exchange(raw, panel, condition):
    """Evaluate four donor combinations while varying one token at a time."""
    target, fd, wd = donor_indices(raw, panel, condition); decoder = np.argmax(raw["receiver_logits"], axis=-1)
    food_tokens = raw["tokens"][fd][:, :, 0]; water_tokens = raw["tokens"][wd][:, :, 1]
    code = 7 * food_tokens[:, :, None] + water_tokens[:, None, :]
    correct = decoder[code] == raw["positions"][target, None, None, :]
    return {
        "all_joint": float(np.all(correct, axis=-1).mean()),
        "all_food": float(correct[..., 0].mean()),
        "all_water": float(correct[..., 1].mean()),
        "food_variation_food": float(correct[:, :, 0, 0].mean()),
        "food_variation_water": float(correct[:, :, 0, 1].mean()),
        "water_variation_food": float(correct[:, 0, :, 0].mean()),
        "water_variation_water": float(correct[:, 0, :, 1].mean()),
    }


def score(raw, panel, condition):
    positions = raw["positions"]; code = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]; decoder = np.argmax(raw["receiver_logits"], axis=-1); action = decoder[code]; correct = action == positions
    send, recv = softmax(raw["sender_log_probs"]), softmax(raw["receiver_logits"]); success = np.all(decoder[None, :, :] == positions[:, None, :], axis=-1)
    q = np.sum(send * recv[:, 0, positions[:, 0]].T * recv[:, 1, positions[:, 1]].T, axis=1); freq = np.bincount(code, minlength=49) / len(code); global_shuffle = success @ freq
    rec = recombination(raw, panel, condition); pairs = partner_pairs(raw, panel, condition); result = {}
    for scope, maps in groups(panel, condition).items():
        base = np.isin(raw["map_id"], maps); result[scope] = {}
        for mask in MASKS:
            use = base if mask == "pooled" else base & (raw["shown"] == MASKS.index(mask) - 1); good = correct[use]; local = np.bincount(code[use], minlength=49) / int(use.sum())
            values = dict(n=int(use.sum()), J=float(good.all(-1).mean()), Q=float(q[use].mean()), food=float(good[:, 0].mean()), water=float(good[:, 1].mean()), blank_J=float(success[use, 0].mean()), shuffle_J=float(global_shuffle[use].mean()), within_shuffle_J=float((success[use] @ local).mean()))
            if scope == "target12":
                for key in ("FW", "WF"):
                    rec_values = np.full(len(code), np.nan); rec_values[rec["target_rows"]] = rec[key][0]
                    values[f"recombine_{key}_J"] = float(rec_values[use].mean())
                if mask == "pooled": values.update(pairs)
            result[scope][mask] = values
    return result


def mean_tree(rows):
    if isinstance(rows[0], dict): return {key: mean_tree([row[key] for row in rows]) for key in rows[0]}
    return float(np.mean(rows))


def auc(curve):
    times = np.asarray([row["update"] for row in curve]); span = times[-1] - times[0]
    def integrate(values):
        if isinstance(values[0], dict): return {key: integrate([value[key] for value in values]) for key in values[0] if key != "n"}
        return float(np.sum(np.diff(times) * (np.asarray(values[1:]) + np.asarray(values[:-1])) / 2) / span)
    return integrate([row["scores"] for row in curve])


def null_raw(raw, panel, condition, permutations):
    ix, fd, wd = donor_indices(raw, panel, condition); decoder = raw["receiver_logits"].argmax(-1); inverse = np.argsort(permutations); oldcodes = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]; f = permutations[:, oldcodes[fd]]; w = permutations[:, oldcodes[wd]]; codes = {"FW": 7 * (f // 7) + (w % 7), "WF": 7 * (w // 7) + (f % 7)}
    result = {"target_rows": ix}
    for name, code in codes.items():
        original = np.take_along_axis(inverse, code.reshape(len(permutations), -1), axis=1).reshape(code.shape)
        result[name] = np.all(decoder[original] == raw["positions"][None, ix, None, :], axis=-1).mean(axis=-1)
    return result


def rank_summary(observed, null):
    values = np.asarray(null); equal = np.isclose(values, observed, atol=1e-12, rtol=0)
    return dict(observed=float(observed), null_mean=float(values.mean()), excess=float(observed - values.mean()), null=values.tolist(), n_less=int(((values < observed) & ~equal).sum()), n_equal=int(equal.sum()), n_greater=int(((values > observed) & ~equal).sum()), upper_tail_fraction=float((1 + ((values > observed) | equal).sum()) / (len(values) + 1)))


def summarize_null(raw, panel, condition, table):
    observed = recombination(raw, panel, condition); ix = table["target_rows"]
    result = {}
    for mask in MASKS:
        use = np.ones(len(ix), bool) if mask == "pooled" else raw["shown"][ix] == MASKS.index(mask) - 1; result[mask] = {}
        for key in ("FW", "WF"):
            observed_values = np.asarray(observed[key])[0]
            null_values = np.asarray(table[key])[:, use].mean(axis=1)
            result[mask][f"recombine_{key}_J"] = rank_summary(float(observed_values[use].mean()), null_values)
    return result


def mean_null(rows):
    return {mask: {key: rank_summary(np.mean([row[mask][key]["observed"] for row in rows]), np.mean([row[mask][key]["null"] for row in rows], axis=0)) for key in rows[0][mask]} for mask in MASKS}


def check_raw(raw, worlds, checks):
    checks.exact({key: raw[key] for key in WORLD_KEYS}, worlds, "complete world")
    n = len(worlds["map_id"]); checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", "two integer tokens"); checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), "token domain")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), "sender table"); checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), "receiver table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), "sender probabilities")


def agreement_curve(folder, times, checks):
    values = []
    for t in times:
        raw = npz(folder / f"agreement_{t:04d}.npz")
        for view in ("full", "food_only", "water_only"):
            for private_type in (0, 1):
                a = raw[f"{view}_type{private_type}_a"]; b = raw[f"{view}_type{private_type}_b"]; checks.require(len(a) == 180 and len(b) == 180, "agreement rows")
        full = [((raw[f"full_type{k}_a"] == raw[f"full_type{k}_b"]).all(-1).mean()) for k in (0, 1)]
        full0 = [((raw[f"full_type{k}_a"][:, 0] == raw[f"full_type{k}_b"][:, 0]).mean()) for k in (0, 1)]
        full1 = [((raw[f"full_type{k}_a"][:, 1] == raw[f"full_type{k}_b"][:, 1]).mean()) for k in (0, 1)]
        food = [((raw[f"food_only_type{k}_a"] == raw[f"food_only_type{k}_b"]).mean()) for k in (0, 1)]
        water = [((raw[f"water_only_type{k}_a"] == raw[f"water_only_type{k}_b"]).mean()) for k in (0, 1)]
        values.append(dict(update=t, agreement=dict(full_message_agreement=float(np.mean(full)), full_token0_agreement=float(np.mean(full0)), full_token1_agreement=float(np.mean(full1)), food_token_agreement=float(np.mean(food)), water_token_agreement=float(np.mean(water)))))
    return values


def analyze(out, checks):
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json")
    checks.exact(complete["status"], "complete", "terminal"); seeds, panels = invocation["seeds"], invocation["partitions"]; checks.exact(seeds, [34101, 34102, 34103, 34104] if invocation["formal"] else [99528], "sources"); checks.exact(panels, [1, 2, 3] if invocation["formal"] else [1], "panels"); checks.exact(invocation["updates"], 2400 if invocation["formal"] else 40, "updates")
    permutations = np.load(ROOT / "code_relabelings.npy"); expected = np.stack([np.random.default_rng(np.random.SeedSequence([33033, 49, i])).permutation(49) for i in range(199)]); checks.exact(permutations, expected, "relabelings"); checks.exact(sha(ROOT / "code_relabelings.npy"), invocation["source_hashes"][str(ROOT / "code_relabelings.npy")], "relabeling binding")
    test_worlds = npz(out / "test_worlds.npz"); checks.exact(len(test_worlds["map_id"]), 180, "test worlds"); checks.exact(test_worlds["positions"], MAPS[test_worlds["map_id"]], "test positions")
    rows = []; raw_hashes = {}; times = [0, 100, 600, 1200, 2100, 2400] if invocation["formal"] else [0, 40]
    for seed, panel, condition in itertools.product(seeds, panels, CONDITIONS):
        folder = out / "social" / f"s{seed}_p{panel}_{condition}"; cfg = read(folder / "config.json"); result = read(folder / "result.json"); saved = read(folder / "curve.json"); checks.exact(cfg["checkpoints"], times, "checkpoints"); checks.exact(result["status"], "complete", "run status"); checks.exact(set(saved[0]["scores"]), {f"team{i}" for i in range(4)}, "team inventory")
        nulls = {}; team_curves = {f"team{i}": [] for i in range(4)}; comp_curves = {f"team{i}": [] for i in range(4)}
        for item in saved:
            t = item["update"]
            for slot in range(4):
                key = f"team{slot}"; path = folder / f"protocol_{t:04d}_{key}.npz"; rel = str(path.relative_to(out)); digest = sha(path); checks.exact(complete["files"][rel], digest, "raw bound"); raw = npz(path); check_raw(raw, test_worlds, checks); metric = score(raw, panel, condition); checks.close(metric, item["scores"][key], "production score recheck"); team_curves[key].append(dict(update=t, scores=metric)); raw_hashes[rel] = digest; checks.coverage["protocol_tables"] += 1; checks.coverage["protocol_worlds"] += len(test_worlds["map_id"])
                if t == invocation["updates"]:
                    checks.close(metric, result["scores"][key], "terminal score"); null = null_raw(raw, panel, condition, permutations); nf = folder / f"recombination_null_{key}.npz"; checks.close(null, npz(nf), "199 relabel rows"); checks.exact(sha(nf), complete["files"][str(nf.relative_to(out))], "null bound"); nulls[key] = summarize_null(raw, panel, condition, null); comp_curves[key].append(component_exchange(raw, panel, condition)); raw_hashes[str(nf.relative_to(out))] = sha(nf); checks.coverage["null_target_scores"] += 2 * 199 * len(null["target_rows"])
        agreement = agreement_curve(folder, times, checks)
        for slot in range(4):
            key = f"team{slot}"; rows.append(dict(seed=seed, partition=panel, condition=condition, team=slot, curve=team_curves[key], scores=team_curves[key][-1]["scores"], auc=auc(team_curves[key]), component=component_exchange(npz(folder / f"protocol_{times[-1]:04d}_{key}.npz"), panel, condition), recombination_null=nulls[key], agreement=agreement))
        checks.coverage["social_runs"] += 1; checks.coverage["agreement_files"] += len(times)
    checks.exact(len(rows), len(seeds) * len(panels) * len(CONDITIONS) * 4, "team row inventory")
    seed_rows = []; aggregate = {}
    for seed, condition in itertools.product(seeds, CONDITIONS):
        selected = [row for row in rows if row["seed"] == seed and row["condition"] == condition]; checks.exact(len(selected), len(panels) * 4, "source team rows")
        curve = [dict(update=t["update"], scores=mean_tree([row["curve"][k]["scores"] for row in selected])) for k, t in enumerate(selected[0]["curve"])]
        agree = [dict(update=t["update"], agreement=mean_tree([row["agreement"][k]["agreement"] for row in selected])) for k, t in enumerate(selected[0]["agreement"])]
        comp = mean_tree([row["component"] for row in selected]); recomb = mean_null([row["recombination_null"] for row in selected]); seed_rows.append(dict(seed=seed, condition=condition, curve=curve, scores=curve[-1]["scores"], auc=auc(curve), agreement=agree, component=comp, recombination_null=recomb))
    for condition in CONDITIONS:
        selected = [row for row in seed_rows if row["condition"] == condition]; curve = [dict(update=t["update"], scores=mean_tree([row["curve"][k]["scores"] for row in selected])) for k, t in enumerate(selected[0]["curve"])]
        agree = [dict(update=t["update"], agreement=mean_tree([row["agreement"][k]["agreement"] for row in selected])) for k, t in enumerate(selected[0]["agreement"])]
        aggregate[condition] = dict(independent_sources=len(seeds), curve=curve, scores=curve[-1]["scores"], auc=auc(curve), agreement=agree, component=mean_tree([row["component"] for row in selected]), recombination_null=mean_null([row["recombination_null"] for row in selected]))
    primary = []
    for seed in seeds:
        one = {row["condition"]: row for row in seed_rows if row["seed"] == seed}; same, comp = one["dual_same_full"], one["dual_complementary"]
        primary.append(dict(seed=seed, single=one["single_full"]["scores"]["target12"]["pooled"]["J"], dual_same_full=same["scores"]["target12"]["pooled"]["J"], dual_complementary=comp["scores"]["target12"]["pooled"]["J"], complementary_difference=comp["scores"]["target12"]["pooled"]["J"] - same["scores"]["target12"]["pooled"]["J"], single_difference=comp["scores"]["target12"]["pooled"]["J"] - one["single_full"]["scores"]["target12"]["pooled"]["J"], complementary_auc_difference=comp["auc"]["target12"]["pooled"]["J"] - same["auc"]["target12"]["pooled"]["J"], full_agreement_single=one["single_full"]["agreement"][-1]["agreement"]["full_message_agreement"], full_agreement_same=same["agreement"][-1]["agreement"]["full_message_agreement"], food_agreement_comp=comp["agreement"][-1]["agreement"]["food_token_agreement"], water_agreement_comp=comp["agreement"][-1]["agreement"]["water_token_agreement"]))
    return dict(status="complete", formal=invocation["formal"], seeds=seeds, partitions=panels, conditions=list(CONDITIONS), rows=rows, seed_rows=seed_rows, aggregate=aggregate, primary=primary, primary_mean=float(np.mean([row["complementary_difference"] for row in primary])), primary_auc_mean=float(np.mean([row["complementary_auc_difference"] for row in primary])), analysis_source_sha256=sha(__file__), training_complete_sha256=sha(out / "training_complete.json"), definitions=dict(J="Saved greedy receiver success on each team protocol; team rows and panels are nested.", component_exchange="Food and water donor token combinations from training support; selective variation is descriptive offline intervention.", agreement="Exact equality between independent same-private-type agents evaluated on the same frozen test worlds; full view uses two tokens and masked views use one token.", recombination="All2x2 same-photo/same-mask donor combinations; full-code relabeling is a structural null."), limits=["No production metrics imported.", "No model calls or optimizer replay by this analysis.", "Four agents reuse two frozen private encoder types; team role labels are designed controls."]), raw_hashes


def self_test():
    checks = Checks(); ids = np.tile(np.repeat(np.arange(30), 3), 2); photo = np.tile(np.array([[0, 3], [1, 3], [2, 3]]), (60, 1)); shown = np.repeat(np.arange(2), 90); raw = dict(map_id=ids, positions=MAPS[ids], photo_ids=photo, shown=shown, tokens=MAPS[ids].copy(), sender_log_probs=np.full((180, 49), -np.log(49.0)), receiver_logits=np.zeros((49, 2, 6)))
    value = score(raw, 1, CONDITIONS[0]); checks.close(value["common30"]["pooled"]["Q"], 1 / 36, "uniform joint"); checks.exact(value["common30"]["pooled"]["J"], 0.0, "uniform argmax")
    return dict(passed=True, checks=checks.count, scope="Uniform protocol score and component table smoke; no model calls.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path); ap.add_argument("--self-test", action="store_true"); args = ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False)); return
    if args.out is None: ap.error("--out required unless --self-test")
    out = args.out.resolve(); checks = Checks(); started = time.monotonic()
    try:
        result, raw_hashes = analyze(out, checks); write(out / "analysis.json", result); qa = dict(passed=True, checks=checks.count, scalar_comparisons=checks.comparisons, maximum_metric_absolute_difference=checks.max_error, coverage=dict(checks.coverage), analysis_source_sha256=sha(__file__), analysis_sha256=sha(out / "analysis.json"), checked_raw_sha256=raw_hashes, production_modules_imported=False, model_calls=0, seconds=time.monotonic() - started); write(out / "raw_validation.json", qa); shutil.copy2(__file__, out / "analyze_results_source.py"); print(json.dumps({key: qa[key] for key in ("passed", "checks", "scalar_comparisons", "maximum_metric_absolute_difference", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(out / f"analysis_failure_{stamp}.json", dict(passed=False, error=repr(error), traceback=traceback.format_exc(), source_sha256=sha(__file__))); raise


if __name__ == "__main__":
    main()
