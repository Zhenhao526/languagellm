"""Independent numerical checks of paired objectives; never creates actors."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import traceback
import numpy as np

from research_program.triadic_coordination_study import runner

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def independent_structure():
    result = []
    for pair in combinations(range(3), 2):
        for site, dest in product(range(4), range(2)):
            a = [0, 0, 0]
            for who in pair:
                other = pair[1] if who == pair[0] else pair[0]
                a[who] = 1+4*site+2*dest+[i for i in range(3) if i != who].index(other)
            result.append(a)
    return np.asarray(result, dtype=np.int64)


JOINT = independent_structure()


def reference(logits, reward):
    # Dense4913 table, independent log-softmax and dense marginalization.
    shifted = logits-np.max(logits, axis=-1, keepdims=True)
    lp = shifted-np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    p = np.exp(lp)
    dense_reward = np.zeros((len(p), 17, 17, 17))
    dense_log_mass = lp[:, 0, :, None, None]+lp[:, 1, None, :, None]+lp[:, 2, None, None, :]
    for t, a in enumerate(JOINT):
        dense_reward[:, a[0], a[1], a[2]] = reward[:, t]
    positive = dense_reward > 0
    reward_logs = np.full_like(dense_reward, -np.inf)
    reward_logs[positive] = np.log(dense_reward[positive])
    weighted_log = dense_log_mass+reward_logs
    high = weighted_log.max(axis=(1, 2, 3), keepdims=True)
    normalized = np.exp(weighted_log-high)
    sums = normalized.sum(axis=(1, 2, 3), keepdims=True)
    posterior = normalized/sums
    log_j = (high+np.log(sums)).reshape(-1)
    marginal = np.stack([posterior.sum(axis=(2, 3)), posterior.sum(axis=(1, 3)), posterior.sum(axis=(1, 2))], axis=1)
    raw_mass = p[:, 0, :, None, None]*p[:, 1, None, :, None]*p[:, 2, None, None, :]*dense_reward
    j = raw_mass.sum(axis=(1, 2, 3))
    raw_marginal = np.stack([raw_mass.sum(axis=(2, 3)), raw_mass.sum(axis=(1, 3)), raw_mass.sum(axis=(1, 2))], axis=1)
    return {"J": j, "log_J": log_j, "log_gradient": marginal-p, "mean_gradient": raw_marginal-p*j[:, None, None],
            "posterior": np.stack([posterior[:, a[0], a[1], a[2]] for a in JOINT], axis=1)}


def audit():
    source_digest = sha(runner.__file__)
    require(sha(runner.base.__file__) == runner.BASE_RUNNER_SHA, "Old baseline source changed")
    old_config = json.dumps(runner.base.CONFIG, sort_keys=True)
    rng = np.random.default_rng(20260920002)
    normal = rng.normal(0, 2, (2, 3, 17))
    rewards = rng.choice(np.asarray([0., .5, 1.]), size=(2, 24))
    tiny = np.full((1, 3, 17), -1000.); tiny[:, :, 1] = 0
    tiny_multiple = np.full((1, 3, 17), -10000.)
    tiny_multiple[0, 0, 5] = 0; tiny_multiple[0, 1, 7] = 0; tiny_multiple[0, 2, 9] = 0
    # Different positive rewards and at least one exact zero; no reward smoothing.
    tiny_reward = np.asarray([[0., .5, 1.]*8])
    cases = (("ordinary", normal, rewards), ("tiny_J", tiny, np.ones((1, 24))), ("tiny_multiple", tiny_multiple, tiny_reward))
    records = []
    derivative_coordinates = 0
    for label, logits, reward in cases:
        terms, ref = runner.objective_terms(logits, reward), reference(logits, reward)
        require(np.allclose(terms["J"], ref["J"], atol=2e-14, rtol=2e-12), "Dense native expectation mismatch")
        require(np.allclose(terms["log_J"], ref["log_J"], atol=2e-11, rtol=2e-12), "Dense log expectation mismatch")
        require(np.allclose(terms["log_J_logit_gradient"], ref["log_gradient"], atol=2e-11, rtol=2e-12), "Dense posterior gradient mismatch")
        require(np.allclose(terms["mean_J_logit_gradient"], ref["mean_gradient"], atol=2e-14, rtol=2e-12), "Dense native gradient mismatch")
        require(np.allclose(terms["posterior_weights"], ref["posterior"], atol=2e-11, rtol=2e-12), "Dense posterior mismatch")
        require(np.allclose(terms["posterior_weights"].sum(-1), 1., atol=2e-14), "Posterior not normalized")
        require((terms["posterior_weights"][reward == 0] == 0).all(), "Zero reward leaked into posterior")
        require(np.max(np.abs(terms["log_J_logit_gradient"].sum(-1))) < 2e-12, "Logit translation gradient nonzero")
        step = 1e-5 if label == "ordinary" else 1e-3
        errors = []
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += step; minus[index] -= step
            numerical = (reference(plus, reward)["log_J"][index[0]]-reference(minus, reward)["log_J"][index[0]])/(2*step)
            errors.append(abs(numerical-terms["log_J_logit_gradient"][index])); derivative_coordinates += 1
        require(max(errors) < 2e-7, "Independent logJ finite difference failed")
        moved = logits+np.asarray([3., -7., 11.])[None, :, None]
        shifted_terms = runner.objective_terms(moved, reward)
        require(np.allclose(shifted_terms["log_J"], terms["log_J"], atol=2e-11)
                and np.allclose(shifted_terms["log_J_logit_gradient"], terms["log_J_logit_gradient"], atol=2e-11), "Actor logit shift changed objective")
        if label != "ordinary":
            require(not terms["J"].any() and np.isfinite(terms["log_J"]).all()
                    and np.max(np.abs(terms["log_J_logit_gradient"])) > .1, "TinyJ stable gradient test not reached")
        records.append({"case": label, "states": len(logits), "finite_difference_step": step,
                        "coordinates": logits.size, "log_J": terms["log_J"].tolist(), "J": terms["J"].tolist(),
                        "logit_gradient_max_abs_error": max(errors),
                        "dense_gradient_max_abs_error": float(np.max(np.abs(terms["log_J_logit_gradient"]-ref["log_gradient"])))})
    # The real batch size is256. This compares complete loss/derivative fields
    # with frozen baseline formulas, without a policy network or update.
    logits = rng.normal(size=(256, 3, 17)); reward = rng.choice(np.asarray([0., .5, 1.]), size=(256, 24))
    terms = runner.objective_terms(logits, reward)
    p, lp = runner.base.policy_distribution(logits)
    j, gj = runner.base.expected_reward_and_logit_gradient(p, reward)
    entropy, gh = runner.base.entropy_and_logit_gradient(p, lp)
    for update in (1, 501, 1001, 6000):
        beta = .001*max(0, 1-(update-1)/1000)
        observed = runner.loss_and_derivative(terms, "mean_J", update)
        require(observed["loss"] == -float(j.mean()+beta*entropy.mean()), "Native control loss not exact old behavior")
        require(np.array_equal(observed["derivative"], -(gj+beta*gh)/256), "Native control derivative not exact old behavior")
        log_observed = runner.loss_and_derivative(terms, "mean_log_J", update)
        require(log_observed["loss"] == -float(terms["log_J"].mean()+beta*entropy.mean())
                and np.array_equal(log_observed["derivative"], -(terms["log_J_logit_gradient"]+beta*gh)/256), "Log objective loss sign/entropy differs")
    for reward in (np.zeros((1, 24)), np.full((1, 24), np.nan), np.full((1, 24), -1.)):
        try:
            runner.objective_terms(np.zeros((1, 3, 17)), reward)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError("Invalid reward/support accepted")
    require(source_digest == sha(runner.__file__) and old_config == json.dumps(runner.base.CONFIG, sort_keys=True), "Audit changed frozen source/config")
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "runner_sha256": source_digest,
            "audit_source_sha256": sha(__file__), "baseline_source_sha256": sha(runner.base.__file__), "cases": records,
            "finite_difference_logit_coordinates": derivative_coordinates, "dense_joint_terms_per_state": 4913,
            "finite_difference_absolute_tolerance": 2e-7, "control_loss_and_derivative_exact_at_updates": [1, 501, 1001, 6000],
            "network_forward_calls": 0, "actors_created": 0, "optimizer_updates": 0, "training_updates": 0,
            "interpretation": "Mean log native expectation changes weighting across states; not pure exploration and not the logarithm of a batch-mean reward"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    require(not args.out.exists(), "Never overwrite or automatically retry an audit")
    args.out.mkdir(parents=True)
    try:
        result = audit(); write(args.out/"verification.json", result)
        print(json.dumps({"status": result["status"], "runner_sha256": result["runner_sha256"],
                          "finite_difference_logit_coordinates": result["finite_difference_logit_coordinates"]}))
    except BaseException as error:
        write(args.out/"failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                                      "automatic_retry": False, "audit_source_sha256": sha(__file__)})
        raise


if __name__ == "__main__":
    main()
