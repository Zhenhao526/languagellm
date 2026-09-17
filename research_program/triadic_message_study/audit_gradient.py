"""Independent finite two-window score/receiver audit. No actor is constructed."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import traceback
import numpy as np

from research_program.triadic_message_study import runner

HERE = Path(__file__).resolve().parent
CORE = ("paired_sender_derivative", "trajectory_objective", "training_gradients", "rollout", "routed_window")


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def core_hashes():
    nodes = ast.parse(Path(runner.__file__).read_text()).body
    return {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            for n in nodes if isinstance(n, ast.FunctionDef) and n.name in CORE}


def distribution(logits):
    z = logits-logits.max(-1, keepdims=True)
    lp = z-np.log(np.exp(z).sum(-1, keepdims=True))
    return np.exp(lp), lp


def structure_and_reward():
    rows, reward = [], []
    needs = ((0, 1), (0, 2), (1, 3))  # wood, short, long; all destinations acceptable
    for i, j in combinations(range(3), 2):
        for site, dest in product(range(4), range(2)):
            a = [0, 0, 0]
            for who, partner in ((i, j), (j, i)):
                a[who] = 1+4*site+2*dest+[k for k in range(3) if k != who].index(partner)
            rows.append(a); reward.append(.5*((site in needs[i])+(site in needs[j])))
    return np.asarray(rows, dtype=np.int64), np.asarray(reward, dtype=np.float64)


JOINT, REWARD = structure_and_reward()
BITS = np.asarray(list(product(range(2), repeat=3)), dtype=np.int64)
FIRST = np.repeat(np.arange(8), 8)
SECOND = np.tile(np.arange(8), 8)


def sender_trajectories(theta):
    first_logits = theta[:6].reshape(3, 2)
    second_logits = theta[6:].reshape(8, 3, 2)
    p1, lp1 = distribution(first_logits); p2, lp2 = distribution(second_logits)
    probabilities = np.empty(64); scores = np.zeros((64, 54))
    for t, (a, b) in enumerate(zip(FIRST, SECOND)):
        first, second = BITS[a], BITS[b]
        probabilities[t] = np.prod(p1[np.arange(3), first])*np.prod(p2[a, np.arange(3), second])
        scores[t, :6] = (np.eye(2)[first]-p1).reshape(-1)
        scores[t, 6+6*a:6+6*(a+1)] = (np.eye(2)[second]-p2[a]).reshape(-1)
    require(np.isclose(probabilities.sum(), 1., atol=1e-14), "Complete trajectory law is not normalized")
    require(np.max(np.abs(probabilities@scores)) < 1e-14, "Mean complete score is not zero")
    return {"q": probabilities, "score": scores, "p1": p1, "p2": p2, "lp1": lp1, "lp2": lp2}


def receiver_reference(logits, beta):
    p, lp = distribution(logits)
    log_mass = np.full((len(logits), 24), -np.inf)
    positive = REWARD > 0
    log_mass[:, positive] = np.log(REWARD[positive])
    for i in range(3):
        log_mass += lp[:, i, JOINT[:, i]]
    high = log_mass.max(1, keepdims=True)
    log_j = high[:, 0]+np.log(np.exp(log_mass-high).sum(1))
    entropy = -(p*lp).sum(-1).mean(1)
    return {"F": log_j+beta*entropy, "log_J": log_j, "entropy": entropy}


def embedded_pairs(law, F):
    """Embed64 binary trajectories into actual [2,4096,2,3,4,8] interface.

    The extra three positions are deterministic and categories2..7 have zero
    floating mass with finite underflow logp. Thus all actual24 scores remain.
    """
    left = np.repeat(np.arange(64), 64); right = np.tile(np.arange(64), 64)
    ids = np.stack((left, right)); B = len(left)
    p = np.zeros((2, B, 2, 3, 4, 8)); p[..., 0] = 1
    lp = np.full_like(p, -1000.); lp[..., 0] = 0
    m = np.zeros(p.shape[:-1], dtype=np.int8)
    for r in range(2):
        p[r, :, 0, :, 0] = 0; lp[r, :, 0, :, 0] = -1000
        p[r, :, 1, :, 0] = 0; lp[r, :, 1, :, 0] = -1000
        p[r, :, 0, :, 0, :2] = law["p1"]
        lp[r, :, 0, :, 0, :2] = law["lp1"]
        p[r, :, 1, :, 0, :2] = law["p2"][FIRST[ids[r]]]
        lp[r, :, 1, :, 0, :2] = law["lp2"][FIRST[ids[r]]]
        m[r, :, 0, :, 0] = BITS[FIRST[ids[r]]]
        m[r, :, 1, :, 0] = BITS[SECOND[ids[r]]]
    actual = runner.paired_sender_derivative(p, lp, m, F[ids])
    weights = law["q"][left]*law["q"][right]
    derivative = actual["derivative"]
    # Actual function averages B rows. We integrate all pairs as outcomes of
    # one state's estimator, so undo its B and weight each pair by q1*q2.
    first = -B*np.einsum("b,rbak->ak", weights, derivative[:, :, 0, :, 0, :2])
    second = np.zeros((8, 3, 2))
    for r in range(2):
        np.add.at(second, FIRST[ids[r]], -B*weights[:, None, None]*derivative[r, :, 1, :, 0, :2])
    require(np.allclose(actual["log_scores"], np.log(law["q"][ids]), atol=2e-14), "Actual complete logscore omits a window")
    require(not derivative[:, :, :, :, 1:].any(), "Deterministic extra position has nonzero score")
    return np.r_[first.reshape(-1), second.reshape(-1)]


def toy_audit(rng):
    theta = rng.normal(0, .7, 54)
    # Deliberate first-window influence on the second-window distribution.
    for context, bits in enumerate(BITS):
        for actor in range(3):
            theta[6+6*context+2*actor+bits[actor]] += 1.7
    law = sender_trajectories(theta)
    # Receiver depends only on window2. Any window1 gradient must travel through
    # the stochastic window2 law, not a direct use of window1 by the receiver.
    bias = rng.normal(0, .4, (3, 17))
    offset = rng.normal(size=(8, 3, 17))*(.1+1.7*BITS.sum(1))[:, None, None]
    logits = (bias[None]+offset)[SECOND]
    records = []; max_sender_error = max_receiver_error = 0.
    for update, beta in ((1, .001), (1001, 0.)):
        ref = receiver_reference(logits, beta); F = ref["F"]
        require(np.array_equal(F.reshape(8, 8), np.repeat(F[:8][None], 8, axis=0)), "Toy receiver unexpectedly reads window1")
        true_gradient = law["score"].T@(law["q"]*F)
        actual_gradient = embedded_pairs(law, F)
        require(np.allclose(actual_gradient, true_gradient, atol=3e-12, rtol=2e-11), "Actual4096-pair LOO mean is biased")
        numerical = np.empty_like(theta); step = 1e-5
        for coordinate in range(len(theta)):
            plus, minus = theta.copy(), theta.copy(); plus[coordinate] += step; minus[coordinate] -= step
            numerical[coordinate] = (sender_trajectories(plus)["q"]@F-sender_trajectories(minus)["q"]@F)/(2*step)
        sender_error = float(np.max(np.abs(numerical-actual_gradient)))
        require(sender_error < 3e-8, "Sender finite difference mismatch")
        require(np.max(np.abs(true_gradient[:6])) > 1e-3, "Toy fails to expose first-window future effect")
        # Wrong construction shares the first window between otherwise separate
        # trajectories. Its expected first-window score advantage is zero.
        shared_first = np.zeros(6)
        for context in range(8):
            ids = np.flatnonzero(FIRST == context)
            first_mass = law["q"][ids].sum(); conditional = law["q"][ids]/first_mass
            for j, k in product(range(8), repeat=2):
                t, u = ids[j], ids[k]
                shared_first += first_mass*conditional[j]*conditional[k]*.5*(F[t]-F[u])*(law["score"][t, :6]-law["score"][u, :6])
        require(np.max(np.abs(shared_first-true_gradient[:6])) > 1e-3, "Shared-first negative control did not expose bias")
        terms, receiver, actual_F = runner.trajectory_objective(logits, np.repeat(REWARD[None], 64, axis=0), update)
        require(np.allclose(actual_F, F, atol=3e-12) and np.isclose(receiver["loss"], -F.mean(), atol=3e-12), "Actual receiver F/loss omits entropy or wrong mean")
        analytic_phi = -64*np.einsum("n,nik->ik", law["q"], receiver["derivative"])
        numerical_phi = np.empty((3, 17))
        for index in np.ndindex(bias.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[:, index[0], index[1]] += step; minus[:, index[0], index[1]] -= step
            numerical_phi[index] = (law["q"]@receiver_reference(plus, beta)["F"]-law["q"]@receiver_reference(minus, beta)["F"])/(2*step)
        receiver_error = float(np.max(np.abs(analytic_phi-numerical_phi)))
        require(receiver_error < 3e-8, "Receiver direct gradient finite difference mismatch")
        entropy_gradient = law["score"].T@(law["q"]*(beta*ref["entropy"]))
        log_only_gradient = embedded_pairs(law, ref["log_J"])
        omitted_entropy_error = float(np.max(np.abs(actual_gradient-log_only_gradient)))
        require(np.allclose(actual_gradient-log_only_gradient, entropy_gradient, atol=3e-12), "Entropy contribution mismatch")
        if beta:
            require(omitted_entropy_error > 1e-7, "Omitting receiver entropy negative control too small")
        else:
            require(omitted_entropy_error == 0, "No entropy but sender return differed")
        records.append({"update": update, "beta": beta, "complete_trajectories": 64, "independent_trajectory_pairs": 4096,
            "actual_LOO_vs_exact_max_abs_error": float(np.max(np.abs(actual_gradient-true_gradient))),
            "sender_finite_difference_coordinates": 54, "sender_finite_difference_max_abs_error": sender_error,
            "receiver_direct_difference_coordinates": 51, "receiver_direct_difference_max_abs_error": receiver_error,
            "first_window_gradient_max_abs": float(np.max(np.abs(true_gradient[:6]))),
            "shared_first_window_wrong_gradient_max_abs_error": float(np.max(np.abs(shared_first-true_gradient[:6]))),
            "omitted_receiver_entropy_sender_error": omitted_entropy_error,
            "inclusive_mean_baseline_half_scale_error": float(np.max(np.abs(.5*actual_gradient-true_gradient)))})
        max_sender_error = max(max_sender_error, sender_error); max_receiver_error = max(max_receiver_error, receiver_error)
    return records, {"theta": theta, "receiver_bias": bias, "receiver_window2_offsets": offset,
                     "trajectory_probabilities": law["q"], "native_reward24": REWARD}, max_sender_error, max_receiver_error


def full_shape_audit(rng):
    B = 3; logits = rng.normal(size=(2, B, 2, 3, 4, 8))
    p, lp = distribution(logits); messages = rng.integers(0, 8, size=logits.shape[:-1], dtype=np.int8)
    F = np.asarray([[-5., -2., -1.], [-3., -6., -2.]])
    observed = runner.paired_sender_derivative(p, lp, messages, F)
    advantage = (F-F[::-1])/2
    def surrogate(z):
        _, logp = distribution(z)
        scores = np.take_along_axis(logp, messages[..., None], axis=-1)[..., 0].sum(axis=(2, 3, 4))
        return -(advantage*scores).sum()/B
    require(np.array_equal(observed["advantage"], advantage), "LOO half/sign wrong")
    require(np.isclose(observed["surrogate_loss"], surrogate(logits), atol=2e-13), "Sender surrogate sign/reduction wrong")
    step = 1e-5; error = 0.
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy(); plus[index] += step; minus[index] -= step
        derivative = (surrogate(plus)-surrogate(minus))/(2*step)
        error = max(error, abs(derivative-observed["derivative"][index]))
    require(error < 3e-8, "Full24-token sender gradient finite difference mismatch")
    return {"shape": list(logits.shape), "coordinates": logits.size, "finite_difference_step": step,
            "max_abs_error": error, "tokens_per_trajectory": 24, "batch_size": B,
            "extra_agent_window_token_divisor": False}


def tiny_audit():
    logits = np.full((2, 3, 17), -1000.)
    logits[:, :, 1] = 0
    logits[1] *= 20
    reward = np.ones((2, 24))
    terms, receiver, F = runner.trajectory_objective(logits, reward, 1)
    require(not terms["J"].any() and np.isfinite(F).all() and np.isfinite(receiver["derivative"]).all(), "tinyJ not stable")
    require(np.allclose(terms["log_J"], [-1000., -20000.], atol=2e-10), "tinyJ known value")
    p = np.full((2, 1, 2, 3, 4, 8), 1/8); lp = np.log(p)
    m = np.zeros(p.shape[:-1], dtype=np.int8); m[1] = 1
    sender = runner.paired_sender_derivative(p, lp, m, F.reshape(2, 1))
    require(np.isfinite(sender["derivative"]).all() and np.max(np.abs(sender["advantage"])) == 9500., "tinyJ advantage unstable or clipped")
    rejected = 0
    for bad in (np.zeros((2, 24)), np.full((2, 24), np.nan)):
        try:
            runner.trajectory_objective(logits, bad, 1)
        except (ValueError, AssertionError):
            rejected += 1
    require(rejected == 2, "Invalid reward support not rejected")
    return {"float_J": terms["J"].tolist(), "stable_log_J": terms["log_J"].tolist(),
            "max_abs_LOO_advantage": float(np.abs(sender["advantage"]).max()), "sender_gradient_finite": True,
            "receiver_gradient_finite": True, "invalid_reward_cases_rejected": rejected}


def audit(out):
    before_core = core_hashes(); before_full = sha(runner.__file__)
    require(set(before_core) == set(CORE), "Missing core functions")
    require(sha(runner.base.__file__) == runner.BASE_SHA and sha(runner.coordination.__file__) == runner.COORDINATION_SHA, "Old dependency changed")
    rng = np.random.default_rng(20260920003)
    toy, fixtures, sender_error, receiver_error = toy_audit(rng)
    full = full_shape_audit(rng); tiny = tiny_audit()
    np.savez_compressed(out/"synthetic_inputs.npz", **fixtures)
    require(core_hashes() == before_core, "Audited core changed during audit")
    result = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
        "runner_full_sha256_at_start": before_full, "runner_full_sha256_at_end": sha(runner.__file__),
        "core_function_AST_sha256": before_core, "baseline_sha256": sha(runner.base.__file__),
        "coordination_sha256": sha(runner.coordination.__file__), "synthetic_inputs_sha256": sha(out/"synthetic_inputs.npz"),
        "toy": toy, "full_shape": full, "tinyJ": tiny,
        "finite_difference_absolute_tolerance": 3e-8, "sender_toy_max_abs_error": sender_error,
        "receiver_toy_max_abs_error": receiver_error, "formal_actors_created": 0, "network_forward_calls": 0,
        "training_updates": 0, "optimizer_updates": 0,
        "scope": "Exact64-path and4096-pair numerical audit tied to actual sender/receiver functions. The causal toy has receiver dependence only on window2. No neural module or training path was executed.",
        "limits": ["Does not replace the separate routing/parameter-sharing/whole-run audit",
                   "Full runner may receive appended execution code; reviewed core AST must still match at formal freeze",
                   "Unbiased score estimator before clipping/Adam does not imply low variance or convergence"]}
    write(out/"verification.json", result)
    report = f'''# 两窗LOO与receiver梯度独立核验

核验通过。以3人×2窗×1二值位置的64条完整轨迹、4096个独立轨迹对作精确积分，并嵌入真实sender函数的4×8接口。第二窗分布依赖第一窗，而合成receiver只直接读取第二窗，因此非零第一窗梯度完全来自后续通信路径。

β=0及β=0.001均通过LOO期望与精确目标梯度、有限差分对照。sender有限差分最大误差{sender_error:.3g}，receiver直接梯度最大误差{receiver_error:.3g}。在真实[2,3,2,3,4,8]形状上逐一核1152个logit坐标，最大误差{full['max_abs_error']:.3g}；负号、两轨迹平均、batch平均以及全部24个score的求和一致，没有额外主体/窗口/token除数。

负对照确认：共用两轨迹的第一窗会漏掉非零早期梯度；包含自身的均值基线会把梯度缩为一半；β>0时sender若遗漏receiver熵会偏离所定义的共同目标，β=0时该差异为0。详细误差与范数见[verification.json](verification.json)。

float J为0、稳定logJ为−1000及−20000的例子中，两轨迹优势最大9500，sender与receiver导数均有限，没有epsilon或优势裁剪。无正收益支持和非有限收益被拒绝。

本次未创建正式actor，没有神经网络前向、优化器更新或训练。审查绑定核心函数AST哈希；尚在追加的完整运行代码须在正式冻结时另核来源和路由。数学无偏性仅适用于裁剪前估计，不保证任务成功或语言形成。
'''
    (out/"独立核验.md").write_text(report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), "No overwrite or automatic retry")
    args.out.mkdir(parents=True)
    try:
        result = audit(args.out)
        print(json.dumps({k: result[k] for k in ("status", "sender_toy_max_abs_error", "receiver_toy_max_abs_error", "network_forward_calls")}, ensure_ascii=False))
    except BaseException as e:
        write(args.out/"failure.json", {"status": "failed", "error": str(e), "traceback": traceback.format_exc(), "automatic_retry": False,
                                      "audit_source_sha256": sha(__file__)})
        raise


if __name__ == "__main__":
    main()
