"""Two synchronous discrete message windows in the unchanged triadic D1 task.

No formal preparation/training occurs on import. Nine independent tanh networks;
sampled sender score gradients and exact native-reward receiver gradients.
"""
from __future__ import annotations

import os
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"

import argparse
from copy import deepcopy
from itertools import combinations, zip_longest
import json
import math
import multiprocessing
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from research_program.triadic_learning_baseline import runner as base
from research_program.triadic_coordination_study import runner as coordination

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEEDS = (47101, 47102, 47103, 47104)
CONDITIONS = ("FI_silent", "FI_live", "PI_silent", "PI_live")
PARTITIONS, CHECKPOINTS = base.PARTITIONS, base.CHECKPOINTS
ALPHABET = ("@", "#", "$", "%", "&", "*", "+", "~")
MODULES = ("sender1", "sender2", "action")
DIMENSIONS = {"sender1": (54, 64, 64, 32), "sender2": (153, 64, 64, 32),
              "action": (252, 64, 64, 17)}
BASE_SHA = "1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a"
COORDINATION_SHA = "9aad4e8bbcb71f645227908ddb05559d103cd919e264dfa5b4bc4eae9aeaba87"
CONFIG = deepcopy(base.CONFIG)
CONFIG.update(seeds=list(SEEDS), conditions=list(CONDITIONS),
    training_objective="mean_log_native_expected_reward_with_two_trajectory_sender_LOO",
    full_information="condition_specific", deployment="greedy_messages_then_independent_greedy_actions",
    sender_windows=2, sender_tokens_per_window=4, alphabet=list(ALPHABET),
    sender_head="four_independent_eight_way_categorical_heads_not_autoregressive",
    dimensions={k:list(v) for k,v in DIMENSIONS.items()}, trajectories_per_state=2,
    sender_entropy_coefficient=0, sender_gradient_reduction="sum_windows_tokens_agents_mean_states_LOO_half",
    receiver_gradient_reduction="mean_two_trajectories_and_states_original_three_actor_mean_entropy",
    initialization_stream="SeedSequence([seed,agent_index,module_index,100])",
    world_stream="SeedSequence([seed,200])",
    message_stream="SeedSequence([seed,agent_index,trajectory_index,window_index,300])",
    same_uniforms_across_conditions=True, silent_retains_own_messages=True,
    run_order="four_spawn_workers_one_seed_each_conditions_serial_canonical_aggregate",
    experiment_type="discrete_symbol_message_co_learning", messages_generated=True, automatic_followon_experiment=False,
    worker_count=4, multiprocessing_start_method="spawn", blas_threads_per_worker=1,
    fixed_formal_message_trajectories_per_world=1)
CONFIG.pop("automatic_symbolic_stage",None)

require, finite, sha = base.require, base.finite, base.sha
read, write_new, json_hash, json_bytes = base.read, base.write_new, base.json_hash, base.json_bytes
array_sha, now = base.array_sha, base.now


def condition_settings(condition):
    require(condition in CONDITIONS, "Unknown information/channel condition")
    return condition.startswith("FI_"), condition.endswith("_live")


def make_network(seed, dimensions):
    rng = np.random.default_rng(seed)
    network = {}
    for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
        network[f"W{layer}"] = rng.normal(0, math.sqrt(2/(left+right)), (left,right))
        network[f"b{layer}"] = np.zeros(right, dtype=np.float64)
    return network


def make_networks(seed):
    """Flat order: A sender1/sender2/action, then B, then C; no shared arrays."""
    networks = [make_network(np.random.SeedSequence([seed,a,m,100]), DIMENSIONS[module])
                for a in range(3) for m,module in enumerate(MODULES)]
    for i,j in combinations(range(9),2):
        require(all(not np.shares_memory(networks[i][k],networks[j][k]) for k in networks[i]),
                "Network parameters are shared")
    return networks


def make_message_rngs(seed):
    return {f"t{t}_w{w}_a{a}":np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300]))
            for t in range(2) for w in range(2) for a in range(3)}


def draw_uniforms(rngs, batch_size):
    """[trajectory,world,window,agent,token]; all streams drawn even if silent."""
    require(batch_size > 0, "Empty message batch")
    uniforms = np.empty((2,batch_size,2,3,4),dtype=np.float64)
    for t in range(2):
        for w in range(2):
            for a in range(3):
                uniforms[t,:,w,a] = rngs[f"t{t}_w{w}_a{a}"].random((batch_size,4))
    return uniforms


def categorical_tokens(probabilities, uniforms=None):
    require(probabilities.shape[-2:] == (4,8), "Sender needs four eight-way heads")
    if uniforms is None:
        return np.argmax(probabilities,axis=-1).astype(np.int8)
    uniforms = np.asarray(uniforms,dtype=np.float64)
    require(uniforms.shape == probabilities.shape[:-1] and np.isfinite(uniforms).all()
            and ((uniforms >= 0) & (uniforms < 1)).all(), "Invalid categorical uniforms")
    cdf = np.cumsum(probabilities,axis=-1)
    # Correct the last floating sum to1; no probability mass or category masking.
    cdf[...,-1] = 1.0
    return (uniforms[...,None] >= cdf).sum(-1).astype(np.int8)


def routed_window(tokens, live):
    """[B,sender,4] -> [B,viewer,99]: all96 onehots then3 visibility bits.

    Missing message is all zero; category0 is a real visible symbol. Identity
    order A/B/C and every token position are fixed. No source observation enters.
    """
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3,4) and tokens.dtype.kind in 'iu'
            and ((tokens >= 0) & (tokens < 8)).all(), "Invalid actual message indices")
    visibility = np.ones((3,3),dtype=np.float64) if live else np.eye(3,dtype=np.float64)
    onehot = np.eye(8,dtype=np.float64)[tokens]
    visible = onehot[:,None,:,:,:] * visibility[None,:,:,None,None]
    bits = np.broadcast_to(visibility[None],(len(tokens),3,3))
    return np.concatenate((visible.reshape(len(tokens),3,96),bits),axis=-1)


def rollout(networks, observations, live, uniforms=None):
    """A full causal rollout; both windows complete before any next-window input.

    observations[B,3,54] contains official per-agent features only. uniforms is
    [B,2,3,4] or None for greedy messages. Returned caches never enter observations.
    """
    x = np.asarray(observations,dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3,54) and len(networks)==9,"Invalid rollout inputs")
    finite(x,"observations")
    if uniforms is not None:
        require(np.shape(uniforms)==(len(x),2,3,4),"Wrong full message trajectory uniforms")
    caches = [None]*9
    sender_logits, sender_probabilities, sender_log_probabilities, messages = [],[],[],[]
    inputs = x
    for window in range(2):
        logits = []
        for a in range(3):
            z,caches[3*a+window] = base.actor_forward(networks[3*a+window],inputs[:,a])
            logits.append(z.reshape(len(x),4,8))
        z = np.stack(logits,axis=1)
        p,lp = base.policy_distribution(z)
        m = categorical_tokens(p, None if uniforms is None else uniforms[:,window])
        sender_logits.append(z); sender_probabilities.append(p); sender_log_probabilities.append(lp); messages.append(m)
        if window == 0:
            inputs = np.concatenate((x,routed_window(m,live)),axis=-1)
    action_inputs = np.concatenate((x,routed_window(messages[0],live),routed_window(messages[1],live)),axis=-1)
    action_logits=[]
    for a in range(3):
        z,caches[3*a+2] = base.actor_forward(networks[3*a+2],action_inputs[:,a])
        action_logits.append(z)
    return {"messages":np.stack(messages,axis=1),"sender_logits":np.stack(sender_logits,axis=1),
        "sender_probabilities":np.stack(sender_probabilities,axis=1),
        "sender_log_probabilities":np.stack(sender_log_probabilities,axis=1),
        "action_logits":np.stack(action_logits,axis=1),"action_inputs":action_inputs,"caches":caches}


def paired_sender_derivative(probabilities, log_probabilities, messages, F):
    """Minimize the stopped two-trajectory score surrogate; no extra token/agent mean.

    arrays [2,B,2,3,4,8], messages without last axis, F[2,B]. This differentiates
    only logq. F and its dependence on messages are NOT differentiated here.
    Both windows use the final F, including window1's effects through window2.
    """
    p,lp,m,F = map(np.asarray,(probabilities,log_probabilities,messages,F))
    require(p.ndim==6 and p.shape[0]==2 and p.shape[2:]==(2,3,4,8)
            and lp.shape==p.shape and m.shape==p.shape[:-1] and F.shape==p.shape[:2],
            "Incorrect paired sender dimensions")
    finite(p,"sender probabilities"); finite(lp,"sender log probabilities"); finite(F,"trajectory F")
    require(m.dtype.kind in 'iu' and ((m>=0)&(m<8)).all(),"Invalid sampled message")
    advantage = np.stack(((F[0]-F[1])/2,(F[1]-F[0])/2))
    selected = np.take_along_axis(lp,m[...,None],axis=-1)[...,0]
    log_scores = selected.sum(axis=(2,3,4))
    score_gradient = np.eye(8,dtype=np.float64)[m]-p
    derivative = -advantage[:,:,None,None,None,None]*score_gradient/F.shape[1]
    surrogate = -float((advantage*log_scores).sum()/F.shape[1])
    finite(derivative,"sender derivative")
    return {"derivative":derivative,"advantage":advantage,"log_scores":log_scores,
            "surrogate_loss":surrogate}


def trajectory_objective(action_logits, rewards, update):
    """Flattened[2B] trajectories: exact receiver mean, plus each state's final F."""
    terms=coordination.objective_terms(action_logits,rewards)
    receiver=coordination.loss_and_derivative(terms,"mean_log_J",update)
    entropy,_=base.entropy_and_logit_gradient(terms["probabilities"],terms["log_probabilities"])
    F=terms["log_J"]+receiver["entropy_coefficient"]*entropy
    return terms,receiver,F


def training_gradients(networks, observations, rewards, live, uniforms, update):
    """One fixed minibatch, without optimizer calls. Discrete channel stops gradients."""
    B=len(observations)
    require(np.shape(uniforms)==(2,B,2,3,4),"Two independent full trajectories required")
    trace=rollout(networks,np.concatenate((observations,observations)),live,uniforms.reshape(2*B,2,3,4))
    terms,receiver,F=trajectory_objective(trace["action_logits"],np.concatenate((rewards,rewards)),update)
    sender=paired_sender_derivative(trace["sender_probabilities"].reshape(2,B,2,3,4,8),
        trace["sender_log_probabilities"].reshape(2,B,2,3,4,8),trace["messages"].reshape(2,B,2,3,4),F.reshape(2,B))
    gradients=[]
    ds=sender["derivative"].reshape(2*B,2,3,4,8)
    for a in range(3):
        for module in range(3):
            d=receiver["derivative"][:,a] if module==2 else ds[:,module,a].reshape(2*B,32)
            gradients.append(base.actor_backward(networks[3*a+module],trace["caches"][3*a+module],d))
    row={"mean_J":float(terms["J"].mean()),"mean_log_J":float(terms["log_J"].mean()),
        "min_log_J":float(terms["log_J"].min()),"max_log_J":float(terms["log_J"].max()),
        "zero_float_J_states":int((terms["J"]==0).sum()),"mean_F":float(F.mean()),
        "receiver_loss":receiver["loss"],"mean_actor_entropy":receiver["mean_actor_entropy"],
        "entropy_coefficient":receiver["entropy_coefficient"],"sender_surrogate_loss":sender["surrogate_loss"],
        "sender_advantage_mean":float(sender["advantage"].mean()),
        "sender_advantage_abs_mean":float(np.abs(sender["advantage"]).mean()),
        "sender_advantage_squared_mean":float(np.square(sender["advantage"]).mean()),
        "sender_advantage_max_abs":float(np.abs(sender["advantage"]).max()),
        "sender_mean_complete_log_score":float(sender["log_scores"].mean()),
        "sampled_messages_sha256":array_sha(trace["messages"])}
    return gradients,row


def make_prepared():
    old=base.make_prepared()
    prepared=deepcopy(old)
    prepared.update(schema="triadic_message_v1",config=deepcopy(CONFIG),
        baseline_prepared_sha256=json_hash(old),baseline_partitions_sha256=json_hash(old["partitions"]),
        runs=[{"seed":seed,"condition":condition,"directory":f"seed_{seed}_{condition}"}
              for seed in SEEDS for condition in CONDITIONS],
        training_state_samples_per_run=6000*256,training_state_samples_total=16*6000*256,
        sampled_complete_message_trajectories_total=16*6000*256*2,
        categorical_message_samples_total=16*6000*256*2*2*3*4,
        weighted_structural_action_contributions=16*6000*256*2*24,
        offline_reward_table_entries_per_worker=143424*24,
        offline_reward_table_entries_actual=4*143424*24,complete_final_world_evaluations=16*143424,
        parameter_count_per_agent=sum((l+1)*r for dimensions in DIMENSIONS.values()
                                      for l,r in zip(dimensions,dimensions[1:])))
    prepared.pop("training_state_samples_per_seed",None)
    require(prepared["partitions"]==old["partitions"] and prepared["actions"]==old["actions"],
            "Original observation domain or action support changed")
    return prepared


def sources():
    require(sha(base.__file__)==BASE_SHA,"Frozen baseline runner changed")
    require(sha(coordination.__file__)==COORDINATION_SHA,"Frozen coordination objective changed")
    paths=(Path(__file__),HERE/"plan.md",HERE/"tests/test_runner.py",HERE/"运行说明.md",
           HERE/"seed_selection_receipt.json",Path(base.__file__),Path(coordination.__file__),Path(base.env.__file__))
    require(all(p.is_file() for p in paths),"Missing code/plan/tests/source before freezing")
    return {str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}


def prepare(output):
    output=Path(output).resolve()
    require(not output.exists(),"Refuse to overwrite preparation")
    hashes=sources()
    prepared=make_prepared()
    plan={"schema":"triadic_message_v1","prepared_at":now(),"config":deepcopy(CONFIG),
          "prepared_sha256":json_hash(prepared),"sources":hashes,
          "runtime":{"python":platform.python_version(),"numpy":np.__version__},
          "no_training_or_network_initialization_by_prepare":True}
    output.mkdir(parents=True,exist_ok=False)
    for relative in hashes:
        target=output/"source_snapshot"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    write_new(output/"prepared.json",prepared); write_new(output/"plan.json",plan)
    write_new(output/"freeze.json",{"plan_sha256":sha(output/"plan.json"),"prepared_sha256":sha(output/"prepared.json")})
    verify(output)
    return {"status":"prepared_not_trained","output":str(output),"plan_sha256":sha(output/"plan.json"),
            "runs":prepared["runs"],"training_state_samples_total":prepared["training_state_samples_total"]}


def verify(output):
    output=Path(output).resolve()
    plan,prepared,freeze=[read(output/n) for n in ("plan.json","prepared.json","freeze.json")]
    require(sha(output/"plan.json")==freeze["plan_sha256"],"Plan changed")
    require(sha(output/"prepared.json")==freeze["prepared_sha256"]==plan["prepared_sha256"],"Prepared input changed")
    require(json_hash(make_prepared())==plan["prepared_sha256"],"Current prepared values differ")
    require(plan["config"]==CONFIG and plan["runtime"]=={"python":platform.python_version(),"numpy":np.__version__},
            "Configuration or runtime changed")
    require(plan["sources"]==sources(),"Current sources changed")
    for relative,digest in plan["sources"].items():
        require(sha(output/"source_snapshot"/relative)==digest,"Frozen source snapshot changed")
    return plan,prepared


def build_arrays(spec):
    arrays=base.build_arrays(spec)
    arrays["x_FI"]=arrays.pop("x")
    private=np.empty_like(arrays["x_FI"])
    for start in range(0,len(arrays["states"]),1024):
        chunk=arrays["states"][start:start+1024]
        views=[{a:base.env.observe(s,a,shared_needs=False,full_information=False) for a in base.AGENTS} for s in chunk]
        private[start:start+len(chunk)]=base.encode_observations(views)
    arrays["x_PI"]=private
    return arrays


def parameter_hash(networks):
    return json_hash({f"agent{a}_{module}_{key}":array_sha(value)
                     for a in range(3) for m,module in enumerate(MODULES)
                     for key,value in networks[3*a+m].items()})


def save_checkpoint(path, networks, optimizer, update, batch_rng, message_rngs):
    require(not Path(path).exists(),"Checkpoint exists")
    payload={f"agent{a}_{module}_{key}":value for a in range(3) for m,module in enumerate(MODULES)
             for key,value in networks[3*a+m].items()}
    for a in range(3):
        for m,module in enumerate(MODULES):
            for moment in ("m","v"):
                payload.update({f"adam_agent{a}_{module}_{moment}_{key}":value
                                for key,value in optimizer[3*a+m][moment].items()})
    payload["update"]=np.array(update,dtype=np.int64)
    payload["batch_rng_json"]=np.array(json.dumps(batch_rng.bit_generator.state,sort_keys=True))
    payload["message_rngs_json"]=np.array(json.dumps({key:rng.bit_generator.state for key,rng in message_rngs.items()},sort_keys=True))
    np.savez_compressed(path,**payload)
    return sha(path)


def load_networks(path):
    """Read nine saved parameter arrays only; used by separately frozen probes."""
    networks=[]
    with np.load(path,allow_pickle=False) as saved:
        for a in range(3):
            for module in MODULES:
                dimensions=DIMENSIONS[module]; network={}
                for layer,(left,right) in enumerate(zip(dimensions,dimensions[1:]),1):
                    for key,shape in ((f"W{layer}",(left,right)),(f"b{layer}",(right,))):
                        value=saved[f"agent{a}_{module}_{key}"].copy()
                        require(value.shape==shape and value.dtype==np.float64,"Saved network shape/dtype mismatch")
                        finite(value,"loaded parameter");network[key]=value
                networks.append(network)
    return networks


def evaluate(networks, arrays, condition, indices=None, *, save_path=None):
    """Always greedy whole messages then greedy actions; no reward-based selection.

    Exact receiver statistics are CONDITIONAL on these greedy messages, not
    expected outcomes under sampling the entire communication policy.
    """
    full,live=condition_settings(condition)
    if indices is None:
        indices=np.arange(len(arrays["states"]),dtype=np.int64)
    else:
        indices=np.asarray(indices,dtype=np.int64)
    require(indices.ndim==1 and len(indices)>0 and len(np.unique(indices))==len(indices)
            and ((indices>=0)&(indices<len(arrays["states"]))).all(),"Invalid evaluation indices")
    n=len(indices)
    probabilities=np.empty((n,3,17),dtype=np.float64)
    choices=np.empty((n,3),dtype=np.int16)
    messages=np.empty((n,2,3,4),dtype=np.int8)
    rewards=np.empty(n,dtype=np.float64)
    satisfied=np.empty((n,3),dtype=bool); executed=np.empty((n,3),dtype=bool)
    expected=np.empty(n); full_probability=np.empty(n); execution_probability=np.empty(n)
    sender_ties=0; action_ties=0
    for start in range(0,n,CONFIG["evaluation_batch_size"]):
        ids=indices[start:start+CONFIG["evaluation_batch_size"]]; dest=slice(start,start+len(ids))
        trace=rollout(networks,arrays["x_FI" if full else "x_PI"][ids],live)
        probs,_=base.policy_distribution(trace["action_logits"])
        greedy=np.argmax(probs,axis=-1)
        action_ties+=int(((probs==probs.max(-1,keepdims=True)).sum(-1)>1).sum())
        sp=trace["sender_probabilities"]
        sender_ties+=int(((sp==sp.max(-1,keepdims=True)).sum(-1)>1).sum())
        probabilities[dest]=probs; choices[dest]=greedy; messages[dest]=trace["messages"]
        expected[dest],full_probability[dest],execution_probability[dest]=base.exact_statistics(probs,arrays["rewards"][ids])
        for k,index in enumerate(ids):
            actions={a:base.ACTIONS[i][greedy[k,i]] for i,a in enumerate(base.AGENTS)}
            outcome=base.env.settle(arrays["states"][index],actions,require_match=True)
            row=start+k; rewards[row]=outcome["reward"]
            for i,a in enumerate(base.AGENTS):
                satisfied[row,i]=outcome["individual_feedback"][a]["own_need_satisfied"]
                executed[row,i]=outcome["individual_feedback"][a]["executed"]
    attempted=(choices!=0).sum(1); physically_executed=executed.sum(1)
    unique,counts=np.unique(choices,axis=0,return_counts=True)
    result={"worlds":n,"condition":condition,"message_mode":"greedy","action_mode":"greedy",
        "greedy_reward_sum":float(rewards.sum()),"greedy_reward_mean":float(rewards.mean()),
        "greedy_full_successes":int((rewards==1).sum()),"greedy_full_success_rate":float((rewards==1).mean()),
        "greedy_non_full_success_worlds":int((rewards!=1).sum()),
        "greedy_reward_counts":{str(r):int((rewards==r).sum()) for r in (0.0,0.5,1.0)},
        "greedy_attempted_transports":int(attempted.sum()),"greedy_executed_transports":int(executed.sum()),
        "greedy_satisfied_agents":int(satisfied.sum()),"greedy_active_count_worlds":{str(k):int((attempted==k).sum()) for k in range(4)},
        "greedy_agent_argmax_ties":action_ties,"greedy_sender_token_argmax_ties":sender_ties,
        "conditional_exact_expected_reward_mean":float(expected.mean()),
        "conditional_exact_full_success_probability_mean":float(full_probability.mean()),
        "conditional_exact_physical_execution_probability_mean":float(execution_probability.mean()),
        "exact_statistics_scope":"Receiver sampling conditional on the saved greedy message rollout; not fully stochastic messages.",
        "greedy_failure_categories":{"all_wait":int((attempted==0).sum()),"single_transport_proposal":int((attempted==1).sum()),
            "overload":int((attempted==3).sum()),"two_unmatched":int(((attempted==2)&(physically_executed==0)).sum()),
            "matched_no_need_satisfied":int(((physically_executed==2)&(rewards==0)).sum()),
            "matched_one_need_satisfied":int((rewards==.5).sum())},
        "greedy_executed_pair_worlds":{base.AGENTS[i]+base.AGENTS[j]:int((executed[:,i]&executed[:,j]).sum())
                                      for i,j in combinations(range(3),2)},
        "distinct_joint_argmax_actions":len(unique),
        "joint_argmax_action_counts":[{"action_indices":a.tolist(),"worlds":int(c)} for a,c in zip(unique,counts)],
        "state_indices_sha256":array_sha(indices),"packed_states_sha256":array_sha(arrays["packed_states"][indices]),
        "messages_sha256":array_sha(messages),"action_indices_sha256":array_sha(choices)}
    if save_path is not None:
        require(not Path(save_path).exists(),"Evaluation output exists")
        np.savez_compressed(save_path,states=arrays["packed_states"][indices],state_indices=indices,
            messages=messages,action_indices=choices,action_probabilities=probabilities,
            greedy_reward=rewards,executed=executed,satisfied=satisfied,
            conditional_exact_expected_reward=expected,conditional_exact_full_success_probability=full_probability,
            conditional_exact_execution_probability=execution_probability)
        result["data_sha256"]=sha(save_path)
    return result


def train_run(seed, condition, prepared, arrays, output):
    full,live=condition_settings(condition)
    output.mkdir(exist_ok=False)
    networks=make_networks(seed); initial_sha=parameter_hash(networks)
    optimizer=base.make_adam(networks)
    batch_rng=np.random.default_rng(np.random.SeedSequence([seed,200]))
    message_rngs=make_message_rngs(seed)
    started=time.perf_counter(); monitor=[]
    def checkpoint(update):
        digest=save_checkpoint(output/f"checkpoint_{update:04d}.npz",networks,optimizer,update,batch_rng,message_rngs)
        scores={name:evaluate(networks,arrays[name],condition,prepared["partitions"][name]["monitor_indices"],
                             save_path=output/f"monitor_{update:04d}_{name}.npz") for name in PARTITIONS}
        row={"update":update,"checkpoint_sha256":digest,"monitor":scores,"elapsed_seconds":time.perf_counter()-started}
        monitor.append(row)
        with (output/"monitor.jsonl").open("a",encoding="utf-8") as stream: stream.write(json_bytes(row).decode())
    checkpoint(0)
    with (output/"training.jsonl").open("x",encoding="utf-8") as stream:
        for update in range(1,CONFIG["updates"]+1):
            ids=batch_rng.integers(0,len(arrays["train"]["states"]),size=CONFIG["batch_size"],dtype=np.int64)
            uniforms=draw_uniforms(message_rngs,len(ids))
            gradients,row=training_gradients(networks,arrays["train"]["x_FI" if full else "x_PI"][ids],
                                             arrays["train"]["rewards"][ids],live,uniforms,update)
            norm,scale=base.adam_step(networks,gradients,optimizer,update)
            row.update(update=update,condition=condition,batch_indices_sha256=array_sha(ids),
                       batch_states_sha256=array_sha(arrays["train"]["packed_states"][ids]),
                       sample_uniforms_sha256=array_sha(uniforms),gradient_norm=norm,gradient_clip_scale=scale,
                       elapsed_seconds=time.perf_counter()-started)
            stream.write(json_bytes(row).decode())
            if update in CHECKPOINTS:
                stream.flush(); checkpoint(update)
    final={name:evaluate(networks,arrays[name],condition,save_path=output/f"final_{name}.npz") for name in PARTITIONS}
    counts={}
    for value in final.values():
        for row in value["joint_argmax_action_counts"]:
            key=tuple(row["action_indices"]); counts[key]=counts.get(key,0)+row["worlds"]
    domain={"worlds":sum(counts.values()),"distinct_joint_argmax_actions":len(counts),
            "joint_argmax_action_counts":[{"action_indices":list(a),"worlds":n} for a,n in sorted(counts.items())]}
    require(domain["worlds"]==143424,"Incomplete full-domain action inventory")
    result={"seed":seed,"condition":condition,"updates":6000,"training_world_samples":6000*256,
            "sampled_complete_message_trajectories":6000*256*2,
            "initial_parameter_sha256":initial_sha,"final_parameter_sha256":parameter_hash(networks),
            "final":final,"full_domain_actions":domain,"monitor":monitor,
            "candidate_threshold_met":all(final[p]["greedy_full_success_rate"]>=.99 for p in PARTITIONS if p!="train"),
            "training_log_sha256":sha(output/"training.jsonl"),"final_checkpoint_sha256":sha(output/"checkpoint_6000.npz"),
            "elapsed_seconds":time.perf_counter()-started}
    write_new(output/"result.json",result)
    return result


def check_pairing(execution, results):
    index={(r["seed"],r["condition"]):r for r in results}
    require(len(results)==16 and set(index)=={(s,c) for s in SEEDS for c in CONDITIONS},"Missing or duplicate conditions")
    checks=[]
    for seed in SEEDS:
        rows=[index[(seed,c)] for c in CONDITIONS]
        require(len({r["initial_parameter_sha256"] for r in rows})==1,"Cross-condition initial parameters differ")
        paths=[execution/f"seed_{seed}_{c}"/"training.jsonl" for c in CONDITIONS]
        streams=[p.open() for p in paths]
        try:
            count=0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines),"Unequal paired training log length")
                group=[json.loads(line) for line in lines]
                require(all(all(row[k]==group[0][k] for k in ("update","batch_indices_sha256","batch_states_sha256",
                    "sample_uniforms_sha256","entropy_coefficient")) for row in group),"Paired world/uniform stream differs")
                count+=1
            require(count==6000,"Missing paired updates")
        finally:
            for stream in streams:stream.close()
        checks.append({"seed":seed,"same_initial_parameters":True,"paired_world_and_message_uniform_updates":count,
                       "same_entropy_schedule":True,"initial_policy_outputs_not_required_equal":True})
    return checks


def primary_comparison(results):
    index={(r["seed"],r["condition"]):r for r in results}
    require(len(results)==16 and set(index)=={(s,c) for s in SEEDS for c in CONDITIONS},"All sixteen runs required")
    rows=[]
    for seed in SEEDS:
        scores={c:index[(seed,c)]["final"]["new_needs_and_layouts"] for c in CONDITIONS}
        require(all(v["worlds"]==8244 for v in scores.values()),"Primary heldout denominator differs")
        rates={c:v["greedy_full_success_rate"] for c,v in scores.items()}
        pi=rates["PI_live"]-rates["PI_silent"]; fi=rates["FI_live"]-rates["FI_silent"]
        rows.append({"seed":seed,"full_success_rates":rates,"PI_live_minus_silent":pi,
                     "FI_live_minus_silent":fi,"difference_in_differences_PI_minus_FI":pi-fi})
    return {"partition":"new_needs_and_layouts","endpoint_update":6000,"metric":"greedy_full_success_rate",
            "primary":"PI_live_minus_silent","seed_pairs":rows,
            "equal_weight_means":{key:sum(row[key] for row in rows)/4 for key in
                ("PI_live_minus_silent","FI_live_minus_silent","difference_in_differences_PI_minus_FI")},
            "independent_paired_initializations":4,"significance_test":None}


def seed_worker(output, seed):
    """Spawn entry: one independent seed, all four conditions in fixed order."""
    output=Path(output).resolve(); _,prepared=verify(output)
    require(seed in SEEDS,"Unknown worker seed")
    execution=output/"execution"
    require((execution/"started.json").is_file(),"Only a started batch may launch workers")
    worker=execution/f"worker_seed_{seed}"
    worker.mkdir(exist_ok=False); started=time.perf_counter()
    write_new(worker/"started.json",{"seed":seed,"pid":os.getpid(),"parent_pid":os.getppid(),
        "started_at":now(),"plan_sha256":sha(output/"plan.json"),
        "thread_environment":{key:os.environ[key] for key in THREAD_KEYS}})
    try:
        arrays={name:build_arrays(prepared["partitions"][name]) for name in PARTITIONS}
        write_new(worker/"input_arrays.json",{name:{"worlds":len(a["states"]),
            "full_information_features_sha256":array_sha(a["x_FI"]),"private_information_features_sha256":array_sha(a["x_PI"]),
            "native_rewards_sha256":array_sha(a["rewards"]),"states_sha256":array_sha(a["packed_states"])} for name,a in arrays.items()})
        results=[]
        for run in prepared["runs"]:
            if run["seed"]!=seed:continue
            result=train_run(seed,run["condition"],prepared,arrays,execution/run["directory"])
            results.append(result)
            print(json.dumps({"completed_run":run,"final_full_success_rates":{k:v["greedy_full_success_rate"] for k,v in result["final"].items()},
                              "worker_elapsed_seconds":time.perf_counter()-started}),flush=True)
        verify(output)
        require([r["condition"] for r in results]==list(CONDITIONS),"Incomplete worker condition sequence")
        write_new(worker/"results.json",{"status":"completed","seed":seed,"runs":results,
            "offline_reward_table_entries":143424*24,"elapsed_seconds":time.perf_counter()-started})
        write_new(worker/"status.json",{"status":"completed","completed_at":now()})
    except BaseException as error:
        write_new(worker/"failure.json",{"status":"failed","seed":seed,"failed_at":now(),
            "error_type":type(error).__name__,"error":str(error),"elapsed_seconds":time.perf_counter()-started})
        write_new(worker/"status.json",{"status":"failed","failed_at":now()})
        raise


def stop_own_workers(processes):
    """Only handles explicitly created by this execute; no global process lookup."""
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:
        if process.pid is None:continue
        process.join(timeout=5)
        if process.is_alive():
            process.kill();process.join(timeout=5)


def wait_for_workers(processes):
    while True:
        failed=[p for p in processes if p.exitcode is not None and p.exitcode!=0]
        if failed:
            codes=[{"name":p.name,"pid":p.pid,"exitcode":p.exitcode} for p in failed]
            stop_own_workers(processes)
            raise RuntimeError("This batch's worker failed; remaining own workers stopped: "+json.dumps(codes))
        if all(p.exitcode==0 for p in processes):
            for process in processes:process.join()
            return
        time.sleep(.2)


def execute(output):
    output=Path(output).resolve(); plan,prepared=verify(output)
    execution=output/"execution"
    require(not execution.exists(),"Never resume or overwrite started execution")
    execution.mkdir(exist_ok=False); started=time.perf_counter()
    write_new(execution/"started.json",{"started_at":now(),"plan_sha256":sha(output/"plan.json"),
              "device":"cpu_numpy","pid":os.getpid(),"thread_environment":{key:os.environ[key] for key in THREAD_KEYS}})
    processes=[]
    try:
        context=multiprocessing.get_context("spawn")
        for seed in SEEDS:
            process=context.Process(target=seed_worker,args=(str(output),seed),name=f"triadic_messages_seed_{seed}")
            processes.append(process);process.start()
        wait_for_workers(processes)
        results=[]
        inputs=[]
        for seed in SEEDS:
            worker=execution/f"worker_seed_{seed}"
            record=read(worker/"results.json")
            require(record["status"]=="completed" and record["seed"]==seed,"Wrong or partial worker result")
            require([r["condition"] for r in record["runs"]]==list(CONDITIONS),"Wrong worker condition order")
            for row in record["runs"]:
                require(row==read(execution/f"seed_{seed}_{row['condition']}"/"result.json"),"Worker aggregate differs from run result")
            results.extend(record["runs"])
            inputs.append(read(worker/"input_arrays.json"))
        require(all(value==inputs[0] for value in inputs),"Workers did not use identical input arrays")
        write_new(execution/"input_arrays.json",{"worker_count":4,"all_worker_arrays_identical":True,
            "arrays":inputs[0],"actual_offline_reward_table_entries":prepared["offline_reward_table_entries_actual"]})
        pairing=check_pairing(execution,results)
        verify(output)
        result={"status":"completed","completed_at":now(),"plan_sha256":sha(output/"plan.json"),
            "runs":results,"completed_run_count":16,"paired_seed_count":4,"pairing_checks":pairing,
            "primary_comparison":primary_comparison(results),"updates_total":16*6000,
            "training_world_samples_total":prepared["training_state_samples_total"],
            "sampled_complete_message_trajectories_total":prepared["sampled_complete_message_trajectories_total"],
            "weighted_structural_action_contributions":prepared["weighted_structural_action_contributions"],
            "offline_reward_table_entries_actual":prepared["offline_reward_table_entries_actual"],
            "worker_processes":[{"name":p.name,"pid":p.pid,"exitcode":p.exitcode} for p in processes],
            "all_seed_candidates_by_condition":{c:all(r["candidate_threshold_met"] for r in results if r["condition"]==c) for c in CONDITIONS},
            "experiment_type":"discrete_symbol_message_co_learning","messages_generated":True,
            "automatic_followon_experiment":False,"language_or_convention_claim_automatically_supported":False,
            "elapsed_seconds":time.perf_counter()-started,
            "scope":"First discrete-message learning comparison with counterfactual-rich centralized training and independent deployment; endpoint channel/content interventions are separate."}
        write_new(execution/"results.json",result)
        write_new(execution/"status.json",{"status":"completed","completed_at":now(),"completed_runs":16})
    except BaseException as error:
        stop_own_workers(processes)
        write_new(execution/"failure.json",{"status":"failed","failed_at":now(),"error_type":type(error).__name__,
                  "error":str(error),"elapsed_seconds":time.perf_counter()-started})
        write_new(execution/"status.json",{"status":"failed","failed_at":now()})
        raise
    return {"status":"completed","output":str(execution),"primary_comparison":result["primary_comparison"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","verify","execute"));parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.command=="prepare":result=prepare(args.out)
    elif args.command=="verify":
        verify(args.out); result={"status":"verified_not_trained","output":str(args.out.resolve())}
    else:result=execute(args.out)
    print(json.dumps(result,ensure_ascii=False))
