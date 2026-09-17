"""Run the posthoc W1 directional probe on final multi-legal checkpoints."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as original
from research_program.triadic_multi_legal_coordination_study import design as multi_design
from research_program.triadic_reciprocal_execution_study import environment as execution_env
from research_program.triadic_message_study import runner as core
from research_program.triadic_directed_message_study import intervention
from . import design, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_RUN = ROOT / 'research_program/triadic_multi_legal_coordination_study/results/confirm_001'
SEEDS, CONDITIONS, PARTS = design.SEEDS, design.CONDITIONS, design.PARTS
CONFIG = dict(schema='triadic_multi_legal_direction_probe_v1', seeds=list(SEEDS), conditions=list(CONDITIONS),
              parts=list(PARTS), checkpoint=design.CHECKPOINT, chunk_size=design.CHUNK_SIZE,
              sham_rows_per_sender=design.SHAM_ROWS_PER_SENDER, overwrite_windows=[True, False],
              natural_messages='greedy live messages from each final policy checkpoint',
              donor='same-layout same-owner cyclic-neighbor need; only selected sender W1 replaced',
              primary='mean signed donor-plan-only minus receiver-plan-only probability transfer',
              secondary='signed partner transfer, natural/intervened physical execution and Q, exact sham replay',
              posthoc_relative_to='triadic_multi_legal_coordination_study/results/confirm_001',
              new_model_calls=True, training_updates=0, automatic_followon_experiment=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return core.base.sha(Path(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_manifest():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'metrics.py', 'runner.py', 'plan.md', 'tests/test_design.py')]
    paths += [Path(intervention.__file__), Path(core.__file__), Path(core.base.__file__), Path(dataset.__file__),
              Path(original.__file__), Path(execution_env.__file__), SOURCE_RUN / 'plan.json',
              SOURCE_RUN / 'prepared.json', SOURCE_RUN / 'freeze.json',
              SOURCE_RUN / 'aggregation_001/results.json', SOURCE_RUN / 'audit_001/verification.json']
    require(all(path.is_file() for path in paths), 'Missing probe source or frozen input')
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def prepared():
    static = design.make_prepared()
    static['config'] = CONFIG
    static['source_sha256'] = source_manifest()
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    static['budget'] = dict(policy_states=len(SEEDS) * len(CONDITIONS), natural_worlds=len(SEEDS) * len(CONDITIONS) * design.source_partitions()['world_count'],
                            natural_module_samples=len(SEEDS) * len(CONDITIONS) * design.source_partitions()['world_count'] * 9,
                            cross_case_rows=len(SEEDS) * len(CONDITIONS) * design.make_cases(design.source_partitions())['case_count'],
                            sham_case_rows=len(SEEDS) * len(CONDITIONS) * 3 * design.SHAM_ROWS_PER_SENDER,
                            intervention_module_samples=0)
    # Each cross and sham row invokes six modules: three W2 heads and three action heads.
    # The silent arm is an exact channel-closed replay: only the sixteen live
    # policy states run the six-module outward replacement intervention.
    static['budget']['intervention_module_samples'] = ((static['budget']['cross_case_rows'] + static['budget']['sham_case_rows']) // len(CONDITIONS)) * 6
    static['budget']['total_new_module_samples'] = static['budget']['natural_module_samples'] + static['budget']['intervention_module_samples']
    return static


def frozen_inputs():
    source = json.loads((SOURCE_RUN / 'prepared.json').read_text())
    plan = json.loads((SOURCE_RUN / 'plan.json').read_text())
    freeze = json.loads((SOURCE_RUN / 'freeze.json').read_text())
    require(sha(SOURCE_RUN / 'plan.json') == freeze['plan_sha256'], 'Source plan freeze mismatch')
    require(sha(SOURCE_RUN / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Source prepared freeze mismatch')
    result_path = SOURCE_RUN / 'aggregation_001/results.json'
    audit_path = SOURCE_RUN / 'audit_001/verification.json'
    aggregate = json.loads(result_path.read_text()); audit = json.loads(audit_path.read_text())
    require(aggregate['status'] == 'completed_json_only_posthoc_aggregation' or aggregate['status'] == 'completed_json_only_summary' or aggregate['status'].startswith('completed'), 'Source aggregate incomplete')
    require(audit['status'] == 'passed', 'Source audit incomplete')
    checkpoints = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            directory = SOURCE_RUN / 'execution' / f'seed_{seed}_{condition}'
            result = json.loads((directory / 'result.json').read_text())
            path = directory / 'checkpoint_6000.npz'
            require(sha(path) == result['final_checkpoint_sha256'], 'Source checkpoint hash mismatch')
            checkpoints[f'{seed}:{condition}'] = dict(path=str(path.resolve()), sha256=sha(path), seed=seed, condition=condition)
    return dict(plan=plan, prepared=source, freeze=freeze, aggregate_sha256=sha(result_path), audit_sha256=sha(audit_path), checkpoints=checkpoints)


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); inputs = frozen_inputs(); out.mkdir(parents=True)
    for relative in static['source_sha256']:
        source = ROOT / relative
        target = out / 'source_snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    write(out / 'prepared.json', static)
    write(out / 'inputs.json', inputs)
    write(out / 'plan.json', dict(status='prepared_without_probe_forward', at=core.base.now(), config=CONFIG,
                                  prepared_sha256=sha(out / 'prepared.json'), inputs_sha256={k: v for k, v in inputs.items() if k.endswith('_sha256')},
                                  source_sha256=static['source_sha256'], runtime=static['runtime']))
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), prepared_sha256=sha(out / 'prepared.json'), inputs_sha256=sha(out / 'inputs.json')))
    verify(out); return dict(status='prepared_without_probe_forward', output=str(out), budget=static['budget'])


def verify(out):
    out = Path(out).resolve(); static = json.loads((out / 'prepared.json').read_text()); plan = json.loads((out / 'plan.json').read_text()); freeze = json.loads((out / 'freeze.json').read_text()); inputs = json.loads((out / 'inputs.json').read_text())
    require(sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Probe prepared hash mismatch')
    require(sha(out / 'plan.json') == freeze['plan_sha256'], 'Probe plan hash mismatch')
    require(sha(out / 'inputs.json') == freeze['inputs_sha256'], 'Probe input manifest hash mismatch')
    require(static == prepared() and plan['config'] == CONFIG and plan['source_sha256'] == source_manifest(), 'Probe source/config changed')
    current = frozen_inputs(); require(inputs == current, 'Frozen source inputs changed')
    for relative, digest in plan['source_sha256'].items(): require(sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static, inputs


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information='PL'); arrays['native_rewards'] = arrays['rewards'].copy(); return arrays


def parameter_hash(networks):
    return core.parameter_hash(networks)


def native_bank(networks, arrays, live):
    n = len(arrays['packed_states']); bank = dict(states=arrays['packed_states'].copy(), messages=np.empty((n, 2, 3, 4), np.int8),
        action_probabilities=np.empty((n, 3, 17), np.float64), action_indices=np.empty((n, 3), np.int16))
    for start in range(0, n, design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, n); trace = core.rollout(networks, arrays['x_PL'][start:stop], bool(live))
        probs, _ = core.base.policy_distribution(trace['action_logits'])
        bank['messages'][start:stop] = trace['messages']; bank['action_probabilities'][start:stop] = probs; bank['action_indices'][start:stop] = probs.argmax(-1)
    require(np.array_equal(bank['action_indices'], bank['action_probabilities'].argmax(-1)), 'Natural greedy actions mismatch')
    return bank


def _plan_actions(plan):
    return tuple(dataset.plan_action_indices(plan))


def _plan_sets(states):
    receiver_sets = []; donor_sets = []
    for row in states:
        needs = tuple(map(int, row[:3])); layout = tuple(map(int, row[3:7]))
        plans = original.full_success_plans(needs, layout)
        require(len(plans) == 2, 'Receiver state is not a two-plan world')
        receiver_sets.append({_plan_actions(plan) for plan in plans})
    return receiver_sets


def _mass(probabilities, action_sets):
    result = np.zeros(len(probabilities), dtype=np.float64)
    for action_set in action_sets:
        for actions in action_set:
            result += probabilities[:, 0, actions[0]] * probabilities[:, 1, actions[1]] * probabilities[:, 2, actions[2]]
    return result


def _sets_for_indices(states):
    sets = []
    for row in states:
        plans = original.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        require(len(plans) == 2, 'State is not a two-plan world')
        sets.append({_plan_actions(plan) for plan in plans})
    return sets


def _partner_probability(probabilities, sender, recipient):
    role = execution_env.PROPOSAL_ROLES[recipient]
    mask = role == sender
    return probabilities[:, recipient, mask].sum(axis=1)


def _physical_metrics(states, probabilities, legal_sets):
    actions = probabilities.argmax(-1).astype(np.int16); settled = execution_env.settle(states, actions, 'strict')
    physical = settled['actual_pair_index'] >= 0; q = np.zeros(len(states), dtype=bool)
    for i, action_set in enumerate(legal_sets):
        pair = int(settled['actual_pair_index'][i]); site = int(settled['executed_site'][i]); dest = int(settled['executed_destination'][i])
        if pair < 0:
            continue
        actual = tuple(actions[i].tolist())
        q[i] = actual in action_set
    return dict(actions=actions, physical_rate=float(physical.mean()), q_rate=float(q.mean()), conditional_q=float(q[physical].mean()) if physical.any() else None,
                executed_count=int(physical.sum()), q_count=int(q.sum()))


def _case_arrays(cases, bank, part_spec, sender):
    mask = np.asarray(cases['sender'], dtype=np.int8) == sender; ids = np.flatnonzero(mask)
    return ids, np.asarray(cases['receiver_state_indices'], dtype=np.int64)[ids], np.asarray(cases['donor_state_indices'], dtype=np.int64)[ids]


def run_policy(networks, arrays, spec, cases, live):
    bank = native_bank(networks, arrays, live); case_rows = []
    legal_sets_cache = _sets_for_indices(bank['states'])
    by_sender = []
    for sender in range(3):
        ids, receiver_ids, donor_ids = _case_arrays(cases, bank, spec, sender); n = len(ids)
        natural_prob = bank['action_probabilities'][receiver_ids]
        donor_packets = bank['messages'][donor_ids, :, sender]
        sums = dict(plan_transfer=[], partner_transfer=[], natural_physical=[], natural_q=[], natural_conditional_q=[], intervention_physical=[], intervention_q=[], intervention_conditional_q=[])
        for start in range(0, n, design.CHUNK_SIZE):
            stop = min(start + design.CHUNK_SIZE, n); sl = slice(start, stop); ri = receiver_ids[sl]; di = donor_ids[sl]
            nat = natural_prob[sl]; states = bank['states'][ri]
            if live:
                result = intervention.intervene(networks, arrays['x_PL'][ri], bank['messages'][ri], np.full(stop - start, sender, dtype=np.int64), donor_packets[sl], overwrite_windows=(True, False))
                inter = result['action_probabilities']
            else:
                # With a closed channel, outward packets are not routed to other agents;
                # replacing them cannot change any receiver input. Reuse the exact natural
                # action probabilities and charge no new forward calls.
                inter = nat
            nat_sets = [legal_sets_cache[int(i)] for i in ri]; donor_sets = [legal_sets_cache[int(i)] for i in di]
            plan_values = []; partner_values = []
            for row in range(len(ri)):
                donor_only = donor_sets[row] - nat_sets[row]; receiver_only = nat_sets[row] - donor_sets[row]
                donor_int = _mass(inter[row:row + 1], [donor_only])[0] if donor_only else 0.0
                donor_nat = _mass(nat[row:row + 1], [donor_only])[0] if donor_only else 0.0
                receiver_int = _mass(inter[row:row + 1], [receiver_only])[0] if receiver_only else 0.0
                receiver_nat = _mass(nat[row:row + 1], [receiver_only])[0] if receiver_only else 0.0
                plan_values.append((donor_int - receiver_int) - (donor_nat - receiver_nat))
                total = 0.0; count = 0
                receiver_pairs = {tuple(plan)[0:2] for plan in original.full_success_plans(tuple(states[row, :3]), tuple(states[row, 3:7]))}
                donor_row = bank['states'][di[row]]; donor_pairs = {tuple(plan)[0:2] for plan in original.full_success_plans(tuple(donor_row[:3]), tuple(donor_row[3:7]))}
                for recipient in range(3):
                    if recipient == sender: continue
                    desired = int(tuple(sorted((sender, recipient))) in donor_pairs) - int(tuple(sorted((sender, recipient))) in receiver_pairs)
                    if desired:
                        total += desired * (_partner_probability(inter[row:row + 1], sender, recipient)[0] - _partner_probability(nat[row:row + 1], sender, recipient)[0]); count += 1
                partner_values.append(total / count if count else 0.0)
            nat_metrics = _physical_metrics(states, nat, nat_sets); inter_metrics = _physical_metrics(states, inter, nat_sets)
            sums['plan_transfer'].extend(plan_values); sums['partner_transfer'].extend(partner_values)
            for key, value in (('natural_physical', nat_metrics['physical_rate']), ('natural_q', nat_metrics['q_rate']), ('natural_conditional_q', nat_metrics['conditional_q'] if nat_metrics['conditional_q'] is not None else 0.0),
                               ('intervention_physical', inter_metrics['physical_rate']), ('intervention_q', inter_metrics['q_rate']), ('intervention_conditional_q', inter_metrics['conditional_q'] if inter_metrics['conditional_q'] is not None else 0.0)):
                sums[key].append((value, len(ri)))
        by_sender.append(metrics.summarize_sender(sender, sums, n))
    # Deterministic sham on the first rows of each sender.
    sham_errors = []
    for sender in range(3):
        ids, receiver_ids, _ = _case_arrays(cases, bank, spec, sender); ids = ids[:design.SHAM_ROWS_PER_SENDER]; receiver_ids = receiver_ids[:design.SHAM_ROWS_PER_SENDER]
        if live:
            result = intervention.intervene(networks, arrays['x_PL'][receiver_ids], bank['messages'][receiver_ids], np.full(len(ids), sender, dtype=np.int64), bank['messages'][receiver_ids, :, sender], overwrite_windows=(True, False))
            sham_errors.append(dict(sender=sender, rows=len(ids), message_equal=bool(np.array_equal(result['generated_messages'], bank['messages'][receiver_ids])), action_equal=bool(np.array_equal(result['action_indices'], bank['action_indices'][receiver_ids])), max_probability_error=float(np.max(np.abs(result['action_probabilities'] - bank['action_probabilities'][receiver_ids])))))
        else:
            sham_errors.append(dict(sender=sender, rows=len(ids), message_equal=True, action_equal=True, max_probability_error=0.0))
    extra_calls = (len(cases['receiver_state_indices']) + 3 * design.SHAM_ROWS_PER_SENDER) * 6 if live else 0
    return dict(case_count=int(len(cases['receiver_state_indices'])), by_sender=by_sender, sham=sham_errors, natural_bank_worlds=len(bank['states']), natural_module_samples=9 * len(bank['states']), intervention_module_samples=extra_calls)


def worker(payload):
    seed, static, execution = payload; execution = Path(execution); out = execution / f'seed_{seed}'; out.mkdir(parents=True)
    spec = static['partition']; cases = static['cases']; arrays = make_arrays(spec); results = []
    for condition in CONDITIONS:
        path = Path(static['inputs']['checkpoints'][f'{seed}:{condition}']['path']); require(sha(path) == static['inputs']['checkpoints'][f'{seed}:{condition}']['sha256'], 'Checkpoint changed')
        networks = core.load_networks(path); started = time.perf_counter(); summary = run_policy(networks, arrays, spec, cases, condition.endswith('_live'))
        row = dict(seed=seed, condition=condition, live=condition.endswith('_live'), checkpoint_sha256=sha(path), parameter_sha256=parameter_hash(networks), elapsed_seconds=time.perf_counter() - started, **summary)
        write(out / f'{condition}.json', row); results.append(row)
    return results


def execute(out):
    out = Path(out).resolve(); plan, static, inputs = verify(out); execution = out / 'execution'; require(not execution.exists(), 'Never overwrite execution'); execution.mkdir(); started = time.perf_counter()
    static = dict(static, inputs=inputs)
    try:
        with multiprocessing.get_context('spawn').Pool(4) as pool: groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        rows = [row for group in groups for row in group]; require(len(rows) == len(SEEDS) * len(CONDITIONS), 'Incomplete policy grid')
        result = dict(status='completed', at=core.base.now(), elapsed_seconds=time.perf_counter() - started, plan_sha256=sha(out / 'plan.json'), rows=rows, budget=static['budget'], no_training_updates=True, posthoc=True)
        write(execution / 'results.json', result); write(execution / 'status.json', dict(status='completed', results_sha256=sha(execution / 'results.json'))); return result
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=core.base.now(), error=repr(error), elapsed_seconds=time.perf_counter() - started)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True); args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]; print(json.dumps(answer, ensure_ascii=False))
