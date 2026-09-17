"""Posthoc directional probe for factorized alternative-partner policies.

The probe loads only the frozen final checkpoints from
``triadic_factorized_neutral_altpartner_study``.  It replaces one sender's
first-window packet with a same-context neighbouring need, then recomputes
the other message window and factorized actions.  No optimizer state is
replayed and no training is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as original
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner
from research_program.triadic_factorized_neutral_altpartner_study import kernel
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment as execution_env
from . import design, intervention, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_RUN = ROOT / 'research_program/triadic_factorized_neutral_altpartner_study/results/altpair_001'
SOURCE_AGGREGATE = SOURCE_RUN.parent / 'aggregation_altpair_001/results.json'
SOURCE_AUDIT = SOURCE_RUN.parent / 'audit_altpair_001/verification.json'

CONFIG = dict(
    schema='triadic_factorized_neutral_altpartner_partner_direction_probe_v1',
    seeds=list(design.SEEDS), schedules=list(design.SCHEDULES),
    conditions=list(design.CONDITIONS), parts=list(design.PARTS),
    checkpoint=design.CHECKPOINT, chunk_size=design.CHUNK_SIZE,
    sham_rows_per_sender=design.SHAM_ROWS_PER_SENDER,
    overwrite_windows=[True, False],
    natural_messages='greedy messages from each frozen final checkpoint',
    donor='same-layout same-owner same-person two-axis composite need endpoint that changes the legal partner set; selected sender W1 outward packet replaced',
    primary='rematching-by-communication interaction on intervention-induced donor-only minus receiver-only partner-edge transfer',
    secondary='full-plan transfer, partner-edge transfer, natural/intervened physical execution and Q, conditional Q, exact sham replay',
    posthoc_relative_to='triadic_factorized_neutral_altpartner_study/results/altpair_001',
    new_model_calls=True, training_updates=0, automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_manifest():
    paths = [HERE / n for n in (
        '__init__.py', 'design.py', 'intervention.py', 'runner.py', 'metrics.py',
        'audit.py', 'aggregate.py', 'plot_results.py', 'plan.md', 'README.md',
        'tests/test_design.py')]
    paths += [
        Path(source_runner.__file__), Path(source_runner.design.__file__),
        Path(source_runner.kernel.__file__), Path(source_runner.remap.__file__),
        Path(core.__file__), Path(core.base.__file__), Path(dataset.__file__),
        Path(original.__file__), Path(execution_env.__file__),
        SOURCE_RUN / 'plan.json', SOURCE_RUN / 'prepared.json', SOURCE_RUN / 'freeze.json',
        SOURCE_AGGREGATE, SOURCE_AUDIT,
    ]
    require(all(path.is_file() for path in paths), 'Missing probe source or frozen input')
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def frozen_inputs():
    """Bind every probe policy to an audited final checkpoint."""
    source_runner.verify(SOURCE_RUN)
    require(SOURCE_AGGREGATE.is_file() and SOURCE_AUDIT.is_file(), 'Source aggregate/audit missing')
    aggregate = json.loads(SOURCE_AGGREGATE.read_text())
    audit = json.loads(SOURCE_AUDIT.read_text())
    require(str(aggregate.get('status', '')).startswith('completed'), 'Source aggregate incomplete')
    require(audit.get('status') == 'passed', 'Source audit incomplete')
    checkpoints = {}
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = SOURCE_RUN / 'execution' / f'seed_{seed}_{condition}'
            result_path = directory / 'result.json'; checkpoint = directory / 'checkpoint_6000.npz'
            require(result_path.is_file() and checkpoint.is_file(), f'Missing source run for {seed}:{condition}')
            result = json.loads(result_path.read_text()); digest = sha(checkpoint)
            require(digest == result['final_checkpoint_sha256'], 'Source checkpoint hash mismatch')
            checkpoints[f'{seed}:{condition}'] = dict(path=str(checkpoint.resolve()), sha256=digest,
                                                        seed=seed, condition=condition)
    return dict(
        source_run=str(SOURCE_RUN), source_plan_sha256=sha(SOURCE_RUN / 'plan.json'),
        source_prepared_sha256=sha(SOURCE_RUN / 'prepared.json'),
        source_freeze_sha256=sha(SOURCE_RUN / 'freeze.json'),
        aggregate_sha256=sha(SOURCE_AGGREGATE), audit_sha256=sha(SOURCE_AUDIT),
        checkpoints=checkpoints,
    )


def prepared():
    static = design.make_prepared()
    static['config'] = CONFIG
    static['source_sha256'] = source_manifest()
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    policy_count = len(design.SEEDS) * len(design.CONDITIONS)
    natural_worlds = int(static['partition']['world_count'])
    case_rows = int(static['cases']['case_count'])
    live_policy_count = policy_count // 2
    natural_samples = policy_count * natural_worlds * 9
    intervention_samples = live_policy_count * (case_rows + 3 * design.SHAM_ROWS_PER_SENDER) * 6
    static['budget'] = dict(
        policy_rows=policy_count, natural_worlds=policy_count * natural_worlds,
        natural_module_samples=natural_samples, case_rows=policy_count * case_rows,
        sham_rows=policy_count * 3 * design.SHAM_ROWS_PER_SENDER,
        intervention_module_samples=intervention_samples,
        total_new_module_samples=natural_samples + intervention_samples,
    )
    return static


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information='PL')
    arrays['native_rewards'] = arrays['rewards'].copy()
    require(arrays['x_PL'].shape == (spec['world_count'], 3, 54), 'Invalid PL feature shape')
    require(np.all(arrays['x_PL'][:, :, 53] == 0), 'Full-information flag leaked into PL')
    require(np.all((arrays['native_rewards'] == 1).sum(axis=1) == 2),
            'Every probe world must have two full plans')
    return arrays


def joint_action_probabilities(logits):
    intent_p, _, proposal_p, _ = kernel.factorized_distribution(logits)
    n = len(logits)
    result = np.zeros((n, 3, 17), dtype=np.float64)
    result[:, :, 0] = intent_p[:, :, 0]
    result[:, :, 1:] = intent_p[:, :, 1, None] * proposal_p
    require(np.allclose(result.sum(axis=-1), 1.0, atol=1e-12, rtol=0),
            'Factorized action probabilities do not normalize')
    return result


def greedy_actions(logits):
    intent_p, _, proposal_p, _ = kernel.factorized_distribution(logits)
    intent = intent_p.argmax(axis=-1).astype(np.int16)
    proposal = proposal_p.argmax(axis=-1).astype(np.int16)
    return np.where(intent == 1, proposal + 1, 0).astype(np.int16)


def native_bank(networks, arrays, live):
    n = len(arrays['packed_states'])
    bank = dict(
        states=arrays['packed_states'].copy(),
        messages=np.empty((n, 2, 3, 4), dtype=np.int8),
        action_probabilities=np.empty((n, 3, 17), dtype=np.float64),
        action_indices=np.empty((n, 3), dtype=np.int16),
    )
    for start in range(0, n, design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, n)
        trace = core.rollout(networks, arrays['x_PL'][start:stop], bool(live))
        probabilities = joint_action_probabilities(trace['action_logits'])
        bank['messages'][start:stop] = trace['messages']
        bank['action_probabilities'][start:stop] = probabilities
        bank['action_indices'][start:stop] = greedy_actions(trace['action_logits'])
    require(np.array_equal(bank['action_indices'], bank['action_probabilities'].argmax(-1)),
            'Natural greedy actions mismatch factorized probabilities')
    return bank


def _plan_actions(plan):
    return tuple(dataset.plan_action_indices(plan))


def _sets_for_indices(states):
    result = []
    for row in states:
        plans = original.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        require(len(plans) == 2, 'Probe state is not a two-plan world')
        result.append({_plan_actions(plan) for plan in plans})
    return result


def _mass(probabilities, action_sets):
    result = np.zeros(len(probabilities), dtype=np.float64)
    for action_set in action_sets:
        for actions in action_set:
            result += (probabilities[:, 0, actions[0]] * probabilities[:, 1, actions[1]]
                       * probabilities[:, 2, actions[2]])
    return result


def _partner_probability(probabilities, sender, recipient):
    mask = execution_env.PROPOSAL_ROLES[recipient] == sender
    return probabilities[:, recipient, mask].sum(axis=1)


def _physical_metrics(states, probabilities, legal_sets):
    actions = probabilities.argmax(-1).astype(np.int16)
    settled = execution_env.settle(states, actions, 'strict')
    physical = settled['actual_pair_index'] >= 0
    q = np.zeros(len(states), dtype=bool)
    for i, action_set in enumerate(legal_sets):
        if physical[i]:
            q[i] = tuple(actions[i].tolist()) in action_set
    return dict(
        actions=actions, physical_rate=float(physical.mean()), q_rate=float(q.mean()),
        conditional_q=float(q[physical].mean()) if physical.any() else 0.0,
        executed_count=int(physical.sum()), q_count=int(q.sum()),
    )


def _case_arrays(cases, bank, sender):
    mask = np.asarray(cases['sender'], dtype=np.int8) == sender
    ids = np.flatnonzero(mask)
    receiver_ids = np.asarray(cases['receiver_state_indices'], dtype=np.int64)[ids]
    donor_ids = np.asarray(cases['donor_state_indices'], dtype=np.int64)[ids]
    return ids, receiver_ids, donor_ids


def run_policy(networks, arrays, spec, cases, live):
    bank = native_bank(networks, arrays, live)
    legal_sets_cache = _sets_for_indices(bank['states'])
    by_sender = []
    for sender in range(3):
        _, receiver_ids, donor_ids = _case_arrays(cases, bank, sender)
        n = len(receiver_ids)
        natural_prob = bank['action_probabilities'][receiver_ids]
        donor_packets = bank['messages'][donor_ids, 0, sender]
        sums = dict(plan_transfer=[], partner_transfer=[], natural_physical=[], natural_q=[],
                    natural_conditional_q=[], intervention_physical=[], intervention_q=[],
                    intervention_conditional_q=[])
        for start in range(0, n, design.CHUNK_SIZE):
            stop = min(start + design.CHUNK_SIZE, n)
            ri = receiver_ids[start:stop]; di = donor_ids[start:stop]
            nat = natural_prob[start:stop]; states = bank['states'][ri]
            if live:
                result = intervention.intervene(
                    networks, arrays['x_PL'][ri], bank['messages'][ri],
                    np.full(stop - start, sender, dtype=np.int64), donor_packets[start:stop])
                inter = result['action_probabilities']
            else:
                inter = nat
            nat_sets = [legal_sets_cache[int(i)] for i in ri]
            donor_sets = [legal_sets_cache[int(i)] for i in di]
            receiver_state_pairs = []
            donor_state_pairs = []
            plan_values = []; partner_values = []
            for row in range(len(ri)):
                donor_only = donor_sets[row] - nat_sets[row]
                receiver_only = nat_sets[row] - donor_sets[row]
                donor_int = _mass(inter[row:row + 1], [donor_only])[0] if donor_only else 0.0
                donor_nat = _mass(nat[row:row + 1], [donor_only])[0] if donor_only else 0.0
                receiver_int = _mass(inter[row:row + 1], [receiver_only])[0] if receiver_only else 0.0
                receiver_nat = _mass(nat[row:row + 1], [receiver_only])[0] if receiver_only else 0.0
                plan_values.append((donor_int - receiver_int) - (donor_nat - receiver_nat))
                receiver_row = bank['states'][ri[row]]; donor_row = bank['states'][di[row]]
                receiver_pairs = {tuple(plan[:2]) for plan in original.full_success_plans(
                    tuple(receiver_row[:3]), tuple(receiver_row[3:7]))}
                donor_pairs = {tuple(plan[:2]) for plan in original.full_success_plans(
                    tuple(donor_row[:3]), tuple(donor_row[3:7]))}
                receiver_state_pairs.append(receiver_pairs); donor_state_pairs.append(donor_pairs)
                effects = []
                for recipient in range(3):
                    if recipient == sender:
                        continue
                    desired = int(tuple(sorted((sender, recipient))) in donor_pairs) - int(
                        tuple(sorted((sender, recipient))) in receiver_pairs)
                    if desired:
                        effect = (_partner_probability(inter[row:row + 1], sender, recipient)[0]
                                  - _partner_probability(nat[row:row + 1], sender, recipient)[0])
                        effects.append(desired * effect)
                partner_values.append(float(np.mean(effects)) if effects else 0.0)
            nat_metrics = _physical_metrics(states, nat, nat_sets)
            inter_metrics = _physical_metrics(states, inter, nat_sets)
            sums['plan_transfer'].extend(plan_values); sums['partner_transfer'].extend(partner_values)
            for key, value in (
                ('natural_physical', nat_metrics['physical_rate']),
                ('natural_q', nat_metrics['q_rate']),
                ('natural_conditional_q', nat_metrics['conditional_q']),
                ('intervention_physical', inter_metrics['physical_rate']),
                ('intervention_q', inter_metrics['q_rate']),
                ('intervention_conditional_q', inter_metrics['conditional_q'])):
                sums[key].append((value, len(ri)))
        by_sender.append(metrics.summarize_sender(sender, sums, n))

    sham = []
    for sender in range(3):
        _, receiver_ids, _ = _case_arrays(cases, bank, sender)
        receiver_ids = receiver_ids[:design.SHAM_ROWS_PER_SENDER]
        if live:
            result = intervention.intervene(
                networks, arrays['x_PL'][receiver_ids], bank['messages'][receiver_ids],
                np.full(len(receiver_ids), sender, dtype=np.int64),
                bank['messages'][receiver_ids, 0, sender])
            sham.append(dict(
                sender=sender, rows=len(receiver_ids),
                message_equal=bool(np.array_equal(result['generated_messages'], bank['messages'][receiver_ids])),
                action_equal=bool(np.array_equal(result['action_indices'], bank['action_indices'][receiver_ids])),
                max_probability_error=float(np.max(np.abs(
                    result['action_probabilities'] - bank['action_probabilities'][receiver_ids]))),
            ))
        else:
            sham.append(dict(sender=sender, rows=len(receiver_ids), message_equal=True,
                             action_equal=True, max_probability_error=0.0))
    extra_calls = (len(cases['receiver_state_indices']) + 3 * design.SHAM_ROWS_PER_SENDER) * 6 if live else 0
    return dict(
        case_count=int(len(cases['receiver_state_indices'])), by_sender=by_sender, sham=sham,
        natural_bank_worlds=len(bank['states']), natural_module_samples=9 * len(bank['states']),
        intervention_module_samples=int(extra_calls),
    )


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); inputs = frozen_inputs(); out.mkdir(parents=True, exist_ok=False)
    for relative in static['source_sha256']:
        source = ROOT / relative; target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    write(out / 'prepared.json', static); write(out / 'inputs.json', inputs)
    write(out / 'plan.json', dict(status='prepared_without_probe_forward', at=core.base.now(),
                                  config=CONFIG, prepared_sha256=sha(out / 'prepared.json'),
                                  inputs_sha256=sha(out / 'inputs.json'),
                                  source_sha256=static['source_sha256'], runtime=static['runtime']))
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'),
                                    prepared_sha256=sha(out / 'prepared.json'),
                                    inputs_sha256=sha(out / 'inputs.json')))
    verify(out)
    return dict(status='prepared_without_probe_forward', output=str(out), budget=static['budget'])


def verify(out):
    out = Path(out).resolve()
    static = json.loads((out / 'prepared.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    freeze = json.loads((out / 'freeze.json').read_text())
    inputs = json.loads((out / 'inputs.json').read_text())
    require(sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'],
            'Probe prepared hash mismatch')
    require(sha(out / 'plan.json') == freeze['plan_sha256'], 'Probe plan hash mismatch')
    require(sha(out / 'inputs.json') == freeze['inputs_sha256'] == plan['inputs_sha256'],
            'Probe input hash mismatch')
    require(static == prepared() and plan['config'] == CONFIG
            and plan['source_sha256'] == source_manifest(), 'Probe source/config changed')
    require(inputs == frozen_inputs(), 'Frozen source inputs changed')
    for relative, digest in plan['source_sha256'].items():
        require(sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static, inputs


def worker(payload):
    seed, static, inputs, execution = payload
    execution = Path(execution); out = execution / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=False)
    spec, cases = static['partition'], static['cases']; arrays = make_arrays(spec); rows = []
    for condition in design.CONDITIONS:
        meta = inputs['checkpoints'][f'{seed}:{condition}']; path = Path(meta['path'])
        require(sha(path) == meta['sha256'], 'Checkpoint binding mismatch')
        networks = source_runner.load_networks(path); started = time.perf_counter()
        summary = run_policy(networks, arrays, spec, cases, condition.endswith('_live'))
        row = dict(seed=seed, condition=condition, schedule=condition.split('_', 1)[0],
                   live=condition.endswith('_live'), checkpoint_sha256=sha(path),
                   parameter_sha256=source_runner.parameter_hash(networks),
                   elapsed_seconds=time.perf_counter() - started, **summary)
        write(out / f'{condition}.json', row); rows.append(row)
    return rows


def execute(out):
    out = Path(out).resolve(); verify(out)
    static = json.loads((out / 'prepared.json').read_text()); inputs = json.loads((out / 'inputs.json').read_text())
    execution = out / 'execution'; require(not execution.exists(), 'Never overwrite probe execution')
    execution.mkdir(parents=False, exist_ok=False); started = time.perf_counter()
    write(execution / 'started.json', dict(started_at=core.base.now(), plan_sha256=sha(out / 'plan.json')))
    try:
        payloads = [(seed, static, inputs, str(execution)) for seed in design.SEEDS]
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            groups = pool.map(worker, payloads)
        rows = [row for group in groups for row in group]
        require(len(rows) == len(design.SEEDS) * len(design.CONDITIONS), 'Incomplete policy grid')
        result = dict(status='completed', at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=sha(out / 'plan.json'), rows=rows, primary=metrics.summarize(rows),
                      budget=static['budget'], no_training_updates=True, posthoc=True,
                      language_claim_automatically_supported=False)
        write(execution / 'results.json', result)
        write(execution / 'status.json', dict(status='completed', results_sha256=sha(execution / 'results.json')))
        return result
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=core.base.now(), error=repr(error),
                                               elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True); args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
