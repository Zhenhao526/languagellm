"""Read-only execution audit of progression_001; never trains or modifies runs.

Independent NumPy reconstruction covers every A training-world digest. Model
initialization is replayed without updates; preparation provenance uses untouched
seed fingerprints and historical comparisons, not a claimed optimizer replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEEDS = (24001, 24002, 24003)
UPDATES = {'A': 1800, 'B': 900, 'C': 1200}
OFFSETS = {'A': 0, 'B': 10000000, 'C': 20000000}
EXPECTED = {
 'A': {
  'known_sequence': dict(known=True, vocab=5, length=2, schedule='course'),
  'hidden_sequence': dict(known=False, vocab=5, length=2, schedule='course'),
  'known_atomic': dict(known=True, vocab=25, length=1, schedule='course'),
  'hidden_atomic': dict(known=False, vocab=25, length=1, schedule='course'),
  'hidden_single': dict(known=False, vocab=5, length=1, schedule='course'),
  'hidden_blocked': dict(known=False, vocab=5, length=2, blocked=True, schedule='course'),
  'hidden_sequence_direct': dict(known=False, vocab=5, length=2, schedule='direct'),
  'hidden_sequence_mixed': dict(known=False, vocab=5, length=2, schedule='mixed'),
  'hidden_sequence_holdout': dict(known=False, vocab=5, length=2, holdout=True)},
 'B': {
  'immediate_continue': dict(known=False, vocab=5, length=2),
  'delay_memory': dict(known=False, vocab=5, length=2, delay=3),
  'delay_reset': dict(known=False, vocab=5, length=2, delay=3, memory_mode='reset'),
  'delay_replay': dict(known=False, vocab=5, length=2, delay=3, memory_mode='replay')},
 'C': {
  'persistent_communication': dict(known=False, vocab=5, length=2, horizon=3),
  'persistent_blocked': dict(known=False, vocab=5, length=2, horizon=3, blocked=True),
  'persistent_channel_removed': dict(known=False, vocab=5, length=2, horizon=3, blocked=True),
  'persistent_known': dict(known=True, vocab=5, length=2, horizon=3),
  'persistent_delayed': dict(known=False, vocab=5, length=2, horizon=3, delay=3)}}
MAPS = np.asarray(list(itertools.permutations(range(4), 2)), np.int64)
WITHHELD = {i for i, p in enumerate(MAPS) if tuple(p) in {(0,1),(1,2),(2,3),(3,0)}}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(states, prefix=None):
    h = hashlib.sha256()
    for i, state in enumerate(states):
        for key, value in sorted(state.items()):
            if prefix is None or key.startswith(prefix):
                h.update(f'{i}/{key}'.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def equal_states(a, b, prefixes=None):
    if len(a) != len(b):
        return False
    for aa, bb in zip(a, b):
        keys = {k for k in aa if prefixes is None or k.startswith(prefixes)}
        if keys != {k for k in bb if prefixes is None or k.startswith(prefixes)}:
            return False
        if any(not torch.equal(aa[k], bb[k]) for k in keys):
            return False
    return True


def load_states(path):
    return torch.load(path, weights_only=True, map_location='cpu')


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = Counter()

    def check(self, value, kind, context=None):
        self.checks[kind] += 1
        if not bool(value):
            self.failures.append(dict(check=kind, context=context))
        return bool(value)


def expected_schedule(seed, stage, plan):
    n = UPDATES[stage]
    stop = n - min(300, n // 6)
    levels = np.full(n, 4, np.int64)
    if plan.get('schedule') in ('course', 'mixed'):
        first = min(400, n // 3)
        levels[:first] = 2; levels[first:2*first] = 3
    identities = np.arange(n)
    if plan.get('schedule') == 'mixed':
        identities[:stop] = np.random.default_rng(seed + 7110000).permutation(stop)
    return levels[identities], identities, stop


def access(seed, identity, level, withheld):
    sites = np.random.default_rng(seed + 7120000 + identity).permutation(4)[:level]
    selected = [i for i, pair in enumerate(MAPS) if all(x in sites for x in pair)]
    if withheld:
        selected = [i for i in selected if i not in WITHHELD]
    return sites.tolist(), np.asarray(selected, np.int64)


def reconstruct_a_world(seed, identity, level, withheld, pools):
    """No policy, Camp environment helper, checkpoint, or training is called."""
    _, maps = access(seed, identity, level, withheld)
    rng = np.random.default_rng(seed * 100000 + identity + 1)
    h = hashlib.sha256()
    for _direction in (0, 1):
        positions = MAPS[rng.choice(maps, 256)].copy()
        ids = np.column_stack([rng.choice(pools[k], 256) for k in (0, 1)])
        goal = rng.integers(2, size=256)
        menu = np.argsort(rng.random((256, 4)), axis=1)
        refill = rng.random(256)
        # The runner draws both unused replacement-photo arrays even in A.
        for k in (0, 1):
            rng.choice(pools[k], 256)
        for array in (positions, ids, goal, menu, refill):
            h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def historical_preparation_states(path):
    value = load_states(path)
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict) and 'project.0.weight' in x]
    if isinstance(value, dict) and 'project.0.weight' in value:
        return [value]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'results/progression_001')
    p.add_argument('--out', type=Path)
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    run = args.root.resolve()
    out = args.out or run / 'audit_execution.json'
    if out.exists():
        raise FileExistsError(f'preserve previous audit: {out}')
    torch.set_num_threads(1)
    audit = Audit()
    invocations = sorted(run.glob('invocation_*.json'))
    if not invocations:
        raise RuntimeError('No recorded invocation; cannot establish source provenance')
    first = read(invocations[0])
    frozen_hashes = first['hashes']
    for invocation in invocations:
        record = read(invocation)
        audit.check(record['hashes'] == frozen_hashes, 'invocation_sources_unchanged', invocation.name)
        audit.check(record['seeds'] == list(SEEDS), 'fixed_seed_set', invocation.name)
        for key, expected in dict(updates_a=1800,updates_b=900,updates_c=1200,batch=512,eval_n=8192).items():
            audit.check(record[key] == expected, 'invocation_budget', [invocation.name,key])
    for path, digest in frozen_hashes.items():
        audit.check(Path(path).exists() and sha(path) == digest, 'original_asset_hash', path)
    camp_hash = frozen_hashes[str(ROOT/'camp.py')]
    runner_hash = frozen_hashes[str(ROOT/'run_stages.py')]
    snapshot = run / f'sources_{camp_hash[:8]}_{runner_hash[:8]}'
    for filename, digest in [('camp.py',camp_hash),('run_stages.py',runner_hash),
        ('渐进探索_固定执行方案.md',frozen_hashes[str(ROOT/'渐进探索_固定执行方案.md')])]:
        audit.check(sha(snapshot/filename) == digest, 'snapshot_hash', filename)
    # Frozen Camp source is executed only for no-update initialization. Keep its
    # original __file__ so the verified old resource module paths resolve.
    ns = {'__file__':str(ROOT/'camp.py'), '__name__':'camp_execution_audit_snapshot'}
    exec(compile((snapshot/'camp.py').read_text(), str(snapshot/'camp.py'), 'exec'), ns)
    entries = read(ROOT.parent/'redesign_v0.4/data/manifest.json')['images']
    pools = {k:np.asarray([i for i,e in enumerate(entries)
             if e['split']=='train' and e['category']==name],np.int64)
             for k,name in enumerate(('food','water'))}
    audit.check([len(pools[k]) for k in (0,1)] == [22,22], 'train_photo_counts')
    train_sha={e['sha256'] for e in entries if e['split']=='train'}
    test_sha={e['sha256'] for e in entries if e['split']=='test'}
    audit.check(len(test_sha)==16 and not (train_sha & test_sha), 'photo_split_disjoint')

    # No full preparation optimizer replay: untouched sender/value tensors
    # provide a specific deterministic fingerprint of each new initial seed.
    prepared, preparation_report, personal_hashes, projection_hashes = {}, [], [], []
    for seed in SEEDS:
        path = run/f'prepared_{seed}.pt'
        if not path.exists():
            preparation_report.append(dict(seed=seed,status='pending')); continue
        states = load_states(path); prepared[seed] = states
        original = [a.state_dict() for a in ns['make_agents'](seed)]
        report = read(run/f'preparation_{seed}.json')
        audit.check(len(states)==2 and report['seed']==seed, 'preparation_seed_and_count', seed)
        audit.check(all(x['updates']==200 and x['choices']==12800 and x['heldout_need_sensitive_choice']>=.9
                        for x in report['two_sites']) and min(report['four_sites'])>=.9,
                    'preparation_budget_and_gate', seed)
        for who, (state, initial) in enumerate(zip(states, original)):
            audit.check(equal_states([state],[initial],('sender.','value.')),
                        'untouched_tensors_match_new_seed', [seed,who])
            audit.check(not equal_states([state],[initial],('project.',)),
                        'prepared_projection_changed_from_random', [seed,who])
            audit.check(not equal_states([state],[initial],('actor.',)),
                        'prepared_actor_changed_from_random', [seed,who])
            ph=state_hash([state]); zh=state_hash([state], 'project.')
            personal_hashes.append(ph); projection_hashes.append(zh)
            preparation_report.append(dict(seed=seed,agent=who,personal_torch_seed=seed*1000+who*137,
                state_sha256=ph,project_sha256=zh,status='seed_fingerprint_verified'))
    audit.check(len(personal_hashes)==len(set(personal_hashes)), 'personal_preparations_unique')
    audit.check(len(projection_hashes)==len(set(projection_hashes)), 'personal_projections_unique')
    history_files = sorted(set((ROOT.parent/'redesign_v0.4/results').glob('**/prepared*.pt')) |
                           set((ROOT/'results').glob('**/prepared*.pt')))
    historical_count, overlaps = 0, []
    for path in history_files:
        if path.resolve().parent == run:
            continue
        for who, state in enumerate(historical_preparation_states(path)):
            historical_count += 1
            if state_hash([state]) in personal_hashes or state_hash([state],'project.') in projection_hashes:
                overlaps.append(dict(path=str(path),agent=who))
    audit.check(not overlaps, 'no_historical_preparation_or_projection_overlap', overlaps)

    completed, pending, rows_by_run, worlds, warmstarts = [], [], {}, {}, []
    expected_paths = {f'{stage}/s{seed}_{name}' for stage,conditions in EXPECTED.items()
                      for seed in SEEDS for name in conditions}
    found_paths = {str(path.relative_to(run)) for stage in EXPECTED for path in (run/stage).glob('s*') if path.is_dir()}
    audit.check(not(found_paths-expected_paths), 'no_extra_study_conditions', sorted(found_paths-expected_paths))
    for stage,conditions in EXPECTED.items():
      for seed in SEEDS:
       for name,plan in conditions.items():
        label=f'{stage}/s{seed}_{name}'; folder=run/label
        if not (folder/'result.json').exists():
            pending.append(label); continue
        completed.append(label)
        config=read(folder/'config.json'); result=read(folder/'result.json')
        n=UPDATES[stage]; levels,identities,stop=expected_schedule(seed,stage,plan)
        checkpoints=sorted(set(x for x in [0,100,300,400,800,n//2,stop,n] if x<=n))
        for key,value in dict(seed=seed,condition=name,plan=plan,updates=n,batch=512,eval_n=8192,
            checkpoints=checkpoints,learning_rate=.0007,entropy_coefficient=.02,
            entropy_off_after=stop,gamma=1,training_stage_seed_offset=OFFSETS[stage],
            source_hashes={'camp.py':camp_hash,'run_stages.py':runner_hash}).items():
            audit.check(config.get(key)==value, 'fixed_run_config', [label,key])
        for key,value in dict(seed=seed,condition=name,plan=plan,updates=n,batch=512,
                              frozen_projection_verified=True).items():
            audit.check(result.get(key)==value, 'result_identity', [label,key])
        initial=load_states(folder/'initial.pt'); final=load_states(folder/'final.pt')
        audit.check(state_hash(initial)==config['initial_sha256']==result['initial_sha256'],
                    'initial_digest',label)
        audit.check(state_hash(final)==result['final_sha256'], 'final_digest',label)
        audit.check(equal_states(initial,final,('project.',)) and
                    equal_states(initial,prepared[seed],('project.',)), 'frozen_project_all_stages',label)
        fresh=ns['remake_agents'](seed,prepared[seed],plan['vocab'],plan['length'])
        trainable=[sum(p.numel() for p in a.parameters() if p.requires_grad) for a in fresh]
        audit.check(config['trainable_parameters']==trainable, 'trainable_parameter_count',label)
        source=run/f'prepared_{seed}.pt'
        if stage=='A':
            expected_initial=[a.state_dict() for a in fresh]
            source_kind='individual preparation only'
        else:
            source_name='hidden_blocked' if name=='persistent_blocked' else 'hidden_sequence'
            source=run/'A'/f's{seed}_{source_name}'/'final.pt'
            expected_initial=load_states(source); source_kind='stage A social checkpoint'
        audit.check(equal_states(initial,expected_initial), 'exact_initialization_source',label)
        recorded_source=config['warmstart_source']
        audit.check(Path(recorded_source['path']).resolve()==source.resolve() and
            recorded_source['sha256']==sha(source) and recorded_source['kind']==source_kind,
            'warmstart_path_and_digest',label)
        warmstarts.append(dict(run=label,source=str(source.relative_to(run)),
             interpretation='always-blocked history' if name=='persistent_blocked' else
                            'fresh social interface' if stage=='A' else 'same A communication source'))
        for update in checkpoints:
            audit.check((folder/f'checkpoint_{update:04d}.pt').exists(), 'checkpoint_present',[label,update])
        audit.check(equal_states(initial,load_states(folder/'checkpoint_0000.pt')), 'checkpoint_zero_is_initial',label)
        audit.check(equal_states(final,load_states(folder/f'checkpoint_{n:04d}.pt')), 'checkpoint_last_is_final',label)
        optimizers=load_states(folder/'final_optimizer.pt')
        for who,optimizer in enumerate(optimizers):
            audit.check(all(g['lr']==.0007 for g in optimizer['param_groups']) and
                        all(float(s['step'])==n for s in optimizer['state'].values()),
                        'fresh_adam_budget', [label,who])
        schedule=read(folder/'training_schedule.json')
        audit.check(schedule['levels']==levels.tolist() and schedule['batch_identities']==identities.tolist(),
                    'fixed_schedule',label)
        rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        rows_by_run[label]=rows
        audit.check(len(rows)==n, 'exact_update_count',label)
        for step,row in enumerate(rows):
            identity=int(identities[step]); level=int(levels[step]); sites,pool=access(seed,identity,level,plan.get('holdout',False))
            audit.check(row['update']==step+1 and row['batch_identity']==identity and
                        row['active_sites']==level and row['allowed_sites']==sites,
                        'training_batch_identity_and_access',[label,step+1])
            audit.check(row['entropy_weight']==(.02 if step<stop else 0.),
                        'training_entropy_weight',[label,step+1])
            audit.check(np.isfinite(row['reward']) and all(np.isfinite(a['loss']) and
                        np.isfinite(a['gradient_norm']) for a in row['agents']),
                        'finite_training',[label,step+1])
            if plan.get('holdout'):
                audit.check(not(set(pool)&WITHHELD), 'heldout_maps_absent_from_training',[label,step+1])
            if stage=='A':
                key=(seed,identity,level,bool(plan.get('holdout')))
                if key not in worlds:
                    worlds[key]=reconstruct_a_world(seed,identity,level,key[-1],pools)
                audit.check(row['world_sha256']==worlds[key], 'independent_A_world_digest',[label,step+1])
        curve=read(folder/'curve.json')
        audit.check([x['update'] for x in curve]==checkpoints, 'fixed_evaluation_checkpoints',label)
        for point in curve:
            for mode,score in point['scores'].items():
                audit.check(score['episodes']==1024 and score['horizon']==plan.get('horizon',1),
                            'checkpoint_evaluation_budget',[label,point['update'],mode])
        audit.check(set(result['scores'])=={'normal','shuffle','blank','stochastic','erase_memory'},
                    'all_final_modes',label)
        for mode,score in result['scores'].items():
            audit.check(score['episodes']==8192 and score['horizon']==plan.get('horizon',1),
                        'final_evaluation_budget',[label,mode])

    matched_curricula=[]; replay=[]
    for seed in SEEDS:
        ckey=f'A/s{seed}_hidden_sequence'; mkey=f'A/s{seed}_hidden_sequence_mixed'
        if ckey in rows_by_run and mkey in rows_by_run:
            course=rows_by_run[ckey]; mixed=rows_by_run[mkey]
            index={r['batch_identity']:r for r in mixed}
            for row in course:
                other=index[row['batch_identity']]
                audit.check(all(row[k]==other[k] for k in
                    ('batch_identity','active_sites','allowed_sites','world_sha256','entropy_weight')),
                    'course_mixed_same_batch_world_and_entropy',[seed,row['batch_identity']])
            audit.check(all(course[i]['batch_identity']==mixed[i]['batch_identity'] for i in range(1500,1800)),
                        'course_mixed_identical_final_300',seed)
            matched_curricula.append(seed)
        ikey=f'B/s{seed}_immediate_continue'; rkey=f'B/s{seed}_delay_replay'
        if ikey in rows_by_run and rkey in rows_by_run:
            a,b=run/ikey,run/rkey
            audit.check(rows_by_run[ikey]==rows_by_run[rkey], 'B_replay_all_training_steps_exact',seed)
            audit.check(read(a/'curve.json')==read(b/'curve.json'), 'B_replay_all_evaluations_exact',seed)
            for path in sorted(a.glob('checkpoint_*.pt'))+[a/'initial.pt',a/'final.pt']:
                audit.check(equal_states(load_states(path),load_states(b/path.name)),
                            'B_replay_all_parameters_exact',[seed,path.name])
            for path in sorted(a.glob('final_*.npz')):
                with np.load(path) as x, np.load(b/path.name) as y:
                    audit.check(set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files),
                                'B_replay_all_final_trace_arrays_exact',[seed,path.name])
            replay.append(seed)

    if not args.allow_incomplete:
        audit.check(not pending and len(completed)==54, 'all_54_planned_runs_complete',pending)
        audit.check(len(personal_hashes)==6, 'all_six_personal_preparations_present')
        audit.check(len(matched_curricula)==3 and len(replay)==3, 'all_three_seed_pair_audits_complete')
    report=dict(status='failed' if audit.failures else 'incomplete' if pending else 'passed',
        planned_runs=54,completed_runs=len(completed),pending=pending,
        independent_pair_seeds=list(SEEDS),checks=dict(audit.checks),failures=audit.failures,
        A_world_updates_verified=audit.checks['independent_A_world_digest'],
        unique_A_world_batches_reconstructed=len(worlds),matched_course_mixed_seeds=matched_curricula,
        exact_B_immediate_replay_seeds=replay,preparations=preparation_report,
        historical_preparations_compared=historical_count,historical_overlap=overlaps,warmstarts=warmstarts,
        frozen_source_hashes=frozen_hashes,audit_script_sha256=sha(__file__),
        boundaries=[
          'No training or optimizer replay. Prepared seed provenance is anchored by unchanged random sender/value tensors, not full preparation trajectory replay.',
          'All completed A training worlds independently reconstructed from fixed seeds, batches and train photo pools.',
          'C worlds become policy-dependent. Full cross-condition C world hashes are deliberately not required to match.',
          'Final reward/transition trace audit is performed by analyze_stages.py; this audit separately checks B replay identity and execution provenance.',
          'Old 44/16 photographs are development data, not new visual confirmation. Statistical replication is three pairs, not episode rows.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],completed=len(completed),pending=len(pending),
                         failures=len(audit.failures),A_worlds=report['A_world_updates_verified'],out=str(out))))
    if audit.failures:
        sys.exit(1)


if __name__=='__main__':
    main()
