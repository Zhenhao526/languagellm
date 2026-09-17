"""Bounded development verification, never the 64-run study.

Four micro-runs compare frozen v0.8 and gain1 at lambda0/1: two updates,
batch16, final eval120. Uses a copied old prepared seed; no new preparation.
Existing output or receipt is never overwritten and failures are not retried.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import textwrap
import time
import traceback


STUDY = Path(__file__).resolve().parent
WORK = STUDY.parents[1]
OUTPUT = STUDY / 'development/preflight_001'
RECEIPT = STUDY / 'preflight_result.json'
SOURCE_PREPARED = WORK / 'redesign_v0.8/results/complementarity_001/prepared_27101.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_checks(runner):
    import torch
    checks = []
    for beta in (0., .02):
        for gain in (1, 3):
            policy = torch.tensor(2., dtype=torch.float64, requires_grad=True)
            value = torch.tensor(3., dtype=torch.float64, requires_grad=True)
            entropy = torch.tensor(4., dtype=torch.float64, requires_grad=True)
            loss = runner.combine_loss(policy, value, entropy, beta, gain)
            gradient = torch.autograd.grad(loss, (policy, value, entropy))
            require([g.item() for g in gradient] == [float(gain), 1., -beta], 'Only policy scalar gradient is gained')
            lp = torch.tensor([-.5, -1., -2., -.25], dtype=torch.float64, requires_grad=True)
            v = torch.tensor([.25, -.5, .75, 0.], dtype=torch.float64, requires_grad=True)
            target = torch.tensor([-1., 0., -.5, -.25], dtype=torch.float64, requires_grad=True)
            ent = torch.tensor([.25, .5, .75, 1.], dtype=torch.float64, requires_grad=True)
            p = -(lp * (target - v).detach()).mean()
            detached = torch.autograd.grad(p, (lp, v, target), allow_unused=True, retain_graph=True)
            require(detached[1] is None and detached[2] is None, 'Policy advantage must detach value and target')
            lv = .5 * torch.nn.functional.mse_loss(v, target)
            combined = runner.combine_loss(p, lv, ent.mean(), beta, gain)
            grads = torch.autograd.grad(combined, (lp, v, target, ent))
            expected = (-(target.detach() - v.detach()) * gain / 4,
                        (v.detach() - target.detach()) / 4,
                        (target.detach() - v.detach()) / 4,
                        torch.full_like(ent, -beta / 4))
            for actual, wanted in zip(grads, expected):
                require(torch.equal(actual, wanted), 'Policy/value/entropy vector gradients')
            checks.append(dict(gain=gain, entropy_weight=beta, scalar_gradients=[g.item() for g in gradient],
                               detached_advantage=True, vector_gradient_checks=4))
    old = ast.parse(textwrap.dedent(inspect.getsource(runner.base.train)))
    new = ast.parse(textwrap.dedent(inspect.getsource(runner.train)))
    def assignment(tree, name):
        return next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    same_assignments = ('ploss', 'vloss', 'loss', 'boundary', 'checkpoints', 'optimizers', 'train_plan')
    for name in same_assignments:
        require(ast.dump(assignment(old, name)) == ast.dump(assignment(new, name)), 'Training expression changed: ' + name)
    return dict(tensor_cases=checks, identical_training_assignments=list(same_assignments))


def recursively_equal(old, new, counts, label='root'):
    import numpy as np
    import torch
    if isinstance(old, torch.Tensor):
        require(isinstance(new, torch.Tensor) and old.dtype == new.dtype and old.shape == new.shape
                and torch.equal(old, new), 'Tensor differs: ' + label)
        counts['tensors'] += 1; counts['tensor_elements'] += old.numel()
    elif isinstance(old, np.ndarray):
        require(isinstance(new, np.ndarray) and old.dtype == new.dtype and old.shape == new.shape
                and np.array_equal(old, new), 'Array differs: ' + label)
        counts['arrays'] += 1; counts['array_elements'] += old.size
    elif isinstance(old, dict):
        require(isinstance(new, dict) and old.keys() == new.keys(), 'Dictionary keys differ: ' + label)
        for key in old:
            recursively_equal(old[key], new[key], counts, f'{label}.{key}')
    elif isinstance(old, (list, tuple)):
        require(type(old) is type(new) and len(old) == len(new), 'Sequence differs: ' + label)
        for i, (a, b) in enumerate(zip(old, new)):
            recursively_equal(a, b, counts, f'{label}[{i}]')
    else:
        require(type(old) is type(new) and old == new, 'Value differs: ' + label)
        counts['non_tensor_values'] += 1


def compare_runs(old_dir, new_dir):
    import numpy as np
    import torch
    counts = Counter()
    files = ('initial.pt', 'checkpoint_0000.pt', 'checkpoint_0002.pt', 'final.pt', 'final_optimizer.pt')
    for name in files:
        a = torch.load(old_dir / name, weights_only=True, map_location='cpu')
        b = torch.load(new_dir / name, weights_only=True, map_location='cpu')
        recursively_equal(a, b, counts, name)
    modes = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
    for mode in modes:
        filename = f'final_{mode}.npz'
        with np.load(old_dir / filename, allow_pickle=False) as a, np.load(new_dir / filename, allow_pickle=False) as b:
            require(a.files == b.files, 'NPZ field order differs')
            for field in a.files:
                recursively_equal(a[field], b[field], counts, filename + '.' + field)
    for name in ('config.json', 'training_schedule.json', 'curve.json', 'result.json'):
        a, b = read(old_dir / name), read(new_dir / name)
        if name in ('config.json', 'result.json'):
            require(b.pop('policy_gain') == 1, 'Bridge must be gain1')
        if name == 'result.json':
            require(a.pop('seconds') >= 0 and b.pop('seconds') >= 0, 'Timing metadata')
        recursively_equal(a, b, counts, name)
    old_lines = [json.loads(x) for x in (old_dir / 'training.jsonl').read_text().splitlines()]
    new_lines = [json.loads(x) for x in (new_dir / 'training.jsonl').read_text().splitlines()]
    require(len(old_lines) == len(new_lines) == 2, 'Exactly two updates per bridge run')
    for row in new_lines:
        require(row.pop('policy_gain') == 1, 'Gain metadata')
        for agent in row['agents']:
            for component in agent['components']:
                require(component.pop('weighted_policy_loss') == component['policy_loss'], 'Gain1 weighted policy loss')
    recursively_equal(old_lines, new_lines, counts, 'training.jsonl')
    return dict(status='exact_match', counts=dict(counts), weight_optimizer_files_compared=list(files),
                full_array_modes_compared=list(modes), training_updates_compared=2,
                excluded_fields=['new config/result/training policy_gain=1',
                    'new training weighted_policy_loss (first verified equal raw policy_loss)', 'result.seconds only'])


def main():
    require(not OUTPUT.exists() and not RECEIPT.exists(), 'Existing preflight evidence; no overwrite or automatic retry')
    sys.path.insert(0, str(STUDY))
    import runner
    import torch
    import numpy as np
    torch.set_num_threads(1)
    runner.validate_base()
    runner_hash = sha(STUDY / 'runner.py')
    original_hash = sha(SOURCE_PREPARED)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE_PREPARED, OUTPUT / SOURCE_PREPARED.name)
    for path in (Path(__file__), STUDY / 'runner.py'):
        shutil.copyfile(path, OUTPUT / path.name)
    write_new(OUTPUT / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        runner_sha256=runner_hash, preflight_sha256=sha(__file__), old_prepared_path=str(SOURCE_PREPARED),
        old_prepared_sha256=original_hash, old_base_sources={str(runner.ROOT / k): v for k, v in runner.EXPECTED_V08.items()},
        development_seed=27101, lambda_values=[0, 1], functions=['frozen_v08_train', 'new_train_gain1'],
        runs=4, updates_per_run=2, total_updates=8, training_batch=16, final_eval_n=120,
        inherited_checkpoint_eval_n=1200, checkpoints=[0, 2], training_worlds=128,
        resource_preparation_updates=0, new_personal_control_runs=0, part_of_formal_64=False))
    started = time.monotonic(); current = None
    try:
        tensor = tensor_checks(runner)
        prepared = torch.load(OUTPUT / SOURCE_PREPARED.name, weights_only=True, map_location='cpu')
        bank = runner.ImageBank()
        comparisons = []
        for kind in ('additive', 'joint'):
            name = 'split1_' + kind
            plan = dict(runner.base.CONDITIONS[name])
            old_dir = OUTPUT / ('old_' + kind)
            new_dir = OUTPUT / ('new_' + kind)
            current = 'old_' + kind
            runner.base.train(27101, name, plan, prepared, bank, old_dir, 2, 16, 120)
            current = 'new_' + kind
            runner.train(27101, name, plan, prepared, bank, new_dir, 2, 16, 120, 1)
            comparisons.append(dict(lambda_value=plan['complementarity'], old_directory=str(old_dir),
                                    new_directory=str(new_dir), **compare_runs(old_dir, new_dir)))
        require(sha(STUDY / 'runner.py') == runner_hash, 'Runner changed during preflight')
        require(sha(SOURCE_PREPARED) == sha(OUTPUT / SOURCE_PREPARED.name) == original_hash, 'Prepared source/copy changed')
        runner.validate_base()
        result = dict(status='passed', runner_sha256=runner_hash, preflight_sha256=sha(__file__),
            completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            development_directory=str(OUTPUT), development_plan_sha256=sha(OUTPUT / 'plan.json'),
            source_prepared_sha256=original_hash, torch=str(torch.__version__), numpy=str(np.__version__),
            tensor_checks=tensor, bridge_comparisons=comparisons, micro_runs=4, total_training_updates=8,
            new_resource_preparation_updates=0, new_personal_control_runs=0, formal_study_runs_started=0,
            scope='Exact limited 2-update gain1 bridges for old seed27101 and lambda0/1; not a full-run or gain3 trajectory reproduction. All original arrays/weights/logs retained.')
        write_new(OUTPUT / 'result.json', result)
        write_new(RECEIPT, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        result = dict(status='failed', runner_sha256=runner_hash, preflight_sha256=sha(__file__),
            development_directory=str(OUTPUT), current_run=current, elapsed_seconds=time.monotonic()-started,
            error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
        write_new(OUTPUT / 'failure.json', result)
        write_new(RECEIPT, result)
        raise


if __name__ == '__main__':
    main()
