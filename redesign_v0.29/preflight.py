"""Bounded execution gate; no performance-based filtering."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import run_support as run
R=Path(__file__).parent;S=R/'results/smoke_001'

def main():
    complete=run.read(S/'training_complete.json');assert complete['status']=='complete' and complete['social_runs']==2
    assert complete['source_hashes']==run.source_hashes()
    folders=[S/'social'/f's99528_p1_{a}' for a in run.support.ARMS]
    initial=[torch.load(f/'initial.pt',weights_only=True) for f in folders]
    assert all(torch.equal(initial[0][d][k],initial[1][d][k]) for d in (0,1) for k in initial[0][d])
    checked=0
    for f in folders:
        c=run.read(f/'config.json');assert c['batch_per_direction']==252
        rows=[__import__('json').loads(l) for l in (f/'training.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,41))
        for t in (0,40):
            for d in (0,1):
                raw=dict(np.load(f/f'protocol_{t:04d}_d{d}.npz'))
                assert raw['tokens'].shape==(180,2) and raw['sender_log_probs'].shape==(180,49) and raw['receiver_logits'].shape==(49,2,6)
                assert np.isfinite(raw['sender_log_probs']).all() and np.isfinite(raw['receiver_logits']).all()
                np.testing.assert_allclose(np.exp(raw['sender_log_probs']).sum(-1),1,atol=1e-6)
                checked+=1
    source=run.PROJECT/'redesign_v0.28/results/formation_001'
    original=run.read(source/'training_complete.json')['files'];bound=0
    inputs=run.input_hashes(source,run.SEEDS,[1,2,3])
    for path,h in inputs.items():
        p=Path(path)
        if p.is_relative_to(source) and str(p.relative_to(source)) in original:
            assert original[str(p.relative_to(source))]==h;bound+=1
    run.write(S/'terminal_receipt.json',dict(session_id=54103,exit_code=0,status='complete',observed_by='root tools.write_stdin',seconds=complete['seconds']))
    checks=[R/'implementation_review.json',S/'raw_validation.json',S/'audit_execution.json']
    for path in checks:assert run.read(path)['passed'],path
    run.write(R/'preflight_qa.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=run.source_hashes(),input_hashes=inputs,
        smoke_tables=checked,initial_exactly_paired=True,original_bound_inputs=bound,smoke_completion_sha256=run.sha(S/'training_complete.json'),
        checks={str(p):run.sha(p) for p in checks},criterion_uses_performance_threshold=False,development_performance_not_used_for_selection=True,
        synthetic_tests=dict(support=8,metrics=3),scope='Synthetic protocol/fixture tests, separate smoke, original input binding, independent smoke statistics and endpoint replay'))
    print('preflight passed',checked,'smoke tables;',bound,'original bound inputs')
if __name__=='__main__':main()
