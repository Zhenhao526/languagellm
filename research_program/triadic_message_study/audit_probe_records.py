"""Independent physical/accounting checks of completed counterfactual records.

Does not load weights or regenerate counterfactual messages. The separate main
audit checks learned natural policies; this audit checks saved interventions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from research_program.triadic_message_study import audit_execution as independent

MODES = ('cross_channels_closed', 'action_only_drop', 'constant_full')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(source, probes, output):
    source, probes, output = map(lambda p: Path(p).resolve(), (source, probes, output))
    assert not output.exists()
    assert read(source/'execution/status.json')['status'] == 'completed'
    assert read(probes/'execution/status.json')['status'] == 'completed'
    root = read(probes/'execution/results.json')
    assert root['status'] == 'completed' and root['run_partitions'] == 64
    expected = {(s,c,p) for s in independent.SEEDS for c in independent.CONDITIONS for p in independent.PARTITIONS}
    assert len(root['rows']) == 64 and {(r['seed'],r['condition'],r['partition']) for r in root['rows']} == expected
    source_hashes = {str(Path(__file__).resolve()):sha(__file__), str(Path(independent.__file__).resolve()):sha(independent.__file__)}
    started = time.perf_counter(); bindings = {}; checks = []; total = 0
    for row in root['rows']:
        name = f"seed_{row['seed']}_{row['condition']}"
        folder = probes/'execution'/name/row['partition']
        assert read(folder/'result.json') == row
        for filename,digest in row['files_sha256'].items():
            assert sha(folder/filename) == digest
            bindings[str(folder/filename)] = digest
        original = source/'execution'/name/f"final_{row['partition']}.npz"
        bindings[str(original)] = sha(original)
        with np.load(original,allow_pickle=False) as f:
            state = f['states']; natural_actions = f['action_indices']; natural_messages = f['messages']
            natural_probs = f['action_probabilities']; natural_reward = f['greedy_reward']
        full = row['condition'].startswith('FI_'); live = row['condition'].endswith('_live')
        n = len(state); checked_modes = {}
        with np.load(folder/'deletions.npz',allow_pickle=False) as f:
            assert np.array_equal(f['states'],state)
            assert np.array_equal(f['natural_action_indices'],natural_actions)
            for mode in MODES:
                a = f[mode+'_actions']; p = f[mode+'_probabilities']; m = f[mode+'_messages']
                assert a.shape == (n,3) and p.shape == (n,3,17) and m.shape == (n,2,3,4)
                assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-14,rtol=0)
                assert np.array_equal(p.argmax(-1),a)
                assert np.array_equal(m[:,0],natural_messages[:,0])
                reward,executed,satisfied = independent.native_settlement(state,a)
                assert np.array_equal(reward,f[mode+'_reward'])
                score = row['deletions'][mode]
                assert score['full_successes'] == int((reward==1).sum())
                assert score['reward_sum'] == float(reward.sum())
                assert score['worlds_with_any_action_change'] == int((a!=natural_actions).any(-1).sum())
                assert score['agent_action_changes'] == (a!=natural_actions).sum(0).tolist()
                assert score['fullsuccess_lost'] == int(((natural_reward==1)&(reward!=1)).sum())
                assert score['fullsuccess_gained'] == int(((natural_reward!=1)&(reward==1)).sum())
                assert score['executed_transports'] == int(executed.sum())
                assert abs(score['full_success_rate_change_from_natural']-((reward==1).mean()-(natural_reward==1).mean())) < 1e-15
                if not live:
                    assert np.array_equal(a,natural_actions) and np.array_equal(m,natural_messages)
                    assert np.allclose(p,natural_probs,atol=2e-12,rtol=0)
                if mode == 'action_only_drop': assert np.array_equal(m,natural_messages)
                checked_modes[mode] = dict(worlds=n,full_successes=score['full_successes'],native_settlement_exact=True)
                total += n
            # Every actual constant-condition action input: official observation,
            # own real message, cross index-0 content only when visible.
            messages = f['constant_full_messages']; actual = f['constant_full_actual_action_inputs_uint8']
            assert actual.shape == (n,3,252) and actual.dtype == np.uint8
            for start in range(0,n,1024):
                stop=min(n,start+1024); expected_inputs=[independent.observed_features(state[start:stop],full)]
                for window in range(2):
                    routed=np.zeros((stop-start,3,99),dtype=np.uint8)
                    for listener in range(3):
                        for sender in range(3):
                            if listener==sender or live:
                                routed[:,listener,96+sender]=1
                                for position in range(4):
                                    token=messages[start:stop,window,sender,position] if listener==sender else np.zeros(stop-start,dtype=int)
                                    routed[np.arange(stop-start),listener,32*sender+8*position+token]=1
                    expected_inputs.append(routed)
                assert np.array_equal(np.concatenate(expected_inputs,-1),actual[start:stop])
        checks.append(dict(seed=row['seed'],condition=row['condition'],partition=row['partition'],modes=checked_modes,constant_actual_inputs_exact=True))
    assert total == 2294784*3
    assert all(sha(p)==h for p,h in source_hashes.items())
    output.mkdir(parents=True)
    result=dict(status='passed',checks=checks,physical_world_settlements=total,source_sha256=source_hashes,input_sha256=bindings,
        no_weight_loads_or_neural_forwards=True,elapsed_seconds=time.perf_counter()-started,
        scope='Independent native settlement and complete accounting of all saved intervention worlds; actual constant inputs checked in every world. Does not independently regenerate second-window messages or action probabilities; content-pair audit is separate.')
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','physical_world_settlements','elapsed_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--probes',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.source,args.probes,args.out)
