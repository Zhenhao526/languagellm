"""Link independent protocol enumeration to actual final world traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/adaptation_001'
path=OUT/'protocol_analysis.json';data=json.loads(path.read_text())
assert data['complete'] and len(data['runs'])==48
counts=Counter();failures=[]

def check(ok,name,context):
    counts[name]+=1
    if not ok:failures.append(dict(check=name,context=context))

baselines={(r['seed'],r['split']):r for r in data['runs'] if r['arm']=='baseline'}
for run in data['runs']:
    if run['arm']=='baseline':continue
    folder=OUT/f"s{run['seed']}_split{run['split']}_{run['arm']}"
    check(hashlib.sha256((folder/'final.pt').read_bytes()).hexdigest()==run['checkpoint_sha256'],'checkpoint_link',folder.name)
    for direction in run['directions']:
        scout=direction['scout'];baseline=baselines[run['seed'],run['split']]['directions'][scout]
        table=np.asarray([r['actions_by_goal'] for r in direction['receiver_decoder_table']])
        check(direction['menu_audit']['all_menu_permutations_physically_equivalent'],'all_menu_invariance',[folder.name,scout])
        lookup={tuple(photos):np.asarray(phase['messages'])[:,i] for phase in direction['phases'].values()
                for i,photos in enumerate(phase['photo_pairs'])}
        check(len(lookup)==32,'calibration_validation_distinct_photos',[folder.name,scout])
        for mode in ('normal','shuffle','blank','erase_memory'):
            with np.load(folder/f'final_{mode}.npz') as archive:
                z={k:archive[k] for k in archive.files}
                ix=np.flatnonzero(z['scout']==scout)
                codes=z['delivered'][ix,0]*7+z['delivered'][ix,1]
                expected=np.take_along_axis(table[codes],z['goals'][ix],axis=1)
                check(np.array_equal(expected,z['place'][ix]),'enumerated_decoder_replays_actual_actions',[folder.name,scout,mode])
                counts['decoder_world_rows_replayed']+=len(ix)
                if mode=='erase_memory':continue
                source_rows=[j for j in ix if tuple(z['photo_ids'][j]) in lookup]
                mids=z['positions'][source_rows,0]*5+z['positions'][source_rows,1]-(z['positions'][source_rows,1]>z['positions'][source_rows,0])
                emitted=np.asarray([lookup[tuple(z['photo_ids'][j])][m] for j,m in zip(source_rows,mids)])
                check(len(source_rows)>0 and np.array_equal(emitted,z['sent'][source_rows]),'independent_photo_message_replay',[folder.name,scout,mode])
                counts['sender_world_rows_replayed']+=len(source_rows)
        if run['arm']=='sender_only':
            check(direction['receiver_decoder_table']==baseline['receiver_decoder_table'],'sender_only_fixed_receiver',[folder.name,scout])
        if run['arm']=='receiver_only':
            check(all(direction['phases'][p]['messages']==baseline['phases'][p]['messages'] for p in ('calibration','validation')),
                  'receiver_only_fixed_sender',[folder.name,scout])
        for phase in ('calibration','validation'):
            for group in ('old','new','all'):
                r=direction['phases'][phase]['bounds'][group]
                check(r['natural_j']<=r['receiver_coverage']+1e-12 and r['natural_j']<=r['sender_optimal_decoder']+1e-12,
                      'natural_under_frozen_function_bounds',[folder.name,scout,phase,group])

report=dict(passed=not failures,failures=failures,counts=dict(counts),
    protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    scope='All deterministic final receiver actions and sender emissions on the 32 independently enumerated photo pairs; stochastic trajectories are excluded from greedy replay.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'protocol_execution_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
assert report['passed']
