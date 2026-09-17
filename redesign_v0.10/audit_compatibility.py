"""Independent full subset enumeration on actual shared codes; no DP import."""
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/generalization_001'
OUT=BATCH/'compatibility_exploratory_002'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=json.loads((OUT/'compatibility_analysis.json').read_text())
    checks=0;enumerated=0;witnesses=0;loss=0;gain=0;change=0
    for row in result['directions']:
        s,p,d=row['seed'],row['partition'],row['direction']
        old=np.asarray(row['old_map_ids']);new=np.asarray(row['added_map_ids'])
        maps=np.array(list(__import__('itertools').permutations(range(6),2)))
        with np.load(BATCH/f'protocol/s{s}_p{p}_base_d{d}.npz') as z:
            ix=(np.r_[old,new][:,None]*16+np.arange(16)).reshape(-1)
            positions=z['positions'][ix];tokens=z['greedy_message'][ix]
            codes=tokens[:,0]*7+tokens[:,1];before=z['receiver_logits'].argmax(-1)
        with np.load(BATCH/f'protocol/s{s}_p{p}_expand_receiver_d{d}.npz') as z:
            after=z['receiver_logits'].argmax(-1)
        oc=np.zeros((49,18),int);nc=np.zeros((49,6),int)
        for i,(code,pos) in enumerate(zip(codes,positions)):
            if i<288:oc[code,np.flatnonzero((maps[old]==pos).all(1))[0]]+=1
            else:nc[code,np.flatnonzero((maps[new]==pos).all(1))[0]]+=1
        assert np.array_equal(oc.T,row['old_counts_by_map_message'])
        assert np.array_equal(nc.T,row['added_counts_by_map_message']);checks+=2
        om=oc.max(1);nm=nc.max(1);shared=np.flatnonzero((om>0)&(nm>0))
        assert len(shared)<=10  # Actual batch is small enough to enumerate fully.
        pairs=[]
        for bits in product((0,1),repeat=len(shared)):
            nsel=np.asarray(bits,dtype=bool)
            o=int(om[nm==0].sum()+om[shared[~nsel]].sum())
            n=int(nm[om==0].sum()+nm[shared[nsel]].sum())
            pairs.append((n,o));enumerated+=1
        candidates=row['retain_baseline_total_frontier']+[row['actual_endpoint_retention_bound'],row['attain_added_CS']['witness']]
        for w in candidates:
            best=max((n,o) for n,o in pairs if o>=w['minimum_old_correct'])
            assert best==(w['added_correct'],w['old_correct']);checks+=1
        cs=int(nm.sum());o_at_cs=max(o for n,o in pairs if n==cs)
        assert max(0,row['baseline']['old_correct']-o_at_cs)==row['attain_added_CS']['minimum_necessary_old_correct_loss'];checks+=1
        for w in candidates+[row['keep_all_old_actions'],row['baseline'],row['receiver_endpoint']]:
            decoder=np.asarray(w['actions_by_resource']);correct=(decoder[codes]==positions).all(1)
            assert (int(correct[:288].sum()),int(correct[288:].sum()))==(w['old_correct'],w['added_correct'])
            checks+=1;witnesses+=1
        strict=np.asarray(row['keep_all_old_actions']['actions_by_resource'])
        assert np.array_equal(strict[codes[:288]],before[codes[:288]]);checks+=1
        # All old-used messages must retain their actions; all other messages can
        # contribute their maximum new frequency. Verify optimality directly.
        used=oc.sum(1)>0
        fixed=sum(nc[m,k] for m in np.flatnonzero(used) for k,pos in enumerate(maps[new]) if np.array_equal(before[m],pos))
        assert int(fixed+nm[~used].sum())==row['keep_all_old_actions']['added_correct'];checks+=1
        b=(before[codes[:288]]==positions[:288]).all(1);a=(after[codes[:288]]==positions[:288]).all(1)
        loss+=int((b&~a).sum());gain+=int((a&~b).sum())
        change+=int((before[codes[:288]]!=after[codes[:288]]).any(1).sum())
        for path,digest in row['source_hashes'].items():assert sha(Path(path))==digest;checks+=1
    assert len(result['directions'])==24
    assert sha(ROOT/'analyze_compatibility.py')==result['script_sha256']
    report=dict(passed=True,checks=checks,actual_shared_code_subsets_enumerated=enumerated,
        witness_tables_replayed=witnesses,directions=24,old_photograph_rows=24*288,
        old_correct_became_wrong=loss,old_wrong_became_correct=gain,old_action_pairs_changed=change,
        note='Exact subset enumeration of every real shared-code old/new allocation, with raw-photo count reconstruction; no dynamic-programming implementation imported.',
        script_sha256=sha(Path(__file__)),analysis_sha256=sha(OUT/'compatibility_analysis.json'))
    (OUT/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
