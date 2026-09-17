"""Evidence gate for the fixed twelve-run batch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import run_joint as run

def main():
    smoke=ROOT/'results/smoke_001';proofs={}
    for p in [ROOT/'prior_integrity_qa.json',ROOT/'self_test_qa.json',ROOT/'synthetic_replay_check.json',ROOT/'code_review.json',
              smoke/'audit_execution.json',smoke/'comparison.json',smoke/'structural_assay_audit.json']:
        x=run.read(p);assert x.get('passed',x.get('status')=='passed'),(p,x.keys());proofs[str(p)]=run.sha(p)
    receipt=run.read(smoke/'training_complete.json');assert receipt['status']=='complete' and receipt['social_runs']==1 and receipt['pair_updates']==40
    hashes=run.source_hashes();assert receipt['source_hashes']==hashes
    for p,h in receipt['files'].items():assert run.sha(smoke/p)==h,p
    for p,h in {**hashes,**receipt['input_hashes']}.items():assert run.sha(p)==h,p
    for name in ('training_complete.json','analysis.json','structural_assay.json'):
        p=smoke/name;proofs[str(p)]=run.sha(p)
    out=dict(passed=True,source_hashes=hashes,evidence_hashes=proofs,development_social_runs=1,development_pair_updates=40,
        development_exit_code=0,development_process_collected=True,independent_development_audit='passed before formal launch',
        formal_matrix=dict(seeds=run.SEEDS,partitions=[1,2,3],new_arms=['joint'],updates=2400,pair_runs=12),
        new_dino_inferences=0,new_private_updates=0,new_cache_inferences=0,primary_fixed='old18 J joint minus historical mean',outcome_tuning=False,
        audit_scope='Initial complete49 distributions/greedy same historical mean, full dev endpoint/statistics, all40world streams and first paired Adam update; frozen cache hashes only')
    run.write(ROOT/'preflight_qa.json',out);print(json.dumps(dict(passed=True,source_files=len(hashes),evidence_files=len(proofs))))
if __name__=='__main__':main()
