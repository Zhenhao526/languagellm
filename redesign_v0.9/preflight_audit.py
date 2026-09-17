"""Verify the final-source one-step three-arm smoke and issue a formal-run gate."""
from datetime import datetime,timezone
import json
import sys
from pathlib import Path
import torch
from audit_adaptation import (ROOT,ARMS,SENDER,RECEIVER,Audit,sha,read,get_pools,
                              audit_sources,audit_one,paired_audit)


def main():
    torch.set_num_threads(1)
    smoke=ROOT/'results/smoke_002';out=ROOT/'preflight_qa.json'
    if out.exists():raise FileExistsError('Preserve prior preflight gate: '+str(out))
    audit=Audit();pools=get_pools(audit)
    hashes=audit_sources(audit,smoke,False,1,512,1200)
    runs={arm:audit_one(audit,smoke/f's27101_split1_{arm}',27101,1,arm,1,512,1200,hashes,pools) for arm in ARMS}
    paired_audit(audit,{(27101,1):runs})
    for arm,prefix in (('sender_only',SENDER),('receiver_only',RECEIVER)):
        for who in (0,1):
            single=runs[arm]['final'][who];both=runs['both']['final'][who];initial=runs[arm]['initial'][who]
            keys=[k for k in initial if k.startswith(prefix)]
            audit.check(all(torch.equal(single[k],both[k]) for k in keys),
                        'first_step_single_module_equals_both_exact',[arm,who])
            audit.check(any(not torch.equal(single[k],initial[k]) for k in keys),
                        'first_step_active_module_really_updated',[arm,who])
        left=runs[arm]['rows'][0];right=runs['both']['rows'][0]
        audit.check(all(left[k]==right[k] for k in ('rng_seed','world_sha256','reward','both_accuracy','single_accuracy','entropy_weight')),
                    'first_step_same_world_actions_and_return',arm)
    qa=dict(passed=not audit.failures,source_sha256=hashes[str(ROOT/'run_adaptation.py')],
        source_hashes=hashes,smoke_directory=str(smoke),smoke_updates=1,smoke_eval_worlds=1200,
        checks=dict(audit.counts),total_checks=sum(audit.counts.values()),failures=audit.failures,
        trace_world_rows_recounted=audit.trace_rows,trace_actions_recounted=2*audit.trace_rows,
        preflight_script_sha256=sha(__file__),audit_helper_sha256=sha(ROOT/'audit_adaptation.py'),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Final-source saved one-step run, including exact active-branch agreement across all three arms. Smoke entropy is zero by its one-step schedule; formal entropy schedule is source/config audited.')
    out.write_text(json.dumps(qa,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=qa['passed'],checks=qa['total_checks'],failures=len(audit.failures),
        source_sha256=qa['source_sha256'],out=str(out)),ensure_ascii=False))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
