"""Issue a frozen-runner gate only after an independent saved-smoke audit."""
import argparse
import sys
from pathlib import Path
import torch
from audit_execution import (ROOT,ARMS,ALL_ARMS,Audit,old,audit_partitions,audit_sources,
    audit_one,paired_audit,write_report,sha)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/smoke_001')
    p.add_argument('--seed',type=int,default=99510);p.add_argument('--partition',type=int,default=1)
    p.add_argument('--out',type=Path,default=ROOT/'preflight_qa.json');args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(1);audit=Audit();audit_partitions(audit);photos=old.get_pools(audit)
    root=args.root.resolve();hashes=audit_sources(audit,root,False,1,1,512,1200)
    runs={a:audit_one(audit,root/f's{args.seed}_p{args.partition}_{a}',args.seed,args.partition,a,1,512,1200,hashes,photos)
          for a in ALL_ARMS}
    paired_audit(audit,{(args.seed,args.partition):runs},first_step=True)
    result=write_report(audit,args.out,runner_sha256=hashes[str(ROOT/'run_generalization.py')],
        source_hashes=hashes,smoke_directory=str(root),smoke_completed_runs=5,preflight_script_sha256=sha(__file__),
        scope='Saved one-step base plus all four one-step arms. Three expansion worlds and active-branch updates exact; stay-old matches RNG identity and budget only. One-step entropy is zero; formal schedules source/config audited.')
    if not result['passed']:sys.exit(1)


if __name__=='__main__':main()
