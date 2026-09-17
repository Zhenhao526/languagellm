"""Descriptive summaries of preregistered gradient and clipping logs."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/joint_001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    invocation=read(OUT/'invocation.json');reference=Path(invocation['reference']);rows=[];inputs={}
    for seed in invocation['seeds']:
        for p in invocation['partitions']:
            for arm,base in [('mean',reference),('attention',reference),('joint',OUT)]:
                folder=base/'social'/f's{seed}_p{p}_{arm}';path=folder/'training.jsonl';inputs[str(path)]=sha(path)
                logs=[json.loads(line) for line in path.read_text().splitlines()];assert len(logs)==2400
                counts={role:sum(row['people'][d]['norms'][role]>2. for row in logs for d in (0,1)) for role in ('sender','receiver')}
                row=dict(seed=seed,partition=p,condition=arm,person_updates=4800,role_clip_counts=counts)
                if arm=='joint':
                    row['initial_contrast_gradient_norms']=[logs[0]['people'][d]['contrast_gradient_norm_before_clip'] for d in (0,1)]
                    assert all(np.isfinite(person['contrast_gradient_norm_before_clip']) for log in logs for person in log['people'])
                    path=folder/'final.pt';inputs[str(path)]=sha(path);states=torch.load(path,weights_only=True)
                    row['final_contrast_weight_norms']=[float(s['focus_sender.contrast.weight'].norm()) for s in states]
                rows.append(row)
    aggregate={}
    for arm in ('mean','attention','joint'):
        rr=[r for r in rows if r['condition']==arm];z=dict(person_updates=sum(r['person_updates'] for r in rr),role_clip_counts={role:sum(r['role_clip_counts'][role] for r in rr) for role in ('sender','receiver')})
        if arm=='joint':
            initial=np.array([x for r in rr for x in r['initial_contrast_gradient_norms']]);final=np.array([x for r in rr for x in r['final_contrast_weight_norms']])
            z.update(initial_contrast_nonzero=int(np.count_nonzero(initial)),initial_contrast_range=[float(initial.min()),float(initial.max())],final_contrast_range=[float(final.min()),float(final.max())])
        aggregate[arm]=z
    output=dict(status='complete',role='Descriptive aggregation of preregistered recorded norms; not causal mediation or full gradient replay',
        source_sha256=sha(Path(__file__)),input_hashes=inputs,rows=rows,aggregate=aggregate)
    (OUT/'process_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(aggregate))
if __name__=='__main__':main()
