"""Small independent raw-log check of the descriptive process summary."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/joint_001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    summary_path=OUT/'process_summary.json';summary=read(summary_path);inv=read(OUT/'invocation.json');reference=Path(inv['reference'])
    count=0;inputs={};observed={};max_error=0.;expected_rows=[];gradient_records=0
    def check(condition,label):
        nonlocal count
        count+=1
        if not bool(condition):raise AssertionError(label)
    check(summary['status']=='complete','summary complete')
    check(summary['source_sha256']==sha(ROOT/'analyze_process.py'),'summary source bound')
    for seed in inv['seeds']:
        for partition in inv['partitions']:
            for arm in ('mean','attention','joint'):
                folder=(OUT if arm=='joint' else reference)/'social'/f's{seed}_p{partition}_{arm}'
                path=folder/'training.jsonl';inputs[str(path)]=sha(path);logs=[json.loads(line) for line in path.read_text().splitlines()]
                check(len(logs)==2400 and [r['update'] for r in logs]==list(range(1,2401)),'complete sequential log rows')
                sample=next(r for r in summary['rows'] if (r['seed'],r['partition'],r['condition'])==(seed,partition,arm))
                check(sample['person_updates']==len(logs)*2,'per-row person update count')
                result=observed.setdefault(arm,dict(person_updates=0,role_clip_counts={'sender':0,'receiver':0},actual_nonunit_coefficients={'sender':0,'receiver':0},maximum_recorded_norm={'sender':0.,'receiver':0.}))
                result['person_updates']+=len(logs)*2
                for role in ('sender','receiver'):
                    values=np.asarray([person['norms'][role] for line in logs for person in line['people']],np.float32)
                    check(np.isfinite(values).all() and (values>=0).all(),'finite nonnegative recorded norms')
                    greater=int(np.count_nonzero(values>np.float32(2.)))
                    # Replay Torch's float32 scalar coefficient, including eps.
                    coefficients=np.minimum(np.float32(2.)/(values+np.float32(1e-6)),np.float32(1.))
                    active=int(np.count_nonzero(coefficients<np.float32(1.)))
                    check(greater==sample['role_clip_counts'][role],'recorded threshold count')
                    check(active==greater,'threshold count matches actual nonunit coefficient for these logs')
                    result['role_clip_counts'][role]+=greater;result['actual_nonunit_coefficients'][role]+=active
                    result['maximum_recorded_norm'][role]=max(result['maximum_recorded_norm'][role],float(values.max()))
                if arm=='joint':
                    values=[person['contrast_gradient_norm_before_clip'] for line in logs for person in line['people']]
                    check(all(math.isfinite(v) and v>=0 for v in values),'all contrast norm records finite')
                    gradient_records+=len(values);first=values[:2]
                    check(first==sample['initial_contrast_gradient_norms'],'first-step contrast records')
                    result.setdefault('initial',[]).extend(first)
                    path=folder/'final.pt';inputs[str(path)]=sha(path);states=torch.load(path,weights_only=True,map_location='cpu')
                    for d,state in enumerate(states):
                        weight=state['focus_sender.contrast.weight'].numpy().astype(np.float64)
                        exact=math.sqrt(math.fsum(float(x*x) for x in weight.ravel()));stored=sample['final_contrast_weight_norms'][d]
                        error=abs(exact-stored);max_error=max(max_error,error)
                        check(error<=8*np.finfo(np.float32).eps*max(1.,exact),'float64 independently accumulated weight norm')
                    result.setdefault('final_recorded',[]).extend(sample['final_contrast_weight_norms'])
                expected_rows.append((seed,partition,arm))
    check(sorted(expected_rows)==sorted((r['seed'],r['partition'],r['condition']) for r in summary['rows']),'exact summary row inventory')
    check(inputs==summary['input_hashes'],'all summary raw sources bound')
    for arm,result in observed.items():
        expected=summary['aggregate'][arm]
        check(result['person_updates']==expected['person_updates'],'aggregate person count')
        check(result['role_clip_counts']==expected['role_clip_counts'],'aggregate clip counts')
        if arm=='joint':
            check(sum(x!=0 for x in result['initial'])==expected['initial_contrast_nonzero'],'nonzero first gradient count')
            check([min(result['initial']),max(result['initial'])]==expected['initial_contrast_range'],'first gradient range')
            check([min(result['final_recorded']),max(result['final_recorded'])]==expected['final_contrast_range'],'final recorded weight norm range')
    output=dict(passed=True,checks=count,failures=[],source_sha256=sha(__file__),summary_sha256=sha(summary_path),input_hashes=inputs,
        observed=observed,contrast_norm_records_checked=gradient_records,max_final_norm_error_float64=max_error,
        scope='All36 logs independently read; totals/ranges recounted. Final24 contrast weights checked by NumPy float64 plus math.fsum; production norms are float32. Recorded gradients are not newly replayed here. Coefficient calculation confirms norm>2 counts are safe for these data.',
        bounds='Shows contrast learns and whether recorded clipping acts; not causal mediation, gradient variance, or complete model-gradient replay.')
    (OUT/'process_summary_qa.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,checks=count,max_final_norm_error_float64=max_error,arms={k:{f:v[f] for f in ('person_updates','role_clip_counts','maximum_recorded_norm')} for k,v in observed.items()})))

if __name__=='__main__':main()
