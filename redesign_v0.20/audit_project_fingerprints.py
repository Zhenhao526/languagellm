"""Read-only supplement: fingerprint actual prepared project tensors only."""
from pathlib import Path
import hashlib,json,itertools
import torch

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/temporal_001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project(state):return {k:v for k,v in state.items() if k.startswith('project.')}
def fingerprint(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name,str(array.dtype),list(array.shape)],separators=(',',':')).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()

def main():
    torch.set_num_threads(1);inv=read(OUT/'invocation.json');done=read(OUT/'training_complete.json');rows=[];failures=[];matches=0
    assert done['status']=='complete' and done['formal'] and done['head_fits']==48
    for seed in inv['seeds']:
        path=OUT/f'prepared_{seed}.pt';states=torch.load(path,weights_only=True)
        for who in inv['directions']:
            p=project(states[who]);digest=fingerprint(p)
            if set(p)!={'project.0.weight','project.0.bias'} or sum(v.numel() for v in p.values())!=65600:failures.append(['unexpected_project_structure',seed,who])
            rows.append(dict(seed=seed,person=who,tensor_sha256=digest,prepared_file_sha256=sha(path),tensor_metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in p.items()}))
            for partition,arm in itertools.product(inv['partitions'],inv['arms']):
                initial=OUT/f's{seed}_p{partition}_d{who}_{arm}'/'initial.pt';actual=project(torch.load(initial,weights_only=True)['agent'])
                if set(actual)!=set(p) or not all(torch.equal(actual[k],p[k]) for k in p):failures.append(['consumed_project_mismatch',str(initial)])
                matches+=1
    unique=len({r['tensor_sha256'] for r in rows})
    if unique!=8:failures.append(['project_tensor_fingerprints_not_all_distinct',unique])
    result=dict(passed=not failures,script_sha256=sha(Path(__file__)),training_complete_sha256=sha(OUT/'training_complete.json'),invocation_sha256=sha(OUT/'invocation.json'),failures=failures,persons=len(rows),unique_project_tensor_fingerprints=unique,exact_initial_project_matches=matches,rows=rows,
        scope='Only the actual learned project.0.weight and project.0.bias tensors are fingerprinted; container filenames and all unused agent parameters are excluded. All48 temporal initial states exactly inherit their respective prepared project.',
        limitation='Distinct project tensors do not establish statistical independence or rerun the resource-preparation optimizer. New seeds and prescribed training provenance remain separate evidence.',new_training=0)
    (OUT/'prepared_project_fingerprints.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
