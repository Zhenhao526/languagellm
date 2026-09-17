"""Verify development execution before the fixed-budget production matrix."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_formation as run

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
    assert set(a)==set(b)
    assert all(torch.equal(a[k],b[k]) for k in a)

def main():
    out=ROOT/'results/smoke_001';done=read(out/'training_complete.json')
    assert done['status']=='complete' and done['private_fits']==4 and done['social_runs']==3
    assert read(out/'invocation.json')['seeds']==[99528]
    assert read(out/'terminal_receipt.json')['exit_code']==0
    for p,h in {**done['source_hashes'],**done['input_hashes']}.items():assert sha(p)==h,p
    for split,n in [('train',720),('test',180)]:
        w=np.load(out/f'{split}_worlds.npz');assert len(w['map_id'])==n
        assert len(set(map(tuple,np.column_stack((w['map_id'],w['photo_ids'],w['shown'])))))==n
    tables=0
    for f in list((out/'private').glob('*/evaluation_*.npz'))+list((out/'social').glob('*/protocol_*.npz')):
        r=np.load(f);assert len(r['map_id'])==180
        assert all(np.isfinite(r[k]).all() for k in r.files)
        if 'sender_log_probs' in r:
            assert r['sender_log_probs'].shape==(180,49) and r['receiver_logits'].shape==(49,2,6)
            assert r['tokens'].shape==(180,2) and ((r['tokens']>=0)&(r['tokens']<7)).all()
        else:assert r['logits'].shape==(180,2,6) and r['h'].shape==(180,96)
        tables+=1
    assert tables==20
    for d in (0,1):
        a=torch.load(out/'private'/f's99528_p1_d{d}_old/initial.pt',weights_only=True)
        b=torch.load(out/'private'/f's99528_p1_d{d}_all/initial.pt',weights_only=True)
        same(a['agent'],b['agent']);same(a['head'],b['head'])
    states=[torch.load(out/'social'/f's99528_p1_{arm}/initial.pt',weights_only=True) for arm in run.ARMS]
    prefixes=run.communication.SENDER_MODULES+run.communication.RECEIVER_MODULES
    for d in (0,1):
        blocks=[{k:v for k,v in s[d].items() if k.startswith(prefixes)} for s in states]
        same(blocks[0],blocks[1]);same(blocks[1],blocks[2]);same(states[1][d],states[2][d])
    fidelity=read(ROOT/'实现保真复核.json')
    result=dict(passed=True,source_hashes=run.source_hashes(),smoke_completion_sha256=sha(out/'training_complete.json'),
        smoke_tables=tables,train_worlds=720,test_worlds=180,initial_states_paired=True,
        source_fidelity_record_sha256=sha(ROOT/'实现保真复核.json'),world_test_log_sha256=sha(ROOT/'world_test_stdout.log'),
        scope='Development completion, raw finite shapes, exact paired initial states and unchanged inputs; independent statistical and endpoint replay results separately required before final reporting.',
        criterion_uses_performance_threshold=False,development_performance_not_used_for_selection=True)
    path=ROOT/'preflight_qa.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,smoke_tables=tables,initial_states_paired=True)))

if __name__=='__main__':main()
