"""Unselected illustrative protocol: fixed33101/p1/d0, first photo pair/mask0."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/attention_001'
sys.path.insert(0,str(ROOT.parent/'redesign_v0.23'));import world
alphabet='@#%&*+='
rows=[];files={}
for arm in ('mean','attention'):
    path=OUT/'social'/f's33101_p1_{arm}'/'protocol_2400_d0.npz';files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=dict(np.load(path));photo=raw['photo_ids'][0]
    idx=np.flatnonzero((raw['shown']==0)&(raw['photo_ids']==photo).all(-1))
    idx=sorted(idx,key=lambda i:int(raw['map_id'][i]))[:6]
    acts=raw['receiver_logits'].argmax(-1);old=set(world.partition(1)['old'])
    for i in idx:
        token=raw['tokens'][i];code=int(token[0]*7+token[1]);target=raw['positions'][i];action=acts[code]
        rows.append(dict(condition=arm,map_id=int(raw['map_id'][i]),photo_ids=photo.tolist(),shown=0,
            social_exposure='old18' if int(raw['map_id'][i]) in old else 'new12',
            target_sites_1based=(target+1).tolist(),tokens=token.tolist(),display_message=''.join(alphabet[int(t)] for t in token),
            receiver_sites_1based=(action+1).tolist(),joint_correct=bool(np.array_equal(target,action))))
payload=dict(selection='Before outcome inspection: source33101/p1/direction0, first photo pair, food-only mask, first6 map IDs; no performance filtering',
    input_hashes=files,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),alphabet=alphabet,rows=rows)
(OUT/'protocol_examples.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload,ensure_ascii=False))
