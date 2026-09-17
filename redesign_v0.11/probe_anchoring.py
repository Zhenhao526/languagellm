"""Pre-fixed dynamic legacy-message constraints; probe after all 36 runs finish."""
from itertools import product
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_anchoring as run
sys.path.insert(0,str(ROOT.parent/'redesign_v0.10'))
import probe_protocol as oldprobe
import analyze_compatibility as compat
from camp import ImageBank,remake_agents,projected_banks,scene_visual,MAPS,write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def extract(agents,bank,photos):
    banks=projected_banks(agents,bank);ix=np.array(list(product(range(30),range(16))))
    positions=MAPS[ix[:,0]];photo_ids=photos[ix[:,1]];out=[]
    codes=torch.tensor(list(product(range(7),repeat=2)));n=len(positions)
    for d in (0,1):
        s=agents[d];r=agents[1-d]
        physical,_,menu_audit=oldprobe.p8.receiver_table(r)
        logits=[]
        for goal in (0,1):
            g=torch.zeros(49,2);g[:,goal]=1
            y,_=r.receive(codes,g,torch.zeros(49,2),torch.zeros(49,18),torch.arange(6).repeat(49,1))
            logits.append(y.numpy())
        receiver_logits=np.stack(logits,axis=1)
        assert menu_audit['all_menu_permutations_physically_equivalent']
        assert np.array_equal(receiver_logits.argmax(-1),physical[:,:,0])
        h=s.observe(scene_visual(positions,photo_ids,banks[d]));state=s.send_context(torch.cat((h,torch.zeros(n,4)),1))
        first=s.send_out(state).numpy();tokens=torch.arange(7).repeat(n)
        second=s.send_out(s.send_recur(s.send_embedding(tokens),state.repeat_interleave(7,0))).numpy().reshape(n,7,7)
        message,_,_,_=s.send(h,torch.zeros(n,2),torch.zeros(n,2),np.random.default_rng(11011),True)
        out.append((dict(positions=positions,photo_ids=photo_ids,greedy_message=message.numpy(),
            sender_first_logits=first,sender_second_logits=second,receiver_logits=receiver_logits),menu_audit))
    return out


def legacy_bound(base,current,p):
    pools=run.v10.partition(p);old=pools['old'];new=pools['added']
    oldix=(old[:,None]*16+np.arange(16)).reshape(-1);newix=(new[:,None]*16+np.arange(16)).reshape(-1)
    bmsg=base['greedy_message'];cmsg=current['greedy_message']
    bid=7*bmsg[:,0]+bmsg[:,1];cid=7*cmsg[:,0]+cmsg[:,1]
    bdec=base['receiver_logits'].argmax(-1);cdec=current['receiver_logits'].argmax(-1)
    oc=np.array([np.bincount(bid[m*16:(m+1)*16],minlength=49) for m in old])
    nc=np.array([np.bincount(cid[m*16:(m+1)*16],minlength=49) for m in new])
    dp=compat.solve(oc,nc);oldpos=base['positions'][oldix];newpos=current['positions'][newix]
    def counts(decoder):
        return (int((decoder[bid[oldix]]==oldpos).all(1).sum()),int((decoder[cid[newix]]==newpos).all(1).sum()))
    initial_old=int((bdec[bid[oldix]]==oldpos).all(1).sum());actual_old,actual_new=counts(cdec)
    def witness(threshold):
        o,n,options=compat.optimum(dp,threshold);table=np.zeros((49,2),dtype=int)
        for m,option in enumerate(options):
            if option==0 and dp['old_best'][m]>0:table[m]=MAPS[old[oc[:,m].argmax()]]
            if option==1 and dp['added_best'][m]>0:table[m]=MAPS[new[nc[:,m].argmax()]]
        assert counts(table)==(o,n)
        return dict(threshold=threshold,old_correct=o,new_correct=n,new_J=n/96,decoder=table.tolist())
    own=witness(actual_old);fixed=witness(initial_old);assert actual_new<=own['new_correct']
    if actual_old>=initial_old:assert actual_new<=fixed['new_correct']
    used=oc.sum(0)>0;strict=bdec.copy()
    for m in np.flatnonzero(~used):
        if dp['added_best'][m]>0:strict[m]=MAPS[new[nc[:,m].argmax()]]
    so,sn=counts(strict);assert so==initial_old
    assert np.array_equal(strict[bid[oldix]],bdec[bid[oldix]])
    max_old_cs=int(dp['old_best'][dp['added_best']==0].sum());csw=witness(max_old_cs)
    assert csw['new_correct']==int(dp['added_best'].sum())
    return dict(old_n=288,new_n=96,initial_old_correct=initial_old,actual_legacy_old_correct=actual_old,
        actual_new_correct=actual_new,new_J=actual_new/96,legacy_old_J=actual_old/288,
        current_sender_old_receiver_old_J=float((bdec[cid[oldix]]==oldpos).all(1).mean()),
        current_old_J=float((cdec[cid[oldix]]==oldpos).all(1).mean()),
        actual_retention_bound=own,fixed_retention_bound=fixed,meets_fixed_old_threshold=actual_old>=initial_old,
        strict_legacy_actions_bound=dict(old_correct=so,new_correct=sn,new_J=sn/96,decoder=strict.tolist()),
        independent_new_CS=float(dp['added_best'].sum()/96),min_old_loss_for_new_CS_pp=100*max(0,initial_old-max_old_cs)/288,
        new_CS_witness=csw,old_message_change=float((bmsg[oldix]!=cmsg[oldix]).any(1).mean()),
        new_using_old_used_codes=float(used[cid[newix]].mean()),old_counts=oc.tolist(),new_counts=nc.tolist())


def scalar(row):
    b=row['legacy'];return dict(new_J=b['new_J'],legacy_old_J=b['legacy_old_J'],
        current_sender_old_receiver_old_J=b['current_sender_old_receiver_old_J'],current_old_J=b['current_old_J'],
        actual_retention_bound=b['actual_retention_bound']['new_J'],fixed_retention_bound=b['fixed_retention_bound']['new_J'],
        strict_legacy_actions_bound=b['strict_legacy_actions_bound']['new_J'],independent_new_CS=b['independent_new_CS'],
        min_old_loss_for_new_CS_pp=b['min_old_loss_for_new_CS_pp'],old_message_change=b['old_message_change'],
        new_using_old_used_codes=b['new_using_old_used_codes'],fixed_threshold_met_fraction=float(b['meets_fixed_old_threshold']))


def main():
    torch.set_num_threads(1);batch=ROOT/'results/anchoring_001'
    assert all((batch/f's{s}_p{p}_{a}/result.json').exists() for s,p,a in product(run.SEEDS,(1,2,3),run.ARMS))
    manifest=json.loads((ROOT/'protocol_fixed_manifest.json').read_text())
    for path,value in manifest['hashes'].items():assert sha(path)==value
    out=batch/'protocol';out.mkdir(exist_ok=False);bank=ImageBank()
    photos=np.array(list(product(bank.pools['test',0][4:8],bank.pools['test',1][4:8])))
    records=[]
    for seed,p in product(run.SEEDS,(1,2,3)):
        source=run.SOURCE/f's{seed}_p{p}_base/final.pt'
        prepared=torch.load(run.SOURCE/f'prepared_{seed}.pt',weights_only=True)
        agents=remake_agents(seed,prepared,7,2,'identity')
        for a,state in zip(agents,torch.load(source,weights_only=True)):a.load_state_dict(state)
        bases=extract(agents,bank,photos)
        phases=[('base',0,source,bases)]
        phases += [(arm,step,batch/f's{seed}_p{p}_{arm}/checkpoint_{step:04d}.pt',None) for arm,step in product(run.ARMS,(300,600))]
        midpoint={}
        for arm,step,path,computed in phases:
            if computed is None:
                for a,state in zip(agents,torch.load(path,weights_only=True)):a.load_state_dict(state)
                computed=extract(agents,bank,photos)
            directions=[]
            for d,(z,menu) in enumerate(computed):
                if step==300 and arm=='anchor':midpoint[d]=z
                if step==300 and arm=='release':assert all(np.array_equal(z[k],midpoint[d][k]) for k in z)
                filename=f's{seed}_p{p}_{arm}_u{step:04d}_d{d}.npz';np.savez_compressed(out/filename,**z)
                directions.append(dict(direction=d,arrays=filename,sha256=sha(out/filename),menu_audit=menu,
                    groups=oldprobe.summarize(z,p),legacy=legacy_bound(bases[d][0],z,p)))
            records.append(dict(seed=seed,partition=p,arm=arm,update=step,source_checkpoint=str(path),
                source_sha256=sha(path),directions=directions))
            print(f'PROBE s{seed} p{p} {arm} {step}',flush=True)
    cells=[];aggregate={}
    for arm,step in [('base',0)]+list(product(run.ARMS,(300,600))):
        label=f'{arm}_{step}';values=[]
        for seed in run.SEEDS:
            selected=[d for r in records if (r['seed'],r['arm'],r['update'])==(seed,arm,step) for d in r['directions']]
            assert len(selected)==6;srows=[scalar(d) for d in selected]
            mean={k:float(np.mean([r[k] for r in srows])) for k in srows[0]}
            cells.append(dict(seed=seed,arm=arm,update=step,metrics=mean));values.append(mean)
        aggregate[label]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    result=dict(complete=True,records=records,seed_cells=cells,aggregate=aggregate,photos=photos.tolist(),
        source_hashes=manifest['hashes'],unit='Four inherited seeds; average 3 partitions and 2 directions within seed.',
        constrained_decoder_class='Deterministic full-code joint-action tables on fixed old-S0 and current-new-S distributions.')
    write_json(batch/'protocol_analysis.json',result)
    lines=['# 旧表达保持与新增表达的协议诊断','',
        '本表固定旧发送者的old18消息分布，新增6图消息来自当前发送者；同图16照片。Tθ是当前接收者使用旧消息时实际取得的旧正确数。界允许分析者用真地图任意重新赋义确定性整码，不保证神经主体可学。','',
        '| 阶段 | 当前新增J | 旧S→当前R旧J | Tθ下新增界 | T0下新增界 | 满足T0方向比例 | 新增码占用旧已用码 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase,m in aggregate.items():
        keys=('new_J','legacy_old_J','actual_retention_bound','fixed_retention_bound','fixed_threshold_met_fraction','new_using_old_used_codes')
        lines.append('| '+phase+' | '+' | '.join(f'{100*m[k]:.2f}%' for k in keys)+' |')
    lines+=['','T0为形成终点的旧正确总数；未满足T0的方向不受该固定门槛限制。原始分布、全部图组N/U/Q/B/C_S、逐例旧动作保持界、最小旧损失及码表见证均在JSON中。封存图未参与兼容优化。','',
        '[完整数据](protocol_analysis.json)；[事前方案](../../协议测量方案.md)。本诊断不包含片段拼接或组合语法验证。']
    (batch/'协议诊断.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,ensure_ascii=False))


if __name__=='__main__':main()
