"""Finite-world and analytic-outcome checks before formal learning."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
import torch
import run_experience as run
import world,metrics


def main():
    torch.set_num_threads(1);bank=run.ImageBank();train=world.table(bank,'train');test=world.table(bank,'test');checks=[]
    assert len(test['map_id'])==960 and np.array_equal(np.bincount(test['map_id']),np.full(30,32))
    for resource in (0,1):assert not set(train['photo_ids'][:,resource])&set(test['photo_ids'][:,resource])
    for p in (1,2,3):
        sets=world.partition(p);assert all(len(sets[k])==n for k,n in [('old',18),('added',6),('sealed',6)])
        assert len(np.unique(np.r_[sets['old'],sets['added'],sets['sealed']]))==30
        for phase in ('private','social'):
            for step in (0,39,2100,2399):
                old=world.fixture(99523,p,0,step,'old',phase,train);full=world.fixture(99523,p,0,step,'all',phase,train)
                n=len(old['indices'])//2
                for f in (old,full):
                    a=world.subset(train,f['indices']);assert np.array_equal(a['map_id'][:n],a['map_id'][n:])
                    assert np.array_equal(a['photo_ids'][:n],a['photo_ids'][n:]);assert (a['shown'][:n]==0).all() and (a['shown'][n:]==1).all()
                assert np.isin(train['map_id'][old['indices']],sets['old']).all()
                assert np.array_equal(old['uniforms'],full['uniforms']) and np.array_equal(old['goals'],full['goals'])
                assert np.array_equal(train['photo_ids'][old['indices']],train['photo_ids'][full['indices']])
        # Exact perfect actor and current-object-only actor have known greedy J.
        logits=np.full((960,2,6),-8.);r=np.arange(960)
        for k in (0,1):logits[r,k,test['positions'][:,k]]=8
        good=metrics.private(dict(test,logits=logits),p);assert all(good[k]['pooled']['J']==1 for k in good)
        scores=np.zeros_like(logits);visible=test['positions'][r,test['shown']]
        for k in (0,1):scores[r,k,visible]=1
        bad=metrics.private(dict(test,logits=scores),p);assert all(bad[k]['pooled']['J']==0 for k in bad)
        # Distinct complete code per map can solve all30; hence J alone is not compositionality.
        tokens=np.c_[test['map_id']//7,test['map_id']%7];rl=np.full((49,2,6),-8.)
        for mid,pos in enumerate(world.MAPS):
            for k in (0,1):rl[mid,k,pos[k]]=8
        lp=np.full((960,49),-30.);lp[r,test['map_id']]=0
        comm=metrics.social(dict(test,tokens=tokens,sender_log_probs=lp,receiver_logits=rl),p)
        assert all(comm[k]['pooled']['J']==1 for k in comm)
        assert np.isclose(comm['common30']['pooled']['shuffle_J'],1/30)
        checks.append(dict(partition=p,paired_masks=True,support_and_random_streams=True,analytic_perfect_J=1,shown_only_J=0,whole_map_code_J=1))
    # Distinguishable synthetic projected vectors verify exact legal rendering.
    projected=torch.arange(len(bank.features)*64,dtype=torch.float32).reshape(-1,64)+1
    f,b=world.frames(projected,test);assert f.shape==(960,2,390) and (b[:,0]==1).all() and (b[:,1]==0).all()
    assert (f[:,0,-6:].sum(-1)==2).all() and (f[:,1,-6:].sum(-1)==1).all()
    for row in range(960):
        visible=int(test['shown'][row]);site=int(test['positions'][row,visible]);photo=int(test['photo_ids'][row,visible])
        assert torch.equal(f[row,1,site*64:(site+1)*64],projected[photo])
    manifest=run.PROJECT/'redesign_v0.22/results/visibility_001/completion_manifest.json';old=run.read(manifest)
    for p,h in old['artifacts'].items():assert run.sha(run.PROJECT/p)==h,p
    run.write(run.ROOT/'prior_integrity_qa.json',dict(passed=True,previous_goal_turn='progress',reason='v22 completed new paired frozen-policy evaluation and changes next causal question',manifest_sha256=run.sha(manifest),artifacts_checked=len(old['artifacts'])))
    result=dict(passed=True,training_run=False,source_hashes=run.source_hashes(),partitions=checks,legal_render_rows=960,train_rows=len(train['map_id']),test_rows=960)
    run.write(run.ROOT/'world_selftest.json',result);print(json.dumps(dict(passed=True,legal_render_rows=960,previous_artifacts=len(old['artifacts']))))
if __name__=='__main__':main()
