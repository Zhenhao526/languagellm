"""Independent NumPy checks for the paired-world v0.8 task; no policy imports.

This module deliberately does not load camp.py or the runner. Its public
functions accept arrays or explicit design metadata supplied by the report
script, so dataset counts and message capacities are never inferred from a
successful run alone.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    require(np.array_equal(actual, expected), f"{context}: arrays differ")


def near(actual, expected, context, tolerance=1e-7):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0),
            f"{context}: {actual} != {expected}")


def describe(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and values.size > 0, "Descriptive sample must be nonempty")
    require(np.isfinite(values).all(), "Descriptive sample contains nonfinite values")
    return {"n": int(values.size), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            "values": values.tolist()}


def selected_mean(reward, mask):
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    return {"n": count, "reward_sum": int(np.asarray(reward)[mask].sum()),
            "mean_reward": float(np.asarray(reward)[mask].mean()) if count else None}


def split_audit(n_sites, split_maps):
    """Verify explicit matching splits and marginal resource/location coverage."""
    all_maps = set(permutations(range(n_sites), 2))
    heldout_sets, records = [], []
    for split_id, maps in split_maps.items():
        heldout = {tuple(map(int, m)) for m in maps}
        require(heldout.issubset(all_maps), "Heldout maps must be legal")
        train = all_maps - heldout
        require(len(heldout) == n_sites == len(maps), "Incorrect holdout size or duplicate maps")
        require(all((water, food) in heldout for food, water in heldout), "Holdout must contain both directions of each matching edge")
        for resource in (0, 1):
            equal(np.bincount([m[resource] for m in heldout], minlength=n_sites),
                  np.ones(n_sites, dtype=int), "Heldout resource/location balance")
            equal(np.bincount([m[resource] for m in train], minlength=n_sites),
                  np.full(n_sites, n_sites - 2), "Training resource/location balance")
        heldout_sets.append(heldout)
        records.append({"split_id": str(split_id), "train_maps": [list(m) for m in sorted(train)],
                        "heldout_maps": [list(m) for m in sorted(heldout)],
                        "train_resource_location_frequency": n_sites - 2,
                        "heldout_resource_location_frequency": 1})
    require(sum(len(s) for s in heldout_sets) == len(set().union(*heldout_sets)),
            "Different matching splits must have disjoint heldout maps")
    return {"status": "passed", "n_sites": n_sites, "all_maps": [list(m) for m in sorted(all_maps)],
            "splits": records, "distinct_heldout_maps": len(set().union(*heldout_sets))}


def outcomes_metrics(successes, goals, weight):
    """Count outcomes in resource order, derive all utilities from the four cells."""
    n=len(successes)
    resource=np.zeros((n,2),dtype=np.int64)
    if n:resource[np.arange(n)[:,None],goals]=successes
    counts={f"{food}{water}":int(((resource[:,0]==food)&(resource[:,1]==water)).sum())
            for food in (0,1) for water in (0,1)}
    food=counts['10']+counts['11'];water=counts['01']+counts['11'];both=counts['11']
    reward_sum=both+(1-weight)*(counts['10']+counts['01'])/2
    second_moment_sum=both+((1-weight)/2)**2*(counts['10']+counts['01'])
    positive=both+(counts['10']+counts['01'] if weight<1 else 0)
    return dict(n=n,decisions=2*n,reward_sum=reward_sum,mean_reward=reward_sum/n if n else None,
        reward_variance=second_moment_sum/n-(reward_sum/n)**2 if n else None,
        positive_rewards=positive,positive_rate=positive/n if n else None,
        single_correct=food+water,single_accuracy=(food+water)/(2*n) if n else None,
        both_correct=both,both_accuracy=both/n if n else None,
        food_correct=food,food_accuracy=food/n if n else None,
        water_correct=water,water_accuracy=water/n if n else None,outcome_counts=counts)


def verify_recorded_metrics(record, computed, context):
    for key in ('n','decisions','single_correct','both_correct','food_correct','water_correct','outcome_counts','positive_rewards'):
        require(record[key]==computed[key],f'{context}: {key} differs')
    for key in ('reward_sum','mean_reward','single_accuracy','both_accuracy','reward_variance'):
        if computed[key] is None:require(record[key] is None,f'{context}: empty {key}')
        else:near(record[key],computed[key],f'{context}: {key}')


def audit_pair_arrays(a, *, n_sites, vocab, length, episodes, mode, blocked, weight, photo_metadata):
    required={'scout','episode','positions','photo_ids','goals','menu','inventory','history',
              'sent','delivered','action','place','successes','reward'}
    require(required.issubset(a),f'Missing paired-world columns: {required-a.keys()}')
    n=int(episodes)
    require(n>0 and n%2==0,'Two equally sized directions required')
    require(all(len(v)==n for v in a.values()),'Inconsistent paired-world lengths')
    shapes={'scout':(n,),'episode':(n,),'positions':(n,2),'photo_ids':(n,2),'goals':(n,2),
            'menu':(n,2,n_sites),'inventory':(n,2),'history':(n,2*(n_sites+3)),
            'sent':(n,length),'delivered':(n,length),'action':(n,2),'place':(n,2),
            'successes':(n,2),'reward':(n,)}
    for key,shape in shapes.items():require(a[key].shape==shape,f'Incorrect {key} shape')
    for key in ('scout','episode','positions','photo_ids','goals','menu','sent','delivered','action','place'):
        require(np.issubdtype(a[key].dtype,np.integer),f'{key} must be integer')
    equal(np.sort(a['menu'],axis=2),np.tile(np.arange(n_sites),(n,2,1)),'Two private menu permutations')
    equal(np.sort(a['goals'],axis=1),np.tile([0,1],(n,1)),'Exactly one query per resource')
    require(((a['positions']>=0)&(a['positions']<n_sites)).all(),'Map range')
    require((a['positions'][:,0]!=a['positions'][:,1]).all(),'Distinct resource positions')
    require(np.isin(a['scout'],[0,1]).all(),'Scout range')
    require(((a['action']>=0)&(a['action']<n_sites)).all(),'Action range')
    equal(a['place'],a['menu'][np.arange(n)[:,None],np.arange(2)[None,:],a['action']],'Query-specific menu routing')
    for d in (0,1):
        ix=a['scout']==d;require(ix.sum()==n//2,'Equal direction count')
        equal(np.sort(a['episode'][ix]),np.arange(n//2),'Direction world IDs')
    for key in ('sent','delivered'):
        require(((a[key]>=0)&(a[key]<vocab)).all(),f'{key}: token range')
    if blocked or mode=='blank':equal(a['delivered'],np.zeros_like(a['delivered']),'Blocked/blank one-message delivery')
    elif mode=='shuffle':
        for d in (0,1):
            for first in (0,1):
                ix=(a['scout']==d)&(a['goals'][:,0]==first)
                x,xn=np.unique(a['sent'][ix],axis=0,return_counts=True)
                y,yn=np.unique(a['delivered'][ix],axis=0,return_counts=True)
                equal(x,y,'Whole-message shuffle values');equal(xn,yn,'Whole-message shuffle counts')
    else:equal(a['sent'],a['delivered'],'Unmodified one-message delivery')
    equal(a['inventory'],np.zeros((n,2)),'Independent empty-inventory queries')
    equal(a['history'],np.zeros((n,2*(n_sites+3))),'No between-query history')
    target=a['positions'][np.arange(n)[:,None],a['goals']]
    successes=(a['place']==target).astype(np.int64)
    equal(successes,a['successes'],'Recomputed two query successes')
    both=np.logical_and(successes[:,0],successes[:,1])
    partial=np.logical_xor(successes[:,0],successes[:,1])
    reward=both.astype(float)+partial.astype(float)*(1-weight)/2
    equal(reward,a['reward'],'Recomputed complementarity utility')
    for resource,category in enumerate(('food','water')):
        ids=a['photo_ids'][:,resource]
        require(((ids>=0)&(ids<len(photo_metadata))).all(),'Photo ID range')
        require(all(photo_metadata[int(i)]['category']==category and photo_metadata[int(i)]['split']=='test'
                    for i in np.unique(ids)),'Photo category and evaluation split')
    return successes,{'status':'passed','worlds_checked':n,'decisions_checked':2*n,
        'checks':['paired_shapes','map','two_menus','both_goal_orders','direction_ids','single_message_delivery',
                  'two_successes','joint_and_partial_utility','zero_context','photo_source']}


def paired_metrics(a, successes, *, n_sites, heldout_maps, weight):
    n=len(successes);maps=list(permutations(range(n_sites),2))
    require(n%(4*len(maps))==0,'Map/first-goal/direction evaluation balance')
    percell=n//(4*len(maps));rows=[];cells=[]
    def count(ix):return outcomes_metrics(successes[ix],a['goals'][ix],weight)
    for food,water in maps:
        mask=(a['positions']==(food,water)).all(1)
        rows.append(dict(food_site=food,water_site=water,**count(mask)))
        for d in (0,1):
            for first in (0,1):
                ix=mask&(a['scout']==d)&(a['goals'][:,0]==first)
                require(ix.sum()==percell,'Unequal map/first-goal/direction cell')
                cells.append(dict(food_site=food,water_site=water,scout=d,first_goal=first,**count(ix)))
    held={tuple(x) for x in heldout_maps};require(held.issubset(set(maps)),'Illegal heldout map')
    unseen=np.array([tuple(x) in held for x in a['positions']],dtype=bool)
    return dict(**outcomes_metrics(successes,a['goals'],weight),episodes=n,horizon=1,
                cases_per_map_firstgoal_direction=percell,
                direction_metrics=[count(a['scout']==d) for d in (0,1)],
                goal_order_metrics=[count(a['goals'][:,0]==g) for g in (0,1)],
                map_subsets={'train_maps':count(~unseen),'heldout_maps':count(unseen)},
                map_rows=rows,balanced_cells=cells)
