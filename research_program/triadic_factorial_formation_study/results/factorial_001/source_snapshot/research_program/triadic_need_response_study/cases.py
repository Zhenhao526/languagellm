"""Static single-need response edges and complete-partition descriptive metrics.

Only native task definitions are imported. No policy, observation encoder, model,
settlement kernel, or saved policy output is read by this module.
"""
from itertools import product
import hashlib
import json

import numpy as np

from research_program.triadic_action_dependency_study import environment as native

AXES=('kind','length','destination')
PAIRS=((0,1),(0,2),(1,2))
PAIR_NAMES=('AB','AC','BC')
STATE_ORDER='need-major, then layout, then owner'
COUNTS=('both_correct','only_first_correct','only_second_correct','one_correct','neither_correct',
        'partner_change','both_executed_pair_change','execution_status_change')
RATES={name:name+'_rate' for name in COUNTS}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def flips(need):
    """Native semantic flips; unspecified acceptance axes stay unspecified."""
    require(type(need) is int and 0<=need<24,'Need must be an integer0..23')
    resource,destination=divmod(need,3)
    if resource<4:
        yield AXES[resource//2],(resource^1)*3+destination
    else:
        yield 'kind',(4+((resource-4)^2))*3+destination
        yield 'length',(4+((resource-4)^1))*3+destination
    if destination!=2:yield 'destination',resource*3+1-destination


def _matrix(value,columns,name):
    array=np.asarray(value)
    require(array.ndim==2 and array.shape[1]==columns and len(array)>0 and array.dtype.kind in 'iu',
            name+' must be a nonempty integer matrix')
    require(len(np.unique(array,axis=0))==len(array),name+' contains duplicate rows')
    return array


def build_cases(spec):
    """Return JSON-only need-level edges; retain the original spec row order."""
    require(spec.get('state_order')==STATE_ORDER,'Unexpected native state index order')
    needs=_matrix(spec['needs'],3,'Needs');layouts=_matrix(spec['layouts'],4,'Layouts');owners=_matrix(spec['private_sites'],3,'Owners')
    require(((needs>=0)&(needs<24)).all(),'Invalid need ids')
    require(np.all(np.sort(layouts,axis=1)==np.arange(4)),'Invalid material permutation')
    require(np.all(np.sort(owners,axis=1)==np.arange(1,4)),'Invalid owner permutation')
    n,l,o=len(needs),len(layouts),len(owners);backgrounds=l*o
    require(type(spec['world_count']) is int and spec['world_count']==n*backgrounds,'Cartesian world count mismatch')
    tuples=[tuple(map(int,row)) for row in needs];lookup={row:i for i,row in enumerate(tuples)}
    plans=[native.full_success_plans(row) for row in tuples]
    require(all(len(p)==1 for p in plans),'Every need world must have one full native execution plan')
    targets=[PAIRS.index(p[0][:2]) for p in plans]
    edges=[]
    for before in tuples:
        for who in range(3):
            for axis,changed in flips(before[who]):
                after=list(before);after[who]=changed;after=tuple(after)
                if after not in lookup or before>=after:continue
                i,j=lookup[before],lookup[after]
                if targets[i]!=targets[j]:edges.append((i,j,who,AXES.index(axis),targets[i],targets[j]))
    edges.sort(key=lambda row:(row[2],row[3],tuples[row[0]],tuples[row[1]]))
    strata=[]
    for who,axis in product(range(3),range(3)):
        selected=[i for i,e in enumerate(edges) if e[2:4]==(who,axis)]
        transitions={f'{a}_{b}':sum(edges[i][4:]==(a,b) for i in selected) for a,b in product(range(3),repeat=2) if a!=b}
        strata.append(dict(changed_person=who,axis=AXES[axis],axis_index=axis,edge_indices=selected,
            need_edges=len(selected),state_edges=len(selected)*backgrounds,empty=not selected,
            target_transition_counts=transitions))
    return dict(schema='single_need_executed_pair_response_v1',partition=spec['partition'],spec_sha256=digest(spec),
        state_order=STATE_ORDER,needs=needs.tolist(),layouts=layouts.tolist(),private_sites=owners.tolist(),
        need_worlds=n,n_backgrounds=backgrounds,world_count=n*backgrounds,
        background_layout_owner_indices=[list(row) for row in product(range(l),range(o))],
        need_target_pairs=targets,edge_need_indices=[[e[0],e[1]] for e in edges],
        changed_person=[e[2] for e in edges],axis_index=[e[3] for e in edges],target_pairs=[[e[4],e[5]] for e in edges],
        need_edges=len(edges),state_edges=len(edges)*backgrounds,strata=strata,
        empty_strata=[dict(changed_person=s['changed_person'],axis=s['axis']) for s in strata if s['empty']],
        weighting='Within each changed-person/axis:all edges equal at each background,then backgrounds equal;all9 strata equal.',
        empty_rule='Keep every structural stratum. If any is empty,overall Q/Q_shuffle/Q_excess and rates are null;never drop or impute a stratum.',
        actor_inputs=False,policy_dependent_filter=False)


def expand_background(cases,background_index):
    """Temporary NumPy index view; no expansion over all backgrounds at once."""
    require(type(background_index) is int and 0<=background_index<cases['n_backgrounds'],'Invalid background index')
    edges=np.asarray(cases['edge_need_indices'],dtype=np.int64).reshape(-1,2)
    li,oi=cases['background_layout_owner_indices'][background_index]
    return dict(state_indices=edges*cases['n_backgrounds']+background_index,
        changed_person=np.asarray(cases['changed_person'],dtype=np.int8),axis_index=np.asarray(cases['axis_index'],dtype=np.int8),
        target_pairs=np.asarray(cases['target_pairs'],dtype=np.int8).reshape(-1,2),background_index=background_index,
        layout_index=li,owner_index=oi,layout=cases['layouts'][li],private_sites=cases['private_sites'][oi])


def validate_partition_arrays(cases,states,state_indices):
    """Reject subsets, duplicates, shuffles and mismatched complete world packing."""
    states=np.asarray(states);indices=np.asarray(state_indices);count=cases['world_count']
    require(states.shape==(count,10) and states.dtype.kind in 'iu','Full integer partition states required')
    require(indices.shape==(count,) and indices.dtype.kind in 'iu','Full integer state indices required')
    nbackgrounds=cases['n_backgrounds'];nowners=len(cases['private_sites']);nlayouts=len(cases['layouts'])
    require(nbackgrounds==nlayouts*nowners and count==len(cases['needs'])*nbackgrounds,'Invalid case Cartesian shape')
    needs=np.asarray(cases['needs']);layouts=np.asarray(cases['layouts']);owners=np.asarray(cases['private_sites'])
    for start in range(0,count,4096):
        stop=min(start+4096,count);ids=np.arange(start,stop,dtype=np.int64)
        require(np.array_equal(indices[start:stop],ids),'Full state index order must be exactly0..world_count-1')
        expected=np.concatenate((needs[ids//nbackgrounds],layouts[(ids//nowners)%nlayouts],owners[ids%nowners]),axis=1)
        require(np.array_equal(states[start:stop],expected),'Packed states contradict frozen need/layout/owner ordering')
    return dict(status='valid',worlds=count,state_order=STATE_ORDER,full_partition=True)


def metrics(cases,executed_pair_indices):
    """All endpoint edges under one settlement, including no-execution outcomes.

The caller validates the saved state/index arrays first. Values must be in the
complete native index order. Analytic shuffle counts use *all* need worlds in
each background, including worlds outside the eligible edge endpoints.
    """
    raw=np.asarray(executed_pair_indices)
    require(raw.shape==(cases['world_count'],) and raw.dtype.kind in 'iu' and ((raw>=-1)&(raw<=2)).all(),
            'Full integer executed-pair vector required;codes-1/0/1/2')
    values=raw.astype(np.int8,copy=False);n=cases['need_worlds'];bcount=cases['n_backgrounds']
    require(n*bcount==len(values) and len(cases['strata'])==9,'Invalid compact cases shape')
    edge=np.asarray(cases['edge_need_indices'],dtype=np.int64).reshape(-1,2)
    targets=np.asarray(cases['target_pairs'],dtype=np.int8).reshape(-1,2)
    require(edge.shape==targets.shape and len(edge)==cases['need_edges'],'Invalid edge/target shape')
    require(((edge>=0)&(edge<n)).all() and ((targets>=0)&(targets<3)).all() and np.all(targets[:,0]!=targets[:,1]),'Invalid eligible edges')
    domains=values.reshape(n,bcount)
    background_rows=[];background_pair_counts=[]
    for b,(li,oi) in enumerate(cases['background_layout_owner_indices']):
        counts=np.bincount(domains[:,b]+1,minlength=4)
        background_pair_counts.append(counts[1:])
        background_rows.append(dict(background_index=b,layout_index=li,owner_index=oi,
            layout=cases['layouts'][li],private_sites=cases['private_sites'][oi],worlds=n,
            actual_pair_counts={name:int(counts[i]) for i,name in enumerate(('none',*PAIR_NAMES))},
            shuffle_ordered_world_pairs=n*(n-1)))
    background_pair_counts=np.asarray(background_pair_counts,dtype=np.int64)
    strata=[];total_counts={k:0 for k in COUNTS}
    for structure in cases['strata']:
        ids=np.asarray(structure['edge_indices'],dtype=np.int64);ecount=len(ids)
        row={k:structure[k] for k in ('changed_person','axis','axis_index','need_edges','state_edges','empty','target_transition_counts')}
        row.update(raw_counts={k:0 for k in COUNTS},backgrounds=[])
        for b in range(bcount):
            if ecount:
                actual=domains[edge[ids],b];truth=targets[ids];correct=actual==truth
                before,after=correct[:,0],correct[:,1];active=actual>=0
                masks=dict(both_correct=before&after,only_first_correct=before&~after,only_second_correct=~before&after,
                    one_correct=before^after,neither_correct=~before&~after,partner_change=actual[:,0]!=actual[:,1],
                    both_executed_pair_change=active.all(axis=1)&(actual[:,0]!=actual[:,1]),execution_status_change=active[:,0]^active[:,1])
                counts={k:int(mask.sum()) for k,mask in masks.items()}
                pair_counts=background_pair_counts[b]
                numerator=int(np.sum(pair_counts[truth[:,0]]*pair_counts[truth[:,1]],dtype=np.int64))
                denominator=n*(n-1)*ecount
                rates={RATES[k]:counts[k]/ecount for k in COUNTS};q=counts['both_correct']/ecount;shuffle=numerator/denominator
                rec=dict(background_index=b,edges=ecount,raw_counts=counts,Q=q,Q_shuffle=shuffle,Q_excess=q-shuffle,
                    shuffle_numerator_sum=numerator,shuffle_denominator=denominator,**rates)
                for key in COUNTS:row['raw_counts'][key]+=counts[key]
            else:
                rec=dict(background_index=b,edges=0,raw_counts={k:0 for k in COUNTS},Q=None,Q_shuffle=None,Q_excess=None,
                    shuffle_numerator_sum=0,shuffle_denominator=0,**{RATES[k]:None for k in COUNTS})
            row['backgrounds'].append(rec)
        scalar_keys=('Q','Q_shuffle','Q_excess',*RATES.values())
        row.update({k:float(np.mean([bg[k] for bg in row['backgrounds']])) if ecount else None for k in scalar_keys})
        for key in COUNTS:total_counts[key]+=row['raw_counts'][key]
        strata.append(row)
    available=all(not row['empty'] for row in strata)
    result={k:float(np.mean([row[k] for row in strata])) if available else None for k in scalar_keys}
    result.update(schema='single_need_response_metrics_v1',partition=cases['partition'],worlds=cases['world_count'],
        need_worlds_per_background=n,background_count=bcount,need_edges=len(edge),state_edges=len(edge)*bcount,
        complete_nine_strata=available,empty_strata=cases['empty_strata'],strata=strata,backgrounds=background_rows,
        raw_counts=total_counts,raw_pooled_Q=(total_counts['both_correct']/(len(edge)*bcount)) if len(edge) else None,
        weighting=cases['weighting'],empty_rule=cases['empty_rule'],
        scope={'Q':'Both endpoints choose their respective different correct executed pairs;none is wrong.',
               'shuffle':'Analytic expectation under a uniform output permutation among allN needs within each background:n_t0*n_t1/[N(N-1)]. No random shuffle is drawn.',
               'raw_counts':'Pooled edge-background counts are descriptive;Q uses equal9 strata,not pooled edges.',
               'partner_change_rate':'Any unequal codes,including transitions to/fromnone.',
               'both_executed_pair_change_rate':'Different pairs with both endpoints executing;excludes execution-on/off alone.',
               'replication':'Overlapping edges and backgrounds are repeated measurements,not independent societies.',
               'input':'Caller must validate full states and indices before passing a complete ordered executed-pair vector.'})
    return result
