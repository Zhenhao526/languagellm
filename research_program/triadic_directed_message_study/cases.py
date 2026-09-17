"""Static mirrored need contexts and first-window recipient selectivity.

These labels/index vectors are researcher metadata, never actor input. This
module neither reads policy outputs nor creates/loads/runs any neural model.
"""
from itertools import product
import numpy as np

from research_program.triadic_need_response_study import cases as original

AXES=original.AXES
PAIRS=original.PAIRS
STATE_ORDER=original.STATE_ORDER


def require(ok,message):
    if not ok:raise ValueError(message)


def build_cases(spec):
    """JSON-only mirror groups, preserving frozen world and background order."""
    previous=original.build_cases(spec)
    needs=[tuple(n) for n in previous['needs']];lookup={n:i for i,n in enumerate(needs)}
    truth=previous['need_target_pairs'];groups={}
    for endpoints,sender,axis in zip(previous['edge_need_indices'],previous['changed_person'],previous['axis_index']):
        recipients=tuple(a for a in range(3) if a!=sender)
        rows=[needs[n] for n in endpoints]
        require(rows[0][sender]<rows[1][sender],'Sender endpoints must use ascending native need ids')
        mirrored=[]
        for row in rows:
            swapped=list(row);swapped[recipients[0]],swapped[recipients[1]]=swapped[recipients[1]],swapped[recipients[0]]
            require(tuple(swapped) in lookup,'Mirror context outside frozen partition')
            mirrored.append(tuple(swapped))
        require(len(set((*rows,*mirrored)))==4,'Mirror must contain four different worlds')
        context0,context1=(rows,mirrored) if tuple(rows[0][r] for r in recipients)<tuple(mirrored[0][r] for r in recipients) else (mirrored,rows)
        indices=[[lookup[n] for n in context] for context in (context0,context1)]
        targets=[[truth[i] for i in context] for context in indices]
        require(targets[0][0]==targets[1][1] and targets[0][1]==targets[1][0] and targets[0][0]!=targets[0][1],
                'Mirror must reverse the unique correct executed pair')
        compatible=[]
        for context in range(2):
            require(all(sender in PAIRS[pair] for pair in targets[context]),'Changed sender must belong to both target pairs')
            row=[]
            for recipient in recipients:
                valid=[d for d in range(2) if recipient in PAIRS[targets[context][d]]]
                require(len(valid)==1,'Exactly one compatible packet need per context/recipient required');row.append(valid[0])
            compatible.append(row)
        key=(sender,axis,context0[0][sender],context0[1][sender],context0[0][recipients[0]],context0[0][recipients[1]])
        value=dict(need_indices=indices,sender=sender,axis_index=axis,recipients=list(recipients),compatible_d=compatible,truth_pair_indices=targets)
        if key in groups:require(groups[key]==value,'Mirror duplicate disagrees')
        groups[key]=value
    ordered=[groups[k] for k in sorted(groups)]
    require(len(ordered)*2==previous['need_edges'],'Every original edge must occur in exactly one two-edge mirror group')
    l=len(previous['layouts']);o=len(previous['private_sites']);b=l*o
    require(l>=2 and l%2==0,'Opposite-index mapping requires an even nontrivial layout support')
    donor=[((li+l//2)%l)*o+oi for li,oi in product(range(l),range(o))]
    require(sorted(donor)==list(range(b)) and all(d!=i and donor[d]==i and d%o==i%o for i,d in enumerate(donor)),
            'Donor mapping must be a same-owner cross-layout involution')
    strata=[]
    for sender,axis in product(range(3),range(3)):
        ids=[i for i,row in enumerate(ordered) if row['sender']==sender and row['axis_index']==axis]
        strata.append(dict(sender=sender,axis=AXES[axis],axis_index=axis,group_indices=ids,group_count=len(ids),
            group_background_count=len(ids)*b,cross_cells=len(ids)*b*8,sham_cells=len(ids)*b*4,empty=not ids))
    g=len(ordered)
    return dict(schema='mirrored_need_W1_packet_cases_v1',partition=previous['partition'],spec_sha256=previous['spec_sha256'],
        state_order=STATE_ORDER,needs=previous['needs'],layouts=previous['layouts'],private_sites=previous['private_sites'],
        world_count=previous['world_count'],n_backgrounds=b,background_layout_owner_indices=previous['background_layout_owner_indices'],
        donor_background_indices=donor,donor_layout_indices=[(li+l//2)%l for li in range(l)],
        group_count=g,group_background_count=g*b,cross_cells=g*b*8,sham_cells=g*b*4,
        original_need_edges=previous['need_edges'],group_need_indices=[v['need_indices'] for v in ordered],
        sender=[v['sender'] for v in ordered],axis_index=[v['axis_index'] for v in ordered],
        recipients=[v['recipients'] for v in ordered],compatible_d=[v['compatible_d'] for v in ordered],
        truth_pair_indices=[v['truth_pair_indices'] for v in ordered],need_target_pairs=truth,
        strata=strata,empty_strata=[dict(sender=s['sender'],axis=s['axis']) for s in strata if s['empty']],
        cross_order='group,background,context,self_endpoint,packet_endpoint',sham_order='group,background,context,self_endpoint',
        donor_rule='cross:canonical context0,packet endpointd,opposite-index layout with same owner;sham:recipient world itself',
        weighting='Within group/background:average sender-own endpoint for each context/recipient contrast,then min4 for L or mean4 for D;equal backgrounds,groups within stratum,all9 strata.',
        empty_rule='Keep empty structural strata;overall metrics null if any of9 is empty.',actor_inputs=False,policy_dependent_selection=False)


def expand(cases,mode='cross'):
    """Full canonical metadata vectors; no deduplication of forward scenarios."""
    require(mode in ('cross','sham'),'Unknown expansion mode')
    g=cases['group_count'];b=cases['n_backgrounds'];per_background=8 if mode=='cross' else 4
    group=np.repeat(np.arange(g,dtype=np.int64),b*per_background)
    background=np.tile(np.repeat(np.arange(b,dtype=np.int64),per_background),g)
    context=np.tile(np.repeat(np.arange(2,dtype=np.int8),per_background//2),g*b)
    endpoint=np.tile(np.repeat(np.arange(2,dtype=np.int8),2 if mode=='cross' else 1),g*b*2)
    packet=np.tile(np.arange(2,dtype=np.int8),g*b*4) if mode=='cross' else endpoint.copy()
    worlds=np.asarray(cases['group_need_indices'],dtype=np.int64).reshape(g,2,2)
    receiver=worlds[group,context,endpoint]*b+background
    if mode=='cross':
        donor_background=np.asarray(cases['donor_background_indices'],dtype=np.int64)[background]
        donor=worlds[group,0,packet]*b+donor_background
    else:donor=receiver.copy();donor_background=background.copy()
    sender=np.asarray(cases['sender'],dtype=np.int8)[group]
    recipients=np.asarray(cases['recipients'],dtype=np.int8).reshape(g,2)[group]
    return dict(group_index=group,background_index=background,context_index=context,self_endpoint=endpoint,
        packet_endpoint=packet,sender=sender,recipient_agents=recipients,receiver_state_indices=receiver,donor_state_indices=donor,
        donor_background_index=donor_background,axis_index=np.asarray(cases['axis_index'],dtype=np.int8)[group],
        compatible_d=np.asarray(cases['compatible_d'],dtype=np.int8).reshape(g,2,2)[group,context],
        receiver_target_pair=np.asarray(cases['truth_pair_indices'],dtype=np.int8).reshape(g,2,2)[group,context,endpoint])


def validate_partition_arrays(cases,states,state_indices):
    """Identical frozen Cartesian packing contract as the original case study."""
    return original.validate_partition_arrays(cases,states,state_indices)


def metrics(cases,recipient_probabilities):
    """Score the complete8-cell canonical cross expansion, without reordering.

Columns follow each group's two non-sender recipients in identity order. Each
entry is the sum of that recipient's eight action probabilities proposing the
sender, with no location/destination or waiting correctness gate.
    """
    p=np.asarray(recipient_probabilities)
    require(p.shape==(cases['cross_cells'],2) and p.dtype.kind in 'fiu' and np.isfinite(p).all() and
            ((p>=-2e-12)&(p<=1+2e-12)).all(),'Expected finite full-cross recipient probabilities[N,2]')
    g=cases['group_count'];b=cases['n_backgrounds']
    require(cases['cross_cells']==g*b*8,'Invalid full-cross case budget')
    require([(s['sender'],s['axis_index']) for s in cases['strata']]==list(product(range(3),range(3))), 'All9 ordered strata required')
    p=p.astype(np.float64,copy=False).reshape(g,b,2,2,2,2)
    packet0=p[:,:,:,:,0,:];packet1=p[:,:,:,:,1,:]
    compatible_zero=np.asarray(cases['compatible_d'],dtype=np.int8).reshape(g,2,2)[:,None,:,None,:]==0
    # Average the within-own-endpoint packet contrast before taking the minimum.
    delta=np.mean(np.where(compatible_zero,packet0-packet1,packet1-packet0),axis=3)
    compatible=np.mean(np.where(compatible_zero,packet0,packet1),axis=3)
    incompatible=np.mean(np.where(compatible_zero,packet1,packet0),axis=3)
    lower=delta.min(axis=(2,3)) if g else np.empty((0,b));average=delta.mean(axis=(2,3))
    comp_mean=compatible.mean(axis=(2,3));incomp_mean=incompatible.mean(axis=(2,3));positive=(lower>0).astype(float)
    arrays=dict(L=lower,D=average,mean_compatible_probability=comp_mean,mean_incompatible_probability=incomp_mean,
                strict_positive_L_fraction=positive)
    rows=[]
    for structure in cases['strata']:
        ids=np.asarray(structure['group_indices'],dtype=np.int64);n=len(ids)
        row={k:structure[k] for k in ('sender','axis','axis_index','group_count','group_background_count','empty')}
        row.update({key:float(np.mean(np.mean(values[ids],axis=1))) if n else None for key,values in arrays.items()})
        row['positive_L_group_backgrounds']=int((lower[ids]>0).sum())
        row['zero_L_group_backgrounds']=int((lower[ids]==0).sum())
        row['negative_L_group_backgrounds']=int((lower[ids]<0).sum())
        row['mean_four_deltas']=(np.mean(np.mean(delta[ids],axis=1),axis=0).tolist() if n else None)
        rows.append(row)
    complete=all(not s['empty'] for s in rows)
    result={key:float(np.mean([r[key] for r in rows])) if complete else None for key in arrays}
    result.update(schema='mirrored_need_W1_selectivity_v1',partition=cases['partition'],group_count=g,n_backgrounds=b,
        group_background_count=g*b,cross_cells=g*b*8,complete_nine_strata=complete,empty_strata=cases['empty_strata'],strata=rows,
        raw_group_background_counts=dict(positive_L=int((lower>0).sum()),zero_L=int((lower==0).sum()),negative_L=int((lower<0).sum())),
        per_group_background=dict(delta=delta.tolist(),compatible_probability=compatible.tolist(),incompatible_probability=incompatible.tolist(),
                                  L=lower.tolist(),D=average.tolist()),
        detail_schema={'delta':'[group,background,context,recipient_in_identity_order],after equal mean over sender-own endpoints',
                       'compatible_probability':'same axes asdelta;mean over sender-own endpoints',
                       'incompatible_probability':'same axes asdelta;mean over sender-own endpoints',
                       'L':'[group,background],minimum of four context/recipient contrasts',
                       'D':'[group,background],mean of four context/recipient contrasts'},
        weighting=cases['weighting'],
        scope={'target':'Recipient proposal probability for sender,summing over8locations/destinations;not executed-pair or all-three-role accuracy.',
               'L':'Contextual recipient selectivity,stronger than physical-task necessity. PositiveoverallL doesnot mean allgroups have fourpositive contrasts.',
               'D':'Mean contrast retained asauxiliary;can be positive under one-sided general suppression evenwhenL isnegative.',
               'phase':'First-window intervention path only;no claim about all message semantics or second-window pathways.',
               'details':'Full group/background numerical arrays may be saved separately with a bound hash;no outcome selection.',
               'replication':'Groups/backgrounds/recipients are repeated measurements;independent units remain policy initialization seeds.'})
    return result
