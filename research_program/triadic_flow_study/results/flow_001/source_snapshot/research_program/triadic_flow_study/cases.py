"""Static incoming/outgoing W1 factorial metadata and generic response contrasts.

Correct-pair labels are researcher-side metadata, never actor observations or
message inputs. No policy file is read. The runner selects the scientific
response and unique primary; the generic scorer retains trailing response axes.
"""
from copy import deepcopy
import numpy as np

from research_program.triadic_directed_message_study import cases as mirrored

AXES=mirrored.AXES
PAIRS=mirrored.PAIRS
STATE_ORDER=mirrored.STATE_ORDER


def require(ok,message):
    if not ok:raise ValueError(message)


def build_cases(spec):
    """Pure JSON support; no cross-layout donors and no outcome-based choice."""
    old=mirrored.build_cases(spec)
    keys=('partition','spec_sha256','state_order','needs','layouts','private_sites','world_count',
          'n_backgrounds','background_layout_owner_indices','group_count','group_background_count',
          'original_need_edges','group_need_indices','sender','axis_index','recipients','compatible_d',
          'truth_pair_indices','need_target_pairs','empty_strata')
    result={key:deepcopy(old[key]) for key in keys}
    strata=[]
    for previous in old['strata']:
        row={key:deepcopy(previous[key]) for key in ('sender','axis','axis_index','group_indices','group_count','group_background_count','empty')}
        row.update(flow_cells=16*row['group_background_count'],arm_cells=4*row['group_background_count'],
                   sham_cells=4*row['group_background_count'])
        strata.append(row)
    gb=old['group_background_count']
    result.update(schema='mirrored_same_background_W1_flow_cases_v1',strata=strata,flow_cells=16*gb,
        arm_cells=4*gb,sham_cells=4*gb,nonsham_cells=12*gb,
        flow_order='group,background,context,self_endpoint,incoming,outgoing',
        incoming_rule='Candidate donor is opposite context, same focus-own endpoint and same public background; replace only other actors W1 delivered to focus.',
        outgoing_rule='Candidate donor is same context, opposite focus-own endpoint and same public background; replace only focus W1 delivered outward.',
        effective_donor_rule='A zero switch uses the receiver world; a one switch uses the corresponding candidate donor. Both zero is natural sham.',
        downstream='Keep all receiver observations, self deliveries, unselected W1 routes and visibility natural; recompute W2 synchronously and actions.',
        scoring_status='Generic paired response contrasts available; response definition and unique primary selection belong to the runner.',
        actor_inputs=False,policy_dependent_selection=False)
    return result


def expand(cases):
    """Complete16-cell arrays; candidate and effective donors are both explicit."""
    g=cases['group_count'];b=cases['n_backgrounds']
    require(cases['schema']=='mirrored_same_background_W1_flow_cases_v1' and cases['flow_cells']==16*g*b,
            'Unexpected flow schema or cell budget')
    group=np.repeat(np.arange(g,dtype=np.int64),b*16)
    background=np.tile(np.repeat(np.arange(b,dtype=np.int64),16),g)
    context=np.tile(np.repeat(np.arange(2,dtype=np.int8),8),g*b)
    endpoint=np.tile(np.repeat(np.arange(2,dtype=np.int8),4),g*b*2)
    incoming=np.tile(np.repeat(np.arange(2,dtype=np.int8),2),g*b*4)
    outgoing=np.tile(np.arange(2,dtype=np.int8),g*b*8)
    worlds=np.asarray(cases['group_need_indices'],dtype=np.int64).reshape(g,2,2)
    targets=np.asarray(cases['truth_pair_indices'],dtype=np.int8).reshape(g,2,2)
    receiver=worlds[group,context,endpoint]*b+background
    donor_in=worlds[group,1-context,endpoint]*b+background
    donor_out=worlds[group,context,1-endpoint]*b+background
    focus=np.asarray(cases['sender'],dtype=np.int8)[group]
    return dict(group_index=group,background_index=background,context_index=context,self_endpoint=endpoint,
        incoming=incoming,outgoing=outgoing,focus_actor=focus,sender=focus.copy(),
        recipient_agents=np.asarray(cases['recipients'],dtype=np.int8).reshape(g,2)[group],
        axis_index=np.asarray(cases['axis_index'],dtype=np.int8)[group],receiver_state_indices=receiver,
        incoming_donor_state_indices=donor_in,outgoing_donor_state_indices=donor_out,
        effective_incoming_donor_state_indices=np.where(incoming,donor_in,receiver),
        effective_outgoing_donor_state_indices=np.where(outgoing,donor_out,receiver),
        receiver_target_pair=targets[group,context,endpoint],
        incoming_donor_target_pair=targets[group,1-context,endpoint],outgoing_donor_target_pair=targets[group,context,1-endpoint],
        compatible_d=np.asarray(cases['compatible_d'],dtype=np.int8).reshape(g,2,2)[group,context],
        sham=(incoming==0)&(outgoing==0))


def validate_partition_arrays(cases,states,state_indices):
    """Require all original states and exactly the original complete index order."""
    return mirrored.validate_partition_arrays(cases,states,state_indices)


def metrics(cases,cell_values):
    """Equal-nine-stratum response summaries, preserving every trailing axis.

Input order is exactly expand(): [group,background,context,self_endpoint,I,O].
Differences are taken within the same group/background/context/own endpoint
before averaging context and own endpoint. Each group/background result then
receives the declared within-stratum weights. No probability gate, execution
filter, argmax, sign filter, fitted weighting or response-dependent selection.

The name S00 denotes the mean of the supplied response at I=0,O=0; it does not
assert that the caller supplied a success probability. For the unique planned
primary, the runner must supply the original-world exact full-success
probability and take I_given_O0. Vector responses use the identical operation.
    """
    raw=np.asarray(cell_values)
    g=cases['group_count'];b=cases['n_backgrounds']
    require(raw.ndim>=1 and raw.shape[0]==cases['flow_cells']==g*b*16 and raw.dtype.kind in 'fiu' and
            np.isfinite(raw).all(),'Expected finite canonical flow values [N,*response_shape]')
    require(len(cases['strata'])==9 and [(s['sender'],s['axis_index']) for s in cases['strata']]==
            [(s,a) for s in range(3) for a in range(3)],'All nine ordered strata required')
    trailing=raw.shape[1:];v=raw.astype(np.float64,copy=False).reshape((g,b,2,2,2,2)+trailing)
    arm={f'S{i}{o}':v[:,:,:,:,i,o] for i in range(2) for o in range(2)}
    paired=dict(I_given_O0=arm['S10']-arm['S00'],I_given_O1=arm['S11']-arm['S01'],
                O_given_I0=arm['S01']-arm['S00'],O_given_I1=arm['S11']-arm['S10'])
    paired['interaction']=paired['I_given_O1']-paired['I_given_O0']
    group_background={key:np.mean(value,axis=(2,3)) for key,value in {**arm,**paired}.items()}
    def numeric(value):
        value=np.asarray(value)
        return float(value) if value.ndim==0 else value.tolist()
    rows=[]
    for structure in cases['strata']:
        ids=np.asarray(structure['group_indices'],dtype=np.int64)
        require(len(ids)==structure['group_count'] and np.all((ids>=0)&(ids<g)),'Invalid stratum group indices')
        row={key:structure[key] for key in ('sender','axis','axis_index','group_count','group_background_count','empty')}
        row.update({key:numeric(np.mean(np.mean(value[ids],axis=1),axis=0)) if len(ids) else None
                    for key,value in group_background.items()})
        rows.append(row)
    require(sum(s['group_count'] for s in rows)==g,'Incomplete nine-stratum group count')
    complete=all(not s['empty'] for s in rows)
    result={key:numeric(np.mean([s[key] for s in rows],axis=0)) if complete else None for key in group_background}
    result.update(schema='same_background_flow_response_metrics_v1',partition=cases['partition'],response_shape=list(trailing),
        group_count=g,n_backgrounds=b,group_background_count=g*b,flow_cells=g*b*16,
        arm_cells=g*b*4,complete_nine_strata=complete,empty_strata=cases['empty_strata'],strata=rows,
        per_group_background={key:value.tolist() for key,value in group_background.items()},
        detail_schema='Every per_group_background value has axes [group,background,*response_shape],after mean over context and own endpoint.',
        weighting='Within each context/own endpoint take paired I/O contrasts,then mean over those four states;equal backgrounds and groups within stratum,then all nine strata equal.',
        empty_rule='Keep every structural stratum;overall summaries are null if any of nine is empty.',
        scope='Generic deterministic response arithmetic only. Full-success probability, action or content responses must be independently supplied and labeled by the caller; no learned output is read here.')
    return result
