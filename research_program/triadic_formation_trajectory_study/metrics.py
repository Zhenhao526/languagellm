"""Fixed-checkpoint trajectory and compatibility summaries; no model work."""
import numpy as np
STEPS=(0,100,500,1500,3000,6000)
AXES=('kind','length','destination')

def require(ok,message):
    if not ok:raise ValueError(message)

def normalized_area(values,steps=STEPS):
    v=np.asarray(values,dtype=float);t=np.asarray(steps,dtype=float)
    require(v.shape==t.shape==(6,) and np.array_equal(t,STEPS) and np.isfinite(v).all(),'Use the fixed six observations')
    return float(np.sum((v[:-1]+v[1:])*(t[1:]-t[:-1])/2)/(t[-1]-t[0]))

def discovery_description(profile):
    chosen=np.asarray(profile['selected_positions']);rates=np.asarray(profile['response_rates']);scores=np.asarray(profile['specificity_scores'])
    require(chosen.shape==(3,3) and rates.shape==scores.shape==(3,3,8),'Discovery shape')
    response=rates[np.arange(3)[:,None],np.arange(3)[None,:],chosen]
    selected_scores=scores[np.arange(3)[:,None],np.arange(3)[None,:],chosen]
    return dict(selected_response=response.tolist(),selected_scores=selected_scores.tolist(),
        mean_selected_response=float(response.mean()),mean_selected_score=float(selected_scores.mean()),
        positive_selected_scores=int((selected_scores>0).sum()),zero_selected_scores=int((selected_scores==0).sum()),
        negative_selected_scores=int((selected_scores<0).sum()),
        boundary='Message-change association, not listener interpretation or onset of a language.')

def temporal_change(earlier,later,earlier_selection,later_selection):
    a,b=np.asarray(earlier),np.asarray(later)
    require(a.shape==b.shape==(419904,2,3,4),'Compare complete identical train endpoint order')
    require(a.dtype.kind in 'iu' and b.dtype.kind in 'iu' and ((a>=0)&(a<8)).all() and ((b>=0)&(b<8)).all(),'Legal message domain')
    diff=a!=b
    selected=np.asarray(earlier_selection['selected_positions'])!=np.asarray(later_selection['selected_positions'])
    return dict(train_token_change_rate=float(diff.mean()),
        by_window_sender_position=diff.mean(0).tolist(),
        sender_whole_packet_change_rate=np.any(diff,axis=(1,3)).mean(0).tolist(),
        selected_position_changed=selected.tolist(),selected_position_change_rate=float(selected.mean()),
        boundary='Exact code/position persistence on a fixed domain; not a semantic equivalence test.')

def trajectory(checkpoints,cross_time):
    require(set(checkpoints)==set(STEPS),'All six checkpoints required')
    require(set(cross_time)=={(r,s) for r in STEPS for s in STEPS},'Complete6x6 grid required')
    curve={key:[float(checkpoints[t]['current'][key]) for t in STEPS] for key in
           ('diagonal_effect','offdiagonal_effect','selectivity','uniform_position_effect','diagonal_minus_uniform_position')}
    retrospective={key:[float(checkpoints[t]['retrospective_final'][key]) for t in STEPS] for key in curve}
    matrix=np.asarray([[cross_time[(r,s)]['contrast']['macro']['target_apt'] for s in STEPS] for r in STEPS])
    adjusted=matrix-np.diag(matrix)[:,None]
    return dict(steps=list(STEPS),current_discovery_curves=curve,
        normalized_areas={key:normalized_area(values) for key,values in curve.items()},
        retrospective_final_position_curves=retrospective,
        retrospective_normalized_areas={key:normalized_area(values) for key,values in retrospective.items()},
        cross_time_target_effect=matrix.tolist(),
        cross_time_minus_receiver_same_time=adjusted.tolist(),
        cross_time_axes=dict(rows='receiver checkpoint',columns='donor-message checkpoint'),
        cross_time_same_time_effect=np.diag(matrix).tolist(),
        interpretation='Area is a trapezoid summary on the fixed sparse checkpoint grid, not an exact onset time or isolated learning speed. Cross-time whole-packet compatibility includes donor-time interaction history and receiver-time action ability.')
