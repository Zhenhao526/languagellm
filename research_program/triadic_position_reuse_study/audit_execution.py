"""Independent position-reuse audit kernels, no import-time model work.

The completed-execution wrapper will bind the final producer schema separately.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from hashlib import sha256
from functools import lru_cache
from fractions import Fraction
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR_AUDIT_SHA='8f001dabd9508dc10d2be8a3bc26ea56cf1e4b1096e1e999e70c79b2073d58bc'
TOL=2e-12

def require(ok,message):
    if not ok:raise AssertionError(message)

@lru_cache(maxsize=1)
def references():
    path=HERE.parent/'triadic_context_transfer_study/audit_execution.py'
    require(sha256(path.read_bytes()).hexdigest()==PRIOR_AUDIT_SHA,'Prior independent audit changed')
    from research_program.triadic_context_transfer_study import audit_execution as previous
    return previous.references()

def vec(value,n,upper,label):
    value=np.asarray(value)
    require(value.dtype.kind in 'iu' and value.shape in ((),(n,)),label+' shape/type')
    require(((value>=0)&(value<upper)).all(),label+' domain')
    return np.full(n,int(value),dtype=np.int64) if value.ndim==0 else value.astype(np.int64,copy=False)

def route(tokens,live,senders=None,positions=None,symbols=None):
    """Independent direct one-hot encoding of actual receiver payloads."""
    tokens=np.asarray(tokens);n=len(tokens)
    require(tokens.shape==(n,3,4) and tokens.dtype.kind in 'iu' and ((tokens>=0)&(tokens<8)).all(),'Token domain')
    require(isinstance(live,(bool,np.bool_)),'Visibility flag')
    replace=senders is not None or positions is not None or symbols is not None
    if replace:
        require(senders is not None and positions is not None and symbols is not None,'Incomplete splice')
        s=vec(senders,n,3,'sender');p=vec(positions,n,4,'position');z=vec(symbols,n,8,'symbol')
    result=np.zeros((n,3,99),dtype=np.float64);rows=np.arange(n)
    for viewer in range(3):
        for speaker in range(3):
            if not (live or viewer==speaker):continue
            sent=tokens[:,speaker].copy()
            if replace and live and viewer!=speaker:
                selected=np.flatnonzero(s==speaker);sent[selected,p[selected]]=z[selected]
            for slot in range(4):result[rows,viewer,32*speaker+8*slot+sent[:,slot]]=1
            result[:,viewer,96+speaker]=1
    return result

def causal_inputs(observations,natural_messages,senders,donor_packets,windows,positions,live,generate_second):
    """Finite scheduler; generate_second is invoked only for live W1 rows."""
    x=np.asarray(observations);m=np.asarray(natural_messages);donor=np.asarray(donor_packets);n=len(x)
    require(x.shape==(n,3,54) and np.isfinite(x).all(),'Observation domain')
    require(m.shape==(n,2,3,4) and m.dtype.kind in 'iu' and ((m>=0)&(m<8)).all(),'Natural message domain')
    require(donor.shape==(n,2,4) and donor.dtype.kind in 'iu' and ((donor>=0)&(donor<8)).all(),'Donor packet domain')
    s=vec(senders,n,3,'sender');w=vec(windows,n,2,'window');p=vec(positions,n,4,'position');rows=np.arange(n)
    symbols=donor[rows,w,p];early=np.flatnonzero(w==0);late=np.flatnonzero(w==1)
    first=route(m[:,0],live)
    if live and len(early):first[early]=route(m[early,0],True,s[early],p[early],symbols[early])
    messages=m.copy()
    if live and len(early):
        next_messages=np.asarray(generate_second(np.concatenate((x[early],first[early]),axis=-1)))
        require(next_messages.shape==(len(early),3,4) and next_messages.dtype.kind in 'iu' and ((next_messages>=0)&(next_messages<8)).all(),'Generated W2 domain')
        messages[early,1]=next_messages
    second=route(messages[:,1],live)
    if live and len(late):second[late]=route(messages[late,1],True,s[late],p[late],symbols[late])
    patched=messages[rows,:,s,:].copy();patched[rows,w,p]=symbols
    return dict(messages=messages,patched_outward_packets=patched,replacement_symbols=symbols,
        first_routes=first,second_routes=second,action_inputs=np.concatenate((x,first,second),axis=-1),
        early_rows=len(early),late_rows=len(late),independent_module_samples=(3*n+3*len(early)) if live else 0)

def forward_intervention(networks,states,information,natural_messages,senders,donor_packets,windows,positions):
    """Only invoked after explicit authorization for a completed main batch."""
    context,old=references();x=context.features(states,information)
    def second(inputs):
        return np.stack([old.probabilities(networks[3*a+1],inputs[:,a]).argmax(-1).astype(np.int8) for a in range(3)],axis=1)
    out=causal_inputs(x,natural_messages,senders,donor_packets,windows,positions,True,second)
    probabilities=np.stack([old.probabilities(networks[3*a+2],out['action_inputs'][:,a]) for a in range(3)],axis=1)
    out.update(action_probabilities=probabilities,action_indices=probabilities.argmax(-1).astype(np.int16))
    return out

def select_positions(response_rates):
    """Rates indexed [sender, axis, eight positions]; retain nonpositive maxima."""
    rates=np.asarray(response_rates,dtype=np.float64)
    require(rates.shape==(3,3,8) and np.isfinite(rates).all() and ((rates>=0)&(rates<=1)).all(),'Rate domain')
    scores=np.empty_like(rates)
    for axis in range(3):scores[:,axis]=rates[:,axis]-rates[:,[q for q in range(3) if q!=axis]].mean(1)
    selected=scores.argmax(-1).astype(np.int8)
    return dict(response_rates=rates,scores=scores,selected_positions=selected,
        selected_scores=np.take_along_axis(scores,selected[:,:,None],axis=-1)[:,:,0])

def discovery(messages,endpoint_indices,axis,sender,listener,weights):
    """Full train natural-message changes only; no action or reward input."""
    m=np.asarray(messages);ends=np.asarray(endpoint_indices);a=np.asarray(axis);s=np.asarray(sender);l=np.asarray(listener);w=np.asarray(weights)
    n=len(ends)
    require(m.ndim==4 and m.shape[1:]==(2,3,4) and m.dtype.kind in 'iu' and ((m>=0)&(m<8)).all(),'Discovery messages')
    require(ends.shape==(n,2) and ends.dtype.kind in 'iu' and ((ends>=0)&(ends<len(m))).all(),'Discovery endpoints')
    a=vec(a,n,3,'axis');s=vec(s,n,3,'sender');l=vec(l,n,3,'listener')
    require((s!=l).all() and w.shape==(n,) and np.isfinite(w).all() and (w>0).all(),'Discovery weights')
    changed=(m[ends[:,0],:,s,:]!=m[ends[:,1],:,s,:]).reshape(n,8)
    rates=np.zeros((3,3,8));counts=np.zeros((3,3,3,8),dtype=np.int64);denominators=np.zeros((3,3,3),dtype=np.int64)
    exact_rates={};exact_scores={};selected=np.zeros((3,3),dtype=np.int8)
    for who in range(3):
        for q in range(3):
            take=(s==who)&(a==q)
            for hearer in range(3):
                if hearer==who:continue
                selected_rows=take&(l==hearer)
                require(abs(w[selected_rows].sum()-1/6)<TOL and np.ptp(w[selected_rows])<=1e-18,'S/L discovery denominator')
                denominators[who,q,hearer]=int(selected_rows.sum());counts[who,q,hearer]=changed[selected_rows].sum(0)
            for slot in range(8):
                value=sum((Fraction(int(counts[who,q,h,slot]),int(denominators[who,q,h])) for h in range(3) if h!=who),Fraction())/2
                exact_rates[who,q,slot]=value;rates[who,q,slot]=float(value)
    scores=np.zeros_like(rates)
    for who in range(3):
        for q in range(3):
            for slot in range(8):
                value=exact_rates[who,q,slot]-sum((exact_rates[who,r,slot] for r in range(3) if r!=q),Fraction())/2
                exact_scores[who,q,slot]=value;scores[who,q,slot]=float(value)
            selected[who,q]=max(range(8),key=lambda slot:exact_scores[who,q,slot])
    return dict(response_rates=rates,scores=scores,selected_positions=selected,
        selected_scores=np.take_along_axis(scores,selected[:,:,None],axis=-1)[:,:,0],
        response_counts=counts,stratum_denominators=denominators,
        exact_response_rates=[[[[exact_rates[s,q,p].numerator,exact_rates[s,q,p].denominator] for p in range(8)] for q in range(3)] for s in range(3)],
        exact_scores=[[[[exact_scores[s,q,p].numerator,exact_scores[s,q,p].denominator] for p in range(8)] for q in range(3)] for s in range(3)])

def matrix_statistics(matrix):
    t=np.asarray(matrix,dtype=np.float64)
    require(t.shape==(3,3) and np.isfinite(t).all(),'Matrix domain')
    diagonal=float(np.trace(t)/3);off=float(t[~np.eye(3,dtype=bool)].mean())
    return dict(T=t,diagonal=diagonal,off_diagonal=off,selectivity=diagonal-off)

def effect_matrix(effects,selected_positions,axis,sender,weights):
    """effects[N,2 directions,8 slots]; no outcome-conditioned filtering."""
    e=np.asarray(effects,dtype=np.float64);p=np.asarray(selected_positions);n=len(e)
    require(e.shape==(n,2,8) and np.isfinite(e).all(),'Effect shape')
    require(p.shape==(3,3) and p.dtype.kind in 'iu' and ((p>=0)&(p<8)).all(),'Selected positions')
    a=vec(axis,n,3,'axis');s=vec(sender,n,3,'sender');w=np.asarray(weights,dtype=float)
    require(w.shape==(n,) and np.isfinite(w).all() and (w>0).all(),'Validation weights')
    mean=e.mean(1);matrix=np.zeros((3,3));uniform=np.zeros(3)
    for r in range(3):
        take=np.flatnonzero(a==r);require(abs(w[take].sum()-1)<TOL,'Validation axis denominator')
        for q in range(3):matrix[q,r]=np.sum(w[take]*mean[take,p[s[take],q]])
        uniform[r]=np.sum(w[take]*mean[take].mean(1))
    result=matrix_statistics(matrix);result.update(uniform8_by_axis=uniform,uniform8_diagonal=float(uniform.mean()),
        selected_diagonal_minus_uniform8=result['diagonal']-float(uniform.mean()))
    return result

def packet_codes(packets):
    """W1[0] is the lowest base8 digit; exact and collision-free."""
    m=np.asarray(packets);require(m.ndim==3 and m.shape[1:]==(2,4) and m.dtype.kind in 'iu' and ((m>=0)&(m<8)).all(),'Packet domain')
    result=np.zeros(len(m),dtype=np.uint32)
    for i in range(8):result+=np.uint32(8**i)*m.reshape(-1,8)[:,i].astype(np.uint32)
    return result

def budget(n_cases=144,live_policies=8,silent_policies=8):
    actual=live_policies*n_cases*2*(8*2+2)
    modules=live_policies*n_cases*2*(2*(4*6+4*3)+(6+3))
    return dict(actual_worlds=actual,module_samples=modules,actual_npz=live_policies*2*(8*2+2),
        silent_alias_worlds=silent_policies*n_cases*2*(8*2+2),silent_records=silent_policies*2*(8*2+2))


def independent_dataset(task_specs):
    """Rebuild new hash selections from independently enumerated old truth."""
    import json
    from itertools import product
    references()
    from research_program.triadic_context_transfer_study import audit_execution as previous
    full={part:previous.independent_dataset(task_specs[part])[0] for part in ('train','new_needs_and_layouts')}
    def rank(salt,payload):
        return sha256((salt+'|'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode()).hexdigest()
    def extract(a,rows):
        fields=('case_index','axis','sender','listener','endpoint_indices','correct_actions','target_materials','recipient_layout_index','owner_index')
        out={k:a[k][rows].copy() for k in fields};out['source_spec_row']=np.asarray(rows,dtype=np.int32)
        out['source_all_case_index']=a['source_case_index'][rows].copy();n=len(rows)
        acts=out['correct_actions'][np.arange(n)[:,None],np.arange(2)[None],out['listener'][:,None]]
        sites=((acts-1)//4).astype(np.int8);dest=(((acts-1)%4)//2).astype(np.int8)
        peer_table=np.asarray([[1,2],[0,2],[0,1]],dtype=np.int8)
        partners=peer_table[out['listener'][:,None],(acts-1)%2]
        fields=np.stack((out['target_materials']//2,out['target_materials']%2,dest,partners),axis=-1).astype(np.int8)
        out.update(listener_correct_actions=acts,target_sites=sites,target_destinations=dest,truth_attributes=fields,
            truth_field_change_mask=fields[:,0]!=fields[:,1])
        require((acts>0).all() and np.array_equal(out['truth_field_change_mask'],np.eye(4,dtype=bool)[out['axis']]),'Independent content attributes')
        return out
    a=full['train'];discovery=extract(a,np.arange(len(a['axis']),dtype=np.int64));discovery['base_within_axis_weight']=a['base_within_axis_weight'].copy()
    a=full['new_needs_and_layouts'];spec=task_specs['new_needs_and_layouts'];layouts=spec['layouts'];owners=spec['private_sites'];needs=spec['needs'];nl,no=len(layouts),len(owners)
    case_rows={int(ci):int(row) for row,ci in reversed(list(enumerate(a['case_index'])))}
    identities={}
    for ci,row in case_rows.items():
        ni=a['endpoint_indices'][row]//no//nl
        identities[ci]=[int(a['axis'][row]),int(a['sender'][row]),int(a['listener'][row]),list(needs[int(ni[0])]),list(needs[int(ni[1])])]
    chosen_rows=[];chosen_cases=[];chosen_shifts=[];selected_case_count=0;case_hashes=[];background_hashes=[];donor_hashes=[]
    for axis,s,l in product(range(3),range(3),range(3)):
        if s==l:continue
        candidates=[ci for ci,ident in identities.items() if ident[:3]==[axis,s,l]]
        candidates.sort(key=lambda ci:(rank('triadic_position_reuse_case_v1',identities[ci]),identities[ci]))
        require(len(candidates)>=4,'Insufficient validation content')
        for ci in candidates[:4]:
            ident=identities[ci];case_hashes.append(rank('triadic_position_reuse_case_v1',ident))
            backgrounds=list(product(range(nl),range(no)))
            backgrounds.sort(key=lambda t:(rank('triadic_position_reuse_background_v1',[ident,layouts[t[0]],owners[t[1]]]),layouts[t[0]],owners[t[1]]))
            first=backgrounds[0];second=next(t for t in backgrounds if t[0]!=first[0])
            for li,oi in (first,second):
                row=(ci*nl+li)*no+oi
                eligible=[k for k in range(nl-1) if a['eligible'][k,row]]
                def donor_key(k):
                    dl=int(a['donor_layout_index'][k,li])
                    return rank('triadic_position_reuse_donor_v1',[ident,layouts[li],owners[oi],layouts[dl]]),layouts[dl]
                shift=min(eligible,key=donor_key)
                chosen_rows.append(row);chosen_cases.append(selected_case_count);chosen_shifts.append(shift)
                background_hashes.append(rank('triadic_position_reuse_background_v1',[ident,layouts[li],owners[oi]]));donor_hashes.append(donor_key(shift)[0])
            selected_case_count+=1
    rows=np.asarray(chosen_rows,dtype=np.int64);shifts=np.asarray(chosen_shifts,dtype=np.int16);validation=extract(a,rows);n=len(rows)
    validation.update(selected_case_index=np.asarray(chosen_cases,dtype=np.int16),donor_shift_index=shifts,
        donor_endpoint_indices=a['donor_endpoint_indices'][shifts,rows].copy(),donor_correct_actions=a['donor_correct_actions'][shifts,rows].copy(),
        donor_layout_index=a['donor_layout_index'][shifts,a['recipient_layout_index'][rows]].copy(),
        eligible_donor_count=a['eligible_donor_count'][rows].copy(),view_equal_LL=a['view_equal_LL'][shifts,rows].copy(),base_within_axis_weight=np.full(n,1/48))
    donor_acts=validation['donor_correct_actions'][np.arange(n)[:,None],np.arange(2)[None],validation['listener'][:,None]]
    validation.update(donor_listener_correct_actions=donor_acts,donor_target_sites=((donor_acts-1)//4).astype(np.int8))
    require(n==144 and selected_case_count==72 and np.all(validation['donor_target_sites']!=validation['target_sites']),'Validation size/eligible truth')
    nd={tuple(n) for n in task_specs['train']['needs']};nv={tuple(n) for n in needs}
    require(not nd&nv and not set(map(tuple,task_specs['train']['layouts']))&set(map(tuple,layouts)),'Static discovery-validation separation')
    return dict(discovery=discovery,validation=validation),dict(case_sha256=case_hashes,background_sha256=background_hashes,donor_sha256=donor_hashes,
        discovery_rows=len(discovery['axis']),validation_rows=n,discovery_unique_endpoint_worlds=len(np.unique(discovery['endpoint_indices'])),validation_unique_endpoint_worlds=len(np.unique(validation['endpoint_indices'])))


SEEDS=(51101,51102,51103,51104)
CONDITIONS=('PL_silent','PL_live','LL_silent','LL_live')
AXES=('kind','length','destination')
PART='new_needs_and_layouts'
PRIOR_RESULT_SHA='5c610cd4f2e79d479ca3859e7e7628d93fbfa270415c539b282407baece914bc'
PRIOR_VERIFICATION_SHA='39447f7ac1410bcbee0d518b62d84a74974350270e89ea2adecf6b72334197a8'
PLAN_SHA='87bf3b412c10fc7a27539b90542e2ec67165e1b2ad4b19476748f2cac2a81eaa'


def read(path):
    import json
    return json.loads(Path(path).read_text())

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    import json
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    require(error<=TOL,label+' numeric mismatch');return error

def verify_selection(expected,actual):
    require(np.array_equal(expected['selected_positions'],actual['selected_positions']),'Exact selected positions')
    require(np.array_equal(expected['response_counts'],actual['stratum_change_counts']),'Discovery integer changes')
    require(np.array_equal(expected['stratum_denominators'],actual['stratum_pair_counts']),'Discovery integer denominators')
    for source,target in [('exact_response_rates','exact_response_rates'),('exact_scores','exact_specificity_scores')]:
        fractions=[[[dict(numerator=n,denominator=d) for n,d in row] for row in actor] for actor in expected[source]]
        require(fractions==actual[target],'Exact Fraction '+target)
    for s in range(3):
        for q in range(3):
            values=[Fraction(n,d) for n,d in expected['exact_scores'][s][q]]
            maximizing=[i for i,v in enumerate(values) if v==max(values)]
            require(actual['maximizing_positions'][s][q]==maximizing,'All exact maximizers')
    close(expected['response_rates'],actual['response_rates'],'Displayed rates')
    close(expected['scores'],actual['specificity_scores'],'Displayed specificity')
    require(actual['axes']==list(AXES) and actual['positions']==[dict(index=i,window=i//4,position=i%4) for i in range(8)],'Position label convention')


def attributes(states,actions,listeners):
    n=len(states);r=np.arange(n);aid=actions[r,listeners];v=np.maximum(aid-1,0)
    site=v//4;destination=(v%4)//2;peer=v%2;material=states[r,3+site]
    partners=np.asarray([[1,2],[0,2],[0,1]],dtype=np.int8)[listeners,peer]
    out=np.stack((material//2,material%2,destination,partners),axis=-1).astype(np.int8)
    out[aid==0]=-1;return out


def measure(spec,b,pool,data,codes):
    context,_=references();n=len(spec['sender']);r=np.arange(n);l=spec['listener'];s=spec['sender']
    rid=spec['endpoint_indices'][:,b];cfid=spec['endpoint_indices'][:,1-b]
    states=pool['states'][rid];cfstates=pool['states'][cfid]
    difference=states!=cfstates;expected_diff=np.zeros_like(difference);expected_diff[r,s]=True
    require(np.array_equal(difference,expected_diff),'Current/counterfactual only sender demand')
    action=data['action_indices'];probability=data['action_probabilities']
    require(action.shape==(n,3) and action.dtype.kind in 'iu' and ((action>=0)&(action<17)).all(),'Action domain')
    require(probability.shape==(n,3,17) and probability.dtype==np.float64 and (probability>=0).all() and (probability<=1).all(),'Probability domain')
    close(probability.sum(-1),np.ones((n,3)),'Probability normalized')
    require(np.array_equal(action,probability.argmax(-1)),'Full17 first argmax')
    reward,executed,satisfied=context.native(states,action);cf,cfexecuted,cfsatisfied=context.native(cfstates,action)
    require(np.array_equal(executed,cfexecuted),'Demand-independent physical execution')
    for key,value in [('greedy_reward',reward),('executed',executed),('satisfied',satisfied),('counterfactual_reward',cf),('counterfactual_satisfied',cfsatisfied)]:
        require(np.array_equal(data[key],value) and data[key].dtype==value.dtype,'Actual outcome '+key)
    cur=spec['correct_actions'][:,b];target=spec['correct_actions'][:,1-b]
    require(np.array_equal(reward==1,np.all(action==cur,axis=1)) and np.array_equal(cf==1,np.all(action==target,axis=1)),'Unique full plan at actual layout')
    actual_attrs=attributes(states,action,l);cur_attrs=attributes(states,cur,l);target_attrs=attributes(states,target,l)
    equality=actual_attrs==target_attrs;unchanged=cur_attrs==target_attrs
    packet=packet_codes(data['patched_outward_packets']);seen=np.zeros(n,dtype=bool)
    for who in range(3):seen[s==who]=np.isin(packet[s==who],codes[who])
    m=pool['messages'][rid];nat=pool['action_indices'][rid,l]
    values=dict(target_apt=action[r,l]==target[r,l],current_apt=action[r,l]==cur[r,l],
        target_probability=probability[r,l,target[r,l]],current_probability=probability[r,l,cur[r,l]],
        natural_current_apt=nat==cur[r,l],listener_action_changed=action[r,l]!=nat,
        native_reward=reward,native_team_full=reward==1,counterfactual_reward=cf,counterfactual_team_full=cf==1,
        physical_pair_executed=executed.sum(-1)==2,unchanged_requirements_apt=np.all(equality|~unchanged,axis=1),
        packet_in_train_endpoint_greedy_set=seen,any_self_generated_message_changed=np.any(data['messages']!=m,axis=(1,2,3)),
        listener_second_message_changed=np.any(data['messages'][r,1,l]!=m[r,1,l],axis=1))
    for i,name in enumerate(('kind','length','destination','partner')):values[name+'_apt']=equality[:,i]
    return values


def contrast(same,opposite):
    require(set(same)==set(opposite),'Paired metric schema')
    require(np.array_equal(same['natural_current_apt'],opposite['natural_current_apt']),'Same natural baseline')
    result={k:np.asarray(opposite[k],float)-np.asarray(same[k],float) for k in same}
    u=~same['packet_in_train_endpoint_greedy_set'].astype(bool)&~opposite['packet_in_train_endpoint_greedy_set'].astype(bool)
    result.update(both_packets_outside_train_greedy_set=u,outside_both_target_apt_gain=u*result['target_apt'])
    return result


def aggregate(spec,values):
    n=len(spec['axis']);require(all(np.shape(v)==(2,n) and np.isfinite(v).all() for v in values.values()),'All directions/rows')
    weights=spec['base_within_axis_weight'];by_axis={};by_sender_listener={};by_direction={}
    for a,label in enumerate(AXES):
        take=spec['axis']==a;require(take.sum()==48 and abs(weights[take].sum()-1)<TOL,'Axis denominator')
        by_axis[label]={k:float(np.sum(v[:,take]*weights[take][None]/2)) for k,v in values.items()}
        by_sender_listener[label]={}
        for s in range(3):
            for l in range(3):
                if s==l:continue
                st=take&(spec['sender']==s)&(spec['listener']==l);require(st.sum()==8,'S/L denominator')
                by_sender_listener[label][f'{s}_{l}']={k:float(np.sum(v[:,st]*weights[st][None]*3)) for k,v in values.items()}
        by_direction[label]={str(b):{k:float(np.sum(v[b,take]*weights[take])) for k,v in values.items()} for b in (0,1)}
    return dict(by_axis=by_axis,macro={k:sum(by_axis[a][k] for a in AXES)/3 for k in values},
        by_sender_listener=by_sender_listener,by_direction=by_direction)


def policy_summary(spec,selection,positions,ref_values,shams):
    selected={};matrix=np.zeros((3,3));p=selection['selected_positions'];n=len(spec['axis'])
    for q,label in enumerate(AXES):
        units=p[spec['sender'],q];selected[label]={}
        for arm in ('same','opposite','contrast'):
            values={}
            for key in positions[0][arm]:
                stacked=np.stack([positions[u][arm][key] for u in range(8)],axis=-1)
                values[key]=stacked[:,np.arange(n),units]
            selected[label][arm]=aggregate(spec,values)
        for r,axis in enumerate(AXES):matrix[q,r]=selected[label]['contrast']['by_axis'][axis]['target_apt']
    uniform={}
    for arm in ('same','opposite','contrast'):
        values={k:np.mean(np.stack([positions[u][arm][k].astype(float) for u in range(8)]),axis=0) for k in positions[0][arm]}
        uniform[arm]=aggregate(spec,values)
    result=matrix_statistics(matrix)
    return dict(matrix_rows_discovered_axis_columns_test_axis=matrix.tolist(),selected=selected,uniform_position=uniform,
        diagonal_effect=result['diagonal'],offdiagonal_effect=result['off_diagonal'],selectivity=result['selectivity'],
        uniform_position_effect=uniform['contrast']['macro']['target_apt'],diagonal_minus_uniform_position=result['diagonal']-uniform['contrast']['macro']['target_apt'],
        all_positions={str(u):{arm:aggregate(spec,v) for arm,v in positions[u].items()} for u in range(8)},
        matched_references={arm:aggregate(spec,v) for arm,v in ref_values.items()},
        shams={str(u):aggregate(spec,v) for u,v in shams.items()})


def compare_tree(expected,actual,counts,path=''):
    if isinstance(expected,dict):
        require(set(expected)<=set(actual),'Missing summary fields '+path)
        for k,v in expected.items():compare_tree(v,actual[k],counts,path+'/'+k)
    elif isinstance(expected,(list,tuple)):
        require(len(expected)==len(actual),'Summary list shape '+path)
        for i,v in enumerate(expected):compare_tree(v,actual[i],counts,path+'/'+str(i))
    elif isinstance(expected,(int,float,np.integer,np.floating)):
        error=close(expected,actual,path);counts['scalars']+=1;counts['max_error']=max(counts['max_error'],error)
    else:require(expected==actual,'Summary value '+path)


def cells():
    from itertools import product
    yield from product(range(8),('same','opposite'),(0,1))
    yield from ((u,'sham',b) for u,b in product((0,4),(0,1)))


def reference_arrays(pool,spec,b,arm):
    context,_=references();r=spec['endpoint_indices'][:,b];cf=spec['endpoint_indices'][:,1-b];s=spec['sender'];n=len(s)
    d=r if arm=='natural' else spec['donor_endpoint_indices'][:,b if arm=='same' else 1-b]
    data={k:pool[k][r].copy() for k in ('messages','action_indices','action_probabilities')}
    data['patched_outward_packets']=pool['messages'][d,:,s,:].copy()
    reward,executed,satisfied=context.native(pool['states'][r],data['action_indices']);cfr,_,cfs=context.native(pool['states'][cf],data['action_indices'])
    data.update(greedy_reward=reward,executed=executed,satisfied=satisfied,counterfactual_reward=cfr,counterfactual_satisfied=cfs)
    return data


def audit_cell(record,path,pool,spec,networks,information,codes):
    context,_=references();u,arm,b=record['unit'],record['arm'],record['direction'];live=record['condition'].endswith('_live')
    n=len(spec['sender']);rows=np.arange(n);s=spec['sender'];r=spec['endpoint_indices'][:,b];cf=spec['endpoint_indices'][:,1-b]
    d=r if arm=='sham' else spec['donor_endpoint_indices'][:,b if arm=='same' else 1-b]
    m=pool['messages'][r];packets=pool['messages'][d,:,s,:];x=context.features(pool['states'][r],information)
    if live:trace=forward_intervention(networks,pool['states'][r],information,m,s,packets,u//4,u%4)
    else:
        def forbidden(_):raise AssertionError('Silent audit must not forward')
        trace=causal_inputs(x,m,s,packets,u//4,u%4,False,forbidden)
        trace.update(action_indices=pool['action_indices'][r].copy(),action_probabilities=pool['action_probabilities'][r].copy())
        require(np.array_equal(trace['action_inputs'],np.concatenate((x,route(m[:,0],False),route(m[:,1],False)),axis=-1)),'Silent invisible intervention changed input')
    expected={k:trace[k] for k in ('messages','patched_outward_packets','replacement_symbols','action_indices','action_probabilities')}
    expected.update(dataset_rows=rows,recipient_indices=r,counterfactual_recipient_indices=cf,donor_indices=d,donor_packets=packets)
    reward,executed,satisfied=context.native(pool['states'][r],expected['action_indices']);cfr,_,cfs=context.native(pool['states'][cf],expected['action_indices'])
    expected.update(greedy_reward=reward,executed=executed,satisfied=satisfied,counterfactual_reward=cfr,counterfactual_satisfied=cfs)
    max_error=0.
    if live:
        require(record['path']==str(path) and sha(path)==record['data_sha256'],'Actual cell bytes/path');data=load_npz(path)
        require(set(data)==set(expected),'Actual cell fields')
        for key,value in expected.items():
            require(data[key].dtype==value.dtype,'Cell dtype '+key)
            if key=='action_probabilities':max_error=close(data[key],value,'Independent17 probability')
            else:require(np.array_equal(data[key],value),'Independent actual '+key)
    else:
        require(record['path'] is None and record['data_sha256'] is None and not path.exists(),'Silent null file');data=expected
    require(record['window']==u//4 and record['position']==u%4 and record['worlds']==144 and record['is_silent_alias'] is (not live),'Record condition/selector')
    require(record['new_forward_worlds']==(n if live else 0) and record['new_network_samples']==trace['independent_module_samples'],'Actual forward samples')
    require(record['neural_forward_calls']==((6 if u<4 else 3) if live else 0),'Actual module calls')
    require(record['outward_patch_visible'] is live,'Visibility metadata')
    require(record['routing_sha256']=={k:context.array_sha(trace[k]) for k in ('first_routes','second_routes','action_inputs')},'All actual route/input hashes')
    require(record['donor_packets_sha256']==context.array_sha(packets),'Only selected sender donor source')
    require(np.array_equal(data['messages'][:,0],m[:,0]) and np.array_equal(data['messages'][rows,1,s],m[rows,1,s]),'Own-generated sender history')
    if u>=4 or not live:require(np.array_equal(data['messages'],m),'Late/silent no W2 update')
    identity_error=0.
    if arm=='sham':
        require(np.array_equal(data['messages'],m) and np.array_equal(data['action_indices'],pool['action_indices'][r]),'Actual sham identity')
        identity_error=close(data['action_probabilities'],pool['action_probabilities'][r],'Actual sham probabilities')
    close(record['max_identity_error'],identity_error,'Sham saved error')
    return measure(spec,b,pool,data,codes),dict(independent_worlds=n if live else 0,independent_modules=trace['independent_module_samples'],max_probability_error=max_error)


def check_reference(record,path,policy,pool,spec,old_records):
    arm,b=record['arm'],record['direction'];live=policy['condition'].endswith('_live')
    expected=reference_arrays(pool,spec,b,arm);used=[]
    if arm!='natural' and live:
        for shift in sorted(set(map(int,spec['donor_shift_index']))):
            take=np.flatnonzero(spec['donor_shift_index']==shift);oldrows=spec['source_spec_row'][take]
            source=old_records[policy['seed'],policy['condition'],f'remote_{arm}_both',shift,b]
            source_path=Path(source['path']);require(sha(source_path)==source['data_sha256'],'Original whole output changed')
            old=load_npz(source_path)
            r=spec['endpoint_indices'][:,b];d=spec['donor_endpoint_indices'][:,b if arm=='same' else 1-b]
            for key,truth in [('recipient_indices',r[take]),('donor_indices',d[take]),('donor_packets',expected['patched_outward_packets'][take])]:
                require(np.array_equal(old[key][oldrows],truth),'Same-row whole '+key)
            for key in expected:
                if key!='patched_outward_packets':expected[key][take]=old[key][oldrows]
            used.append(dict(path=str(source_path),sha256=source['data_sha256'],shift=shift,selected_rows=len(take)))
    require(record['whole_sources']==used,'Whole source coverage/order')
    require(record['path']==str(path) and sha(path)==record['data_sha256'],'Materialized reference bytes')
    require(record['worlds']==144 and record['new_forward_worlds']==record['new_network_samples']==0,'References have no new forward')
    require(record['is_silent_alias'] is (not live) and record['source_natural']==policy['endpoints'][PART],'Reference cohort/source')
    require(record['kind']==('natural' if arm=='natural' else 'whole'),'Reference kind')
    data=load_npz(path);require(set(data)==set(expected),'Reference NPZ fields')
    for key,value in expected.items():require(data[key].dtype==value.dtype and np.array_equal(data[key],value),'Zero-forward reference copy '+key)
    return data


def audit(run,out):
    import platform
    from itertools import product
    context,old=references();run=Path(run).resolve();out=Path(out).resolve();execution=run/'execution'
    require(read(execution/'status.json')['status']=='completed' and not (execution/'failure.json').exists(),'Main must be complete without failure before audit')
    plan=read(run/'plan.json');main=read(execution/'results.json');frozen=read(run/'frozen_spec.json')
    require(sha(run/'plan.json')==PLAN_SHA,'Authorized position plan SHA')
    require(main['status']=='completed' and main['plan_sha256']==read(run/'freeze.json')['plan_sha256']==sha(run/'plan.json'),'Main completion/freeze chain')
    require(read(execution/'started.json')['plan_sha256']==sha(run/'plan.json'),'Started plan')
    require(plan['runtime']==frozen['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Runtime binding')
    require(frozen['config']==plan['config'] and frozen['sources_sha256']==plan['sources_sha256'] and frozen['discovery_started'] is False,'Pre-discovery specification freeze')
    cfg=plan['config'];expected_budget=budget()
    for key,value in dict(seeds=list(SEEDS),conditions=list(CONDITIONS),checkpoint=6000,validation=PART,validation_rows=144,
        positions=list(range(8)),arms=['same','opposite'],directions=[0,1],shams=[0,4],actual_worlds=41472,
        actual_module_samples=186624,actual_files=288,silent_alias_records=288,silent_logical_worlds=41472,
        parameter_loads=8,reference_cells=96,reference_row_occurrences=13824,training_updates=0,new_natural_forward_worlds=0,
        positivity_filter=False,force_distinct_positions=False,validation_selection=False,
        discovery_ties='exact_fraction_score_then_lowest_position').items():require(cfg[key]==value,'Frozen configuration '+key)
    artifacts={str(p):sha(p) for p in (run/'plan.json',run/'freeze.json',run/'frozen_spec.json',execution/'started.json',execution/'status.json',execution/'results.json')}
    root=HERE.parents[1]
    for path,digest in plan['sources_sha256'].items():
        require(sha(path)==digest,'Frozen source changed');snapshot=run/'source_snapshot'/Path(path).relative_to(root)
        require(sha(snapshot)==digest,'Frozen snapshot changed');artifacts[path]=digest;artifacts[str(snapshot)]=digest
    for group in ('inputs_sha256','prepared_files_sha256'):
        for path,digest in plan[group].items():require(sha(path)==digest,'Frozen input '+path);artifacts[path]=digest
    prior=HERE.parent/'triadic_context_transfer_study/results/transfer_001'
    require(sha(prior/'execution/results.json')==PRIOR_RESULT_SHA and sha(prior/'audit_execution_001/verification.json')==PRIOR_VERIFICATION_SHA,'Prior full independent proof chain')
    prior_proof=read(prior/'audit_execution_001/verification.json');require(prior_proof['status']=='passed' and prior_proof['audit_source_sha256']==PRIOR_AUDIT_SHA,'Prior audited source')
    prior_main=read(prior/'execution/results.json');prior_plan=read(prior/'plan.json')
    old_records={(r['seed'],r['condition'],r['mode'],r['shift_index'],r['direction']):r for r in prior_main['records'] if r['partition']==PART}
    task_specs,_,_=context.independent_specs();expected_data,static_proof=independent_dataset(task_specs)
    dataset={}
    for part in ('discovery','validation'):
        dataset[part]=load_npz(run/'dataset'/(part+'.npz'));require(set(dataset[part])==set(expected_data[part]),'Static all fields')
        for key,truth in expected_data[part].items():
            actual=dataset[part][key];require(actual.dtype==truth.dtype,'Static dtype '+key)
            if truth.dtype.kind=='f':close(actual,truth,'Static weight '+key)
            else:require(np.array_equal(actual,truth),'Independent static '+key)
    validation_meta=read(run/'dataset/validation.json')
    for key,group,field in [('case_sha256','selected_cases','case_sha256'),('background_sha256','selected_backgrounds','background_sha256'),('donor_sha256','selected_backgrounds','donor_sha256')]:
        require(static_proof[key]==[r[field] for r in validation_meta[group]],'Independent ranked metadata '+key)
    spec=dataset['validation'];ds=dataset['discovery']
    require([(p['seed'],p['condition']) for p in plan['policies']]==list(product(SEEDS,CONDITIONS)),'All16 ordered policies')
    require([(r['seed'],r['condition'],r['unit'],r['arm'],r['direction']) for r in main['records']]==[(s,c,u,a,b) for s,c in product(SEEDS,CONDITIONS) for u,a,b in cells()],'Complete canonical576 cells')
    cell_lookup={(r['seed'],r['condition'],r['unit'],r['arm'],r['direction']):r for r in main['records']}
    ref_lookup={(r['seed'],r['condition'],r['arm'],r['direction']):r for r in main['references']}
    require(len(main['references'])==len(ref_lookup)==96 and set(ref_lookup)==set(product(SEEDS,CONDITIONS,('natural','same','opposite'),(0,1))),'All96 reference copies')
    summary_lookup={(r['seed'],r['condition']):r for r in main['policy_summaries']}
    require(len(main['policy_summaries'])==len(summary_lookup)==16,'All16 policy summaries')
    reports=[];counts=dict(scalars=0,max_error=0.);worlds=modules=loads=actual_files=silent_records=0;max_error=0.;actual_paths=set();reference_paths=set()
    for policy in plan['policies']:
        seed,condition=policy['seed'],policy['condition'];information=condition.split('_')[0];live=condition.endswith('_live')
        original=next(p for p in prior_plan['policies'] if p['seed']==seed and p['condition']==condition)
        require(policy['checkpoint']==original['checkpoint'] and policy['endpoints']=={part:original['endpoints'][part] for part in ('train',PART)},'No cross-policy natural/weights')
        expected_whole=[r for r in prior_main['records'] if r['seed']==seed and r['condition']==condition and r['partition']==PART and r['mode'] in ('remote_same_both','remote_opposite_both') and r['shift_index'] in set(map(int,spec['donor_shift_index']))]
        require(policy['whole_reference_records']==expected_whole,'Bound complete matched whole sources')
        with np.load(policy['endpoints']['train'],allow_pickle=False) as z:training_messages=z['messages']
        require(training_messages.shape==(419904,2,3,4),'Full train natural domain')
        selected=discovery(training_messages,ds['endpoint_indices'],ds['axis'],ds['sender'],ds['listener'],ds['base_within_axis_weight'])
        stored_selection=read(policy['selection']);verify_selection(selected,stored_selection)
        codes=[np.unique(packet_codes(training_messages[:,:,s,:])) for s in range(3)];stored_codes=load_npz(policy['train_packet_codes'])
        require(set(stored_codes)=={'0','1','2'},'Three sender reference sets')
        for s in range(3):require(stored_codes[str(s)].dtype==np.uint32 and np.array_equal(stored_codes[str(s)],codes[s]),'Full training greedy packet set')
        del training_messages
        pool=load_npz(policy['endpoints'][PART]);require(np.array_equal(pool['states'],context.packed(task_specs[PART])),'Natural validation world order')
        networks=None
        if live:networks,_,_=old.checkpoint(policy['checkpoint'],6000,seed);loads+=1
        directory=execution/f'seed_{seed}_{condition}';values={};shams={}
        for u,arm,b in cells():
            record=cell_lookup[seed,condition,u,arm,b];name=f'unit{u}_{arm}_direction{b}';path=directory/(name+'.npz');jp=directory/(name+'.json')
            require(read(jp)==record and record['natural_source']==policy['endpoints'][PART],'Cell source/individual record');artifacts[str(jp)]=sha(jp)
            measured,receipt=audit_cell(record,path,pool,spec,networks,information,codes)
            worlds+=receipt['independent_worlds'];modules+=receipt['independent_modules'];max_error=max(max_error,receipt['max_probability_error'])
            if live:actual_files+=1;actual_paths.add(path);artifacts[str(path)]=record['data_sha256']
            else:silent_records+=1
            if arm=='sham':shams[u,b]=measured
            else:values[u,arm,b]=measured
        positions={}
        for u in range(8):
            position={arm:{k:np.stack([values[u,arm,b][k] for b in (0,1)]) for k in values[u,arm,0]} for arm in ('same','opposite')}
            position['contrast']=contrast(position['same'],position['opposite']);positions[u]=position
        reference_values={}
        for arm,b in product(('natural','same','opposite'),(0,1)):
            record=ref_lookup[seed,condition,arm,b];path=directory/f'reference_{arm}_direction{b}.npz'
            data=check_reference(record,path,policy,pool,spec,old_records);reference_values[arm,b]=measure(spec,b,pool,data,codes)
            reference_paths.add(path);artifacts[str(path)]=record['data_sha256']
        refs={arm:{k:np.stack([reference_values[arm,b][k] for b in (0,1)]) for k in reference_values[arm,0]} for arm in ('natural','same','opposite')}
        refs['contrast']=contrast(refs['same'],refs['opposite'])
        sham_values={u:{k:np.stack([shams[u,b][k] for b in (0,1)]) for k in shams[u,0]} for u in (0,4)}
        expected_summary=policy_summary(spec,selected,positions,refs,sham_values)
        sr=summary_lookup[seed,condition];sp=directory/'summary.json'
        require(sr['path']==str(sp) and sha(sp)==sr['sha256'],'Policy summary bytes');official=read(sp)
        require(official['seed']==seed and official['condition']==condition and official['selection']==stored_selection,'Summary/selection cohort')
        compare_tree(expected_summary,official,counts,f'{seed}/{condition}')
        if not live:
            for value in (expected_summary['matrix_rows_discovered_axis_columns_test_axis'],expected_summary['diagonal_effect'],expected_summary['selectivity'],expected_summary['uniform_position_effect']):close(value,np.zeros_like(value),'Silent structural zero')
        artifacts[str(sp)]=sr['sha256'];require(set(directory.glob('*.npz'))=={p for p in actual_paths|reference_paths if p.parent==directory},'No extra/missing NPZ')
        report=dict(seed=seed,condition=condition,selected_positions=selected['selected_positions'].tolist(),training_packet_set_sizes=[len(z) for z in codes],summary=expected_summary)
        write(out/f'seed_{seed}_{condition}.json',report);reports.append(report)
        print(__import__('json').dumps(dict(audit_stage='policy_complete',seed=seed,condition=condition,independent_worlds=worlds,independent_modules=modules)),flush=True)
    require(worlds==41472 and modules==186624 and loads==8 and actual_files==288 and silent_records==288 and len(reference_paths)==96,'Full independent actual budget')
    for key,value in dict(actual_worlds=41472,actual_module_samples=186624,actual_files=288,silent_alias_records=288,silent_logical_worlds=41472,parameter_loads=8,reference_cells=96,reference_row_occurrences=13824).items():require(main['totals'][key]==value,'Saved final budget '+key)
    require(main['training_updates']==main['new_natural_forward_worlds']==0,'No new training/natural forward')
    paired=[]
    for seed in SEEDS:
        p=next(r for r in reports if r['seed']==seed and r['condition']=='PL_live')['summary'];l=next(r for r in reports if r['seed']==seed and r['condition']=='LL_live')['summary']
        paired.append(dict(seed=seed,PL=p['selectivity'],LL=l['selectivity'],difference=p['selectivity']-l['selectivity']))
    primary=dict(paired_seeds=paired,mean_difference=sum(p['difference'] for p in paired)/4);compare_tree(primary,main['primary'],counts,'primary')
    for path,digest in artifacts.items():require(sha(path)==digest,'Source changed during audit')
    policy_receipts=[dict(seed=r['seed'],condition=r['condition'],path=str(out/f"seed_{r['seed']}_{r['condition']}.json"),sha256=sha(out/f"seed_{r['seed']}_{r['condition']}.json")) for r in reports]
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=sha(run/'plan.json'),artifacts_sha256=artifacts,
        static_checks=static_proof,policies=policy_receipts,primary=primary,summary_comparison=counts,max_probability_error=max_error,
        scope=dict(actual_cells=288,silent_alias_records=288,materialized_reference_npz=96,reference_rows=13824,
            independent_forward_worlds=worlds,independent_module_samples=modules,parameter_loads=loads,
            full_train_natural_domains_read=16,full_train_worlds_per_domain=419904,validation_rows=144,
            new_natural_forwards=0,training_updates=0,optimizer_replays=0),
        limits=['All saved post-endpoint single-position interventions independently forwarded; natural and whole reference copies anchored to prior complete audit without new forward.',
            'All numeric fields in per-policy position/selected/uniform/sham/reference summaries independently checked, including fixed direction and S/L strata.',
            'Full-train greedy endpoint packet membership is not proof of absence from stochastic training trajectories.',
            'Developmental position-function reuse; selectivity alone is not compositionality or language-origin evidence.'])


def main():
    import argparse,time,traceback
    from datetime import datetime,timezone
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out).resolve();require(not out.exists(),'Refuse audit overwrite/retry');out.mkdir(parents=True);start=__import__('time').perf_counter()
    write(out/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),audit_source_sha256=sha(__file__),automatic_retry=False))
    try:
        result=audit(args.run,out);result.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        write(out/'verification.json',result)
        (out/'独立核验.md').write_text('完整独立审计通过。288实际干预NPZ、288静默引用、96复制参照均核验；独立后继前向41,472世界、186,624模块样本，8份权重加载，0训练/优化器重放。\n\n完整训练内容对子的位置发现以整数计数和Fraction独立重算，8位参考集合覆盖每政策419,904个末点贪心训练世界。全部新位置输入/消息/行动、原生与反事实结算及每政策数值汇总通过检查。参考集合外不等于实际随机训练从未见过，位置作用不等于组合性。详见verification及16政策收据。\n')
        print(__import__('json').dumps(dict(status='passed',scope=result['scope'],max_probability_error=result['max_probability_error'],summary_comparison=result['summary_comparison'])),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,source_sha256=sha(__file__),automatic_retry=False));raise

if __name__=='__main__':main()
