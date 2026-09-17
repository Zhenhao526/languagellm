"""Complete static audit of a candidate reciprocal-execution physical rule.

No model imports, weights, policy outputs, neural inference or agent training.
The only inherited file read is the SHA-pinned original static prepared.json.
Canonical-layout proofs extend by a bijection to all 24 material layouts;
private-site owner permutations do not affect physical settlement.
"""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations,permutations,product
from pathlib import Path
import argparse,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'triadic_action_dependency_study/results/context_001/prepared.json'
SOURCE_SHA='e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299'
PAIRS=tuple(combinations(range(3),2))
RESOURCE_BITS=(3,12,5,10,1,2,4,8)
DEST_BITS=(1,2,3)
ACCEPT=np.asarray([[[bool((RESOURCE_BITS[n//3]>>m)&1) and bool((DEST_BITS[n%3]>>d)&1)
                    for d in range(2)] for m in range(4)] for n in range(24)])
ALL_ACTIONS=np.asarray(list(product(range(17),repeat=3)),dtype=np.int16)


def action(who,partner,site,destination):
    return 1+4*site+2*destination+[p for p in range(3) if p!=who].index(partner)


def decode(actions):
    actions=np.asarray(actions);raw=np.maximum(actions.astype(np.int64)-1,0)
    site,destination=raw//4,(raw//2)%2
    partner=np.stack([np.asarray([p for p in range(3) if p!=who])[raw[...,who]%2]
                      for who in range(3)],axis=-1)
    return actions!=0,site,destination,partner


def execution_index(actions,reciprocal):
    active,site,dest,partner=decode(actions)
    found=np.full(np.shape(actions)[:-1],-1,dtype=np.int16);matches=np.zeros(found.shape,dtype=np.int8)
    for pair,(first,second) in enumerate(PAIRS):
        matched=(active[...,first]&active[...,second]&(partner[...,first]==second)&(partner[...,second]==first)
                 &(site[...,first]==site[...,second])&(dest[...,first]==dest[...,second]))
        if not reciprocal:matched&=active.sum(-1)==2
        found[matched]=(pair*8+2*site[...,first]+dest[...,first])[matched];matches+=matched
    assert (matches<=1).all(),'A person cannot simultaneously reciprocate two people'
    return found


def structural_actions():
    rows=[]
    for first,second in PAIRS:
        for site,destination in product(range(4),range(2)):
            row=[0,0,0]
            row[first]=action(first,second,site,destination);row[second]=action(second,first,site,destination)
            rows.append(row)
    return np.asarray(rows,dtype=np.int16)


STRUCTURAL=structural_actions()


def counts(needs,layout=(0,1,2,3)):
    """Satisfied executors (0,1,2) for each of the 24 structural transports."""
    needs=np.asarray(needs);output=np.zeros((len(needs),24),dtype=np.uint8)
    for index,(pair,site,destination) in enumerate(product(range(3),range(4),range(2))):
        first,second=PAIRS[pair]
        output[:,index]=ACCEPT[needs[:,first],layout[site],destination].astype(np.uint8)+ACCEPT[needs[:,second],layout[site],destination]
    return output


def index(actions):
    return np.asarray(actions)@np.asarray([289,17,1])


def ratio(numerator,denominator):
    f=Fraction(int(numerator),int(denominator))
    return dict(numerator=int(numerator),denominator=int(denominator),fraction=str(f),value=float(f))


def proposal_roles(actions):
    active,_,_,partner=decode(actions)
    return np.where(active,partner,-1)


def bound_certificate(needs):
    needs=np.asarray(needs,dtype=np.int16);r=counts(needs);target=STRUCTURAL[(r==2).argmax(1)]
    assert ((r==2).sum(1)==1).all()
    actors=[]
    for who in range(3):
        rows=[];full=role=old=0
        for need in range(24):
            local=target[needs[:,who]==need,who];hist=np.bincount(local,minlength=17)
            roles=np.where(local==0,0,1+(local-1)%2);rolehist=np.bincount(roles,minlength=3)
            full+=int(hist[1:].max());role+=int(rolehist[1:].max());old+=int(rolehist.max())
            rows.append(dict(own_need=need,worlds=int(len(local)),target_action_counts=hist.tolist(),
                target_role_counts=rolehist.tolist(),maximum_active_action_count=int(hist[1:].max()),
                maximum_active_partner_count=int(rolehist[1:].max())))
        actors.append(dict(actor=who,active_correct_action_bound=ratio(full,len(needs)),
            active_correct_partner_bound=ratio(role,len(needs)),old_three_role_bayes_bound=ratio(old,len(needs)),by_own_need=rows))
    return dict(needs=len(needs),actors=actors,
        new_full_success_upper=ratio(sum(row['active_correct_action_bound']['numerator'] for row in actors),2*len(needs)),
        new_correct_execution_pair_upper=ratio(sum(row['active_correct_partner_bound']['numerator'] for row in actors),2*len(needs)),
        scope='Uniform needs, any public material layout and independent owner assignment; deterministic or randomized PL-silent proposals. Bounds need not be attainable.')


def role_witness(needs):
    """A constructive finite lookup witness, not an optimality certificate.

Coordinate search over deterministic own-need partner tables is a static
combinatorial calculation; this is not learned agent performance. Every
proposal uses site0/destination0, so any mutual pair physically executes.
    """
    needs=np.asarray(needs,dtype=np.int16);r=counts(needs);target_pair=(r==2).argmax(1)//8
    pair_counts=np.zeros((3,24,24),dtype=np.int64)
    for pi,(i,j) in enumerate(PAIRS):
        selected=target_pair==pi
        np.add.at(pair_counts[pi],(needs[selected,i],needs[selected,j]),1)
    rng=np.random.default_rng(81031);best=-1;chosen=None;attempts=[]
    for attempt in range(32):
        policy=np.asarray([[others[v] for v in rng.integers(0,2,24)]
            for who in range(3) for others in [[p for p in range(3) if p!=who]]])
        for iteration in range(100):
            before=policy.copy()
            for who in range(3):
                scores=np.zeros((24,3),dtype=np.int64)
                for pair,(i,j) in enumerate(PAIRS):
                    if who==i:scores[:,j]=pair_counts[pair]@(policy[j]==i)
                    if who==j:scores[:,i]=pair_counts[pair].T@(policy[i]==j)
                others=[p for p in range(3) if p!=who]
                policy[who]=np.asarray(others)[scores[:,others].argmax(1)]
            if np.array_equal(policy,before):break
        proposals=np.zeros((len(needs),3),dtype=np.int16)
        for who in range(3):
            others=[p for p in range(3) if p!=who]
            proposals[:,who]=np.where(policy[who,needs[:,who]]==others[0],1,2)
        executed=execution_index(proposals,True);correct=(executed>=0)&(executed//8==target_pair)
        value=int(correct.sum());attempts.append(dict(attempt=attempt,iterations=iteration+1,correct_pairs=value))
        if value>best:best=value;chosen=policy.copy()
    return dict(partner_by_actor_own_need=chosen.tolist(),proposal_site=0,proposal_destination=0,
        correct_execution_pair_rate=ratio(best,len(needs)),search_seed=81031,restarts=32,maximum_sweeps=100,
        attempts=attempts,not_an_optimality_claim=True,not_a_model_or_policy_training_result=True)


def audit():
    source_bytes=SOURCE.read_bytes();assert sha256(source_bytes).hexdigest()==SOURCE_SHA
    original=json.loads(source_bytes);all_needs=np.asarray(list(product(range(24),repeat=3)),dtype=np.int16)
    all_counts=counts(all_needs);needs=all_needs[(all_counts==2).sum(1)==1];r=counts(needs)
    assert len(needs)==5376
    train=np.asarray(original['partitions']['train']['needs'],dtype=np.int16)
    held=np.asarray(original['partitions']['new_needs']['needs'],dtype=np.int16)
    assert set(map(tuple,needs))==set(map(tuple,train))|set(map(tuple,held))
    assert not set(map(tuple,train))&set(map(tuple,held))
    old=execution_index(ALL_ACTIONS,False);new=execution_index(ALL_ACTIONS,True)
    assert np.array_equal(old[index(STRUCTURAL)],np.arange(24))
    assert int((old>=0).sum())==24 and int((new>=0).sum())==408
    assert np.array_equal(np.bincount(new[new>=0],minlength=24),np.full(24,17))
    layouts=[]
    for layout in permutations(range(4)):
        altered=counts(needs,layout)
        remap=np.asarray([pair*8+2*layout[site]+destination for pair,site,destination in product(range(3),range(4),range(2))])
        assert np.array_equal(altered,r[:,remap])
        layouts.append(dict(layout=list(layout),canonical_plan_permutation=remap.tolist(),exact=True))
    old_hist=np.zeros(3,dtype=np.int64);new_hist=np.zeros(3,dtype=np.int64)
    for start in range(0,len(needs),256):
        table=np.c_[r[start:start+256],np.zeros(min(256,len(needs)-start),dtype=np.uint8)]
        old_hist+=np.bincount(table[:,np.where(old>=0,old,24)].ravel(),minlength=3)
        new_hist+=np.bincount(table[:,np.where(new>=0,new,24)].ravel(),minlength=3)
    assert new_hist[1]==17*old_hist[1] and old_hist[2]==5376 and new_hist[2]==17*5376
    # Every positive old physical plan loses all execution after any one change.
    unilateral_old=[]
    for plan,row in enumerate(STRUCTURAL):
        neighboring=np.asarray([np.where(np.arange(3)==who,replacement,row)
            for who in range(3) for replacement in range(17) if replacement!=row[who]])
        assert len(neighboring)==48 and (old[index(neighboring)]==-1).all()
        unilateral_old.append(dict(plan=plan,changed_actions=48,neighbors_with_execution=0))
    physical_rows=[ALL_ACTIONS[new==plan] for plan in range(24)]
    path_counts=Counter();partial_strict_stalls=0;examples={};cases=0
    def transfer(profile,source_plan,target_plan):
        source_pair=set(PAIRS[source_plan//8]);target_pair=set(PAIRS[target_plan//8])
        assert source_pair!=target_pair
        shared=next(iter(source_pair&target_pair));newcomer=next(iter(target_pair-source_pair))
        prepared=profile.copy();prepared[:,newcomer]=STRUCTURAL[target_plan,newcomer]
        finished=prepared.copy();finished[:,shared]=STRUCTURAL[target_plan,shared]
        assert np.array_equal(new[index(prepared)],np.full(len(profile),source_plan))
        assert np.array_equal(new[index(finished)],np.full(len(profile),target_plan))
        return [prepared,finished]
    for ni,table in enumerate(r):
        full=int(np.flatnonzero(table==2)[0]);target=STRUCTURAL[full];active=target!=0
        positive_by_pair=[np.flatnonzero(table[8*pi:8*pi+8]>0)+8*pi for pi in range(3)]
        assert all(len(values)>0 for values in positive_by_pair),'Every pair can satisfy at least one nonempty need'
        for partial in np.flatnonzero(table==1):
            partial=int(partial);start=physical_rows[partial];path=[start]
            if partial//8!=full//8:path+=transfer(start,partial,full)
            else:
                other_pair=next(pi for pi in range(3) if pi!=full//8)
                bridge=int(positive_by_pair[other_pair][0]);assert table[bridge]==1
                path+=transfer(start,partial,bridge);path+=transfer(path[-1],bridge,full)
            indices=np.stack([new[index(x)] for x in path]);rewards=table[indices]
            assert (np.diff(rewards.astype(np.int16),axis=0)>=0).all() and (rewards[-1]==2).all()
            changes=np.stack([np.count_nonzero(b!=a,axis=1) for a,b in zip(path,path[1:])])
            assert (changes<=1).all()
            for value,count in zip(*np.unique(changes.sum(0),return_counts=True)):path_counts[int(value)]+=int(count)
            matched=(start[:,active]==target[active]).sum(1)
            partial_strict_stalls+=int((matched==0).sum());cases+=len(start)
            example_key='different_pair' if partial//8!=full//8 else 'same_pair_different_content'
            if example_key not in examples:
                # Last spectator choice usually illustrates the ignored active proposal.
                which=16
                examples[example_key]=dict(needs=needs[ni].tolist(),start_plan=partial,full_plan=full,
                    candidate_joint_actions=[p[which].tolist() for p in path],
                    candidate_rewards=(rewards[:,which]/2).tolist(),
                    original_rewards=[float(table[old[index(p[which])]])/2 if old[index(p[which])]>-1 else 0. for p in path],
                    full_target_actions=target.tolist(),
                    final_proposal_roles=proposal_roles(path[-1][which:which+1])[0].tolist(),
                    correct_execution_roles=proposal_roles(target[None])[0].tolist())
    assert cases==int(new_hist[1]) and max(path_counts)<=4
    # Any zero-reward profile reaches full by replacing the two true executors.
    # After first replacement reward is >=0; after second the true pair executes.
    zero_target_check=0
    for target in STRUCTURAL:
        working=ALL_ACTIONS.copy()
        for who in np.flatnonzero(target):working[:,who]=target[who]
        assert (new[index(working)]==new[index(target)]).all()
        zero_target_check+=len(working)
    certificates={name:bound_certificate(values) for name,values in (('full_domain',needs),('train',train),('heldout',held))}
    assert certificates['heldout']['new_full_success_upper']['fraction']=='19/62'
    assert certificates['heldout']['new_correct_execution_pair_upper']['fraction']=='1/2'
    witness=role_witness(held)
    assert witness['correct_execution_pair_rate']['fraction']=='23/62'
    return dict(status='completed_static_candidate_audit',candidate_not_adopted=True,
        source=dict(path=str(SOURCE),sha256=SOURCE_SHA,scope='Original static task preparation only'),
        domains=dict(all_need_triples=24**3,unique_full_plan_need_triples=5376,train_need_triples=len(train),heldout_need_triples=len(held),
            joint_action_profiles=17**3,canonical_need_action_cells=len(needs)*17**3,material_layouts=24,owner_permutations=6,
            structural_plans=24,old_executable_profiles=24,new_executable_profiles=408,
            new_profiles_per_execution_plan=17,all_layout_invariance_certificates=layouts),
        rewards=dict(values=[0,.5,1],canonical_cells_by_reward={'old':old_hist.tolist(),'candidate':new_hist.tolist()},
            old_full_support_per_need=1,new_full_support_per_need=17,partial_support_multiplier=17,
            positive_support_multiplier=17,physical_execution_support_multiplier=17),
        paths=dict(all_candidate_partial_profiles_checked=cases,nondecreasing_path_length_histogram=dict(sorted(path_counts.items())),
            all_partial_profiles_reach_full=True,maximum_unilateral_changes=4,
            candidate_partial_profiles_without_a_strictly_improving_single_change=partial_strict_stalls,
            old_strict_suboptimal_local_maxima=int(old_hist[1]),candidate_strict_suboptimal_local_maxima=0,
            original_structural_neighbor_proof=unilateral_old,zero_reward_to_full_two_changes_physical_checks=zero_target_check,
            examples=examples,interpretation='Neutral unilateral preparation is required in many states; existence of a nondecreasing path is not a policy-gradient convergence guarantee.'),
        silent_bounds=certificates,heldout_actual_pair_lookup_witness=witness,
        unresolved='The exact optimum for PL-silent correct executed pair is not established: constructive lower bound 23/62, proven upper bound 1/2. Do not reuse 23/62 as a proven cap on this changed role metric.',
        metrics=dict(proposal_role='All three proposed partner/wait values; spectator mismatch can coexist with native full success.',
            execution_role='Actual executed mutually matched pair, inactive third; compare with the unique full-success pair.',
            native_full_success='Both executed needs satisfied, requiring correct pair, material/site and destination.',
            reward='Number of satisfied actual executors divided by 2; ignored third actor is never satisfied or rewarded.'),
        neural_forward_samples=0,model_parameter_loads=0,agent_training_updates=0,
        static_lookup_search='32 deterministic coordinate-search restarts certify one feasible role-pair lookup only, not optimality or learned-agent performance.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    output=Path(args.out);assert not output.exists();start=time.perf_counter()
    result=audit();result.update(source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start)
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','domains','rewards','paths','elapsed_seconds') if k not in ('domains','paths')},ensure_ascii=False))


if __name__=='__main__':main()
