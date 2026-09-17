"""Independent PL live/silent completed-run audit, frozen before new training.

No new production runner, cases, environment or kernel is imported. All endpoint
networks are replayed through pinned old independent PL features and two-window
routing. The pinned independent response audit reconstructs cases and Q.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse,json,platform,re,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
from research_program.triadic_need_response_study import audit_response as response

prior=response.prior;env=prior.env;message=prior.message;old=prior.old
require,read,sha,array_sha,json_bytes,close,compare,finite_tree=prior.require,prior.read,prior.sha,prior.array_sha,prior.json_bytes,prior.close,prior.compare,prior.finite_tree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=(59101,59102,59103,59104);CONDITIONS=('PL_live','PL_silent');PARTS=env.PARTS;STEPS=env.STEPS
RESPONSE_SHA='589ca516ff962f56f3bda0e23d253bd48a02b2e513d03d84d310eae1d51527ac'
ORIGINAL_SHA=response.ORIGINAL_SHA
PLAN_SHA='b5126ca468e16a41db4ee9e96c77f0a7e465c3c2123589d2f423911fc5bf5555'


def references():
    refs=response.references();path=Path(response.__file__).resolve()
    require(sha(path)==RESPONSE_SHA,'Pinned independent response source');refs[str(path)]=RESPONSE_SHA;return refs


def name(seed,condition):return f'seed_{seed}_{condition}'


def evaluate_saved(entry,path,states,indices,live,mode,networks=None,cases=None):
    path=Path(path);ids=np.asarray(indices,np.int64);n=len(ids)
    require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['data_sha256'],'Evaluation path/hash')
    require(type(live) is bool and mode in ('natural','closed') and (mode!='closed' or not live),'Expected channel intervention')
    require(entry['rule']=='reciprocal' and entry['information']=='PL' and entry['live'] is live and entry['mode']==mode,'PL evaluation identity')
    require(entry['intervention']==('close_cross_agent_channel_from_window_1' if mode=='closed' else None),'Intervention identity')
    value=env.load_npz(path)
    require(value['states'].dtype==np.int16 and np.array_equal(value['states'],states[ids]),'Evaluation ordered states')
    require(value['state_indices'].dtype==np.int64 and np.array_equal(value['state_indices'],ids),'Evaluation ordered indices')
    data=prior.settle(states[ids],value['action_indices'],'reciprocal')
    keys=('conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability','conditional_full_posterior_mass')
    require(set(value)==set(data)|set(keys)|{'states','state_indices','messages','action_indices','action_probabilities'},'Exact evaluation schema')
    for key,v in data.items():require(value[key].dtype==v.dtype and np.array_equal(value[key],v),'Independent settlement '+key)
    p=value['action_probabilities'];a=value['action_indices'];m=value['messages']
    require(a.dtype==np.int16 and a.shape==(n,3),'Action dtype/shape')
    require(m.dtype==np.int8 and m.shape==(n,2,3,4) and np.all((m>=0)&(m<8)),'Message dtype/domain')
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Probability dtype/domain')
    require(np.array_equal(p.argmax(-1),a),'Greedy action identity')
    for key in keys:require(value[key].dtype==np.float64 and value[key].shape==(n,) and np.isfinite(value[key]).all(),'Conditional dtype/domain '+key)
    if cases is not None:response.validate_partition(cases,value['states'],value['state_indices'])
    else:require('need_response' not in entry,'No monitor Q')
    max_p=max_stat=0.;ties=0
    for begin in range(0,n,1024):
        sl=slice(begin,begin+1024);chunk=states[ids[sl]];native=env.rewards(chunk);logs=None
        if networks is not None:
            mm,pp,logs,tt=old.forward_final(networks,chunk,'PL',live);ties+=tt
            require(np.array_equal(mm,m[sl]) and np.array_equal(pp.argmax(-1),a[sl]),'Complete independent PL route/discrete replay')
            max_p=max(max_p,close(pp,p[sl],'Complete final probability replay'))
        terms=prior.conditional_statistics(p[sl],native,'reciprocal',logs)
        for key in keys:max_stat=max(max_stat,close(value[key][sl],terms[key],'Independent conditional '+key))
    metrics=prior.summarize(states[ids],a,'reciprocal')
    metrics.update(information='PL',live=live,mode=mode,intervention='close_cross_agent_channel_from_window_1' if mode=='closed' else None,
        expected_reward_given_greedy_messages=float(value['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(value['conditional_exact_full_success_probability'].mean()),
        execution_probability_given_greedy_messages=float(value['conditional_exact_execution_probability'].mean()),
        full_posterior_mass_given_greedy_messages=float(value['conditional_full_posterior_mass'].mean()),state_indices_sha256=array_sha(ids))
    if cases is not None:metrics['need_response']=response.metrics(cases,data['actual_pair_index'])
    compare(entry,metrics,'Independent PL evaluation summary/')
    return value,dict(sha256=sha(path),worlds=n,max_probability_error=max_p,max_conditional_error=max_stat,
        independent_forward_worlds=n if networks is not None else 0,independent_network_samples=9*n if networks is not None else 0,
        independent_final_sender_ties=ties if networks is not None else 0,metrics=metrics)


def training_row(row,seed,condition,update,expected):
    require((row['seed'],row['condition'],row['rule'],row['update'])==(seed,condition,'reciprocal',update),'PL training identity')
    # The old independent audit's same-kernel algebra is condition-invariant.
    checked=dict(row,condition='FI_silent');prior.training_row(checked,seed,'reciprocal',update,expected)


def primary(values):
    require(set(values)==set(product(SEEDS,CONDITIONS)),'Exactly eight natural primary cells')
    rows=[]
    for seed in SEEDS:
        live=values[seed,'PL_live']['need_response']['Q'];silent=values[seed,'PL_silent']['need_response']['Q']
        require(all(isinstance(v,(float,int)) and not isinstance(v,bool) and np.isfinite(v) and 0<=v<=1 for v in (live,silent)),'Defined Q')
        rows.append(dict(seed=seed,Q_live=live,Q_silent=silent,difference=live-silent))
    return dict(metric='Q',partition='new_needs_and_layouts',contrast='PL_live_minus_PL_silent',independent_paired_seed_blocks=4,
        paired_seeds=rows,mean_difference=float(np.mean([r['difference'] for r in rows])))


def budget(specs,cases):
    worlds=sum(s['world_count'] for s in specs.values());mon=sum(len(s['monitor_indices']) for s in specs.values())
    require(worlds==774144 and mon==21504,'Fixed original world support')
    return dict(runs=8,training_updates=48000,training_world_samples=48000*256,message_trajectories=48000*256*2,
        categorical_symbol_samples=48000*256*2*24,training_forward_module_samples=48000*256*2*9,checkpoints=48,
        natural_monitor_files=192,natural_final_files=32,closed_final_files=16,actual_monitor_files=192,actual_final_files=48,aliased_evaluations=0,
        actual_monitor_worlds=8*6*mon,natural_final_worlds=8*worlds,closed_final_worlds=4*worlds,actual_final_worlds=12*worlds,
        monitor_forward_module_samples=8*6*mon*9,final_forward_module_samples=12*worlds*9,
        evaluation_forward_module_samples=(8*6*mon+12*worlds)*9,need_response_full_partition_evaluations=48,
        need_response_edge_background_evaluations=12*sum(c['state_edges'] for c in cases.values()),need_response_additional_forward_samples=0)


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution';refs=references();artifacts=dict(refs)
    require(expected_plan_sha==PLAN_SHA,'Only the pretraining frozen plan is accepted')
    require(read(execution/'status.json')['status']=='completed' and not (execution/'failure.json').exists(),'No incomplete-run audit')
    result=read(execution/'results.json');finite_tree(result);plan=read(run/'plan.json');prepared=read(run/'prepared.json');freeze=read(run/'freeze.json')
    require(result['status']=='completed' and len(result['runs'])==8,'Completed eight runs')
    require(sha(run/'plan.json')==expected_plan_sha==freeze['plan_sha256']==result['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==freeze['prepared_sha256']==plan['prepared_sha256'],'Prepared chain')
    require(read(execution/'started.json')['plan_sha256']==expected_plan_sha,'Started plan')
    require(read(execution/'status.json')['results_sha256']==sha(execution/'results.json'),'Completion hash')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Numerical runtime')
    compare(plan['config'],dict(seeds=list(SEEDS),conditions=list(CONDITIONS),execution_rule='reciprocal',partitions=list(PARTS),
        updates=6000,batch_size=256,checkpoints=list(STEPS),features=54,dtype='float64',learning_rate=.001,global_gradient_clip=5.,
        entropy_initial=.001,entropy_zero_after_updates=1000,sender_entropy_coefficient=0,trajectories_per_state=2,sender_windows=2,sender_tokens_per_window=4,
        actions_per_actor=17,worker_count=4,multiprocessing_start_method='spawn',dimensions=dict(sender1=[54,64,64,32],sender2=[153,64,64,32],action=[252,64,64,17]),
        primary='6000_full_double_holdout_Q_PL_live_minus_PL_silent_then_mean_four_paired_seeds',
        information='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
        channel_closure='from_window_1_cross_actor_messages_and_visibility_bits_zero; own_messages_retained'), 'Frozen core configuration/')
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:require(digest==expected,'Changed artifact '+str(path))
        artifacts[str(path)]=digest;return digest
    for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json',HERE/'audit_freeze_001.json'):bind(path)
    af=read(HERE/'audit_freeze_001.json')
    require(af['status']=='frozen_before_main_training' and af['plan_sha256']==expected_plan_sha,'Pretraining independent audit freeze')
    require(set(af['source_sha256'])=={str(HERE/'audit_execution.py'),str(HERE/'tests/test_audit.py'),str(HERE/'audit_preflight_001.json')},'Audit source coverage')
    for p,d in af['source_sha256'].items():bind(p,d)
    require(af['at']<read(execution/'started.json')['at'],'Audit freeze must precede actual main start')
    for p,d in plan['source_sha256'].items():bind(p,d);bind(run/'source_snapshot'/Path(p).relative_to(ROOT),d)
    for p,d in plan['inputs_sha256'].items():bind(p,d)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json';bind(original,ORIGINAL_SHA)
    require(prepared['source_prepared_sha256']==ORIGINAL_SHA and prepared['schema']=='triadic_private_partner_v1' and prepared['independent_initializations']==4,'Prepared source/schema')
    independent,_,_=env.independent_specs();specs=read(original)['partitions']
    require(prepared['partitions']==specs,'Anchored all partition metadata')
    for part in PARTS:compare(specs[part],independent[part],'Independent core support/')
    cases={p:response.build_cases(specs[p]) for p in PARTS}
    compare(prepared['need_response_cases'],cases,'Independent full case support/')
    require(prepared['budget']==result['budget']==budget(specs,cases),'Exact planned budget')
    require([(r['seed'],r['condition']) for r in result['runs']]==list(product(SEEDS,CONDITIONS)),'Canonical eight paired runs')
    states={p:env.packed(specs[p]) for p in PARTS};hashes={}
    for part,s in states.items():
        x=env.features(s,'PL');require(np.all(x[:,:,53]==0),'PL FI flag hidden')
        for viewer,other in product(range(3),repeat=2):
            if viewer!=other:require(np.all(x[:,viewer,7*other:7*other+7]==0),'Other needs hidden')
        hashes[part]=dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)),x_PL=array_sha(x))
    del x
    require(result['array_hashes']==hashes,'Complete independent PL arrays')
    for seed in SEEDS:
        path=execution/f'seed_{seed}_arrays.json';bind(path);require(read(path)['array_hashes']==hashes,'Worker arrays')
    by={(r['seed'],r['condition']):r for r in result['runs']};counts=Counter();errors=defaultdict(float)
    evaluations={};final_metrics={};initial_hashes={};ckpt_paths=set();evaluation_paths=set()
    for seed in SEEDS:
        rng_at={};final_nets={};initial=[]
        for condition in CONDITIONS:
            row=by[seed,condition];directory=execution/name(seed,condition)
            require(row['rule']=='reciprocal' and row['updates']==6000,'Run identity')
            for file in ('result.json','training.jsonl','monitor.jsonl','status.json'):bind(directory/file)
            require(read(directory/'result.json')==row,'Per-run result identity')
            require(read(directory/'status.json')['status']=='completed' and read(directory/'status.json')['result_sha256']==sha(directory/'result.json'),'Per-run completion')
            require(sha(directory/'training.jsonl')==row['training_log_sha256'],'Training log hash')
            monitor=[json.loads(line) for line in (directory/'monitor.jsonl').read_text().splitlines()]
            require(monitor==row['monitor'] and [m['update'] for m in monitor]==list(STEPS),'Fixed monitor grid')
            rng_at[condition]={}
            for step,m in zip(STEPS,monitor):
                path=directory/f'checkpoint_{step:04d}.npz';bind(path,m['checkpoint_sha256']);ckpt_paths.add(path)
                nets,wr,mr=message.checkpoint(path,step,seed);rng_at[condition][step]=(wr,mr);counts['checkpoints']+=1
                if step==0:
                    h=message.network_hash(nets);require(h==row['initial_parameter_sha256'],'Initial parameters');initial.append(h)
                if step==6000:
                    require(message.network_hash(nets)==row['final_parameter_sha256'] and sha(path)==row['final_checkpoint_sha256'],'Final parameter identity');final_nets[condition]=nets
        require(len(set(initial))==1,'Paired initialization');initial_hashes[seed]=initial[0]
        wr=np.random.default_rng(np.random.SeedSequence([seed,200]));mr={f't{t}_w{w}_a{a}':np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300])) for t,w,a in product(range(2),range(2),range(3))}
        def rng_check(step):
            now=(wr.bit_generator.state,{k:g.bit_generator.state for k,g in mr.items()})
            for condition in CONDITIONS:require(rng_at[condition][step]==now,'Checkpoint world/message stream state')
        streams=[(execution/name(seed,c)/'training.jsonl').open() for c in CONDITIONS]
        try:
            rng_check(0)
            for update in range(1,6001):
                u=wr.random((256,3));ids=env.sample(specs['train'],u);mu=np.empty((2,256,2,3,4),np.float64)
                for t,w,a in product(range(2),range(2),range(3)):mu[t,:,w,a]=mr[f't{t}_w{w}_a{a}'].random((256,4))
                expected=dict(world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),batch_states_sha256=array_sha(states['train'][ids]),sample_uniforms_sha256=array_sha(mu))
                for condition,stream in zip(CONDITIONS,streams):
                    line=stream.readline();require(bool(line),'Missing training row');training_row(json.loads(line),seed,condition,update,expected);counts['training_rows']+=1
                if update in STEPS:rng_check(update)
            require(all(stream.read()=='' for stream in streams),'Excess training rows')
        finally:
            for stream in streams:stream.close()
        initial_w1={}
        for condition in CONDITIONS:
            row=by[seed,condition];directory=execution/name(seed,condition);end_monitor={};live=condition=='PL_live'
            for checkpoint in row['monitor']:
                step=checkpoint['update'];require(set(checkpoint['monitor'])==set(PARTS),'Monitor parts')
                for part in PARTS:
                    ids=np.asarray(specs[part]['monitor_indices'],np.int64);path=directory/f'monitor_{step:04d}_{part}.npz'
                    value,receipt=evaluate_saved(checkpoint['monitor'][part],path,states[part],ids,live,'natural')
                    bind(path,receipt['sha256']);evaluation_paths.add(path);evaluations[str(path)]=receipt;counts['monitor_files']+=1;counts['monitor_worlds']+=len(ids)
                    errors['conditional']=max(errors['conditional'],receipt['max_conditional_error'])
                    if step==0:
                        if live:initial_w1[part]=value['messages'][:,0].copy()
                        else:require(np.array_equal(initial_w1[part],value['messages'][:,0]),'Paired initial first-window messages')
                    if step==6000:end_monitor[part]=value
            require(set(row['final'])==set(PARTS),'Final parts')
            for part in PARTS:
                require(set(row['final'][part])==({'natural','closed'} if live else {'natural'}),'Exact natural/closed evaluation grid')
                ids=np.arange(specs[part]['world_count'],dtype=np.int64);natural_w1=None
                for mode in (('natural','closed') if live else ('natural',)):
                    path=directory/f'final_{part}_{mode}.npz';route=live if mode=='natural' else False
                    value,receipt=evaluate_saved(row['final'][part][mode],path,states[part],ids,route,mode,final_nets[condition],cases[part])
                    bind(path,receipt['sha256']);evaluation_paths.add(path);evaluations[str(path)]=receipt;counts['final_files']+=1;counts[mode+'_final_files']+=1
                    counts['response_edge_background_evaluations']+=cases[part]['state_edges']
                    for key in ('independent_forward_worlds','independent_network_samples','independent_final_sender_ties'):counts[key]+=receipt[key]
                    for key in ('conditional','probability'):errors[key]=max(errors[key],receipt['max_'+key+'_error'])
                    if mode=='natural':
                        natural_w1=value['messages'][:,0].copy()
                        for key,v in end_monitor[part].items():
                            selected=value[key][specs[part]['monitor_indices']]
                            if v.dtype.kind=='f':close(v,selected,'Final monitor subset '+key)
                            else:require(np.array_equal(v,selected),'Final monitor subset '+key)
                        if part=='new_needs_and_layouts':final_metrics[seed,condition]=receipt['metrics']
                    else:require(np.array_equal(natural_w1,value['messages'][:,0]),'Same-policy channel closure leaves W1 messages unchanged')
                    print(json.dumps(dict(stage='final_audited',seed=seed,condition=condition,part=part,mode=mode)),flush=True)
            require(set(directory.glob('*.npz'))=={p for p in ckpt_paths|evaluation_paths if p.parent==directory},'Unexpected NPZ')
        print(json.dumps(dict(stage='paired_seed_audited',seed=seed)),flush=True)
    require(len(set(initial_hashes.values()))==4,'Four distinct initializations')
    compare(dict(counts),dict(training_rows=48000,checkpoints=48,monitor_files=192,monitor_worlds=1032192,final_files=48,natural_final_files=32,closed_final_files=16,
        independent_forward_worlds=9289728,independent_network_samples=83607552,response_edge_background_evaluations=3027456),'Complete independent scope/')
    require({p for p in execution.glob('seed_*_*') if p.is_dir()}=={execution/name(s,c) for s,c in product(SEEDS,CONDITIONS)},'Exact run directories')
    calculated=primary(final_metrics);compare(result['primary'],calculated,'Independent unique primary/')
    for p,d in artifacts.items():require(sha(p)==d,'Artifact changed during audit '+p)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,scope=dict(counts),max_errors=dict(errors),
        runtime=dict(python=platform.python_version(),numpy=np.__version__),historical_helpers_sha256=refs,artifacts_sha256=artifacts,evaluations=evaluations,primary=calculated,
        limits=['No optimizer/gradient/intermediate-checkpoint forward replay; all world/message uniform streams reconstructed.',
            'Training objective rows algebraically/range checked; sampled-message hashes are not independently regenerated.',
            'All48 complete endpoints freshly replayed with PL observations and legal live/closed routing; monitor statistics recomputed from saved probabilities.',
            'Q cases and exact shuffle reconstructed independently; edges/backgrounds are repeated observations, not replication units.',
            'Primary estimates reciprocal-rule communication opportunity effect within four fresh paired societies; not rule-by-communication interaction or evidence of language/composition.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    out=Path(a.out).resolve();require(not out.exists(),'Never overwrite audit');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),run=a.run,expected_plan_sha256=a.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        v=audit(a.run,a.plan_sha);v.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(v));print(json.dumps({k:v[k] for k in ('status','elapsed_seconds','scope','max_errors')}))
    except BaseException as e:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)));raise

if __name__=='__main__':main()
