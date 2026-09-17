"""At most one single-thread 60s HiGHS MILP per ecology. No neural policies."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
import warnings
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, save_npz, load_npz
import scipy.optimize._highspy._core as highscore

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parent.parent
OPTIONS={'disp':True,'presolve':True,'time_limit':60.0,'mip_rel_gap':0.0,
         'mip_abs_gap':0.0,'threads':1,'random_seed':20260916}
ROLES=[['wait']+[b for b in 'ABC' if b!=a] for a in 'ABC']


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
def rate(f):return {'fraction':str(f),'numerator':f.numerator,'denominator':f.denominator,'float':float(f)}


def bits_compatible(needs,i,j):
    resources=(3,12,5,10);destinations=(1,2,3)
    return bool(resources[needs[i]//3]&resources[needs[j]//3]) and bool(destinations[needs[i]%3]&destinations[needs[j]%3])


def supports():
    layers=[]
    for d in product(range(3),repeat=3):
        tables={'unique':[],'multiple':[]}
        for resources in product(range(4),repeat=3):
            n=tuple(3*resources[i]+d[i] for i in range(3))
            pairs=[p for p in combinations(range(3),2) if bits_compatible(n,*p)]
            if pairs:tables['unique' if len(pairs)==1 else 'multiple'].append(list(n))
        if all(tables.values()):layers.append({'destinations':list(d),'supports':tables})
    assert len(layers)==21
    return layers


def x_index(agent,need,role):return (agent*12+need)*3+role


def construct(layers,ecology):
    multiple=math.lcm(*(len(x['supports'][ecology]) for x in layers))
    states=[];zs=[]
    for di,layer in enumerate(layers):
        count=len(layer['supports'][ecology])
        for n in layer['supports'][ecology]:
            si=len(states);states.append({'D_index':di,'needs':n,'integer_weight':multiple//count})
            for i,j in combinations(range(3),2):
                if bits_compatible(n,i,j):zs.append({'state_index':si,'pair':[i,j],'integer_weight':multiple//count})
    nvars=108+len(zs);rows=[];cols=[];values=[];lb=[];ub=[]
    def constraint(entries,lower,upper):
        k=len(lb)
        for c,v in entries:rows.append(k);cols.append(c);values.append(v)
        lb.append(lower);ub.append(upper)
    for a,n in product(range(3),range(12)):
        constraint([(x_index(a,n,r),1.) for r in range(3)],1.,1.)
    c=np.zeros(nvars)
    for zi,z in enumerate(zs):
        i,j=z['pair'];k=({0,1,2}-{i,j}).pop();n=states[z['state_index']]['needs']
        xids=(x_index(i,n[i],ROLES[i].index('ABC'[j])),x_index(j,n[j],ROLES[j].index('ABC'[i])),x_index(k,n[k],0))
        for xid in xids:constraint([(108+zi,1.),(xid,-1.)],-np.inf,0.)
        c[108+zi]=-z['integer_weight']
    matrix=coo_matrix((values,(rows,cols)),shape=(len(lb),nvars)).tocsc()
    integrality=np.zeros(nvars,dtype=np.uint8);integrality[:108]=1
    assert sum(s['integer_weight'] for s in states)==21*multiple
    return {'ecology':ecology,'states':states,'z_variables':zs,'integer_denominator':21*multiple,
            'choice_variables':108,'continuous_z_variables':len(zs),'constraints':len(lb)}, matrix, c,integrality,np.asarray(lb),np.asarray(ub)


def prepare():
    out=HERE/'prepared'
    out.mkdir(exist_ok=False)
    layers=supports()
    audit=STUDY/'design_audit_002/verification.json'
    old=STUDY/'design_audit_002/prepared_inspected.json'
    assert read(audit)['status']=='passed'
    assert sha(STUDY/'design.py')==read(audit)['design_sha256']
    assert layers==read(old)['common_destination_layers']
    files={str(p):sha(p) for p in [Path(__file__),HERE/'方案与约化证明.md',STUDY/'design.py',audit,old,HERE/'install.log']}
    artifacts={};summaries=[]
    for ecology in ('unique','multiple'):
        info,A,c,integrality,lb,ub=construct(layers,ecology)
        write(out/f'{ecology}_inputs.json',info)
        save_npz(out/f'{ecology}_matrix.npz',A)
        np.savez(out/f'{ecology}_arrays.npz',c=c,integrality=integrality,lb=lb,ub=ub)
        for suffix in ('inputs.json','matrix.npz','arrays.npz'):
            p=out/f'{ecology}_{suffix}';artifacts[str(p)]=sha(p)
        summaries.append({k:info[k] for k in ('ecology','integer_denominator','choice_variables','continuous_z_variables','constraints')})
    solver_version='.'.join(str(getattr(highscore,'HIGHS_VERSION_'+k)) for k in ('MAJOR','MINOR','PATCH'))
    manifest={'status':'prepared_no_solves','prepared_at_utc':now(),'sources_sha256':files,'artifacts_sha256':artifacts,
        'options':OPTIONS,'order':['unique','multiple'],'max_solver_calls':2,'seconds_per_solver_call':60.,'threads':1,
        'integer_objective_bound_safety_margin':1e-5,'solution_integrality_tolerance':1e-7,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                   'highs':solver_version,'executable':sys.executable,'platform':platform.platform()},
        'problem_sizes':summaries,'role_labels':ROLES,'model_forward_calls':0,'training_updates':0}
    write(out/'manifest.json',manifest)
    write(HERE/'freeze.json',{'manifest_sha256':sha(out/'manifest.json')})
    print(json.dumps({'status':'prepared_no_solves','manifest_sha256':sha(out/'manifest.json'),'sizes':summaries,'runtime':manifest['runtime']},ensure_ascii=False))


def exact_score(info,role_map):
    accepted_resources=({0,1},{2,3},{0,2},{1,3})
    accepted_destinations=({0},{1},{0,1})
    total=Fraction(0);numerator=0;rows=[];per_D=[Fraction(0) for _ in range(21)];pair_mass={p:Fraction(0) for p in ('AB','AC','BC')}
    for state in info['states']:
        needs=state['needs'];chosen=[ROLES[a][role_map[a][needs[a]]] for a in range(3)]
        active=[a for a,role in enumerate(chosen) if role!='wait'];success=False;pair=None
        if len(active)==2:
            i,j=active
            mutual=chosen[i]=='ABC'[j] and chosen[j]=='ABC'[i]
            compatible=any(m in accepted_resources[needs[i]//3] and m in accepted_resources[needs[j]//3]
                and d in accepted_destinations[needs[i]%3] and d in accepted_destinations[needs[j]%3]
                for m,d in product(range(4),range(2)))
            success=bool(mutual and compatible)
            if mutual:pair=''.join('ABC'[a] for a in active)
        weight=Fraction(state['integer_weight'],info['integer_denominator'])
        if success:
            total+=weight;numerator+=state['integer_weight'];per_D[state['D_index']]+=weight;pair_mass[pair]+=weight
        rows.append({**state,'chosen_roles':chosen,'compatibility_role_success':success,'mutual_pair':pair,'weight':str(weight)})
    assert total==Fraction(numerator,info['integer_denominator'])
    return {'score':rate(total),'integer_objective':numerator,'D_conditional_scores':[rate(v*21) for v in per_D],
            'successful_pair_target_mass':{p:rate(v) for p,v in pair_mass.items()},'states':rows}


def solve(ecology):
    freeze=read(HERE/'freeze.json');manifest=read(HERE/'prepared/manifest.json')
    assert sha(HERE/'prepared/manifest.json')==freeze['manifest_sha256']
    assert manifest['options']==OPTIONS
    for p,h in {**manifest['sources_sha256'],**manifest['artifacts_sha256']}.items():assert sha(p)==h,p
    out=HERE/f'solve_{ecology}'
    out.mkdir(exist_ok=False)
    if ecology=='multiple':assert (HERE/'solve_unique/result.json').exists() or (HERE/'solve_unique/failure.json').exists()
    write(out/'started.json',{'started_at_utc':now(),'pid':os.getpid(),'ecology':ecology,'manifest_sha256':freeze['manifest_sha256'],'options':OPTIONS})
    info=read(HERE/f'prepared/{ecology}_inputs.json')
    A=load_npz(HERE/f'prepared/{ecology}_matrix.npz')
    arrays=np.load(HERE/f'prepared/{ecology}_arrays.npz',allow_pickle=False)
    started=time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=milp(c=arrays['c'],integrality=arrays['integrality'],bounds=Bounds(0.,1.),
                        constraints=LinearConstraint(A,arrays['lb'],arrays['ub']),options=OPTIONS)
        elapsed=time.perf_counter()-started
        write(out/'warnings.json',[{'category':w.category.__name__,'message':str(w.message)} for w in caught])
        raw={}
        for k,v in result.items():
            if isinstance(v,np.ndarray):continue
            if isinstance(v,np.generic):v=v.item()
            raw[k]=v
        write(out/'solver_result.json',raw)
        summary={'status':'solver_returned','ecology':ecology,'solver_calls':1,'elapsed_seconds':elapsed,
                 'runtime':manifest['runtime'],'options':OPTIONS,'raw':raw,'feasible_map_returned':False,
                 'certified_optimal_by_solver':False,'integer_lattice_gap_closed_with_safety_margin':False,
                 'model_forward_calls':0,'training_updates':0}
        if result.x is not None:
            np.save(out/'solution_vector.npy',result.x)
            x=result.x[:108].reshape(3,12,3)
            assert np.max(np.abs(x-np.rint(x)))<=manifest['solution_integrality_tolerance']
            binary=np.rint(x).astype(int)
            assert np.array_equal(binary.sum(axis=-1),np.ones((3,12),dtype=int))
            role_map=np.argmax(binary,axis=-1).tolist()
            check=exact_score(info,role_map)
            assert abs(check['integer_objective']+float(result.fun))<1e-5
            write(out/'exact_evaluation.json',check)
            write(out/'role_map.json',{'role_indices':role_map,'role_labels':ROLES,
                'maps':{a:[{'need_id':n,'resource_requirement':('wood','fiber','short','long')[n//3],
                           'destination_requirement':('L','R','L_or_R')[n%3],
                           'role':ROLES[i][role_map[i][n]]} for n in range(12)] for i,a in enumerate('ABC')}})
            summary.update(feasible_map_returned=True,exact_score=check['score'],integer_objective=check['integer_objective'],
                certified_optimal_by_solver=bool(result.status==0 and result.success))
            dual=getattr(result,'mip_dual_bound',None)
            if dual is not None and math.isfinite(dual):
                upper=-float(dual);safe_upper=math.floor(upper+manifest['integer_objective_bound_safety_margin'])
                assert safe_upper>=check['integer_objective']
                summary.update(solver_maximization_upper_integer_objective=upper,
                    solver_upper_rate=upper/info['integer_denominator'],
                    conservative_integer_upper=rate(Fraction(min(safe_upper,info['integer_denominator']),info['integer_denominator'])),
                    integer_lattice_gap_closed_with_safety_margin=safe_upper==check['integer_objective'])
        for p,h in manifest['sources_sha256'].items():assert sha(p)==h,p
        summary['completed_at_utc']=now()
        write(out/'result.json',summary)
        print(json.dumps(summary,ensure_ascii=False))
    except Exception as error:
        write(out/'failure.json',{'status':'failed','ecology':ecology,'error_type':type(error).__name__,'error':str(error),
                                'elapsed_seconds':time.perf_counter()-started,'solver_retries':0})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','unique','multiple'))
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:solve(args.command)
