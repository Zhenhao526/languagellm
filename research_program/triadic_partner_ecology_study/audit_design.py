"""No-network/no-actor independent finite-domain preparation audit."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
from pathlib import Path
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from research_program.triadic_partner_ecology_study import design


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def independent_layers():
    resources = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
    destinations = ({0}, {1}, {0, 1})
    rows, excluded = [], []
    for d in product(range(3), repeat=3):
        support = {'unique': [], 'multiple': []}
        for r in product(range(4), repeat=3):
            edges = sum(bool(resources[r[i]] & resources[r[j]]) and
                        bool(destinations[d[i]] & destinations[d[j]])
                        for i, j in combinations(range(3), 2))
            if edges:
                support['unique' if edges == 1 else 'multiple'].append([3*r[i]+d[i] for i in range(3)])
        row = {'destinations': list(d), 'supports': support}
        (rows if all(support.values()) else excluded).append(row)
    return rows, excluded


def unpack(spec, ids):
    no, nl = len(spec['private_sites']), len(spec['layouts'])
    out = []
    for index in ids:
        q, owner = divmod(int(index), no)
        need, layout = divmod(q, nl)
        out.append((tuple(n % 3 for n in spec['needs'][need]), layout, owner, need))
    return out


def reference_index(spec, u):
    di = int(math.floor(float(u[0])*21))
    layer = spec['demand_strata'][di]
    need = layer['need_indices'][int(math.floor(float(u[1])*len(layer['need_indices'])))]
    layout = int(math.floor(float(u[2])*len(spec['layouts'])))
    owner = int(math.floor(float(u[3])*6))
    # Reconstruct lexicographic Cartesian index without calling source pairing.
    return need*(len(spec['layouts'])*6) + layout*6 + owner


def main():
    out = HERE/'design_audit_001'
    out.mkdir(exist_ok=False)
    source = Path(design.__file__)
    source_hash = sha(source)
    old = ROOT/'research_program/triadic_message_study/results/messages_001/prepared.json'
    old_source = ROOT/'research_program/triadic_task/environment.py'
    before = {str(p): sha(p) for p in (source, old, old_source, Path(__file__))}
    # Block constructor, neural forward, gradient and optimizer paths in this
    # audit process. Preparation remains permitted; source files stay untouched.
    blocked_calls = []
    def blocked(*args, **kwargs):
        blocked_calls.append(True)
        raise AssertionError('Unexpected actor/forward/optimizer call')
    for module, names in [(design.r, ['make_networks', 'make_network', 'rollout', 'training_gradients']),
                          (design.r.base, ['make_actor', 'actor_forward', 'actor_backward', 'make_adam', 'adam_step', 'build_arrays'])]:
        for name in names:
            setattr(module, name, blocked)
    try:
        prepared = design.make_prepared()
        assert prepared == design.make_prepared()
        expected_layers, excluded = independent_layers()
        assert expected_layers == prepared['common_destination_layers']
        assert excluded == prepared['excluded_destination_layers']
        assert len(expected_layers) == 21 and len(excluded) == 6
        old_prepared = json.loads(old.read_text())['partitions']
        assert prepared['seeds'] == [49101, 49102, 49103, 49104]
        grid = [(seed, ecology, condition) for seed in prepared['seeds']
                for ecology in ('unique', 'multiple') for condition in ('FI_silent','FI_live','PI_silent','PI_live')]
        assert [(r['seed'],r['ecology'],r['condition']) for r in prepared['runs']] == grid
        assert len(grid) == len(set(grid)) == 32
        assert prepared['training_state_samples_total'] == 32*6000*256
        assert prepared['sampled_complete_message_trajectories_total'] == 32*6000*256*2
        assert prepared['full_natural_worlds_total'] == 16*(324+996)*144
        assert prepared['full_closed_worlds_total'] == 8*(324+996)*144
        monitor_records = {}; reports=[]; raw_arrays={}; all_full=all_boundary=0
        independent_marginals={}
        for ecology in ('unique', 'multiple'):
            need_list=[n for layer in expected_layers for n in layer['supports'][ecology]]
            margins=[[Fraction(0) for _ in range(12)] for _ in range(3)]
            for layer in expected_layers:
                w=Fraction(1,21*len(layer['supports'][ecology]))
                for needs in layer['supports'][ecology]:
                    for a,need in enumerate(needs):margins[a][need]+=w
            independent_marginals[ecology]=[[str(v) for v in row] for row in margins]
            assert independent_marginals[ecology] == prepared['exact_personal_need_marginals'][ecology]
            for pi,part in enumerate(('train','heldout_layouts')):
                spec=prepared['partitions'][ecology][part]
                assert spec['needs']==need_list
                assert spec['layouts']==old_prepared['train' if part=='train' else 'new_layouts']['layouts']
                assert spec['private_sites']==[list(p) for p in permutations((1,2,3))]
                nl=len(spec['layouts']);nphys=nl*6;n=len(need_list)*nphys
                assert spec['world_count']==n
                if part=='heldout_layouts':
                    assert not set(map(tuple,spec['layouts'])) & set(map(tuple,prepared['partitions'][ecology]['train']['layouts']))
                centers=[]; expected_weights=[]; strata_for_need=[]
                for di,layer in enumerate(expected_layers):
                    count=len(layer['supports'][ecology])
                    expected_weights.extend([float(Fraction(1,21*count*nphys))]*(count*nphys))
                    for rank in range(count):
                        strata_for_need.append(di)
                        for li,oi in product(range(nl),range(6)):
                            centers.append([(di+.5)/21,(rank+.5)/count,(li+.5)/nl,(oi+.5)/6])
                centers=np.asarray(centers)
                ids=design.sample_indices(spec,centers)
                assert np.array_equal(ids,np.arange(n,dtype=np.int64))
                all_full+=n
                fields=design.pairing_fields(spec,ids)
                independent=unpack(spec,ids)
                assert np.array_equal(fields['destinations'],np.asarray([r[0] for r in independent]))
                assert np.array_equal(fields['layout_indices'],np.asarray([r[1] for r in independent]))
                assert np.array_equal(fields['owner_indices'],np.asarray([r[2] for r in independent]))
                actual_w=design.evaluation_weights(spec)
                assert np.array_equal(actual_w,np.asarray(expected_weights))
                assert abs(actual_w.sum()-1)<1e-14
                boundary=[]
                for di,layer in enumerate(expected_layers):
                    sizes=(21,len(layer['supports'][ecology]),nl,6)
                    for dimension,count in enumerate(sizes):
                        values=[0.,np.nextafter(0.,1.),np.nextafter(1.,0.)]
                        for k in range(1,count):
                            u=k/count
                            values.extend([np.nextafter(u,0.),u,np.nextafter(u,1.)])
                        for u in values:
                            row=[(di+.5)/21,.5,.5,.5];row[dimension]=u;boundary.append(row)
                boundary=np.asarray(boundary)
                actual_boundary=design.sample_indices(spec,boundary)
                assert np.array_equal(actual_boundary,np.asarray([reference_index(spec,u) for u in boundary]))
                assert ((actual_boundary>=0)&(actual_boundary<n)).all()
                all_boundary+=len(boundary)
                mon=spec['monitor_indices'];mon_w=design.evaluation_weights(spec,mon)
                assert len(mon)==len(set(mon))==672
                assert np.array_equal(mon_w,np.full(672,1/672))
                decoded=unpack(spec,mon)
                dc=Counter(r[0] for r in decoded);assert set(dc.values())=={32} and len(dc)==21
                assert len({r[:3] for r in decoded})==672
                reconstructed=[];paired_by_key={}
                for di,layer in enumerate(expected_layers):
                    rng=np.random.default_rng(np.random.SeedSequence([61917001,pi,di]))
                    physical=rng.choice(nphys,32,replace=False);us=rng.random(32)
                    layer_start=sum(len(x['supports'][ecology]) for x in expected_layers[:di])
                    for draw,(phys,u) in enumerate(zip(physical,us)):
                        ni=layer_start+math.floor(float(u)*len(layer['supports'][ecology]))
                        index=ni*nphys+int(phys); reconstructed.append(index)
                        li,oi=divmod(int(phys),6)
                        key=(tuple(layer['destinations']),li,oi)
                        paired_by_key[key]={'draw':draw,'u':float(u),'need':spec['needs'][ni],'index':index}
                assert sorted(reconstructed)==mon
                monitor_records[(ecology,part)]={'ordered':decoded,'keyed':paired_by_key}
                raw_arrays[f'{ecology}_{part}_weights']=actual_w
                raw_arrays[f'{ecology}_{part}_monitor_indices']=np.asarray(mon)
                raw_arrays[f'{ecology}_{part}_boundary_u']=boundary
                raw_arrays[f'{ecology}_{part}_boundary_ids']=actual_boundary
                reports.append({'ecology':ecology,'partition':part,'full_worlds':n,'full_ids_center_checked':n,
                    'weights_sum':float(actual_w.sum()),'max_weight_abs_error':0.,'boundary_rows_checked':len(boundary),
                    'monitor_worlds':672,'monitor_each_D':32,'monitor_distinct_physical_per_D':32,
                    'monitor_weight':'1/672 sampled-subset weight; not population-importance weight',
                    'weight_values':sorted(set(expected_weights))})
        assert independent_marginals['unique']==independent_marginals['multiple']
        # Same-U pairing: exercise each D with a fixed artificial U grid. No
        # formal seed or actor is instantiated; conditional needs can differ.
        pair_reports=[]
        for part in ('train','heldout_layouts'):
            u=np.asarray([[(d+.5)/21, v, l, o] for d in range(21)
                for v in (0.,.019,.25,.5,.731,np.nextafter(1.,0.))
                for l in (.0,.31,np.nextafter(1.,0.)) for o in (0.,.417,np.nextafter(1.,0.))])
            mapped={}
            for ecology in ('unique','multiple'):
                spec=prepared['partitions'][ecology][part]
                ids=design.sample_indices(spec,u)
                for _condition in prepared['conditions']:
                    assert np.array_equal(design.sample_indices(spec,u),ids)
                mapped[ecology]=np.asarray([x[:3] for x in unpack(spec,ids)],dtype=object)
            assert mapped['unique'].tolist()==mapped['multiple'].tolist()
            a=monitor_records[('unique',part)];b=monitor_records[('multiple',part)]
            assert set(a['keyed'])==set(b['keyed'])
            assert all(a['keyed'][k]['draw']==b['keyed'][k]['draw'] and a['keyed'][k]['u']==b['keyed'][k]['u'] for k in a['keyed'])
            position_matches=sum(x[:3]==y[:3] for x,y in zip(a['ordered'],b['ordered']))
            pair_reports.append({'partition':part,'same_U_rows_per_ecology':len(u),'eight_arm_D_layout_owner_pairing':True,
                'same_ecology_four_conditions_exact_ids':True,'cross_ecology_need_or_index_equality_not_required':True,
                'monitor_paired_by_D_layout_owner':672,'monitor_same_draw_and_resource_U':True,
                'monitor_physical_pairing_equal_at_same_sorted_row':position_matches,
                'monitor_join_key':'destinations + layout_index + owner_index; independently sorted rows are not a cross-ecology join key'})
            raw_arrays[f'{part}_paired_artificial_u']=u
        invalid=[[[1.,0.,0.,0.]],[[-.01,0.,0.,0.]],[[float('nan'),0.,0.,0.]],[[float('inf'),0.,0.,0.]],[[0.,0.,0.]], [0.,0.,0.,0.]]
        rejected=0
        spec=prepared['partitions']['unique']['train']
        for value in invalid:
            try:design.sample_indices(spec,value)
            except (ValueError,AssertionError):rejected+=1
        assert rejected==len(invalid)
        try:design.evaluation_weights(spec,spec['monitor_indices'][::-1])
        except (ValueError,AssertionError):monitor_wrong_order_rejected=True
        else:monitor_wrong_order_rejected=False
        assert monitor_wrong_order_rejected
        assert not blocked_calls
        assert before=={p:sha(Path(p)) for p in before}
        np.savez_compressed(out/'audit_arrays.npz',**raw_arrays)
        write(out/'prepared_inspected.json',prepared)
        result={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':before,
            'design_sha256':source_hash,'independent_support_rows':1728,'common_D':21,'excluded_D':6,
            'unique_needs':324,'multiple_needs':996,'runs':32,'full_sampling_center_rows':all_full,
            'artificial_boundary_rows':all_boundary,'invalid_U_inputs_rejected':rejected,
            'wrong_monitor_order_rejected':monitor_wrong_order_rejected,'partitions':reports,'pairing':pair_reports,
            'exact_personal_need_marginals':independent_marginals,
            'model_constructors_called':0,'model_forward_calls':0,'training_updates':0,
            'limits':['This verifies design functions and preparation, not a yet-to-be-reviewed execution runner.',
                      'Boundary mapping uses the specified float64 floor(U*n); finite-precision U bins are not claimed mathematically continuous.',
                      'The frozen monitor has equal per-stratum counts but is a sample, not the full population.',
                      'Join monitors across ecology by D/layout/owner, never by independently sorted row number.'],
            'artifacts_sha256':{p.name:sha(p) for p in (out/'audit_arrays.npz',out/'prepared_inspected.json')}}
        write(out/'verification.json',result)
        print(json.dumps({'status':'passed','full_worlds_checked':all_full,'boundary_rows':all_boundary,'pairing':pair_reports},ensure_ascii=False))
    except Exception as error:
        write(out/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'source_sha256':before})
        raise


if __name__=='__main__':
    main()
