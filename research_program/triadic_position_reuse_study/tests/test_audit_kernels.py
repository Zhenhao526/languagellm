"""Independent synthetic identification and scheduling tests; no learned model."""
from itertools import product
import unittest
import numpy as np
from research_program.triadic_position_reuse_study import audit_execution as a

class PureTests(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(210916)
    def second(self,x):
        n=len(x);m=np.zeros((n,3,4),dtype=np.int8)
        for who,pos in product(range(3),range(4)):
            value=x[:,who,:54].sum(-1).astype(int)
            for sender in range(3):value+=(sender+1)*x[:,who,54+sender*32+pos*8:54+sender*32+(pos+1)*8].argmax(-1)
            m[:,who,pos]=value%8
        return m
    def test_all_single_position_routes_and_self_visibility(self):
        from research_program.triadic_position_reuse_study import channel_intervention as c
        triples=list(product(range(3),range(4),range(8)));n=len(triples)
        s,p,z=np.asarray(triples,dtype=np.int8).T;m=self.rng.integers(0,8,(n,3,4),dtype=np.int8)
        for live in (True,False):
            x=a.route(m,live,s,p,z)
            np.testing.assert_array_equal(x,c.replace_outward(m,s,p,z,live))
            np.testing.assert_array_equal(x[np.arange(n),s],a.route(m,live)[np.arange(n),s])
            if not live:np.testing.assert_array_equal(x,a.route(m,False))
    def test_mixed_windows_recompute_only_early_and_preserve_self(self):
        n=24;x=self.rng.integers(0,2,(n,3,54)).astype(float);m=self.rng.integers(0,8,(n,2,3,4),dtype=np.int8)
        m[:,1]=self.second(np.concatenate((x,a.route(m[:,0],True)),axis=-1))
        s=np.arange(n)%3;w=np.arange(n)%2;p=np.arange(n)%4;donor=self.rng.integers(0,8,(n,2,4),dtype=np.int8)
        calls=[]
        def second(xx):calls.append(len(xx));return self.second(xx)
        out=a.causal_inputs(x,m,s,donor,w,p,True,second)
        self.assertEqual(calls,[12]);self.assertEqual(out['independent_module_samples'],108)
        np.testing.assert_array_equal(out['messages'][w==1],m[w==1])
        np.testing.assert_array_equal(out['messages'][np.arange(n),1,s],m[np.arange(n),1,s])
        for i in range(n):
            single=a.causal_inputs(x[i:i+1],m[i:i+1],int(s[i]),donor[i:i+1],int(w[i]),int(p[i]),True,self.second)
            np.testing.assert_array_equal(out['action_inputs'][i],single['action_inputs'][0])
    def test_all_eight_shams_and_silent_no_generator(self):
        n=24;x=self.rng.integers(0,2,(n,3,54)).astype(float);s=np.arange(n)%3
        for live in (True,False):
            m=self.rng.integers(0,8,(n,2,3,4),dtype=np.int8)
            m[:,1]=self.second(np.concatenate((x,a.route(m[:,0],live)),axis=-1))
            own=m[np.arange(n),:,s,:];natural=np.concatenate((x,a.route(m[:,0],live),a.route(m[:,1],live)),axis=-1)
            for w,p in product(range(2),range(4)):
                out=a.causal_inputs(x,m,s,own,w,p,live,self.second)
                np.testing.assert_array_equal(out['action_inputs'],natural)
                if not live:
                    def forbidden(_):raise AssertionError('Silent must not generate')
                    altered=a.causal_inputs(x,m,s,(own+1)%8,w,p,False,forbidden)
                    np.testing.assert_array_equal(altered['action_inputs'],natural)
                    self.assertEqual(altered['independent_module_samples'],0)
    def test_additive_axis_and_position_strength_cancel(self):
        rows=np.array([.2,-.3,.4]);cols=np.array([-.5,.1,.9])
        self.assertAlmostEqual(a.matrix_statistics(rows[:,None]+cols[None,:])['selectivity'],0.)
        t=-np.ones((3,3));np.fill_diagonal(t,0.)
        out=a.matrix_statistics(t);self.assertEqual(out['diagonal'],0.);self.assertEqual(out['selectivity'],1.)
    def test_shared_selection_and_uniform_position_are_not_extra_evidence(self):
        n=18;axis=np.repeat(np.arange(3),6);sender=np.tile([0,0,1,1,2,2],3);weights=np.full(n,1/6)
        effect=self.rng.uniform(-1,1,(n,2,8));selected=np.zeros((3,3),dtype=np.int8)
        r=a.effect_matrix(effect,selected,axis,sender,weights)
        np.testing.assert_allclose(r['T'],np.broadcast_to(r['T'][0],(3,3)),rtol=0,atol=1e-15)
        self.assertAlmostEqual(r['selectivity'],0.)
        uniform=np.broadcast_to(r['uniform8_by_axis'],(3,3));self.assertAlmostEqual(a.matrix_statistics(uniform)['selectivity'],0.)
    def test_discovery_keeps_negative_and_shared_ties(self):
        rates=np.ones((3,3,8));rates[:,0]=0
        r=a.select_positions(rates)
        self.assertTrue(np.all(r['selected_positions']==0));self.assertTrue(np.all(r['selected_scores'][:,0]==-1))
        n=18;axis=np.repeat(np.arange(3),6);sender=np.tile([0,0,1,1,2,2],3);listener=np.tile([1,2,0,2,0,1],3)
        ends=np.arange(2*n).reshape(n,2);m=np.zeros((2*n,2,3,4),dtype=np.int8)
        for row in range(n):m[ends[row,1],0,sender[row],axis[row]]=1
        discovered=a.discovery(m,ends,axis,sender,listener,np.full(n,1/6))
        np.testing.assert_array_equal(discovered['selected_positions'],np.tile([0,1,2],(3,1)))
    def test_legal_unseen_packet_and_exact_budget(self):
        natural=np.zeros((2,2,4),dtype=np.int8);natural[1]=7;mix=natural[:1].copy();mix[0,0,0]=7
        codes=a.packet_codes(natural);self.assertEqual(int(codes[0]),0);self.assertEqual(int(codes[1]),8**8-1)
        self.assertNotIn(int(a.packet_codes(mix)[0]),set(map(int,codes)))
        self.assertEqual(a.budget(),dict(actual_worlds=41472,module_samples=186624,actual_npz=288,silent_alias_worlds=41472,silent_records=288))
    def test_static_dataset_all_arrays_and_hash_selected_rows(self):
        from research_program.triadic_position_reuse_study import dataset
        context,_=a.references();specs,_,_=context.independent_specs()
        expected,proof=a.independent_dataset(specs)
        actual=dataset.make_prepared(dataset.load_sources())
        for name in ('discovery','validation'):
            self.assertEqual(set(expected[name]),set(actual[name]['arrays']))
            for key,value in expected[name].items():
                saved=actual[name]['arrays'][key]
                self.assertEqual(saved.dtype,value.dtype)
                if value.dtype.kind=='f':np.testing.assert_allclose(saved,value,atol=1e-18,rtol=0)
                else:np.testing.assert_array_equal(saved,value)
        meta=actual['validation']['metadata']
        self.assertEqual(proof['case_sha256'],[r['case_sha256'] for r in meta['selected_cases']])
        self.assertEqual(proof['background_sha256'],[r['background_sha256'] for r in meta['selected_backgrounds']])
        self.assertEqual(proof['donor_sha256'],[r['donor_sha256'] for r in meta['selected_backgrounds']])
        self.assertEqual(proof['discovery_unique_endpoint_worlds'],326592)
        self.assertEqual(proof['validation_unique_endpoint_worlds'],284)
    def test_actual_fraction_discovery_schema(self):
        from research_program.triadic_position_reuse_study import discovery
        n=18;axis=np.repeat(np.arange(3),6);sender=np.tile([0,0,1,1,2,2],3);listener=np.tile([1,2,0,2,0,1],3)
        ends=np.arange(2*n).reshape(n,2);m=self.rng.integers(0,8,(2*n,2,3,4),dtype=np.int8)
        spec=dict(endpoint_indices=ends,axis=axis,sender=sender,listener=listener)
        expected=a.discovery(m,ends,axis,sender,listener,np.full(n,1/6))
        a.verify_selection(expected,discovery.select_positions(spec,m))
        np.testing.assert_array_equal(a.packet_codes(m[:,:,0]),discovery.packet_codes(m[:,:,0]))
    def test_actual_silent_record_and_native_metrics_schema(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from research_program.triadic_position_reuse_study import runner,metrics
        context,_=a.references();specs,_,_=context.independent_specs();data,_=a.independent_dataset(specs);spec=data['validation']
        states=context.packed(specs[a.PART]);n=len(states)
        pool=dict(states=states,messages=np.zeros((n,2,3,4),dtype=np.int8),action_indices=np.zeros((n,3),np.int16),action_probabilities=np.zeros((n,3,17)))
        pool['action_probabilities'][:,:,0]=1;codes=[np.array([0],np.uint32)]*3
        with TemporaryDirectory() as directory:
            for u,b in ((0,0),(4,1)):
                ids=spec['endpoint_indices'][:,b];cf=spec['endpoint_indices'][:,1-b];x=context.features(states[ids],'PL')
                output,record=runner.execute_cell(None,x,pool,spec,u,'opposite',b,False)
                record.update(condition='PL_silent',path=None,data_sha256=None)
                expected,receipt=a.audit_cell(record,Path(directory)/f'{u}.npz',pool,spec,None,'PL',codes)
                natural={k:pool[k][ids] for k in ('messages','action_indices','action_probabilities')}
                actual=metrics.row_values(spec,b,states[ids],natural,output,codes,counterfactual_states=states[cf])
                self.assertEqual(set(actual),set(expected))
                for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
                self.assertEqual(receipt['independent_modules'],0)
    def test_all_summary_numeric_fields_on_synthetic_rows(self):
        from research_program.triadic_position_reuse_study import metrics
        context,_=a.references();specs,_,_=context.independent_specs();data,_=a.independent_dataset(specs);spec=data['validation'];n=len(spec['axis'])
        positions={}
        for u in range(8):
            same=dict(target_apt=self.rng.integers(0,2,(2,n)).astype(bool),current_probability=self.rng.random((2,n)),natural_current_apt=np.zeros((2,n),bool),packet_in_train_endpoint_greedy_set=self.rng.integers(0,2,(2,n)).astype(bool))
            opposite={k:v.copy() for k,v in same.items()};opposite['target_apt']=self.rng.integers(0,2,(2,n)).astype(bool)
            positions[u]=dict(same=same,opposite=opposite,contrast=a.contrast(same,opposite))
        selected=self.rng.integers(0,8,(3,3),dtype=np.int8);selection=dict(selected_positions=selected)
        expected=a.policy_summary(spec,selection,positions,{'natural':positions[0]['same']},{0:positions[0]['same'],4:positions[4]['same']})
        actual=metrics.selection_summary(spec,selection,positions);counts=dict(scalars=0,max_error=0.)
        a.compare_tree({k:v for k,v in expected.items() if k in actual},actual,counts)
        self.assertGreater(counts['scalars'],1000);self.assertLess(counts['max_error'],1e-14)

if __name__=='__main__':unittest.main()
