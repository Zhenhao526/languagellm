"""Small synthetic audit fixtures; no formal policies or results are opened."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_program.triadic_role_decoder_study import audit_execution as a


def states():
    # Needs12: material0,destination0;15: material1,destination0.
    return np.asarray([[12,12,15,0,1,2,3,1,2,3],
                       [12,15,12,3,2,1,0,2,3,1],
                       [15,12,12,1,0,3,2,3,1,2]],dtype=np.int16)


def checkpoint_payload(seed=17):
    value={};nets=[]
    for actor in range(3):
        rng=np.random.default_rng(np.random.SeedSequence([seed,actor,730]));net={}
        for layer,(left,right) in enumerate(zip(a.DIMS,a.DIMS[1:]),1):
            net[f'W{layer}']=rng.normal(0,math.sqrt(2/(left+right)),(left,right))
            net[f'b{layer}']=np.zeros(right)
        for key,item in net.items():
            value[f'agent{actor}_{key}']=item
            for moment in ('m','v'):value[f'adam_agent{actor}_{moment}_{key}']=np.zeros_like(item)
        nets.append(net)
    state=np.random.default_rng(np.random.SeedSequence([seed,800])).bit_generator.state
    value.update(schema=np.array('triadic_role_decoder_checkpoint_v1'),dimensions=np.array(a.DIMS,dtype=np.int64),
        update=np.array(0,dtype=np.int64),world_rng_json=np.array(json.dumps(state,sort_keys=True)),
        optimizer_config_json=np.array(json.dumps(a.OPTIMIZER,sort_keys=True)))
    return value,nets,state


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env,cls.sender=a.helpers()

    def test_truth_pair_identity_and_reject_nonunique(self):
        s=states();expected=np.asarray([[1,1,0],[2,0,1],[0,2,2]],dtype=np.int8)
        np.testing.assert_array_equal(a.role_truth(s),expected)
        reward=self.env.rewards(s);self.assertTrue(np.all((reward==1).sum(1)==1))
        actions=self.env.JOINT[(reward==1).argmax(1)]
        np.testing.assert_array_equal(np.where(actions==0,0,1+(actions-1)%2),expected)
        bad=s.copy();bad[:,0:3]=12
        with self.assertRaisesRegex(AssertionError,'exactly one'):a.role_truth(bad)

    def test_routes_positions_and_all_198_control_channels(self):
        s=states();m=(np.arange(72,dtype=np.int8).reshape(3,2,3,4)%8)
        for window in (0,1):
            route=a.route(m[:,window]);np.testing.assert_array_equal(route,self.sender.route(m[:,window],True))
            self.assertTrue(np.all(route.sum(-1)==15))
            for batch in range(3):
                for viewer in range(3):
                    for sender in range(3):
                        for pos in range(4):self.assertEqual(route[batch,viewer,sender*32+pos*8+m[batch,window,sender,pos]],1.)
        for view in ('Own','FI'):
            x=a.inputs(s,view,m,self.env)
            self.assertFalse(x[:,:,54:].any())
            np.testing.assert_array_equal(x[:,:,:54],self.env.features(s,'FI' if view=='FI' else 'PL'))
            self.assertTrue(np.all(x[:,:,53]==(view=='FI')))
        own=a.inputs(s,'Own',m,self.env)
        for viewer in range(3):
            for other in range(3):
                if other!=viewer:self.assertFalse(own[:,viewer,7*other:7*(other+1)].any())
        transcript=a.inputs(s,'Final',m,self.env)
        np.testing.assert_array_equal(transcript[:,:,54:153],a.route(m[:,0]))
        np.testing.assert_array_equal(transcript[:,:,153:252],a.route(m[:,1]))
        np.testing.assert_array_equal(a.inputs(s,'Initial',m,self.env),transcript)
        bad=m.copy();bad[0,0,0,0]=8
        with self.assertRaises(AssertionError):a.inputs(s,'Final',bad,self.env)

    def test_independent_forward_and_stable_ce(self):
        # Exactly one synthetic forward of 3 heads x2 samples, no parameter file.
        nets=[{k:np.zeros(shape) for k,shape in a.SHAPES.items()} for _ in range(3)]
        for actor in range(3):nets[actor]['b3']=np.asarray([float(actor),0.,-float(actor)])
        p,lp=a.probe_forward(nets,np.zeros((2,3,252)))
        for actor in range(3):
            raw=nets[actor]['b3'];expected=np.exp(raw)/np.exp(raw).sum()
            np.testing.assert_allclose(p[:,actor],np.repeat(expected[None,:],2,axis=0),atol=1e-15)
        labels=np.asarray([[0,1,2],[2,1,0]],np.int8)
        expected=-np.asarray([[lp[b,w,labels[b,w]] for w in range(3)] for b in range(2)]).mean(1)
        np.testing.assert_array_equal(a.cross_entropy(lp,labels),expected)
        # CE accepts stable log probabilities even where exp(logp) underflows.
        logs=np.zeros((1,3,3));logs[:,:,2]=-1000
        np.testing.assert_array_equal(a.cross_entropy(logs,np.full((1,3),2,np.int8)),[1000.])

    def test_saved_summaries_include_illegal_joint_predictions(self):
        target=a.role_truth(states());pred=np.asarray([[1,1,0],[2,0,1],[0,0,0]],np.int8)
        p=np.full((3,3,3),.1);p[np.arange(3)[:,None],np.arange(3),pred]=.8
        out,choices,nll,joint=a.evaluation_summary(p,target,np.arange(3,dtype=np.int64))
        self.assertEqual(out['joint_accuracy'],2/3);self.assertEqual(out['indiv_accuracy'],7/9)
        self.assertEqual(out['true_pair_strata']['BC'],dict(worlds=1,joint_accuracy=0.))
        self.assertIn(dict(labels=[0,0,0],worlds=1),out['predicted_joint_role_counts'])
        np.testing.assert_array_equal(choices,pred);np.testing.assert_array_equal(joint,[True,True,False])
        self.assertAlmostEqual(nll[2],-(math.log(.8)+2*math.log(.1))/3)
        bad=p.copy();bad[0,0,0]=0
        with self.assertRaises(AssertionError):a.evaluation_summary(bad,target,np.arange(3))

    def test_probe_checkpoint_independent_initialization_and_corruptions(self):
        value,nets,state=checkpoint_payload()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'scratch.npz';np.savez_compressed(path,**value)
            actual=a.probe_checkpoint(path,0,17,state)
            self.assertEqual(a.parameter_hash(actual),a.parameter_hash(nets))
            for kind in ('weight','moment','rng','schema','extra'):
                bad={k:v.copy() for k,v in value.items()}
                if kind=='weight':bad['agent0_W1'][0,0]+=.01
                if kind=='moment':bad['adam_agent0_v_W1'][0,0]=-1
                if kind=='rng':
                    wrong=json.loads(str(bad['world_rng_json'].item()));wrong['state']['state']+=1
                    bad['world_rng_json']=np.array(json.dumps(wrong))
                if kind=='schema':bad['schema']=np.array('wrong')
                if kind=='extra':bad['extra']=np.array(1)
                np.savez_compressed(path,**bad)
                with self.subTest(kind=kind),self.assertRaises(AssertionError):a.probe_checkpoint(path,0,17,state)

    def test_nested_primary_aliases_and_negative_blocks(self):
        values=(-.1,.02,.08,.01);runs=[]
        for job in a.expected_jobs():
            score={'Own':.25,'FI':.98,'Initial':.5,'Final':.5}[job['view']]
            if job['view']=='Final':score+=values[a.OLD_SEEDS.index(job['old_seed'])]+(.01 if job['payoff']=='a50' else -.01)
            runs.append(dict(job=job,final={'new_needs_and_layouts':dict(joint_accuracy=score)}))
        logical,primary=a.logical_and_primary(runs)
        self.assertEqual(len(logical),96);self.assertEqual(len({r['actual_job_id'] for r in logical}),42)
        self.assertAlmostEqual(primary['mean_difference'],.0025)
        for block,expected in zip(primary['protocol_seed_blocks'],values):self.assertAlmostEqual(block['mean_final_minus_initial'],expected)
        self.assertEqual(a.logical_and_primary(list(reversed(runs))),(logical,primary))
        with self.assertRaises(AssertionError):a.logical_and_primary(runs[:-1])
        with self.assertRaises(AssertionError):a.logical_and_primary(runs[:-1]+[runs[0]])

    def test_metadata_corefields_and_budget_counts(self):
        # Extra anchored descriptive fields are allowed; altered core is not.
        a.compare(dict(core=1,description='anchored separately'),dict(core=1))
        with self.assertRaises(AssertionError):a.compare(dict(core=2,description='x'),dict(core=1))
        parts={p:dict(world_count=n,monitor_indices=range(m)) for p,n,m in zip(a.PARTS,(419904,160704,139968,53568),(7776,2976,7776,2976))}
        budget=a.expected_budget(parts)
        self.assertEqual(budget['actual_decoder_runs'],42);self.assertEqual(budget['updates'],252000)
        self.assertEqual(budget['initial_sender_module_samples'],18579456)
        self.assertEqual(budget['actual_final_worlds']*3,97542144)
        self.assertEqual(budget['actual_monitor_files'],1008)

    def test_feature_stream_hash_and_world_sampling(self):
        s=states()
        for info in ('FI','PL'):self.assertEqual(a.feature_hash(s,info,self.env),a.array_sha(self.env.features(s,info)))
        spec=dict(needs=[[12,12,15],[12,15,12]],layouts=[[0,1,2,3],[3,2,1,0]],private_sites=[[1,2,3],[3,2,1]],world_count=8)
        uniform=np.asarray([[0.,0.,0.],[.999,.999,.999],[.1,.8,.2]])
        ids=self.env.sample(spec,uniform)
        np.testing.assert_array_equal(ids,[0,7,2])
        packed=self.env.packed(spec,ids)
        np.testing.assert_array_equal(packed[:,3:7],[[0,1,2,3],[3,2,1,0],[3,2,1,0]])
        self.assertEqual(len(a.array_sha(packed)),64)


if __name__=='__main__':unittest.main()
