"""Scratch-only probe tests: no formal policies, task outputs or study training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from research_program.triadic_role_decoder_study import learner as l


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.networks=l.make_networks(913)
        rng=np.random.default_rng(919)
        self.x=rng.normal(0,.3,(2,3,252))
        self.labels=np.asarray([[0,1,2],[2,0,1]],dtype=np.int16)

    def test_fixed_initialization_and_independent_heads(self):
        repeated=l.make_networks(913);different=l.make_networks(914)
        self.assertEqual(l.parameter_hash(self.networks),l.parameter_hash(repeated))
        self.assertNotEqual(l.parameter_hash(self.networks),l.parameter_hash(different))
        for actor in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([913,actor,730]))
            for layer,(left,right) in enumerate(zip(l.DIMENSIONS,l.DIMENSIONS[1:]),1):
                np.testing.assert_array_equal(self.networks[actor][f'W{layer}'],rng.normal(0,np.sqrt(2/(left+right)),(left,right)))
                np.testing.assert_array_equal(self.networks[actor][f'b{layer}'],np.zeros(right))
            for other in range(actor):
                for first in self.networks[actor].values():
                    for second in self.networks[other].values():self.assertFalse(np.shares_memory(first,second))
                self.assertFalse(np.array_equal(self.networks[actor]['W1'],self.networks[other]['W1']))
        for seed in (-1,True,1.5):
            with self.assertRaises(ValueError):l.make_networks(seed)

    def test_probability_shapes_normalization_and_actor_isolation(self):
        p=l.probabilities(self.networks,self.x)
        self.assertEqual(p.shape,(2,3,3));self.assertEqual(p.dtype,np.float64)
        np.testing.assert_allclose(p.sum(-1),1,rtol=0,atol=2e-16)
        changed=self.x.copy();changed[:,0]+=2.
        other=l.probabilities(self.networks,changed)
        np.testing.assert_array_equal(p[:,1:],other[:,1:])
        self.assertFalse(np.array_equal(p[:,0],other[:,0]))

    def test_gradient_finite_differences_all_layers_all_heads(self):
        before=l.parameter_hash(self.networks)
        gradients,row=l.training_gradients(self.networks,self.x,self.labels)
        self.assertEqual(len(gradients),3)
        def objective():
            p=l.probabilities(self.networks,self.x)
            return -np.log(p[np.arange(2)[:,None],np.arange(3),self.labels]).mean()
        self.assertAlmostEqual(row['loss'],objective(),places=14)
        epsilon=1e-5
        for actor in range(3):
            self.assertEqual(set(gradients[actor]),set(l.PARAMETER_KEYS))
            for key in l.PARAMETER_KEYS:
                parameter=self.networks[actor][key]
                for index in (tuple(0 for _ in parameter.shape),tuple(n-1 for n in parameter.shape)):
                    original=float(parameter[index]);parameter[index]=original+epsilon;upper=objective()
                    parameter[index]=original-epsilon;lower=objective();parameter[index]=original
                    self.assertAlmostEqual((upper-lower)/(2*epsilon),gradients[actor][key][index],delta=4e-10,
                                           msg=f'{actor}/{key}/{index}')
        self.assertEqual(l.parameter_hash(self.networks),before)

    def test_cross_entropy_B_times_three_reduction_and_first_ties(self):
        zeros=[{key:np.zeros(shape,dtype=np.float64) for key,shape in l.PARAMETER_SHAPES.items()} for _ in range(3)]
        gradients,row=l.training_gradients(zeros,self.x,self.labels)
        self.assertEqual(set(row),{'loss','indiv_accuracy','joint_accuracy'})
        self.assertAlmostEqual(row['loss'],np.log(3))
        self.assertAlmostEqual(row['indiv_accuracy'],1/3)
        self.assertEqual(row['joint_accuracy'],0)
        for actor in range(3):
            expected=(np.full((2,3),1/3)-np.eye(3)[self.labels[:,actor]]).sum(0)/6
            np.testing.assert_allclose(gradients[actor]['b3'],expected,atol=2e-17)
            for key in ('W1','b1','W2','b2','W3'):np.testing.assert_array_equal(gradients[actor][key],0)

    def test_stable_cross_entropy_when_correct_probability_underflows(self):
        zeros=[{key:np.zeros(shape,dtype=np.float64) for key,shape in l.PARAMETER_SHAPES.items()} for _ in range(3)]
        for network in zeros:network['b3'][:]=[1000.,-1000.,-1000.]
        labels=np.full((2,3),2,dtype=np.int64)
        gradients,row=l.training_gradients(zeros,self.x,labels)
        self.assertEqual(row['loss'],2000.)
        for gradient in gradients:np.testing.assert_allclose(gradient['b3'],[1/3,0,-1/3],atol=0)
        probability,logs=l.prediction_terms(zeros,self.x)
        np.testing.assert_array_equal(probability[:,:,2],0)
        np.testing.assert_array_equal(logs[:,:,2],-2000.)
        self.assertEqual(-float(logs[np.arange(2)[:,None],np.arange(3),labels].mean()),row['loss'])

    def test_labels_never_enter_forward_inputs(self):
        calls=[];forward=l.base.actor_forward
        def capture(network,x):
            calls.append(x.copy());return forward(network,x)
        with patch.object(l.base,'actor_forward',side_effect=capture):
            first,_=l.training_gradients(self.networks,self.x,self.labels)
            second,_=l.training_gradients(self.networks,self.x,(self.labels+1)%3)
        self.assertEqual(len(calls),6)
        for actor in range(3):
            np.testing.assert_array_equal(calls[actor],self.x[:,actor])
            np.testing.assert_array_equal(calls[actor],calls[3+actor])
        self.assertFalse(np.array_equal(first[0]['b3'],second[0]['b3']))

    def test_invalid_inputs_and_labels_fail_before_forward(self):
        invalid_x=[self.x[:,:,:251],self.x[:0],np.full_like(self.x,np.nan),np.full(self.x.shape,'string'),
                   self.x.astype(complex)]
        invalid_y=[self.labels.astype(float),self.labels.astype(bool),np.zeros((2,2),dtype=int),
                   np.full((2,3),-1),np.full((2,3),3)]
        with patch.object(l.base,'actor_forward',side_effect=AssertionError('Invalid input reached network')):
            for x in invalid_x:
                with self.assertRaises(ValueError):l.training_gradients(self.networks,x,self.labels)
            for labels in invalid_y:
                with self.assertRaises(ValueError):l.training_gradients(self.networks,self.x,labels)

    def test_bad_network_shape_nonfinite_and_sharing(self):
        variants=[]
        bad=deepcopy(self.networks);bad[0]['W3']=bad[0]['W3'][:,:2];variants.append(bad)
        bad=deepcopy(self.networks);bad[0]['W1'][0,0]=np.inf;variants.append(bad)
        bad=deepcopy(self.networks);bad[0]['W1']=bad[0]['W1'].astype(np.float32);variants.append(bad)
        bad=deepcopy(self.networks);bad[1]['W1']=bad[0]['W1'];variants.append(bad)
        for networks in variants:
            with self.assertRaises(ValueError):l.probabilities(networks,self.x)

    def test_one_adam_step_decreases_loss_and_checkpoint_roundtrip(self):
        optimizer=l.base.make_adam(self.networks)
        gradients,before=l.training_gradients(self.networks,self.x,self.labels)
        norm,scale=l.base.adam_step(self.networks,gradients,optimizer,1)
        _,after=l.training_gradients(self.networks,self.x,self.labels)
        self.assertGreater(norm,0);self.assertTrue(0<scale<=1);self.assertLess(after['loss'],before['loss'])
        rng=np.random.default_rng(921);rng.random((7,3));rng.integers(0,9,dtype=np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scratch.npz';digest=l.save_checkpoint(path,self.networks,optimizer,1,rng)
            loaded,moments,step,restored_rng=l.load_checkpoint(path)
            self.assertEqual(digest,l.sha(path));self.assertEqual(step,1)
            self.assertEqual(l.parameter_hash(loaded),l.parameter_hash(self.networks))
            self.assertEqual(restored_rng.bit_generator.state,rng.bit_generator.state)
            np.testing.assert_array_equal(restored_rng.random((5,3)),rng.random((5,3)))
            for actor in range(3):
                for key in l.PARAMETER_KEYS:
                    self.assertFalse(np.shares_memory(loaded[actor][key],self.networks[actor][key]))
                    for moment in ('m','v'):np.testing.assert_array_equal(moments[actor][moment][key],optimizer[actor][moment][key])
            with self.assertRaisesRegex(ValueError,'exists'):l.save_checkpoint(path,self.networks,optimizer,1,rng)

    def test_initial_checkpoint_and_invalid_moments_rng_and_update(self):
        optimizer=l.base.make_adam(self.networks);rng=np.random.default_rng(929)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'initial.npz';l.save_checkpoint(path,self.networks,optimizer,0,rng)
            _,moments,step,_=l.load_checkpoint(path);self.assertEqual(step,0)
            for state in moments:
                for moment in ('m','v'):
                    for value in state[moment].values():self.assertFalse(value.any())
            for step in (-1,6001,True,1.5):
                with self.assertRaises(ValueError):l.save_checkpoint(Path(directory)/'bad.npz',self.networks,optimizer,step,rng)
            bad=deepcopy(optimizer);bad[0]['v']['b3'][0]=-1
            with self.assertRaises(ValueError):l.save_checkpoint(Path(directory)/'bad.npz',self.networks,bad,1,rng)
            bad=deepcopy(optimizer);bad[0]['m']['b3'][0]=1
            with self.assertRaises(ValueError):l.save_checkpoint(Path(directory)/'bad.npz',self.networks,bad,0,rng)
            with self.assertRaises(ValueError):l.save_checkpoint(Path(directory)/'bad.npz',self.networks,optimizer,0,np.random.Generator(np.random.MT19937(1)))

    def test_checkpoint_corruption_schema_parameter_optimizer_and_rng(self):
        optimizer=l.base.make_adam(self.networks);rng=np.random.default_rng(933)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'initial.npz';l.save_checkpoint(path,self.networks,optimizer,0,rng)
            with np.load(path,allow_pickle=False) as saved:original={key:saved[key].copy() for key in saved.files}
            cases=[]
            bad=deepcopy(original);bad['unexpected']=np.array(1);cases.append(bad)
            bad=deepcopy(original);del bad['agent1_W1'];cases.append(bad)
            bad=deepcopy(original);bad['schema']=np.array('wrong');cases.append(bad)
            bad=deepcopy(original);bad['dimensions']=np.array([252,64,64,17]);cases.append(bad)
            bad=deepcopy(original);bad['agent1_b3']=np.zeros(17);cases.append(bad)
            bad=deepcopy(original);bad['agent1_W1']=bad['agent1_W1'].astype(np.float32);cases.append(bad)
            bad=deepcopy(original);bad['agent1_W1'][0,0]=np.nan;cases.append(bad)
            bad=deepcopy(original);bad['adam_agent2_v_b3'][0]=-1;cases.append(bad)
            bad=deepcopy(original);bad['update']=np.array(1.,dtype=np.float64);cases.append(bad)
            bad=deepcopy(original);bad['update']=np.array(6001,dtype=np.int64);cases.append(bad)
            bad=deepcopy(original);state=json.loads(str(bad['world_rng_json'].item()));state['state']['inc']=2
            bad['world_rng_json']=np.array(json.dumps(state));cases.append(bad)
            bad=deepcopy(original);config=json.loads(str(bad['optimizer_config_json'].item()));config['learning_rate']=.1
            bad['optimizer_config_json']=np.array(json.dumps(config));cases.append(bad)
            for index,case in enumerate(cases):
                corrupt=Path(directory)/f'bad_{index}.npz';np.savez_compressed(corrupt,**case)
                with self.subTest(index=index),self.assertRaises(ValueError):l.load_checkpoint(corrupt)


if __name__=='__main__':unittest.main()
