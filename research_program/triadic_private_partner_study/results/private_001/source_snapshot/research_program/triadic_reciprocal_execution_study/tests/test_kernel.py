"""Scratch-only numerical checks; no formal checkpoint/result/optimizer use."""
from itertools import combinations,product
import unittest
from unittest.mock import patch
import numpy as np
from research_program.triadic_reciprocal_execution_study import kernel as k

PAIRS=tuple(combinations(range(3),2))
PROFILES=np.asarray(list(product(range(17),repeat=3)),dtype=np.int16)
RESOURCE=((0,1),(2,3),(0,2),(1,3),(0,),(1,),(2,),(3,))
DEST=((0,),(1,),(0,1))
WORLDS=(((0,1,6),(0,1,2,3)),((0,1,9),(3,2,1,0)),((0,1,12),(2,0,3,1)))


def settle(world,choices,rule):
    needs,layout=world
    decoded=[]
    for who,choice in enumerate(choices):
        if choice==0:decoded.append(None)
        else:
            number=int(choice)-1
            decoded.append((number//4,(number//2)%2,[p for p in range(3) if p!=who][number%2]))
    if rule=='strict' and sum(value is not None for value in decoded)!=2:return 0.,False
    selected=[]
    for i,j in PAIRS:
        a,b=decoded[i],decoded[j]
        if a is not None and b is not None and a[:2]==b[:2] and a[2]==j and b[2]==i:selected.append((i,j))
    assert len(selected)<=1
    if not selected:return 0.,False
    i,j=selected[0];site,destination,_=decoded[i];material=layout[site]
    satisfied=sum(material in RESOURCE[needs[who]//3] and destination in DEST[needs[who]%3] for who in (i,j))
    return satisfied/2,True


def tables():
    value=np.asarray([[settle(world,choice,'strict')[0] for choice in k.JOINT] for world in WORLDS])
    assert ((value==1).sum(1)==1).all()
    return value


class KernelTests(unittest.TestCase):
    def test_all_4913_joint_actions_and_all_51_conditional_actions(self):
        logits=np.random.default_rng(710).normal(size=(3,3,17));native=tables()
        for rule in k.RULES:
            terms=k.objective_terms(logits,native,rule);p=terms['probabilities']
            for row,world in enumerate(WORLDS):
                reward,executed=map(np.asarray,zip(*(settle(world,choice,rule) for choice in PROFILES)))
                mass=np.prod(p[row,np.arange(3),PROFILES],axis=1)
                self.assertAlmostEqual(float(mass@reward),terms['J'][row],delta=2e-16)
                self.assertAlmostEqual(float(mass@(reward==1)),terms['full_success_probability'][row],delta=2e-16)
                self.assertAlmostEqual(float(mass@executed),terms['execution_probability'][row],delta=2e-16)
                for who in range(3):
                    for action in range(17):
                        selected=PROFILES[:,who]==action
                        value=float(mass[selected]@reward[selected])/p[row,who,action]
                        self.assertAlmostEqual(value,terms['action_conditioned_expected_reward'][row,who,action],delta=3e-16)
                conditional=terms['action_conditioned_expected_reward'][row]
                np.testing.assert_allclose((p[row]*conditional).sum(-1),terms['J'][row],rtol=3e-15,atol=2e-16)
                np.testing.assert_allclose(p[row]*(conditional-terms['J'][row]),terms['mean_J_logit_gradient'][row],rtol=3e-14,atol=3e-17)

    def test_uniform_support_and_integrating_third_actor(self):
        native=tables();logits=np.zeros((3,3,17))
        strict=k.objective_terms(logits,native,'strict');reciprocal=k.objective_terms(logits,native,'reciprocal')
        for key in ('J','full_success_probability','partial_success_probability','execution_probability'):
            np.testing.assert_allclose(reciprocal[key],17*strict[key],rtol=2e-15)
        np.testing.assert_allclose(reciprocal['execution_probability'],24/17**2,rtol=2e-15)
        np.testing.assert_allclose(strict['execution_probability'],24/17**3,rtol=2e-15)
        np.testing.assert_allclose(reciprocal['log_J'],strict['log_J']+np.log(17),rtol=2e-15)

    def test_strict_exact_old_utility_compatibility(self):
        native=tables();logits=np.random.default_rng(716).normal(size=(3,3,17))
        old=k.previous.objective_terms(logits,native,.5);new=k.objective_terms(logits,native,'strict')
        for key in old:np.testing.assert_array_equal(new[key],old[key],err_msg=key)

    def test_finite_differences_both_logit_objectives_all_actions(self):
        logits=np.random.default_rng(714).normal(0,1.5,size=(3,3,17));native=tables();epsilon=1e-5
        for rule in k.RULES:
            terms=k.objective_terms(logits,native,rule)
            for index in np.ndindex(logits.shape):
                plus=logits.copy();minus=logits.copy();plus[index]+=epsilon;minus[index]-=epsilon
                above=k.objective_terms(plus,native,rule);below=k.objective_terms(minus,native,rule);row=index[0]
                self.assertAlmostEqual((above['log_J'][row]-below['log_J'][row])/(2*epsilon),terms['log_J_logit_gradient'][index],delta=3e-10)
                self.assertAlmostEqual((above['J'][row]-below['J'][row])/(2*epsilon),terms['mean_J_logit_gradient'][index],delta=3e-12)
            np.testing.assert_allclose(terms['log_J_logit_gradient'].sum(-1),0,atol=3e-16)

    def test_spectator_is_integrated_not_forced_to_wait(self):
        reward=np.zeros((1,24));reward[0,0]=1
        logits=np.random.default_rng(715).normal(size=(1,3,17));changed=logits.copy();changed[:,2,0]+=20
        first=k.objective_terms(logits,reward,'reciprocal');second=k.objective_terms(changed,reward,'reciprocal')
        np.testing.assert_array_equal(first['J'],second['J']);np.testing.assert_array_equal(first['log_J'],second['log_J'])
        np.testing.assert_array_equal(first['log_J_logit_gradient'][:,2],0)
        np.testing.assert_array_equal(first['mean_J_logit_gradient'][:,2],0)
        np.testing.assert_array_equal(first['action_conditioned_expected_reward'][:,2],np.repeat(first['J'][:,None],17,axis=1))
        old_first=k.objective_terms(logits,reward,'strict');old_second=k.objective_terms(changed,reward,'strict')
        self.assertGreater(old_second['J'][0],old_first['J'][0])

    def test_log_stability_without_floors(self):
        logits=np.full((3,3,17),-1000.);logits[:,:,0]=0.
        for rule in k.RULES:
            terms=k.objective_terms(logits,tables(),rule)
            np.testing.assert_array_equal(terms['J'],0)
            self.assertTrue(np.isfinite(terms['log_J']).all())
            self.assertTrue((terms['log_J']<-1000).all())
            self.assertTrue(np.isfinite(terms['log_J_logit_gradient']).all())
            np.testing.assert_allclose(terms['posterior_weights'].sum(-1),1,atol=3e-16)

    def test_input_rejection_before_any_model(self):
        x=np.zeros((3,3,54));native=tables();mu=np.zeros((2,3,2,3,4))
        common=dict(networks=[{}]*9,observations=x,rewards=native,live=True,uniforms=mu,update=1,rule='reciprocal')
        for name,value in (('rule','other'),('rule',True),('rewards',np.zeros((3,24))),
                           ('rewards',native*.1),('observations',x[:,:,:53]),('observations',x+np.nan),
                           ('live',1),('uniforms',mu+1),('update',0),('update',1.2)):
            kwargs=dict(common);kwargs[name]=value
            with self.subTest(name=name),patch.object(k.core,'rollout',side_effect=AssertionError('No model allowed')),self.assertRaises(ValueError):
                k.training_gradients(**kwargs)

    def test_strict_training_exact_compatibility_including_sender(self):
        nets=k.core.make_networks(918);rng=np.random.default_rng(927)
        x=rng.normal(size=(2,3,54));mu=rng.random((2,2,2,3,4));native=tables()[:2]
        before=k.core.parameter_hash(nets)
        old_grad,old_row=k.previous.training_gradients(nets,x,native,False,mu,51,.5)
        new_grad,new_row=k.training_gradients(nets,x,native,False,mu,51,'strict')
        for old,new in zip(old_grad,new_grad):
            for key in old:np.testing.assert_array_equal(new[key],old[key])
        for key in old_row:self.assertEqual(new_row[key],old_row[key],key)
        self.assertEqual(k.core.parameter_hash(nets),before)

    def test_small_network_parameter_finite_differences_with_stopped_sender_advantage(self):
        nets=k.core.make_networks(925);rng=np.random.default_rng(931);B=2
        x=rng.normal(0,.2,(B,3,54));mu=rng.random((2,B,2,3,4));native=tables()[:B]
        doubled_x=np.concatenate((x,x));doubled_r=np.concatenate((native,native))
        grad,row=k.training_gradients(nets,x,native,True,mu,47,'reciprocal')
        trace=k.core.rollout(nets,doubled_x,True,mu.reshape(2*B,2,3,4))
        terms=k.objective_terms(trace['action_logits'],doubled_r,'reciprocal')
        entropy,_=k.base.entropy_and_logit_gradient(terms['probabilities'],terms['log_probabilities'])
        F=(terms['log_J']+k.base.entropy_coefficient(47)*entropy).reshape(2,B)
        advantage=np.stack(((F[0]-F[1])/2,(F[1]-F[0])/2))
        sampled=trace['messages'].copy()
        def receiver_loss():
            trial=k.core.rollout(nets,doubled_x,True,mu.reshape(2*B,2,3,4))
            np.testing.assert_array_equal(trial['messages'],sampled)
            values=k.objective_terms(trial['action_logits'],doubled_r,'reciprocal')
            return k.core.coordination.loss_and_derivative(values,'mean_log_J',47)['loss']
        def sender_loss():
            trial=k.core.rollout(nets,doubled_x,True,mu.reshape(2*B,2,3,4))
            np.testing.assert_array_equal(trial['messages'],sampled)
            lp=trial['sender_log_probabilities'].reshape(2,B,2,3,4,8)
            message=sampled.reshape(2,B,2,3,4)
            scores=np.take_along_axis(lp,message[...,None],axis=-1)[...,0].sum((2,3,4))
            return -float((advantage*scores).sum()/B)
        self.assertAlmostEqual(receiver_loss(),row['receiver_loss'],places=14)
        self.assertAlmostEqual(sender_loss(),row['sender_surrogate_loss'],places=14)
        epsilon=1e-5
        for head,objective in ((0,sender_loss),(4,sender_loss),(8,receiver_loss)):
            for key,index in (('W3',(2,1)),('b3',(1,))):
                original=float(nets[head][key][index]);nets[head][key][index]=original+epsilon;upper=objective()
                nets[head][key][index]=original-epsilon;lower=objective();nets[head][key][index]=original
                self.assertAlmostEqual((upper-lower)/(2*epsilon),grad[head][key][index],delta=5e-9,
                    msg=f'{head}/{key}/{index}')


if __name__=='__main__':unittest.main()
