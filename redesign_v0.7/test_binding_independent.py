"""Independent matrix, locality, initialization and information-route tests."""
import copy
from collections import Counter
from itertools import permutations, product
from types import MethodType
import unittest

import numpy as np
import torch

import camp
import run_experiment as run

REPRS=('identity','permute','orthogonal')
SEEDS=(26101,26102,26103,26104)


def raw_slots(n=7):
    x=torch.from_numpy(np.random.default_rng(7311).normal(size=(n,6,65)).astype(np.float32))
    x[:,:,64]=0
    x[:,1,64]=1;x[:,4,64]=1
    flat=torch.cat((x[:,:,:64].reshape(n,384),x[:,:,64]),1)
    return x,flat


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.prepared=[a.state_dict() for a in camp.make_agents(990071)]
        cls.bank=camp.ImageBank()

    def agents(self,representation='identity',seed=26101):
        return camp.remake_agents(seed,self.prepared,7,2,representation)

    def test_fixed_52_run_design_and_direct_budget(self):
        self.assertEqual(run.SEEDS,list(SEEDS))
        expected={f'split{s}_{r}' for s in (1,2,3) for r in REPRS}
        expected|={f'full_{r}' for r in REPRS}|{'full_blocked'}
        self.assertEqual(set(run.CONDITIONS),expected)
        self.assertEqual(len(expected)*len(SEEDS),52)
        for name,plan in run.CONDITIONS.items():
            self.assertEqual((plan['schedule'],plan['vocab'],plan['length'],plan['known']),('direct',7,2,False))
            self.assertEqual(plan['blocked'],name=='full_blocked')
        levels,ids=run.training_schedule(SEEDS[0],2400,'direct')
        self.assertTrue(np.all(levels==6) and np.array_equal(ids,np.arange(2400)))

    def test_matrices_are_private_deterministic_orthogonal_and_six_cycles(self):
        nonidentity=[]
        for seed,who in product(SEEDS,(0,1)):
            for representation in REPRS:
                m=camp.binding_matrix(seed,who,representation)
                self.assertTrue(torch.equal(m,camp.binding_matrix(seed,who,representation)))
                self.assertFalse(m.requires_grad)
                self.assertTrue(torch.allclose(m.T@m,torch.eye(6),atol=3e-7,rtol=0))
                self.assertTrue(torch.allclose(m@m.T,torch.eye(6),atol=3e-7,rtol=0))
                if representation=='identity':self.assertTrue(torch.equal(m,torch.eye(6)))
                elif representation=='permute':
                    self.assertTrue(((m==0)|(m==1)).all() and (m.sum(0)==1).all() and (m.sum(1)==1).all())
                    permutation=m.argmax(1).tolist();visited=[];position=0
                    for _ in range(6):visited.append(position);position=permutation[position]
                    self.assertEqual(set(visited),set(range(6)));self.assertEqual(position,0)
                    self.assertTrue(all(i!=p for i,p in enumerate(permutation)))
                else:
                    self.assertTrue((m.abs()>1e-6).sum(1).min()>1)
                    singular=torch.linalg.svdvals(m)
                    self.assertLess(float(singular.max()/singular.min()),1.000002)
                    nonidentity.append(m.numpy().tobytes())
        self.assertEqual(len(nonidentity),len(set(nonidentity)))

    def test_transform_generation_does_not_advance_model_rng(self):
        torch.manual_seed(8421);state=torch.get_rng_state().clone()
        for representation in REPRS:camp.binding_matrix(26101,0,representation)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_all_trainable_initial_parameters_and_receivers_match(self):
        for seed in SEEDS:
            groups=[self.agents(r,seed) for r in REPRS]
            for who in (0,1):
                baseline=dict(groups[0][who].named_parameters())
                for agents in groups:
                    actual=dict(agents[who].named_parameters())
                    self.assertEqual(set(actual),set(baseline))
                    self.assertTrue(all(torch.equal(v,actual[k]) and v.requires_grad==actual[k].requires_grad
                                        for k,v in baseline.items()))
                    self.assertNotIn('input_transform',actual)
                    self.assertIn('input_transform',dict(agents[who].named_buffers()))
                msg=torch.tensor([[1,2],[5,6]]);goal=torch.eye(2);inv=torch.zeros(2,2)
                hist=torch.zeros(2,18);menu=torch.tensor([[1,4,0,3,2,5],[5,0,4,1,3,2]])
                logits=[a[who].receive(msg,goal,inv,hist,menu)[0] for a in groups]
                self.assertTrue(all(torch.equal(logits[0],x) for x in logits))
            pointers=[p.data_ptr() for a in groups for person in a for p in person.parameters()]
            pointers += [person.input_transform.data_ptr() for a in groups for person in a]
            self.assertEqual(len(pointers),len(set(pointers)))
            self.assertEqual(len({run.state_sha(a,trainable_only=True) for a in groups}),1)
            self.assertEqual(len({run.state_sha(a,receiver_only=True) for a in groups}),1)
            self.assertEqual(len({run.state_sha(a) for a in groups}),3)

    def test_visual_presence_repacking_and_no_unmixed_bypass(self):
        slots,flat=raw_slots()
        for representation in REPRS:
            agent=self.agents(representation)[0];observed=[]
            handle=agent.slot_phi.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
            encoded=agent.encode_slots(flat);handle.remove()
            reference=torch.stack([sum(agent.input_transform[i,j]*slots[:,j] for j in range(6)) for i in range(6)],1)
            self.assertTrue(torch.allclose(observed[0],reference,atol=1e-6,rtol=0))
            self.assertEqual(tuple(encoded.shape),(7,390))
            direct=agent.memory(torch.cat((agent.slot_phi(reference).flatten(1),torch.ones(7,1)),1),torch.zeros(7,96))
            self.assertTrue(torch.allclose(agent.observe(flat),direct,atol=1e-6,rtol=0))
            reconstructed=torch.einsum('ji,bjd->bid',agent.input_transform,reference)
            self.assertTrue(torch.allclose(reconstructed,slots,atol=2e-6,rtol=0))
            self.assertTrue(torch.allclose(reference.square().sum((1,2)),slots.square().sum((1,2)),atol=1e-4,rtol=1e-6))

    def test_phi_is_shared_and_independent_between_examples(self):
        _,flat=raw_slots()
        for representation in REPRS:
            agent=self.agents(representation)[0]
            together=agent.encode_slots(flat)
            separate=torch.cat([agent.encode_slots(flat[i:i+1]) for i in range(len(flat))])
            self.assertTrue(torch.allclose(together,separate,atol=1e-6,rtol=0))
            reordered=agent.encode_slots(flat.flip(0)).flip(0)
            self.assertTrue(torch.allclose(together,reordered,atol=1e-6,rtol=0))
            x=flat.clone().requires_grad_();agent.encode_slots(x)[0].sum().backward()
            self.assertEqual(int(torch.count_nonzero(x.grad[1:])),0)
            self.assertGreater(int(torch.count_nonzero(x.grad[0])),0)
            self.assertEqual(len(list(agent.slot_phi.parameters())),2)

    def test_permutation_commutes_with_phi_and_gru_column_relabeling(self):
        _,flat=raw_slots();identity=self.agents('identity')[0];permuted=self.agents('permute')[0]
        a=identity.encode_slots(flat).reshape(7,6,65)
        b=permuted.encode_slots(flat).reshape(7,6,65)
        expected=torch.einsum('ij,bjd->bid',permuted.input_transform,a)
        self.assertTrue(torch.allclose(b,expected,atol=1e-6,rtol=0))
        permutation=permuted.input_transform.argmax(1)
        with torch.no_grad():
            columns=identity.memory.weight_ih[:,:390].reshape(288,6,65)
            permuted.memory.weight_ih[:,:390].copy_(columns[:,permutation].reshape(288,390))
        self.assertTrue(torch.allclose(identity.observe(flat),permuted.observe(flat),atol=2e-6,rtol=0))

    def test_all_representations_have_identical_world_streams_and_heldout_support(self):
        train_photos={i for i,e in enumerate(self.bank.entries) if e['split']=='train'}
        for split in (0,1,2,3):
            outputs=[]
            for representation in REPRS:
                agents=self.agents(representation);banks=camp.projected_banks(agents,self.bank)
                name=f'split{split}_{representation}' if split else f'full_{representation}'
                plan=run.training_batch_plan(26101,723,6,run.CONDITIONS[name])
                result=run.rollout(agents,banks,self.bank,plan,26101*100000+40000000+724,
                    128,training=True,greedy=False,trace=True,split='train')
                outputs.append(result)
                for row in result[2]:
                    self.assertTrue(set(row['photo_ids'].flatten())<=train_photos)
                    self.assertTrue(set(map(tuple,row['positions']))<=set(map(tuple,camp.MAPS[camp.split_maps(split)[0]])))
                    self.assertTrue((row['history']==0).all() and (row['inventory']==0).all())
            self.assertEqual(len({o[0]['world_sha256'] for o in outputs}),1)
            for other in outputs[1:]:
                for a,b in zip(outputs[0][2],other[2]):
                    self.assertTrue(all(np.array_equal(a[k],b[k]) for k in ('positions','photo_ids','goals','menu','refill_uniform')))

    def test_gradients_are_personal_and_frozen_matrix_never_receives_gradients(self):
        for representation in REPRS:
            agents=self.agents(representation);banks=camp.projected_banks(agents,self.bank)
            initial=[a.input_transform.clone() for a in agents]
            plan=run.training_batch_plan(26101,9,6,run.CONDITIONS[f'split1_{representation}'])
            _,learning,_=run.rollout(agents,banks,self.bank,plan,7321,128,training=True,greedy=False,split='train')
            loss=sum(-lp.mean()+.5*(value-target).square().mean() for lp,ent,value,target in learning[0])
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in agents[0].slot_phi.parameters()))
            self.assertTrue(all(p.grad is None for p in agents[1].parameters()))
            self.assertTrue(all(p.grad is None for a in agents for p in a.project.parameters()))
            for a,m in zip(agents,initial):
                self.assertIsNone(a.input_transform.grad);self.assertFalse(a.input_transform.requires_grad)
                self.assertTrue(torch.equal(a.input_transform,m))

    def test_private_goal_balanced_evaluation_and_native_blocking(self):
        for representation in REPRS:
            agents=self.agents(representation);banks=camp.projected_banks(agents,self.bank);goals_seen=[]
            for a in agents:
                original=a.send
                def capture(self,h,goal,inventory,rng,greedy,_original=original):
                    goals_seen.append(goal.detach().clone());return _original(h,goal,inventory,rng,greedy)
                a.send=MethodType(capture,a)
            _,_,rows=run.rollout(agents,banks,self.bank,run.CONDITIONS[f'split1_{representation}'],7322,120,trace=True)
            self.assertTrue(all(torch.count_nonzero(g)==0 for g in goals_seen))
            for row in rows:
                counts=Counter((tuple(p),int(g)) for p,g in zip(row['positions'],row['goals']))
                self.assertEqual(counts,Counter({(p,g):1 for p in permutations(range(6),2) for g in (0,1)}))
                correct=row['positions'][np.arange(60),row['goals']]
                self.assertTrue(np.array_equal(row['reward'],(row['place']==correct).astype(float)))
        agents=self.agents();banks=camp.projected_banks(agents,self.bank)
        _,_,normal=run.rollout(agents,banks,self.bank,run.CONDITIONS['full_identity'],7323,120,greedy=False,trace=True)
        _,_,blocked=run.rollout(agents,banks,self.bank,run.CONDITIONS['full_blocked'],7323,120,greedy=False,trace=True)
        self.assertTrue(all(np.array_equal(a['sent'],b['sent']) and (b['delivered']==0).all() for a,b in zip(normal,blocked)))


if __name__=='__main__':unittest.main(verbosity=2)
