"""Independent payoff, sampling, gradient and information-route tests for v0.8."""
from collections import Counter
from itertools import permutations, product
from types import MethodType
from unittest.mock import patch
import unittest

import numpy as np
import torch

import camp
import run_experiment as run


class ComplementarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.prepared=[a.state_dict() for a in camp.make_agents(990081)]
        cls.bank=camp.ImageBank()

    def agents(self):
        agents=camp.remake_agents(27101,self.prepared,7,2,'identity')
        return agents,camp.projected_banks(agents,self.bank)

    def training(self,kind='additive',split=1):
        name=f'split{split}_{kind}' if split else f'full_{kind}'
        return run.training_batch_plan(27101,41,6,run.CONDITIONS[name])

    def test_fixed_sixty_run_design_and_two_action_budget(self):
        self.assertEqual(run.SEEDS,[27101,27102,27103,27104])
        self.assertEqual(run.REWARDS,{'additive':0.,'mixed':.5,'joint':1.})
        names={f'split{s}_{k}' for s in (1,2,3) for k in run.REWARDS}
        names|={f'{p}_{k}' for p in ('full','blocked') for k in run.REWARDS}
        self.assertEqual(set(run.CONDITIONS),names);self.assertEqual(4*len(names),60)
        for name,plan in run.CONDITIONS.items():
            self.assertEqual((plan['vocab'],plan['length'],plan['known'],plan['representation'],plan['schedule']),
                             (7,2,False,'identity','direct'))
            self.assertEqual(plan['blocked'],name.startswith('blocked_'))
        levels,ids=run.training_schedule(27101,2400,'direct')
        self.assertTrue((levels==6).all() and np.array_equal(ids,np.arange(2400)))

    def test_four_payoffs_order_invariance_and_non_affine_change(self):
        x=np.asarray([[0,0],[0,1],[1,0],[1,1]],np.float32)
        for lam in (0,.5,1):
            reward=run.utility(x,lam)
            self.assertTrue(np.array_equal(reward,[0,(1-lam)/2,(1-lam)/2,1]))
            self.assertTrue(np.array_equal(reward,run.utility(x[:,::-1],lam)))
            self.assertEqual(float(reward[3]-reward[1]-reward[2]+reward[0]),lam)

    def test_no_information_bounds_all_matching_supports(self):
        all_maps=list(permutations(range(6),2))
        for split in (1,2,3):
            train,held=camp.split_maps(split)
            for pool in (train,held,np.arange(30)):
                maps=camp.MAPS[pool];m=len(maps);scores=[]
                for first,second in product(range(6),repeat=2):
                    s=np.column_stack((maps[:,0]==first,maps[:,1]==second)).astype(float)
                    scores.append([s.mean(),s.prod(1).mean(),*[run.utility(s,l).mean() for l in (0,.5,1)]])
                maxes=np.asarray(scores).max(0)
                self.assertTrue(np.allclose(maxes,[1/6,1/m,*[(1-l)/6+l/m for l in (0,.5,1)]],atol=1e-7))
                independent=[int(a==f and b==w) for f,w in maps for a,b in product(range(6),repeat=2)]
                self.assertEqual(sum(independent),m);self.assertEqual(len(independent),m*36)
            fixed=all_maps[int(train[0])]
            self.assertTrue(not((camp.MAPS[held]==fixed).all(1)).any())

    def test_metrics_use_resource_identity_not_query_order(self):
        by_resource=np.asarray([[0,0],[0,1],[1,0],[1,1]],np.float32)
        goals=np.asarray([[0,1],[1,0],[1,0],[0,1]])
        successes=np.take_along_axis(by_resource,goals,1)
        stats=run.pair_metrics(run.utility(successes,.5),successes,goals)
        self.assertEqual(stats,dict(n=4,decisions=8,reward_sum=1.5,mean_reward=.375,reward_variance=.140625,positive_rewards=3,
            single_correct=4,single_accuracy=.5,both_correct=1,both_accuracy=.25,
            food_correct=2,water_correct=2,outcome_counts={'00':1,'01':1,'10':1,'11':1}))
        empty=run.pair_metrics(np.empty(0),np.empty((0,2)),np.empty((0,2),int))
        self.assertEqual(empty['n'],0);self.assertIsNone(empty['both_accuracy'])

    def test_sender_once_joint_logprob_entropy_baseline_and_shared_return(self):
        agents,banks=self.agents();sends=[];receives=[];draws=[]
        for who,a in enumerate(agents):
            old_send=a.send;old_receive=a.receive
            def capture_send(self,h,goal,inv,rng,greedy,_original=old_send,_who=who):
                result=_original(h,goal,inv,rng,greedy);sends.append((_who,goal.clone(),result));return result
            def capture_receive(self,msg,goal,inv,hist,menu,_original=old_receive,_who=who):
                result=_original(msg,goal,inv,hist,menu)
                receives.append((_who,msg.clone(),inv.clone(),hist.clone(),result[1]));return result
            a.send=MethodType(capture_send,a);a.receive=MethodType(capture_receive,a)
        original=run.draw
        def capture_draw(logits,rng,greedy):
            result=original(logits,rng,greedy);draws.append(result);return result
        with patch.object(run,'draw',capture_draw):
            _,learning,rows=run.rollout(agents,banks,self.bank,self.training('mixed'),74101,128,
                training=True,greedy=False,trace=True,split='train')
        self.assertEqual([x[0] for x in sends],[0,1]);self.assertEqual(len(receives),4)
        self.assertTrue(all((x[1]==0).all() for x in sends))
        for direction,scout in enumerate((0,1)):
            sender_index=0 if scout==0 else 1;receiver_index=0 if scout==0 else 1
            send_learning=learning[scout][sender_index]
            receive_learning=learning[1-scout][receiver_index]
            self.assertTrue(torch.equal(send_learning[0],sends[direction][2][1]))
            self.assertTrue(torch.equal(receive_learning[0],draws[2*direction][1]+draws[2*direction+1][1]))
            self.assertTrue(torch.equal(receive_learning[1],draws[2*direction][2]+draws[2*direction+1][2]))
            self.assertTrue(torch.equal(receive_learning[2],(receives[2*direction][4]+receives[2*direction+1][4])/2))
            target=torch.from_numpy(rows[direction]['reward']-1)
            self.assertTrue(torch.equal(send_learning[3],target) and torch.equal(receive_learning[3],target))
            a,b=receives[2*direction:2*direction+2]
            self.assertTrue(torch.equal(a[1],b[1]) and (a[2]==0).all() and (b[2]==0).all()
                            and (a[3]==0).all() and (b[3]==0).all())

    def test_receiver_uniforms_are_consecutive_independent_draws(self):
        agents,banks=self.agents();seed=74102;draw_seeds=[];original=run.draw
        def inspect_rng(logits,rng,greedy):
            state=rng.bit_generator.state
            probe=np.random.default_rng();probe.bit_generator.state=state
            draw_seeds.append(probe.random(len(logits)))
            return original(logits,rng,greedy)
        with patch.object(run,'draw',inspect_rng):
            run.rollout(agents,banks,self.bank,self.training(),seed,512,training=True,greedy=False,split='train')
        for scout in (0,1):
            rng=np.random.default_rng(seed+102+scout*1000)
            first,second=rng.random(256),rng.random(256)
            self.assertTrue(np.array_equal(draw_seeds[2*scout],first))
            self.assertTrue(np.array_equal(draw_seeds[2*scout+1],second))
            self.assertFalse(np.array_equal(first,second))

    def test_first_action_cannot_change_second_query(self):
        agents,banks=self.agents();plan=self.training();seed=74103
        _,_,base=run.rollout(agents,banks,self.bank,plan,seed,120,greedy=False,trace=True)
        original=run.draw;count=0
        def force_first(logits,rng,greedy):
            nonlocal count
            a,lp,ent=original(logits,rng,greedy)
            if count%2==0:a=(a+1)%6
            count+=1;return a,lp,ent
        with patch.object(run,'draw',force_first):
            _,_,changed=run.rollout(agents,banks,self.bank,plan,seed,120,greedy=False,trace=True)
        for before,after in zip(base,changed):
            self.assertFalse(np.array_equal(before['action'][:,0],after['action'][:,0]))
            self.assertTrue(np.array_equal(before['action'][:,1],after['action'][:,1]))
            self.assertTrue(np.array_equal(before['successes'][:,1],after['successes'][:,1]))

    def test_query_order_swap_preserves_greedy_resource_outcomes(self):
        agents,banks=self.agents();plan=run.CONDITIONS['split1_mixed'];seed=74104
        stats,_,rows=run.rollout(agents,banks,self.bank,plan,seed,120,trace=True)
        original=run.fixture
        def reverse(*args,**kwargs):
            world=original(*args,**kwargs)
            world['goals']=world['goals'][:,::-1].copy();world['menu']=world['menu'][:,::-1].copy()
            return world
        with patch.object(run,'fixture',reverse):
            flipped,_,other=run.rollout(agents,banks,self.bank,plan,seed,120,trace=True)
        for a,b in zip(rows,other):
            self.assertTrue(np.array_equal(a['sent'],b['sent']))
            self.assertTrue(np.array_equal(a['place'],b['place'][:,::-1]))
            self.assertTrue(np.array_equal(a['successes'],b['successes'][:,::-1]))
            self.assertTrue(np.array_equal(a['reward'],b['reward']))
        self.assertEqual(stats['outcome_counts'],flipped['outcome_counts'])

    def test_same_worlds_initial_actions_messages_across_reward_conditions(self):
        agents,banks=self.agents();outputs=[]
        train_ids={i for i,e in enumerate(self.bank.entries) if e['split']=='train'}
        for kind in run.REWARDS:
            plan=self.training(kind)
            stats,learning,rows=run.rollout(agents,banks,self.bank,plan,74105,512,
                training=True,greedy=False,trace=True,split='train')
            outputs.append((stats,learning,rows))
            for row in rows:
                self.assertTrue(set(row['photo_ids'].flatten())<=train_ids)
                self.assertTrue(set(map(tuple,row['positions']))<=set(map(tuple,camp.MAPS[plan['map_pool']])))
        self.assertEqual(len({x[0]['world_sha256'] for x in outputs}),1)
        for other in outputs[1:]:
            for a,b in zip(outputs[0][2],other[2]):
                self.assertTrue(all(np.array_equal(a[k],b[k]) for k in (*run.WORLD_KEYS,'sent','delivered','action','place','successes')))

    def test_balanced_worlds_and_native_blocked_modes(self):
        agents,banks=self.agents()
        for kind in run.REWARDS:
            scores=run.evaluate(agents,banks,self.bank,run.CONDITIONS[f'blocked_{kind}'],74106,120)
            self.assertEqual(set(scores),{'normal','shuffle','blank','stochastic','erase_memory'})
            _,_,rows=run.rollout(agents,banks,self.bank,run.CONDITIONS[f'blocked_{kind}'],74106,120,
                greedy=False,trace=True)
            for row in rows:
                self.assertTrue((row['delivered']==0).all())
                self.assertTrue(np.array_equal(np.sort(row['goals'],axis=1),np.tile([0,1],(60,1))))
                counts=Counter((tuple(p),int(g)) for p,g in zip(row['positions'],row['goals'][:,0]))
                self.assertEqual(counts,Counter({(p,g):1 for p in permutations(range(6),2) for g in (0,1)}))
            for value in scores.values():
                self.assertEqual((value['n'],value['decisions']),(120,240))
                self.assertEqual(sum(value['outcome_counts'].values()),120)

    def test_gradients_remain_personal_and_projection_frozen(self):
        for kind in run.REWARDS:
            agents,banks=self.agents()
            _,learning,_=run.rollout(agents,banks,self.bank,self.training(kind),74107,128,
                training=True,greedy=False,split='train')
            loss=sum(-(lp*(target-value).detach()).mean()+.5*(value-target).square().mean()
                     for lp,ent,value,target in learning[0])
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in agents[0].slot_phi.parameters()))
            self.assertTrue(all(p.grad is None for p in agents[1].parameters()))
            self.assertTrue(all(p.grad is None for a in agents for p in a.project.parameters()))
            self.assertTrue(all(a.input_transform.grad is None and not a.input_transform.requires_grad for a in agents))


if __name__=='__main__':unittest.main(verbosity=2)
