"""Independent finite-environment and information-route tests; no training."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb
from types import MethodType
import unittest

import numpy as np
import torch

import camp
import run_experiment as run


REFERENCE_MAPS = list(permutations(range(6), 2))
REFERENCE_MATCHINGS = {
    1: ((0, 1), (2, 3), (4, 5)),
    2: ((0, 2), (1, 4), (3, 5)),
    3: ((0, 3), (1, 5), (2, 4)),
}


def reference_train(split):
    omitted = set()
    if split:
        for a, b in REFERENCE_MATCHINGS[split]:
            omitted.update(((a, b), (b, a)))
    return [i for i, pair in enumerate(REFERENCE_MAPS) if pair not in omitted]


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank = camp.ImageBank()
        cls.prepared = [a.state_dict() for a in camp.make_agents(990061)]

    def agents(self, vocab=7, length=2):
        return camp.remake_agents(990061, self.prepared, vocab, length)

    def test_exact_map_space_and_balanced_disjoint_holdouts(self):
        self.assertEqual(camp.SITES, 6)
        self.assertEqual(camp.HISTORY, 18)
        self.assertEqual(camp.MAPS.tolist(), [list(p) for p in REFERENCE_MAPS])
        held_sets = []
        for split in range(4):
            train, held = camp.split_maps(split)
            self.assertEqual(train.tolist(), reference_train(split))
            self.assertEqual(set(train) | set(held), set(range(30)))
            self.assertFalse(set(train) & set(held))
            if split:
                self.assertEqual(len(train), 24); self.assertEqual(len(held), 6)
                for resource in (0, 1):
                    self.assertEqual(Counter(camp.MAPS[train, resource]), Counter({i:4 for i in range(6)}))
                    self.assertEqual(Counter(camp.MAPS[held, resource]), Counter({i:1 for i in range(6)}))
                held_sets.append(set(held))
        self.assertTrue(all(not(a & b) for a,b in combinations(held_sets,2)))

    def test_weighted_access_has_uniform_maps_and_exact_information_bounds(self):
        for split in range(4):
            train = set(reference_train(split))
            for level in (2, 4, 6):
                options = run.access_options(split, level)
                self.assertEqual(len({tuple(s) for s,_ in options}), len(options))
                total = sum(len(pool) for _,pool in options)
                self.assertEqual(total, len(train) * comb(4,level-2))
                probabilities = Counter()
                bound = Fraction(0)
                for sites,pool in options:
                    self.assertEqual(len(sites), level)
                    self.assertTrue(pool)
                    self.assertEqual(set(pool), {i for i in train if set(REFERENCE_MAPS[i]) <= set(sites)})
                    # No surviving orientation can uniquely reveal resource type.
                    pairs = {REFERENCE_MAPS[i] for i in pool}
                    self.assertTrue(all((b,a) in pairs for a,b in pairs))
                    self.assertGreaterEqual(len(pool),2)
                    counts = [Counter(REFERENCE_MAPS[i][r] for i in pool) for r in (0,1)]
                    subset_best = Fraction(max(counts[0].values())+max(counts[1].values()), 2*len(pool))
                    bound += Fraction(len(pool),total) * subset_best
                    for i in pool:
                        probabilities[i] += Fraction(1,total)
                self.assertEqual(set(probabilities),train)
                self.assertTrue(all(v==Fraction(1,len(train)) for v in probabilities.values()))
                wanted = {2:Fraction(1,2),4:Fraction(7,24),6:Fraction(1,6)} if split else {
                    2:Fraction(1,2),4:Fraction(1,4),6:Fraction(1,6)}
                self.assertEqual(bound,wanted[level])

    def test_access_sampler_uses_map_count_weights_and_never_leaks_holdouts(self):
        for split in (1,2,3):
            for level in (2,4,6):
                options=run.access_options(split,level)
                weights=np.asarray([len(p) for _,p in options],float)
                cumulative=np.cumsum(weights/weights.sum())
                for batch in range(40):
                    seed=25101
                    u=np.random.default_rng(seed+37120000+batch).random()
                    expected=min(int(np.searchsorted(cumulative,u,side='right')),len(options)-1)
                    plan=run.training_batch_plan(seed,batch,level,dict(split=split))
                    self.assertEqual(plan['allowed_sites'],list(options[expected][0]))
                    self.assertEqual(plan['map_pool'],options[expected][1])
                    self.assertTrue(set(plan['map_pool']) <= set(reference_train(split)))

    def test_fixed_56_run_scope_and_matched_curriculum_multiset(self):
        self.assertEqual(run.SEEDS,[25101,25102,25103,25104])
        wanted={f'split{s}_{kind}' for s in (1,2,3) for kind in ('course','mixed','direct','atomic_direct')}
        wanted |= {'full_direct','full_blocked'}
        self.assertEqual(set(run.CONDITIONS),wanted)
        self.assertEqual(len(run.SEEDS)*len(wanted),56)
        for name,plan in run.CONDITIONS.items():
            self.assertFalse(plan['known'])
            self.assertEqual(plan['vocab']**plan['length'],49)
            self.assertEqual(plan['blocked'],name=='full_blocked')
        for seed in run.SEEDS:
            c,ci=run.training_schedule(seed,2400,'course')
            m,mi=run.training_schedule(seed,2400,'mixed')
            d,di=run.training_schedule(seed,2400,'direct')
            self.assertEqual(c.tolist(),[2]*600+[4]*600+[6]*1200)
            self.assertTrue(np.array_equal(np.sort(mi),np.arange(2400)))
            self.assertTrue(np.array_equal(m,c[mi]))
            self.assertTrue(np.array_equal(ci,np.arange(2400)))
            self.assertTrue(np.array_equal(mi[2100:],ci[2100:]))
            self.assertTrue(np.all(d==6) and np.array_equal(di,ci))
            weights=np.r_[np.full(2100,.02),np.zeros(300)]
            self.assertTrue(np.array_equal(weights,weights[mi]))

    def test_collection_reference_exhaustive_states_and_refill_bins(self):
        cases=list(product(range(30),range(3),range(3),range(2),range(6),range(5)))
        values=np.asarray(cases)
        positions=np.asarray(REFERENCE_MAPS)[values[:,0]]
        inventory=values[:,1:3]
        goals,places,bins=values[:,3],values[:,4],values[:,5]
        u=(bins+.5)/5
        result=camp.collect(positions,inventory,goals,places,u,replenish=True)
        nxt,after,reward,gathered,overflow=result
        for i,(mapid,f,w,goal,place,binid) in enumerate(cases):
            p=list(REFERENCE_MAPS[mapid]); stock=[f,w]
            picked=[int(place==p[r]) for r in (0,1)]
            excess=[max(0,stock[r]+picked[r]-2) for r in (0,1)]
            for r in (0,1): stock[r]=min(2,stock[r]+picked[r])
            success=int(stock[goal]>0)
            stock[goal]-=success
            for r in (0,1):
                if picked[r]: p[r]=[x for x in range(6) if x!=p[1-r]][binid]
            self.assertEqual(after[i].tolist(),stock)
            self.assertEqual(int(reward[i]),success)
            self.assertEqual(gathered[i].tolist(),picked)
            self.assertEqual(overflow[i].tolist(),excess)
            self.assertEqual(nxt[i].tolist(),p)
        no_refill=camp.collect(positions,inventory,goals,places,u,replenish=False)
        self.assertTrue(np.array_equal(no_refill[0],positions))
        for actual,expected in zip(no_refill[1:],result[1:]):
            self.assertTrue(np.array_equal(actual,expected))

    def test_six_location_visual_binding_and_postprojection_empty_mask(self):
        projected=torch.arange(4*64,dtype=torch.float32).reshape(4,64)+1
        positions=np.asarray(REFERENCE_MAPS)
        ids=np.tile([0,1],(30,1))
        visual=camp.scene_visual(positions,ids,projected)
        self.assertEqual(tuple(visual.shape),(30,390))
        slots=visual[:,:384].reshape(30,6,64); exists=visual[:,384:]
        for row,pair in enumerate(REFERENCE_MAPS):
            for site in range(6):
                if site in pair:
                    self.assertTrue(torch.equal(slots[row,site],projected[pair.index(site)]))
                    self.assertEqual(float(exists[row,site]),1)
                else:
                    self.assertTrue(torch.equal(slots[row,site],torch.zeros(64)))
                    self.assertEqual(float(exists[row,site]),0)

    def test_goal_branch_and_all_720_private_menus(self):
        agent=self.agents()[0]
        menus=torch.tensor(list(permutations(range(6))))
        n=len(menus); messages=torch.tensor([[2,5]]).repeat(n,1)
        inventory=torch.zeros(n,2);history=torch.zeros(n,18)
        context=torch.cat((agent.receive_embedding(messages).flatten(1),inventory,history),-1)
        raw=agent.actor(context).reshape(n,2,6)
        for need in (0,1):
            goal=torch.zeros(n,2);goal[:,need]=1
            scores,_=agent.receive(messages,goal,inventory,history,menus)
            self.assertTrue(torch.equal(scores,raw[:,need].gather(1,menus)))
        self.assertEqual(agent.memory.input_size,391)

    def test_personal_storage_frozen_projection_and_gradient_isolation(self):
        agents=self.agents();again=self.agents()
        for a,b in zip(agents,again):
            self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
            self.assertTrue(all(not p.requires_grad for p in a.project.parameters()))
        pointers=[p.data_ptr() for a in agents for p in a.parameters()]
        self.assertEqual(len(pointers),len(set(pointers)))
        plan=run.training_batch_plan(25101,11,4,run.CONDITIONS['split1_course'])
        banks=camp.projected_banks(agents,self.bank)
        _,learning,_=run.rollout(agents,banks,self.bank,plan,731101,128,training=True,greedy=False,split='train')
        loss=sum(-lp.mean()+.5*(value-target).square().mean() for lp,ent,value,target in learning[0])
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in agents[0].parameters()))
        self.assertTrue(all(p.grad is None for p in agents[1].parameters()))
        self.assertTrue(all(p.grad is None for a in agents for p in a.project.parameters()))

    def test_balanced_evaluation_heldout_pools_and_hidden_goal_route(self):
        agents=self.agents();banks=camp.projected_banks(agents,self.bank)
        sent_goals=[]
        for agent in agents:
            original=agent.send
            def captured(self,h,visible_goal,inventory,rng,greedy,_original=original):
                sent_goals.append(visible_goal.detach().clone())
                return _original(h,visible_goal,inventory,rng,greedy)
            agent.send=MethodType(captured,agent)
        plan=run.CONDITIONS['split2_direct']
        _,_,records=run.rollout(agents,banks,self.bank,plan,731102,120,trace=True)
        testids={i for i,e in enumerate(self.bank.entries) if e['split']=='test'}
        for record in records:
            pairs=Counter((tuple(p),int(g)) for p,g in zip(record['positions'],record['goals']))
            self.assertEqual(pairs,Counter({(p,g):1 for p in REFERENCE_MAPS for g in (0,1)}))
            self.assertTrue(set(record['photo_ids'].flatten()) <= testids)
            self.assertEqual(record['history'].shape,(60,18))
            self.assertTrue((record['history']==0).all() and (record['inventory']==0).all())
            correct=record['positions'][np.arange(60),record['goals']]
            self.assertTrue(np.array_equal(record['reward'],(record['place']==correct).astype(float)))
        self.assertTrue(sent_goals and all(torch.count_nonzero(g)==0 for g in sent_goals))
        scores=run.evaluate(agents,banks,self.bank,plan,731102,120)
        self.assertEqual(len({v['world_sha256'] for v in scores.values()}),1)
        for value in scores.values():
            self.assertEqual(value['map_groups']['seen']['n'],96)
            self.assertEqual(value['map_groups']['unseen']['n'],24)

    def test_training_support_access_and_matched_course_mixed_worlds(self):
        agents=self.agents();banks=camp.projected_banks(agents,self.bank)
        trainids={i for i,e in enumerate(self.bank.entries) if e['split']=='train'}
        c,ci=run.training_schedule(25101,2400,'course')
        m,mi=run.training_schedule(25101,2400,'mixed')
        for split in (1,2,3):
            for identity in (0,599,600,1199,1200,2099,2100,2399):
                index=int(np.flatnonzero(mi==identity)[0])
                plans=[run.training_batch_plan(25101,identity,int(c[identity]),run.CONDITIONS[f'split{split}_course']),
                       run.training_batch_plan(25101,int(mi[index]),int(m[index]),run.CONDITIONS[f'split{split}_mixed'])]
                outputs=[run.rollout(agents,banks,self.bank,p,25101*100000+30000000+identity+1,
                        128,training=True,greedy=False,trace=True,split='train') for p in plans]
                self.assertEqual(outputs[0][0]['world_sha256'],outputs[1][0]['world_sha256'])
                for record in outputs[0][2]:
                    self.assertTrue(set(record['photo_ids'].flatten()) <= trainids)
                    self.assertTrue(set(map(tuple,record['positions'])) <= {REFERENCE_MAPS[i] for i in reference_train(split)})
                    self.assertTrue(np.isin(record['place'],plans[0]['allowed_sites']).all())

    def test_blocked_native_keeps_draws_and_never_delivers_messages(self):
        agents=self.agents();banks=camp.projected_banks(agents,self.bank)
        _,_,normal=run.rollout(agents,banks,self.bank,run.CONDITIONS['full_direct'],
                               731103,120,greedy=False,trace=True)
        _,_,blocked=run.rollout(agents,banks,self.bank,run.CONDITIONS['full_blocked'],
                                731103,120,greedy=False,trace=True)
        for a,b in zip(normal,blocked):
            self.assertTrue(np.array_equal(a['sent'],b['sent']))
            self.assertTrue(np.array_equal(a['positions'],b['positions']))
            self.assertTrue((b['delivered']==0).all())
        plan=run.training_batch_plan(25101,0,6,run.CONDITIONS['full_blocked'])
        _,_,rows=run.rollout(agents,banks,self.bank,plan,731104,128,
                            training=True,greedy=False,trace=True,split='train')
        self.assertTrue(all((r['delivered']==0).all() for r in rows))


if __name__=='__main__':
    unittest.main(verbosity=2)
