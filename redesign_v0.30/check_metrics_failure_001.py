"""Known protocols validate conditional information, use, and upper bounds."""
import itertools,unittest
import numpy as np
import metrics,support


def oracle():
    positions=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
    mids=np.tile(np.repeat(np.arange(30),3),2);pos=positions[mids];codes=7*pos[:,0]+pos[:,1]
    lp=np.full((180,49),-100.,np.float32);lp[np.arange(180),codes]=0
    logits=np.full((49,2,6),-100.,np.float32)
    for c in range(49):logits[c,0,min(c//7,5)]=0;logits[c,1,min(c%7,5)]=0
    return dict(map_id=mids,photo_ids=np.tile([[2,11],[6,11],[9,11]],(60,1)),positions=pos,
        shown=np.repeat((0,1),90),tokens=pos.copy(),sender_log_probs=lp,receiver_logits=logits)


class Tests(unittest.TestCase):
    def test_known_composition_full_information(self):
        raw=oracle()
        for p,arm in itertools.product((1,2,3),support.ARMS):
            score=metrics.social(raw,p,arm)
            for group,b in score.items():
                for mask,v in b.items():
                    self.assertEqual(v['J'],1);self.assertAlmostEqual(v['Q'],1)
            for mask,v in score['target12'].items():
                self.assertAlmostEqual(v['within_shuffle_J'],1/12)
                self.assertEqual(v['recombine_FW_J'],1);self.assertEqual(v['recombine_WF_J'],0)
                for role,other in [('food','water'),('water','food')]:
                    self.assertEqual(v[f'swap_same_{role}_J'],0)
                    self.assertEqual(v[f'swap_same_{role}_{role}'],1);self.assertEqual(v[f'swap_same_{role}_{other}'],0)
                    self.assertEqual(v[f'pair_same_{role}_J'],1)
                    self.assertAlmostEqual(v[f'I_{other}_given_{role}_bits'],1)
                    self.assertEqual(v[f'I_greedy_{other}_given_{role}_bits'],1)
                self.assertAlmostEqual(v['bayes_message_Q'],1);self.assertEqual(v['bayes_greedy_J'],1)

    def test_fixed_food_information_cannot_exceed_half(self):
        raw=oracle();f=raw['positions'][:,0];raw['tokens']=np.c_[f,np.zeros(len(f),np.int64)]
        raw['sender_log_probs'][:]=-100;raw['sender_log_probs'][np.arange(len(f)),7*f]=0
        raw['receiver_logits'][:]=-100
        for c in range(49):
            site=min(c//7,5);raw['receiver_logits'][c,0,site]=0;raw['receiver_logits'][c,1,(site+1)%6]=0
        for arm in support.ARMS:
            s=metrics.social(raw,1,arm)['target12']['pooled']
            self.assertEqual(s['J'],.5);self.assertAlmostEqual(s['bayes_message_Q'],.5);self.assertEqual(s['bayes_greedy_J'],.5)
            self.assertAlmostEqual(s['I_water_given_food_bits'],0);self.assertEqual(s['I_greedy_water_given_food_bits'],0)
            self.assertEqual(s['pair_same_food_J'],0);self.assertEqual(s['swap_same_food_J'],.5)

    def test_constant_channel_and_uniform_actions(self):
        raw=oracle();raw['tokens'][:]=0;raw['sender_log_probs'][:]=0;raw['receiver_logits'][:]=0
        s=metrics.social(raw,1,support.ARMS[0])['target12']['pooled']
        self.assertEqual(s['J'],0);self.assertAlmostEqual(s['Q'],1/36)
        self.assertAlmostEqual(s['bayes_message_Q'],1/12);self.assertAlmostEqual(s['bayes_greedy_J'],1/12)
        for name in ('I_water_given_food_bits','I_food_given_water_bits','I_greedy_water_given_food_bits','I_greedy_food_given_water_bits'):self.assertAlmostEqual(s[name],0)

    def test_two_fixed_recombination_assignments(self):
        raw=oracle();identity=np.arange(49);swap=7*(identity%7)+identity//7
        r=metrics.recombination(raw,1,support.ARMS[0],np.stack([identity,swap]))
        self.assertEqual(r['FW'].shape,(2,72));np.testing.assert_array_equal(r['FW'][0],np.ones(72));np.testing.assert_array_equal(r['WF'][0],np.zeros(72))
        np.testing.assert_array_equal(r['FW'][1],np.zeros(72));np.testing.assert_array_equal(r['WF'][1],np.ones(72))
if __name__=='__main__':unittest.main(verbosity=2)
