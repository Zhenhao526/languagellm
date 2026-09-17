"""Known compositional/constant protocols test the intervention estimands."""
import itertools,unittest
import numpy as np
import metrics,support


def oracle():
    positions=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
    mids=np.tile(np.repeat(np.arange(30),3),2)
    pos=positions[mids];codes=7*pos[:,0]+pos[:,1]
    lp=np.full((180,49),-100.,np.float32);lp[np.arange(180),codes]=0
    logits=np.full((49,2,6),-100.,np.float32)
    for c in range(49):
        logits[c,0,min(c//7,5)]=0;logits[c,1,min(c%7,5)]=0
    return dict(map_id=mids,photo_ids=np.tile([[2,11],[6,11],[9,11]],(60,1)),positions=pos,
        shown=np.repeat((0,1),90),tokens=pos.copy(),sender_log_probs=lp,receiver_logits=logits)


class Tests(unittest.TestCase):
    def test_known_factored_protocol_and_shuffle_supports(self):
        raw=oracle()
        for p,arm in itertools.product((1,2,3),support.ARMS):
            s=metrics.social(raw,p,arm)
            for g,b in s.items():
                for m,v in b.items():
                    self.assertEqual(v['J'],1);self.assertAlmostEqual(v['Q'],1)
                    self.assertAlmostEqual(v['shuffle_J'],1/30)
            self.assertAlmostEqual(s['common_target6']['pooled']['within_shuffle_J'],1/6)
            self.assertEqual(s['common_target6']['pooled']['recombine_FW_J'],1)
            self.assertEqual(s['common_target6']['pooled']['recombine_WF_J'],0)

    def test_uniform_action_probability(self):
        raw=oracle();raw['receiver_logits'][:]=0
        s=metrics.social(raw,1,support.ARMS[0])
        self.assertEqual(s['common30']['pooled']['J'],0)
        self.assertAlmostEqual(s['common30']['pooled']['Q'],1/36)

    def test_identity_bijection_and_coordinate_swap(self):
        raw=oracle();identity=np.arange(49);swap=7*(identity%7)+identity//7
        r=metrics.recombination(raw,1,support.ARMS[0],np.stack([identity,swap]))
        np.testing.assert_array_equal(r['FW'][0],np.ones(36));np.testing.assert_array_equal(r['WF'][0],np.zeros(36))
        np.testing.assert_array_equal(r['FW'][1],np.zeros(36));np.testing.assert_array_equal(r['WF'][1],np.ones(36))
        for perm in (identity,swap):
            inverse=np.argsort(perm);old=7*raw['tokens'][:,0]+raw['tokens'][:,1]
            np.testing.assert_array_equal(raw['receiver_logits'][inverse[perm[old]]],raw['receiver_logits'][old])

if __name__=='__main__':unittest.main(verbosity=2)
