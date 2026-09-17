"""Fake-forward routing checks; no learned actors, parameters or training."""
from itertools import permutations
from unittest import TestCase, mock
import numpy as np
from research_program.triadic_context_transfer_study import channel_intervention as c


def fake_forward(net, x):
    # Handwritten finite arithmetic dependent on every input coordinate.
    value = np.rint(x @ (1 + np.arange(x.shape[1]) % 7)).astype(int)
    if net % 3 == 2:
        z = np.full((len(x), 17), -4.)
        z[np.arange(len(x)), (value + net) % 17] = 4.
    else:
        z = np.full((len(x), 4, 8), -4.)
        for t in range(4):
            z[np.arange(len(x)), t, (value + net + t) % 8] = 4.
        z = z.reshape(len(x), 32)
    return z, None


def natural(x, live=True):
    r = c.core.rollout(tuple(range(9)), x, live)
    p, _ = c.core.base.policy_distribution(r['action_logits'])
    return dict(messages=r['messages'], action_inputs=r['action_inputs'],
                action_probabilities=p, action_indices=p.argmax(-1))


class ChannelTests(TestCase):
    def test_all_outgoing_viewers_patched_self_and_bits_preserved(self):
        tokens = np.arange(3*3*4).reshape(3,3,4) % 8
        sender = np.arange(3); replacement = (tokens[np.arange(3),sender]+3)%8
        expected = c.core.routed_window(tokens, True)
        before = tokens.copy()
        got = c.replace_outward(tokens, sender, replacement, True)
        np.testing.assert_array_equal(got[:,:,-3:], expected[:,:,-3:])
        for row in range(3):
            for viewer in range(3):
                for source in range(3):
                    block = got[row,viewer,32*source:32*(source+1)]
                    wanted = replacement[row] if source==sender[row] and viewer!=sender[row] else tokens[row,source]
                    np.testing.assert_array_equal(block,np.eye(8)[wanted].reshape(32))
        np.testing.assert_array_equal(tokens,before)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_sham_replays_natural_and_exactly_six_fake_calls(self, forward):
        x = np.random.default_rng(271).integers(0,2,(3,3,54)).astype(float)
        original = natural(x); senders=np.arange(3)
        donor=original['messages'][np.arange(3),:,senders]
        forward.reset_mock()
        got=c.intervene(tuple(range(9)),x,original['messages'],senders,donor,True)
        self.assertEqual(forward.call_count,6)
        for key in ('messages','action_inputs','action_indices','action_probabilities'):
            np.testing.assert_array_equal(got[key],original[key])
        self.assertEqual(got['neural_forward_samples'],18)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_local_both_non_sender_input_identity_and_sender_self_retained(self, forward):
        x=np.zeros((3,3,54));senders=np.arange(3)
        donor_x=x.copy();donor_x[np.arange(3),senders,0]=1
        original=natural(x);donor=natural(donor_x)
        packet=donor['messages'][np.arange(3),:,senders]
        got=c.intervene(tuple(range(9)),x,original['messages'],senders,packet,True)
        for row,sender in enumerate(senders):
            np.testing.assert_array_equal(got['messages'][row,:,sender], original['messages'][row,:,sender])
            for viewer in range(3):
                if viewer!=sender:
                    np.testing.assert_array_equal(got['action_inputs'][row,viewer],donor['action_inputs'][row,viewer])
        # Receiver observations are never taken from the donor.
        np.testing.assert_array_equal(got['action_inputs'][:,:,:54],x)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_silent_alias_needs_no_forward_and_exact_routing(self, forward):
        x=np.zeros((2,3,54));original=natural(x,False)
        forward.reset_mock()
        got=c.silent_routes(x,original['messages'],np.array([0,2]),np.full((2,2,4),7))
        self.assertEqual(forward.call_count,0)
        np.testing.assert_array_equal(got['action_inputs'],original['action_inputs'])
        self.assertTrue(got['reused_natural'])

    def test_probability_rounding_tie_uses_softmax_first_argmax(self):
        def tiny(net,x):
            z=np.zeros((len(x),32 if net%3!=2 else 17))
            if net%3!=2:
                z[:,1::8]=1e-18
            else:
                z[:,1]=1e-18
            return z,None
        with mock.patch.object(c.core.base,'actor_forward',side_effect=tiny):
            got=c.intervene(tuple(range(9)),np.zeros((1,3,54)),np.zeros((1,2,3,4),dtype=int),
                            np.array([0]),np.ones((1,2,4),dtype=int),True)
        np.testing.assert_array_equal(got['messages'][:,1],0)
        np.testing.assert_array_equal(got['action_indices'],0)

    def test_new_24_need_observations_and_native_physics(self):
        needs=next(n for n in c.env.support() if max(n)>=12)
        rows=[];acts=[]
        for layout in permutations(range(4)):
            s=c.env.State(needs,layout,(3,1,2))
            rows.append(s.needs+s.layout+s.private_sites)
            witness=c.env.sufficient_information_witness(s)
            acts.append([c.env.all_actions(a).index(witness[a]) for a in c.env.AGENTS])
        rows=np.asarray(rows);acts=np.asarray(acts)
        pl=c.observations(rows,'PL');ll=c.observations(rows,'LL')
        self.assertEqual(pl.shape,(24,3,54))
        self.assertTrue(np.any(pl!=ll))
        np.testing.assert_array_equal(c.settle(rows,acts)['greedy_reward'],1)
        with self.assertRaises(ValueError):c.observations(rows,'FI')

    def test_invalid_donor_symbols_refused(self):
        with self.assertRaises(ValueError):
            c.replace_outward(np.zeros((1,3,4),dtype=int),np.array([0]),np.full((1,4),8),True)
        with self.assertRaises(ValueError):
            c.replace_outward(np.zeros((1,3,4),dtype=int),np.array([0]),np.zeros((1,4)),True)
