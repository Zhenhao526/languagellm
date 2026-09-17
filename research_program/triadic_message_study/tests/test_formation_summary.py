"""Synthetic descriptions only, no real results or trained parameters."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_program.triadic_message_study import formation_summary as f


class DescriptionTests(unittest.TestCase):
    def test_all_code_bins_and_position_frequencies(self):
        codes=np.arange(4096)
        digits=(codes[:,None]//np.array([512,64,8,1]))%8
        m=np.broadcast_to(digits[:,None,None,:],(4096,2,3,4)).astype(np.int8)
        full,tokens=f.frequency_counts(m)
        np.testing.assert_array_equal(full,1)
        np.testing.assert_array_equal(tokens,512)
        with self.assertRaisesRegex(ValueError,'Invalid generated messages'):
            f.frequency_counts(np.full((2,2,3,4),8,dtype=np.int8))

    def test_changed_token_message_and_world_denominators(self):
        before=np.zeros((3,2,3,4),dtype=np.int8);after=before.copy()
        after[0,0,1,:]=1;after[1,1,2,0]=7
        result=f.transition_counts(before,after)
        self.assertEqual(result['worlds'],3)
        self.assertEqual(result['worlds_with_any_generated_message_change'],2)
        self.assertEqual(np.sum(result['changed_complete_messages_by_window_agent']),2)
        self.assertEqual(np.sum(result['changed_tokens_by_window_agent_position']),5)

    def test_paired_directions_and_incomplete_batch_rejection(self):
        rows=[]
        for i,s in enumerate(f.SEEDS):
            for c in f.CONDITIONS:
                for p in f.PARTITIONS:
                    rates={'FI_silent':.3,'FI_live':.4,'PI_silent':.2,'PI_live':.1+i*.1}
                    rows.append({'seed':s,'condition':c,'partition':p,'greedy_full_success_rate':rates[c],
                                 'greedy_reward_mean':.5})
        contrasts,means=f.endpoint_contrasts(rows)
        self.assertEqual(len(contrasts),16);self.assertEqual(len(means),4)
        self.assertLess(contrasts[0]['PI_live_minus_silent'],0)
        self.assertAlmostEqual(means[0]['contrasts']['PI_live_minus_silent']['mean'],.05)
        with self.assertRaisesRegex(ValueError,'All64'):f.endpoint_contrasts(rows[:-1])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'execution').mkdir()
            (p/'execution/status.json').write_text(json.dumps({'status':'running','completed_runs':2}))
            out=p/'summary'
            with self.assertRaisesRegex(ValueError,'Complete sixteen-run'):
                f.analyze(p,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
