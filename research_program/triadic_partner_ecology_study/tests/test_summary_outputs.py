"""Pure fabricated descriptions; no reading of formal or partial results."""
from copy import deepcopy
import unittest
from research_program.triadic_partner_ecology_study import summarize_results as s
from research_program.triadic_partner_ecology_study import plot_results as p


def fixture():
    records=[];aliases=[]
    for seed in s.SEEDS:
        for eco in s.ECOLOGIES:
            for condition in s.CONDITIONS:
                for part in s.PARTITIONS:
                    for stage,updates in [('final',(6000,)),('monitor',s.CHECKPOINTS)]:
                        for update in updates:
                            for mode in ('natural','closed'):
                                row=dict(seed=seed,ecology=eco,condition=condition,partition=part,stage=stage,update=update,mode=mode)
                                if mode=='closed' and condition.endswith('_silent'):
                                    aliases.append(dict(**row,source_mode='natural'));continue
                                i=s.SEEDS.index(seed)
                                gain=([.02,-.03,.06,.01][i] if eco=='unique' else .01)
                                value=.5
                                if condition=='PI_live':value+=gain
                                if condition=='FI_live':value+=(-.02 if eco=='unique' else .03)
                                row.update(worlds=672 if stage=='monitor' else s.COUNTS[eco][part],
                                    weighted={m:value for m in s.METRICS})
                                records.append(row)
    return dict(status='completed',contract={k:list(v) for k,v in [('seeds',s.SEEDS),('ecologies',s.ECOLOGIES),
        ('conditions',s.CONDITIONS),('partitions',s.PARTITIONS),('checkpoints',s.CHECKPOINTS)]},
        evaluations=records,closed_aliases=aliases)


class SummaryTests(unittest.TestCase):
    def test_actual_grid_aliases_and_all_negative_contrasts_retained(self):
        data=fixture();index=p.validate(data)
        self.assertEqual(len(data['evaluations']),672)
        self.assertEqual(len(data['closed_aliases']),224)
        self.assertEqual(len(index),896)
        pairs=s.paired_contrasts(index)
        rows=[r for r in pairs['seed_values'] if r['partition']=='heldout_layouts' and r['metric']=='compatible_role_rate']
        for row,truth in zip(rows,(.01,-.04,.05,0)):
            self.assertAlmostEqual(row['PI_DiD_unique_minus_multiple'],truth)
            self.assertAlmostEqual(row['FI_DiD_unique_minus_multiple'],-.05)
        mean=next(x for x in pairs['equal_seed_means'] if x['partition']=='heldout_layouts' and x['metric']=='compatible_role_rate')
        self.assertAlmostEqual(mean['PI_DiD_unique_minus_multiple'],.005)
        aggregates=s.aggregate_curves_and_cells(index)
        self.assertEqual(len(aggregates),1120)
        self.assertTrue(all(list(x['seed_values'])==list(map(str,s.SEEDS)) for x in aggregates))

    def test_incomplete_actual_data_or_duplicate_alias_rejected(self):
        data=fixture();data['evaluations'].pop()
        with self.assertRaisesRegex(ValueError,'672 actual'):p.validate(data)
        data=fixture();data['closed_aliases'][1]=deepcopy(data['closed_aliases'][0])
        with self.assertRaisesRegex(ValueError,'Invalid silent alias'):p.validate(data)

    def test_all_proposals_distinguish_fixed_pair_and_role_changes(self):
        fixed=s.proposal_inventory([dict(action_indices=[1,1,0],worlds=7),dict(action_indices=[5,5,0],worlds=3)],10)
        self.assertEqual(fixed['globally_fixed_mutual_active_pair'],'AB')
        self.assertEqual(fixed['globally_always_waiting_agents'],['C'])
        self.assertEqual(fixed['agents'][0]['proposed_site_worlds'],{'0':7,'1':3,'2':0,'3':0})
        varied=s.proposal_inventory([dict(action_indices=[1,1,0],worlds=5),dict(action_indices=[2,0,1],worlds=5)],10)
        self.assertIsNone(varied['globally_fixed_mutual_active_pair'])
        self.assertEqual(varied['globally_always_waiting_agents'],[])
        self.assertEqual(varied['agents'][0]['partner_support'],['B','C'])
        overload=s.proposal_inventory([dict(action_indices=[1,2,1],worlds=10)],10)
        self.assertIsNone(overload['globally_fixed_mutual_active_pair'])
        self.assertTrue(all(x['transport_proposal_worlds']==10 for x in overload['agents']))
        with self.assertRaisesRegex(ValueError,'Incomplete full-domain'):
            s.proposal_inventory([dict(action_indices=[1,1,0],worlds=9)],10)


if __name__=='__main__':unittest.main()
