import unittest
from . import runner

class PrimaryTests(unittest.TestCase):
    def records(self):
        records=[]
        for i,seed in enumerate(runner.SEEDS):
            for rule in runner.RULES:
                for part in runner.PARTS:
                    q=.1+i*.01 if rule=='reciprocal' else .9
                    records.append(dict(seed=seed,training_rule=rule,part=part,settlements={
                        'reciprocal':dict(Q=q,Q_shuffle=.12,Q_excess=q-.12),
                        'strict':dict(Q=0,Q_shuffle=0,Q_excess=0)}))
        return records
    def test_unique_primary_includes_negative(self):
        value=runner.primary(self.records())
        self.assertAlmostEqual(value['mean_Q_excess'],-.005)
        self.assertEqual(value['paired_units'],4)
        self.assertEqual(len(value['by_seed']),4)
    def test_rejects_duplicate_or_missing(self):
        rows=self.records()
        with self.assertRaises(AssertionError):runner.primary(rows[:-1])
        rows[-1]=rows[0]
        with self.assertRaises(AssertionError):runner.primary(rows)

if __name__=='__main__':unittest.main()
