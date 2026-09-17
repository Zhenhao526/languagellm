import unittest
import numpy as np
from research_program.triadic_partner_ecology_study import design as d


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.prepared=d.make_prepared()

    def test_all_support_and_marginals(self):
        p=self.prepared
        self.assertEqual(p['exact_personal_need_marginals']['unique'],p['exact_personal_need_marginals']['multiple'])
        self.assertEqual(len(p['common_destination_layers']),21)
        self.assertEqual(len(p['runs']),32)
        self.assertEqual(p['full_natural_worlds_total'],3041280)
        for e in d.ECOLOGIES:
            for spec in p['partitions'][e].values():
                self.assertEqual(sum(len(x['need_indices']) for x in spec['demand_strata']),len(spec['needs']))
                self.assertTrue(all((len(d.r.base.env.compatible_pairs(n))==1 if e=='unique' else len(d.r.base.env.compatible_pairs(n))>=2) for n in spec['needs']))

    def test_sampling_boundaries_and_pairing(self):
        u=np.array([[0,0,0,0],[np.nextafter(1.,0.)]*4,[.2,.4,.6,.8]])
        for part in d.PARTITIONS:
            fields=[]
            for e in d.ECOLOGIES:
                spec=self.prepared['partitions'][e][part];ids=d.sample_indices(spec,u)
                self.assertTrue(((ids>=0)&(ids<spec['world_count'])).all())
                fields.append(d.pairing_fields(spec,ids))
            for key in fields[0]:np.testing.assert_array_equal(fields[0][key],fields[1][key])
        for bad in (np.ones((2,4)),np.zeros((2,3)),np.full((2,4),np.nan)):
            with self.assertRaises(Exception):d.sample_indices(spec,bad)

    def test_full_weights_stratified_and_not_uniform(self):
        for e in d.ECOLOGIES:
            for part in d.PARTITIONS:
                spec=self.prepared['partitions'][e][part];w=d.evaluation_weights(spec)
                self.assertAlmostEqual(w.sum(),1,places=12)
                physical=len(spec['layouts'])*6
                self.assertGreater(np.ptp(w),0)
                for layer in spec['demand_strata']:
                    ids=np.concatenate([np.arange(i*physical,(i+1)*physical) for i in layer['need_indices']])
                    self.assertAlmostEqual(w[ids].sum(),1/21,places=13)

    def test_frozen_monitor_equal_strata_and_shared_physical_worlds(self):
        for part in d.PARTITIONS:
            values=[]
            for e in d.ECOLOGIES:
                spec=self.prepared['partitions'][e][part];ids=np.array(spec['monitor_indices'])
                self.assertEqual(len(ids),672);self.assertEqual(len(set(ids)),672)
                w=d.evaluation_weights(spec,ids);self.assertAlmostEqual(w.sum(),1,places=12)
                fields=d.pairing_fields(spec,ids)
                records={}
                for D,l,o in zip(fields['destinations'],fields['layout_indices'],fields['owner_indices']):
                    records.setdefault(tuple(D),set()).add((int(l),int(o)))
                self.assertEqual(len(records),21);self.assertTrue(all(len(x)==32 for x in records.values()))
                values.append(records)
            self.assertEqual(values[0],values[1])


if __name__=='__main__':unittest.main()
