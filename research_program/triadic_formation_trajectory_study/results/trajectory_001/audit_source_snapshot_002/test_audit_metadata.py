"""Regression for full annotated metadata versus independent domain projection."""
from pathlib import Path
from copy import deepcopy
import unittest
from research_program.triadic_formation_trajectory_study import audit_execution_v2 as a
class MetadataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,context,_=a.references();cls.specs,_,_=context.independent_specs()
        cls.original=a.read(a.HERE.parent/'triadic_action_dependency_study/results/context_001/prepared.json')['partitions']
        cls.meta=a.read(a.HERE/'results/trajectory_001/dataset/pool.json')
    def test_exact_existing_static_definition_plus_original_annotations(self):
        a.validate_pool_metadata(self.meta,self.specs,self.original)
        for key,part in (('training_spec','train'),('validation_spec',a.PART)):
            self.assertEqual(len(self.specs[part]),8)
            self.assertEqual(set(self.meta[key])-set(self.specs[part]),{'partition','state_order','weighting','monitor_scope','semantic_pair_summary'})
    def test_changed_annotation_rejected_by_source_identity(self):
        m=deepcopy(self.meta);m['training_spec']['weighting']='changed'
        with self.assertRaisesRegex(AssertionError,'Complete source'):a.validate_pool_metadata(m,self.specs,self.original)
    def test_shared_source_data_change_rejected_by_independent_definition(self):
        m=deepcopy(self.meta);old=deepcopy(self.original)
        m['training_spec']['needs'][0][0]=23;old['train']['needs'][0][0]=23
        with self.assertRaisesRegex(AssertionError,'Independent pool'):a.validate_pool_metadata(m,self.specs,old)
