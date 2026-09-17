"""Static fixtures and six stub sender calls; no real model activity."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_role_decoder_study import dataset as d


def spec():
    return dict(needs=[(0,1,6),(0,6,1),(1,0,6)],layouts=[(0,1,2,3),(3,1,0,2)],
                private_sites=[(1,2,3),(2,3,1)],world_count=12,monitor_indices=[0,3,4,7,8,11])


class DatasetTests(unittest.TestCase):
    def test_vector_features_equal_official_encoder_and_labels(self):
        actual=d.make_arrays(spec())
        official=d.original_dataset.make_arrays(spec())
        self.assertEqual(set(actual),{'packed_states','x_PL','x_FI','labels'})
        for name in ('packed_states','x_PL','x_FI'):
            np.testing.assert_array_equal(actual[name],official[name])
        expected=np.repeat(np.array([[2,0,1],[1,1,0],[0,2,2]],dtype=np.int8),4,axis=0)
        np.testing.assert_array_equal(actual['labels'],expected)
        self.assertEqual(actual['labels'].dtype,np.int8)
        self.assertEqual(actual['x_PL'].dtype,np.float64)
        self.assertFalse(np.shares_memory(actual['x_PL'],actual['x_FI']))
        # Permuting material locations or their private owners cannot change partner labels.
        self.assertTrue(all(np.unique(row,axis=0).shape[0]==1 for row in actual['labels'].reshape(3,4,3)))

    def test_make_arrays_rejects_bad_support_without_forward(self):
        bad=[]
        a=spec();a['world_count']=13;bad.append(a)
        a=spec();a['layouts'][0]=(0,0,2,3);bad.append(a)
        a=spec();a['private_sites'][0]=(1,1,3);bad.append(a)
        a=spec();a['needs'][0]=(0,0,0);bad.append(a) # Multiple full plans.
        a=spec();a['needs'][0]=(24,1,6);bad.append(a)
        a=spec();a['needs'][0]=(0.,1.,6.);bad.append(a)
        with patch.object(d.core.base,'actor_forward',side_effect=AssertionError('No model call')):
            for value in bad:
                with self.subTest(spec=value),self.assertRaises(ValueError):d.make_arrays(value)

    def test_transcript_routes_both_windows_and_repeated_indices(self):
        arrays=d.make_arrays(spec())
        messages=(np.arange(12*2*3*4).reshape(12,2,3,4)%8).astype(np.int8)
        ids=np.array([7,0,7,11],dtype=np.int64)
        out=d.build_inputs(arrays,ids,'Transcript',messages)
        self.assertEqual(out.shape,(4,3,252))
        np.testing.assert_array_equal(out[:,:,:54],arrays['x_PL'][ids])
        for b,world in enumerate(ids):
            for viewer in range(3):
                for window in range(2):
                    start=54+99*window
                    expected=np.zeros(99)
                    for sender in range(3):
                        for position in range(4):
                            expected[sender*32+position*8+messages[world,window,sender,position]]=1
                    expected[-3:]=1
                    np.testing.assert_array_equal(out[b,viewer,start:start+99],expected)
        before=deepcopy(arrays);before_messages=messages.copy()
        out[:]=99
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[key])
        np.testing.assert_array_equal(messages,before_messages)

    def test_own_and_fi_ignore_all_messages_and_labels(self):
        arrays=d.make_arrays(spec());ids=np.array([2,0,2])
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('Forbidden hidden input read')
        arrays['labels']=Poison();arrays['rewards']=Poison();arrays['correct_plan']=Poison()
        for view,key in (('Own','x_PL'),('FI','x_FI')):
            out=d.build_inputs(arrays,ids,view,messages=Poison())
            np.testing.assert_array_equal(out[:,:,:54],arrays[key][ids])
            np.testing.assert_array_equal(out[:,:,54:],0)
            self.assertEqual(out.shape,(3,3,252))
            self.assertEqual(out.dtype,np.float64)

    def test_rejects_invalid_views_indices_and_selected_messages(self):
        arrays=d.make_arrays(spec());messages=np.zeros((12,2,3,4),np.int8)
        for ids in ([],[.2],[-1],[12],[[0]]):
            with self.subTest(indices=ids),self.assertRaises(ValueError):d.build_inputs(arrays,ids,'Own')
        with self.assertRaises(ValueError):d.build_inputs(arrays,[0],'Labels',messages)
        with self.assertRaises(ValueError):d.build_inputs(arrays,[0],'Transcript')
        for invalid in (messages.astype(np.int16),messages[:1],messages.astype(float)):
            with self.subTest(shape=invalid.shape,dtype=invalid.dtype),self.assertRaises(ValueError):
                d.build_inputs(arrays,[0],'Transcript',invalid)
        for value in (-1,8):
            bad=messages.copy();bad[0,0,0,0]=value
            with self.assertRaises(ValueError):d.build_inputs(arrays,[0],'Transcript',bad)
        bad=deepcopy(arrays);bad['x_PL'][0,0,0]=np.nan
        with self.assertRaises(ValueError):d.build_inputs(bad,[0],'Own')

    def test_initial_messages_has_six_synchronous_sender_calls_only(self):
        x=d.make_arrays(spec())['x_PL'][:3]
        expected=np.empty((3,2,3,4),np.int8)
        for b in range(3):
            for window in range(2):
                for actor in range(3):expected[b,window,actor]=(3*actor+window+np.arange(4)+b)%8
        calls=[]
        def forward(module,observed):
            self.assertNotIn(module,(2,5,8),'Action head called')
            actor,window=divmod(module,3)
            if window==0:np.testing.assert_array_equal(observed,x[:,actor])
            else:
                self.assertEqual(calls[:3],[0,3,6])
                route=np.zeros((3,99))
                for b in range(3):
                    for sender in range(3):
                        for p in range(4):route[b,sender*32+p*8+expected[b,0,sender,p]]=1
                    route[b,-3:]=1
                np.testing.assert_array_equal(observed,np.concatenate((x[:,actor],route),axis=1))
            calls.append(module)
            logits=np.zeros((3,4,8))
            for b in range(3):
                for p in range(4):logits[b,p,expected[b,window,actor,p]]=4
            return logits.reshape(3,32),None
        with patch.object(d.core.base,'actor_forward',side_effect=forward), \
             patch.object(d.core,'rollout',side_effect=AssertionError('Whole rollout forbidden')):
            out=d.initial_messages(list(range(9)),x)
        self.assertEqual(calls,[0,3,6,1,4,7])
        np.testing.assert_array_equal(out['messages'],expected)
        self.assertEqual(out['messages'].dtype,np.int8)
        self.assertEqual(out['neural_forward_calls'],6)
        self.assertEqual(out['neural_forward_samples'],18)
        self.assertEqual(out['action_forward_calls'],0)
        self.assertEqual(out['action_forward_samples'],0)

    def test_initial_messages_invalid_inputs_rejected_before_forward(self):
        with patch.object(d.core.base,'actor_forward',side_effect=AssertionError('Forward forbidden')):
            for networks,x in (([],np.zeros((1,3,54))), (list(range(9)),np.zeros((0,3,54))),
                               (list(range(9)),np.zeros((1,3,54),np.float32)),
                               (list(range(9)),np.full((1,3,54),np.nan))):
                with self.assertRaises(ValueError):d.initial_messages(networks,x)

    def test_saved_loader_verifies_hash_rows_and_reads_only_required_arrays(self):
        arrays=d.make_arrays(spec());states=arrays['packed_states']
        messages=(np.arange(12*24).reshape(12,2,3,4)%8).astype(np.int8)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'fixture.npz'
            np.savez_compressed(path,states=states,state_indices=np.arange(12),messages=messages,
                                action_probabilities=np.array([object()],dtype=object))
            digest=d.sha(path)
            loaded=d.load_final_messages(path,states,expected_sha256=digest)
            np.testing.assert_array_equal(loaded,messages)
            with self.assertRaises(ValueError):d.load_final_messages(path,states,expected_sha256='0'*64)
            changed=states.copy();changed[0,0]=1
            with self.assertRaises(ValueError):d.load_final_messages(path,changed,expected_sha256=digest)
            for i,change in enumerate(('indices','shape','dtype','symbol')):
                target=Path(td)/f'bad{i}.npz';indices=np.arange(12);m=messages.copy()
                if change=='indices':indices=indices[::-1]
                if change=='shape':m=m[:1]
                if change=='dtype':m=m.astype(np.int16)
                if change=='symbol':m[0,0,0,0]=8
                np.savez_compressed(target,states=states,state_indices=indices,messages=m)
                with self.assertRaises(ValueError):d.load_final_messages(target,states,expected_sha256=d.sha(target))

    def test_source_manifest_is_static_and_binds_both_initial_arms(self):
        with (patch.object(d.np,'load',side_effect=AssertionError('Array loading forbidden in manifest')),
             patch.object(d.core,'load_networks',side_effect=AssertionError('Parameter load forbidden')),
             patch.object(d.core,'make_networks',side_effect=AssertionError('Model construction forbidden')),
             patch.object(d.core.base,'actor_forward',side_effect=AssertionError('Forward forbidden'))):
            manifest=d.source_manifest()
        self.assertEqual(len(manifest['policies']),8)
        self.assertEqual(len(manifest['initial_sources']),4)
        self.assertEqual(manifest['counts']['final_npz'],32)
        self.assertEqual(manifest['counts']['monitor0_npz'],32)
        self.assertEqual(sum(p['world_count'] for p in manifest['partitions'].values()),774144)
        for row in manifest['checkpoint0_equalities']:
            self.assertTrue(row['byte_identical'])
            self.assertEqual(row['a50']['sha256'],row['a10']['sha256'])
        for policy in manifest['policies']:
            for group in ('final','monitor0'):
                for item in policy[group].values():
                    self.assertEqual(manifest['source_sha256'][item['path']],item['sha256'])


if __name__=='__main__':unittest.main()
