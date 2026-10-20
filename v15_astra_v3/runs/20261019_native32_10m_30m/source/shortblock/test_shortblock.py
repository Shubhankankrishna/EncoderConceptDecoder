"""Behavioral tests for short-block conversion, causality and diagnostics."""
from dataclasses import replace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from ..model import AstraHRM, Config
from ..raw import raw_windows
from ..tasks import group_split
from .codec import resize_root, LEVELS
from .evaluate import text_metrics
from .data import exclude_probed_test_families


class CodecConversionTests(unittest.TestCase):
    def test_down_preserves_piecewise_constant_input(self):
        torch.manual_seed(4)
        weight=torch.randn(5,3,128,dtype=torch.float64)
        short=torch.randn(2,3,32,dtype=torch.float64)
        original=F.conv1d(short.repeat_interleave(4,-1),weight,stride=128)
        converted=F.conv1d(short,resize_root(weight,down=True),stride=32)
        torch.testing.assert_close(original,converted,atol=1e-12,rtol=1e-12)

    def test_up_matches_group_average(self):
        torch.manual_seed(5)
        weight=torch.randn(3,5,128,dtype=torch.float64)
        x=torch.randn(2,3,1,dtype=torch.float64)
        original=F.conv_transpose1d(x,weight,stride=128).reshape(2,5,32,4).mean(-1)
        converted=F.conv_transpose1d(x,resize_root(weight,down=False),stride=32)
        torch.testing.assert_close(original,converted,atol=1e-12,rtol=1e-12)

    def test_linear_root_is_same_operation_as_convolution(self):
        torch.manual_seed(6)
        x=torch.randn(3,7,32)
        weight=torch.randn(7,7,32);bias=torch.randn(7)
        expected=F.conv1d(x,weight,bias,stride=32)
        actual=F.linear(x.reshape(3,-1),weight.reshape(7,-1),bias)[:,:,None]
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)
        root=torch.randn(3,7,1)
        expected=F.conv_transpose1d(root,weight,bias,stride=32)
        actual=F.linear(root[:,:,0],weight.reshape(7,-1).T).reshape(3,7,32)+bias[None,:,None]
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)

    def test_native_hierarchy_has_no_padding_expansion(self):
        self.assertEqual(LEVELS[-1],32)
        self.assertEqual(sum(LEVELS),161)
        self.assertTrue(all(a<b for a,b in zip(LEVELS,LEVELS[1:])))


class ContextContractTests(unittest.TestCase):
    def test_probe_exclusion_preserves_fit_splits(self):
        raw=dict(examples=[dict(split='test',group='probed'),dict(split='test',group='untouched'),
            dict(split='train',group='training'),dict(split='valid',group='validation')])
        result=exclude_probed_test_families(raw,['probed'])
        self.assertEqual([e['group'] for e in result['examples']],['untouched','training','validation'])
        self.assertEqual(result['excluded_probed_test_windows'],1)

    def test_full_context_and_no_cross_document_windows(self):
        documents=[]
        for split in ('train','valid','test'):
            number=next(i for i in range(10000) if group_split(f'duplicate:d{i}')==split)
            base=(number+1)*1000
            documents.append(dict(doc_id=f'd{number}',family=f'f{number}',duplicate_cluster=f'd{number}',
                license='test fixture',source='fixture',case_sensitive=False,
                text=' '.join(str(base+i) for i in range(320))))
        raw=raw_windows(documents,lambda s:list(map(int,s.split())),history=8,block_size=32)
        self.assertEqual(len(raw['examples']),6)
        for e in raw['examples']:
            context=raw['block_ids'][e['context']].reshape(-1)
            target=raw['block_ids'][e['target']]
            self.assertEqual(len(context),256)
            self.assertEqual(len(target),32)
            self.assertEqual(int(context[-1])+1,int(target[0]))

    def test_future_target_level_does_not_change_earlier_decoder_prediction(self):
        torch.manual_seed(7)
        config=Config(width=24,code_dim=4,vocab=16,levels=(1,2,4),heads=2,ff=32,
            ctx_layers=1,head_layers=1,slots=2,state_length=2,context_source='codes',context_pooling='hybrid')
        model=AstraHRM(config).eval()
        observed=torch.randint(16,(1,8,7));target=torch.randint(16,(1,7))
        with torch.no_grad():
            memory=model.encode_context(observed,None);state,_=model.reason(memory)
            first=model.decode_hidden(memory,state,target,level=0)
            changed=target.clone();changed[:,1:]=(changed[:,1:]+3)%16
            second=model.decode_hidden(memory,state,changed,level=0)
        torch.testing.assert_close(first,second,atol=0,rtol=0)


class DiagnosticTests(unittest.TestCase):
    def test_repetition_and_seams_are_descriptive(self):
        result=text_metrics('cup cup cup cup cup cup',['cup cup cup','cup cup cup'])
        self.assertEqual(result['dominant_word_fraction'],1.)
        self.assertEqual(result['same_word_seam_count'],1)
        self.assertEqual(result['duplicate_chunk_fraction'],.5)
        self.assertGreater(result['repeated_trigram_fraction'],0)

    def test_empty_output_metrics_are_finite(self):
        result=text_metrics('',['',''])
        for key,value in result.items():
            if isinstance(value,(float,int)):self.assertTrue(np.isfinite(value))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
