import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from ..isolation import ROOT
from ..raw import build_raw,RawCodeDataset,raw_windows
from ..tasks import group_split
from ..train import parser


class LaunchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_raw_preparation_signal_resume_and_identity(self):
        # A tiny stand-in codec tests the production file/stop protocol only.
        # Actual frozen-codec fidelity is audited by real preparation separately.
        class BPE:
            def encode(self,text):return [ord(c) for c in text]
        class Codec:
            interrupt=True
            def __init__(self,*a,**kw):
                self.bpe=BPE();self.pad=0;self.model=torch.nn.Linear(1,1);self.identity={'codec_id':'unit_only'}
            def encode_ids(self,ids):
                if Codec.interrupt:
                    Codec.interrupt=False;signal.raise_signal(signal.SIGTERM)
                return ids.repeat(1,6)[:,:641]
            def decode_logits(self,codes):return F.one_hot(codes[:,:128],num_classes=128).float()
        docs=[]
        for split in ('train','valid','test'):
            i=next(i for i in range(1000) if group_split('duplicate:id'+str(i))==split)
            docs.append(dict(doc_id=str(i),family='f'+str(i),duplicate_cluster='id'+str(i),
                license='synthetic unit fixture',source='unit',case_sensitive=False,text=chr(65+len(docs))*400))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);source=root/'documents.jsonl';source.write_text(''.join(json.dumps(d)+'\n' for d in docs))
            out=root/'cache'
            with patch('v15_astra_v3.isolation.active_legacy_processes',return_value=[]),patch('v15_astra_v3.codec.FrozenNCMCodec',Codec):
                with self.assertRaises(SystemExit) as stopped:build_raw(source,out,batch=2,audit_blocks=6)
                self.assertEqual(stopped.exception.code,75)
                progress=json.loads((out/'preparation.json').read_text());self.assertEqual(progress['encoded_blocks'],2)
                with self.assertRaises(ValueError):build_raw(source,out,batch=3,audit_blocks=6)
                manifest=build_raw(source,out,batch=2,audit_blocks=6)
                self.assertTrue(manifest['codec_state_unchanged']);self.assertEqual(manifest['decode_audit']['nonpad_token_accuracy'],1.)
                data=RawCodeDataset(out);self.assertEqual(set(data.indices),{'train','valid','test'})
                self.assertEqual(build_raw(source,out,batch=2,audit_blocks=6),manifest)

    def test_shared_raw_blocks_do_not_cross_evaluation_windows(self):
        docs=[]
        for n,split in enumerate(('train','valid')):
            i=next(i for i in range(1000) if group_split('duplicate:id'+str(i))==split)
            docs.append(dict(doc_id=str(i),family='f'+str(i),duplicate_cluster='id'+str(i),license='fixture',
                source='unit',case_sensitive=False,text='same'+chr(65+n)*12))
        raw=raw_windows(docs,lambda s:list(s.encode()),history=2,block_size=4)
        self.assertGreater(raw['cross_split_shared_blocks'],0)
        blocks={s:set() for s in ('train','valid')}
        for e in raw['examples']:blocks[e['split']].update(e['context']+[e['target']])
        self.assertFalse(blocks['train']&blocks['valid'])

    def test_explicit_absolute_deadline_and_schedule_are_separate(self):
        args=parser().parse_args(['--run','unused','--data','unused','--arm','code_gram','--task','raw',
            '--steps','100000000','--schedule-steps','130000','--seconds','86400','--deadline','1792500000'])
        self.assertEqual(args.schedule_steps,130000);self.assertEqual(args.seconds,86400)
        self.assertEqual(args.deadline,1792500000)


if __name__=='__main__':unittest.main(verbosity=2)
