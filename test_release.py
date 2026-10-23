"""CPU checks for release additions and unchanged evaluation inputs."""
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dataset_utils import canonical_book_id, book_split
import generate_RNN as generator
import window_retention_audit as audit

ROOT=Path(__file__).resolve().parent


class ReleaseHelpers(unittest.TestCase):
    def test_book_identity_and_stable_split(self):
        for value in ('20462', '060495_20462-8.txt', '20462:56320'):
            self.assertEqual(canonical_book_id(value),'20462')
        self.assertEqual(book_split('20462'),book_split(canonical_book_id('060495_20462-8.txt')))
        with self.assertRaises(ValueError):canonical_book_id('unidentified.txt')

    def test_frozen_audit_integrity(self):
        for name,count in (('no_assist_1024',1024),('trainer_validation_256',256)):
            directory=ROOT/'results'/'audits'/name
            manifest=json.loads((directory/'manifest.json').read_text())
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(audit.digest(manifest),summary['metadata']['manifest_sha256'])
            self.assertEqual(len(manifest['samples']),count)
            for sample in manifest['samples']:
                audit.validate_sample(sample,256,50257,5)
            if name=='trainer_validation_256':
                self.assertEqual(manifest,json.loads((directory/'trainer_validation_manifest.json').read_text()))

    def test_history_write_schedule(self):
        calls=[]
        def forward(tokens,memory,write):
            calls.append((list(tokens),memory,write))
            return 9, (0 if memory is None else memory)+1 if write else memory
        history,memory=generator.generate_history(forward,lambda _:7,[1,2],4,3)
        self.assertEqual(len(history),12)
        self.assertEqual(memory,3)
        self.assertEqual([x[0] for x in calls if x[2]],[[1,2,7,7],[7,7,7,7],[7,7,7,7]])
        self.assertTrue(all(len(tokens)==4 for tokens,_,write in calls if write))

    def test_downloader_records_revision_and_deduplicates(self):
        import types
        import download_gutenberg
        books=[dict(id='1',text='alpha'), dict(id='1-8',text='beta'),
               dict(id='2',text='alpha'), dict(id='3',text='gamma')]
        dataset_module=types.SimpleNamespace(load_dataset=lambda *a,**k:books)
        api=types.SimpleNamespace(dataset_info=lambda *a,**k:types.SimpleNamespace(sha='a'*40))
        hub_module=types.SimpleNamespace(HfApi=lambda **k:api)
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, datasets=dataset_module, huggingface_hub=hub_module):
                with patch.object(sys,'argv',['download_gutenberg.py','--output',temp]):
                    download_gutenberg.main()
            manifest=json.loads((Path(temp)/'download_manifest.json').read_text())
            self.assertEqual(manifest['revision'],'a'*40)
            self.assertEqual([b['source_id'] for b in manifest['books']],['1','3'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('tiktoken'),
                         'PyTorch and tiktoken required')
    def test_prose_split_keeps_editions_together_and_honors_qa(self):
        import config
        from data_loader import split_set
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'build_manifest.json').write_text(json.dumps(dict(books=[dict(book_id='1',split='validation') ])))
            files=[Path('000001_1.txt'),Path('000002_1-8.txt'),Path('000003_2.txt')]
            with patch.object(config,'GUTENBERG_QA_DIR',str(root)):
                train,valid=split_set(files)
                self.assertTrue(set(files[:2]).issubset(valid))
                self.assertFalse(set(files[:2]) & set(train))
                self.assertEqual((train,valid),split_set(list(reversed(files))))

    @unittest.skipUnless(importlib.util.find_spec('tiktoken'),'tiktoken required')
    def test_builder_materialization_and_cropping(self):
        import tiktoken
        import build_gutenberg_qa as builder
        import gutenberg_qa_loader as loader
        enc=tiktoken.get_encoding('gpt2');width=64
        prose='The captain carried a golden lantern. Eleanor visited the village market. '
        ids=enc.encode_ordinary(prose*200)[:5*width+1]
        qas=builder.window_questions(ids,enc,width,random.Random(42))
        self.assertEqual({q['source_window'] for q in qas},set(range(4)))
        record=dict(id='1:0',schema_version=1,block_size=width,tokenizer='gpt2',tokens=ids,qas=qas)
        for q in qas:
            a=q['source_window']*width+q['local_start'];b=q['source_window']*width+q['local_end']
            self.assertEqual(enc.decode(ids[a:b]),q['evidence'])
            self.assertIn(q['answer'],q['evidence'])
        for count in range(2,6):
            for seed in range(12):
                tokens,start,end,q=loader.materialize_record(dict(record,_sequence_windows=count),random.Random(seed))
                self.assertEqual(len(tokens),count*width+1)
                answer_position=(count-1)*width+start+1
                self.assertEqual(tokens[answer_position:answer_position+len(q['answer_tokens'])],q['answer_tokens'])
                self.assertEqual(tokens[(count-1)*width+end+1],198)
                base=q['source_window']*width
                self.assertEqual(enc.decode(tokens[base+q['local_start']:base+q['local_end']]),q['evidence'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
    def test_recurrent_gradients_and_activation_checkpoint_parity(self):
        # Separate interpreter prevents tiny test config from contaminating imports.
        settings=dict(block_size=16,emb_dims=16,memory_slots=2,memory_slot_dims=16,
            head_num=2,num_blocks=1,num_encoder_blocks=1,memory_pooling_heads=2,
            memory_routing_passes=3,memory_write_chunk_size=16,dropout=0.,MEMORY_WINDOWS_PER_SEQUENCE=3)
        code='''
import copy,torch
from miniGPT_model_RNN import MiniGPTRecall
from training_RNN_mixed import memory_training_forward
torch.set_num_threads(1);torch.manual_seed(42)
a=MiniGPTRecall(256);b=copy.deepcopy(a)
assert a.lm_head.weight is a.token_embedding_table.weight
x=torch.randint(0,256,(2,3,16));y=torch.randint(0,256,(2,3,16))
starts=torch.tensor([2,3]);ends=torch.tensor([3,5])
for i in range(2):y[i,-1,ends[i]]=198
results=[]
for model,check in [(a,False),(b,True)]:
 _,memory,lm,recall=memory_training_forward(model,x,y,starts,ends,check)
 (lm+recall).backward()
 assert memory.shape==(2,2,16)
 assert model.memory_encoder.blocks[0].attention.query.weight.grad.abs().sum()>0
 assert model.memory_compressor.value.weight.grad.abs().sum()>0
 assert all(g.weight_hh.grad.abs().sum()>0 for g in model.memory.grus)
 results.append((memory.detach(),lm.detach(),recall.detach()))
for lhs,rhs in zip(results[0],results[1]):torch.testing.assert_close(lhs,rhs)
for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
 assert n==m
 if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
print('CPU recurrent gradients and activation-checkpoint parity passed')
'''
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'tiny.json';config.write_text(json.dumps(settings))
            import os
            env=dict(os.environ,MINIGPT_CONFIG=str(config),OMP_NUM_THREADS='1')
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
