import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('window_retention_audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Enc:
    def encode_ordinary(self, text):
        return list(text.encode())


class Features(np.ndarray):
    def split(self, width, dim):
        return np.split(self, list(range(width, self.shape[dim], width)), axis=dim)


class HelperTests(unittest.TestCase):
    def test_label_modes_have_identical_eligibility(self):
        raw = dict(id='x', schema_version=1, block_size=64, tokenizer='gpt2', tokens=[1]*321,
                   qas=[dict(source_window=3, prompt_tokens=[1, 2], answer_tokens=[3], evidence='key'),
                        dict(source_window=0, prompt_tokens=[1]*50, answer_tokens=[3], evidence='too big')])
        labeled = audit.prepare_record(raw, 64, 'labeled', Enc())
        bare = audit.prepare_record(raw, 64, 'unlabeled', Enc())
        self.assertEqual(len(labeled['qas']), 1)
        self.assertEqual(len(bare['qas']), 1)
        self.assertEqual(bare['_sequence_windows'], 3)
        self.assertEqual(bare['_support_window'], 'source')
        self.assertEqual(bare['qas'][0]['_support_label_length'], 0)
        self.assertEqual(bare['qas'][0]['_support_tokens'], list(b'key\n'))
        self.assertNotIn('_support_tokens', raw['qas'][0])
        self.assertIsNone(audit.prepare_record(raw, 64, 'unlabeled', Enc(), [1]))

    def test_chunk_updates_and_boundary_only_resets(self):
        audit.torch = types.SimpleNamespace(zeros_like=np.zeros_like)
        features = np.arange(8, dtype=float).reshape(1, 4, 2).view(Features)
        memory = np.ones((1, 1, 2))*5
        original = memory.copy()
        calls = []
        def compress(chunk, prior):
            calls.append(prior.copy())
            return chunk.sum(axis=1, keepdims=True)+prior*.1
        model = types.SimpleNamespace(memory_compressor=compress,
            memory=lambda candidate, prior: candidate+prior*.8)
        normal, trace = audit.write_features(model, features, memory, 2)
        self.assertEqual(len(trace), 2)
        np.testing.assert_array_equal(calls[1], trace[0])
        calls.clear()
        both, _ = audit.write_features(model, features, memory, 2, True, True)
        zero, _ = audit.write_features(model, features, np.zeros_like(memory), 2)
        np.testing.assert_allclose(both, zero)
        calls.clear()
        reset, partial = audit.write_features(model, features, memory, 2, True, False)
        np.testing.assert_array_equal(calls[0], 0)
        np.testing.assert_array_equal(calls[1], partial[0])
        np.testing.assert_array_equal(memory, original)
        self.assertFalse(np.allclose(normal, reset))

    def test_paired_counts_and_error_partition(self):
        rows = []
        for i, (a, b) in enumerate([(1, 0), (0, 1), (0, 0), (1, 1), (1, 0)]):
            for name, value in [('before', a), ('after', b)]:
                rows.append(dict(index=i, source=0, condition=name, exact=value,
                                 ce_sum=1.0, answer_tokens=1))
        result = audit.paired(rows, 'before', 'after', 0, 100, 42)
        self.assertEqual(result['correct_to_incorrect'], 2)
        self.assertEqual(result['incorrect_to_correct'], 1)
        self.assertEqual(result['both_incorrect'], 1)
        self.assertEqual(result['both_correct'], 1)
        self.assertEqual(result['accuracy_delta_pp'], -20)
        self.assertEqual(result['correct_to_incorrect']+result['both_incorrect'], 3)

    def test_token_weighted_ce_and_accuracy(self):
        result = audit.aggregate([dict(exact=1, first=1, ce_sum=1, answer_tokens=1),
                                  dict(exact=0, first=1, ce_sum=6, answer_tokens=3)])
        self.assertEqual(result['answer_ce'], 1.75)
        self.assertEqual(result['exact_accuracy'], .5)

    def test_validation_checks_alignment_and_fingerprint(self):
        tokens = [0]*25
        tokens[18] = 3
        sample = dict(index=0, tokens=tokens, tokens_sha256=audit.digest(tokens),
                      source=0, start=1, end=2, answer_tokens=[3])
        audit.validate_sample(sample, 8, 10)
        bad = dict(sample, start=0)
        with self.assertRaises(ValueError):
            audit.validate_sample(bad, 8, 10)
        with self.assertRaises(ValueError):
            audit.validate_sample(dict(sample, tokens_sha256='wrong'), 8, 10)

    def test_shuffle_never_uses_identical_history(self):
        samples = [dict(tokens=[i]*25) for i in [1, 1, 2, 3]]
        donors = audit.donor_indices(samples, 8, 42)
        for i, j in enumerate(donors):
            self.assertNotEqual(samples[i]['tokens'][:16], samples[j]['tokens'][:16])
        with self.assertRaises(ValueError):
            audit.donor_indices([samples[0], samples[1]], 8, 42)

    def test_manifest_reuse_and_index_validation(self):
        samples = []
        for i in range(2):
            tokens = [i]*25
            samples.append(dict(index=i, source=i, start=1, end=2, answer_tokens=[i],
                                tokens=tokens, tokens_sha256=audit.digest(tokens)))
        manifest = dict(schema_version=1, block_size=8, sequence_windows=3,
                        assistance='none', sampling={}, samples=samples)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'manifest.json'
            path.write_text(json.dumps(manifest))
            args = types.SimpleNamespace(manifest=path, sequence_windows=3)
            self.assertEqual(audit.collect_samples(args, None, None, None, 8, 10, 'none'), manifest)
            manifest['samples'][1]['index'] = 0
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit.collect_samples(args, None, None, None, 8, 10, 'none')


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        audit.torch = torch

    def test_read_controls_and_teacher_forced_coordinates(self):
        class Attention(torch.nn.Module):
            def forward(self, x, memory):
                return memory.expand(-1, x.shape[1], -1)
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                block = torch.nn.Module()
                block.memory_attention = Attention()
                self.blocks = torch.nn.ModuleList([block])
            def forward(self, x, memory, update_memory):
                assert not update_memory
                return self.blocks[0].memory_attention(x, memory), None, memory
        model = Reader()
        x = torch.zeros(1, 4, dtype=torch.long)
        y = torch.tensor([[0, 3, 3, 0]])
        memory = torch.zeros(1, 1, 8)
        memory[:, :, 3] = 10
        samples = [dict(start=1, end=3)]
        normal = audit.score_batch(model, x, y, memory, samples)
        off = audit.score_batch(model, x, y, memory, samples, True)
        self.assertEqual(normal[0]['exact'], 1)
        self.assertEqual(off[0]['exact'], 0)
        self.assertEqual(normal[0]['answer_tokens'], 2)
        self.assertFalse(model.blocks[0].memory_attention._forward_hooks)

    def test_torch_chunk_resets_equal_zero_prior(self):
        class Writer:
            memory_compressor = staticmethod(lambda features, prior: features.mean(1, keepdim=True)+prior*.1)
            memory = staticmethod(lambda candidate, prior: candidate+prior*.8)
        features = torch.randn(2, 8, 4)
        prior = torch.randn(2, 1, 4)
        reset, trace = audit.write_features(Writer(), features, prior, 2, True, True)
        isolated, _ = audit.write_features(Writer(), features, torch.zeros_like(prior), 2)
        self.assertEqual(len(trace), 4)
        self.assertTrue(torch.allclose(reset, isolated))


if __name__ == '__main__':
    unittest.main()
