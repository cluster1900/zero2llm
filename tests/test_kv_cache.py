"""P11: full S0 reference, offset/mask mutations, explicit cache isolation."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from torch_test_support import torch_status, torch_reason

READY, VERSION, PRESENT = torch_status()
if READY:
    import torch
    spec = importlib.util.spec_from_file_location('cache_test', Path(__file__).resolve().parents[1] / 'code/12_kv_cache.py')
    kv = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kv
    spec.loader.exec_module(kv)


@unittest.skipUnless(READY, torch_reason('P11', VERSION, PRESENT))
class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.model = kv.s0.BabyGPT(17, 16,4,2,16).eval()
        self.tokens = torch.randint(17,(2,12))

    def compare_chunks(self, sizes):
        full = self.model(self.tokens)[0]
        cache, outputs, start = None, [], 0
        for size in sizes:
            out, cache = kv.cached_forward(self.model,self.tokens[:,start:start+size],cache)
            outputs.append(out)
            start += size
        self.assertEqual(start,12)
        torch.testing.assert_close(torch.cat(outputs,1), full, atol=3e-6, rtol=1e-5)
        return cache

    def test_chapter_rectangular_mask_hand_calculation(self):
        allowed, weights, outputs, wrong = kv.rectangular_mask_walkthrough()
        self.assertEqual(allowed.int().tolist(),[[1,1,1,1,0],[1,1,1,1,1]])
        torch.testing.assert_close(weights,torch.tensor([[.1,.2,.3,.4,0],
            [1/15,2/15,3/15,4/15,5/15]],dtype=torch.float64))
        torch.testing.assert_close(outputs,torch.tensor([30.,110/3],dtype=torch.float64))
        torch.testing.assert_close(wrong,torch.tensor([10.,50/3],dtype=torch.float64))
        self.assertFalse(torch.allclose(outputs,wrong))
        self.assertEqual(kv.expected_cache_bytes(kv.s0.GPTConfig(57),1,5),3840)

    def test_token_and_chunk_prefill_equivalence(self):
        for sizes in ([1]*12,[3,4,2,3],[12],[11,1]):
            with self.subTest(sizes=sizes):
                self.compare_chunks(sizes)
        logits, cache = kv.cached_forward(self.model,self.tokens[:,:1])
        torch.testing.assert_close(logits,self.model(self.tokens[:,:1])[0])
        self.assertEqual(cache.length,1)

    def test_memory_formula_actual_dtype(self):
        for dtype in (torch.float32,torch.float64):
            self.model.to(dtype)
            _, cache = kv.cached_forward(self.model,self.tokens)
            self.assertEqual(kv.cache_bytes(cache), 2*2*2*4*12*4*torch.empty((),dtype=dtype).element_size())
            self.assertEqual(kv.cache_bytes(cache), kv.expected_cache_bytes(self.model.config,2,12,dtype))
            self.assertEqual(cache.layers[0][0].shape,(2,4,12,4))
        self.assertEqual(kv.expected_cache_bytes(self.model.config,1,0),0)

    def test_reset_and_branch_do_not_mutate_prior_cache(self):
        _, first = kv.cached_forward(self.model,self.tokens[:,:4])
        snapshot = first.layers[0][0].clone()
        _, extended = kv.cached_forward(self.model,self.tokens[:,4:7],first)
        self.assertEqual(first.length,4)
        self.assertEqual(extended.length,7)
        torch.testing.assert_close(snapshot,first.layers[0][0],atol=0,rtol=0)
        second = self.tokens.flip(1)
        fresh, _ = kv.cached_forward(self.model,second,None)
        torch.testing.assert_close(fresh,self.model(second)[0])
        a = kv.generate_cached(self.model,self.tokens[:,:3],4)
        kv.generate_cached(self.model,second[:,:3],4)
        self.assertTrue(torch.equal(a,kv.generate_cached(self.model,self.tokens[:,:3],4)))

    def test_greedy_matches_uncached_loop(self):
        expected = self.tokens[:,:3].clone()
        for _ in range(5):
            logits,_ = self.model(expected)
            expected = torch.cat((expected,logits[:,-1].argmax(-1,keepdim=True)),1)
        self.assertTrue(torch.equal(expected,kv.generate_cached(self.model,self.tokens[:,:3],5)))

    def test_assigned_parameters_and_conversions_invalidate_cache(self):
        _, cache = kv.cached_forward(self.model, self.tokens[:, :4])
        other = kv.s0.BabyGPT(17,16,4,2,16).eval()
        # Replacing objects can preserve _version counters. Identity must count.
        self.assertEqual(tuple(p._version for p in self.model.parameters()),
                         tuple(p._version for p in other.parameters()))
        self.model.load_state_dict(other.state_dict(), assign=True)
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            kv.cached_forward(self.model, self.tokens[:, 4:5], cache)
        _, fresh = kv.cached_forward(self.model, self.tokens[:, :4])
        self.model.double()
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            kv.cached_forward(self.model, self.tokens[:, 4:5], fresh)
        # Reset works after both transformations and still agrees with S0.
        logits, reset = kv.cached_forward(self.model, self.tokens)
        torch.testing.assert_close(logits, self.model(self.tokens)[0])
        self.assertEqual(reset.layers[0][0].dtype, torch.float64)

    def test_tiny_temperature_matches_s0_greedy(self):
        prompt = self.tokens[:, :3]
        self.assertTrue(torch.equal(kv.generate_cached(self.model, prompt, 5, 1e-300),
                                   kv.generate_cached(self.model, prompt, 5, 0)))

    def test_malformed_cache_structure_shape_and_dtype(self):
        _, cache = kv.cached_forward(self.model, self.tokens[:, :4])
        k, v = cache.layers[0]
        cases = [
            ((), 'layer count'),
            ((cache.layers[0],), 'layer count'),
            (((k,), cache.layers[1]), 'K/V pair'),
            (((None, v), cache.layers[1]), 'shape'),
            (((k[0], v[0]), cache.layers[1]), 'shape'),
            (((k[:, :2], v[:, :2]), cache.layers[1]), 'shape'),
            (((k[..., :2], v[..., :2]), cache.layers[1]), 'shape'),
            (((k, v[:, :, :3]), cache.layers[1]), 'shape'),
            ((cache.layers[0], (k[:, :, :3], v[:, :, :3])), 'shape'),
            (((k.double(), v.double()), cache.layers[1]), 'dtype/device'),
        ]
        for layers, message in cases:
            with self.subTest(message=message, layers=len(layers)):
                malformed = kv.KVCache(layers, self.model, cache.versions)
                with self.assertRaisesRegex(ValueError, message):
                    kv.cached_forward(self.model, self.tokens[:, 4:5], malformed)
        # Same-shaped tensors on another device must fail before attention.
        malformed = kv.KVCache(((k.to('meta'), v.to('meta')), cache.layers[1]),
                               self.model, cache.versions)
        with self.assertRaisesRegex(ValueError, 'dtype/device'):
            kv.cached_forward(self.model, self.tokens[:, 4:5], malformed)

    def test_boundaries_and_foreign_stale_cache(self):
        _,cache = kv.cached_forward(self.model,self.tokens)
        with self.assertRaisesRegex(ValueError,'shape'):
            kv.cached_forward(self.model,self.tokens[:1,:1],cache)
        with self.assertRaisesRegex(ValueError,'max_seq_len'):
            kv.cached_forward(self.model,self.tokens[:,:5],cache)
        _, full_cache = kv.cached_forward(self.model,self.tokens[:,:4],cache)
        self.assertEqual(full_cache.length,16)
        other = kv.s0.BabyGPT(17,16,4,2,16).eval()
        with self.assertRaisesRegex(ValueError,'different'):
            kv.cached_forward(other,self.tokens[:,:1],cache)
        with torch.no_grad():
            self.model.head.weight.add_(.01)
        with self.assertRaisesRegex(ValueError,'changed'):
            kv.cached_forward(self.model,self.tokens[:,:1],cache)
        self.model.train()
        with self.assertRaisesRegex(ValueError,'eval'):
            kv.cached_forward(self.model,self.tokens)
        self.model.eval()
        for x in (self.tokens[:,:0],torch.zeros(1,17,dtype=torch.long),self.tokens.float()):
            with self.assertRaises(ValueError):
                kv.cached_forward(self.model,x)
        with self.assertRaises(ValueError):
            kv.generate_cached(self.model,self.tokens,5)
        with self.assertRaises(ValueError):
            kv.generate_cached(self.model,self.tokens,1,float('nan'))
        self.assertTrue(torch.equal(kv.generate_cached(self.model,self.tokens,0),self.tokens))

    def test_wrong_position_and_rectangular_mask_mutations_are_detected(self):
        # Execute isolated in-memory variants: never modify production files.
        source = Path(kv.__file__).read_text()
        changes = [
            ('pos = torch.arange(offset, offset + T, device=idx.device)',
             'pos = torch.arange(0, T, device=idx.device)'),
            ('allowed = key_positions <= query_positions',
             'allowed = key_positions <= torch.arange(T, device=x.device)[:, None]'),
        ]
        reference = self.model(self.tokens)[0][:,4:8]
        for old,new in changes:
            self.assertIn(old,source)
            scope = dict(kv.__dict__)
            exec(compile(source.replace(old,new),kv.__file__,'exec'),scope)
            _,cache = scope['cached_forward'](self.model,self.tokens[:,:4])
            mutated,_ = scope['cached_forward'](self.model,self.tokens[:,4:8],cache)
            self.assertGreater((reference-mutated).abs().max().item(),1e-3)

    def test_benchmark_smoke_and_contract(self):
        result = kv.benchmark(self.model,self.tokens,prompt_length=4,repeats=2,warmup=1)
        self.assertEqual(set(result),{'full_first_ms','cache_first_ms','full_later_ms','cache_later_ms'})
        for values in result.values():
            self.assertLessEqual(0,values['min'])
            self.assertLessEqual(values['min'],values['median'])
            self.assertLessEqual(values['median'],values['max'])


if __name__ == '__main__':
    unittest.main()
