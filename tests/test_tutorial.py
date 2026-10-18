"""Small CPU regression tests for the tutorial's formulas and teaching code."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'code' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TutorialTests(unittest.TestCase):
    def test_vectors(self):
        m = load('01_vector_similarity')
        self.assertAlmostEqual(m.cosine_similarity([8,9,5],[7.8,8.5,4.8]), .9999, places=4)
        self.assertAlmostEqual(m.euclidean_distance([8,9,5],[7.8,8.5,4.8]), .5745, places=4)
        with self.assertRaises(ValueError): m.cosine_similarity([0,0],[1,0])

    def test_manual_attention(self):
        m = load('02_manual_attention')
        self.assertEqual(m.matmul_vec([[.5,.2],[.1,.8]],[1,.5]), [.6,.5])
        self.assertAlmostEqual(sum(m.softmax([10000,10001,9999])),1.)
        # Symmetric raw scores do not imply symmetric row-softmax weights.
        scores = torch.tensor([[1.,2.],[2.,4.]])
        weights = scores.softmax(-1)
        self.assertNotAlmostEqual(weights[0,1].item(),weights[1,0].item())

    def test_sdpa(self):
        with contextlib.redirect_stdout(io.StringIO()): load('03_scaled_dot_product').compare_with_pytorch()

    def test_multihead_reference_and_mask(self):
        m=load('04_multi_head_attention').SimpleMultiHeadAttention(8,2)
        x=torch.randn(2,4,8)
        mask=torch.ones(4,4).tril()
        out,w=m(x,mask=mask)
        q,k,v=[layer(x).view(2,4,2,4).transpose(1,2) for layer in [m.W_q,m.W_k,m.W_v]]
        ref=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        ref=m.W_o(ref.transpose(1,2).contiguous().view(2,4,8))
        torch.testing.assert_close(out,ref)
        self.assertTrue(torch.equal(w.triu(1),torch.zeros_like(w)))
        torch.testing.assert_close(w.sum(-1),torch.ones(2,2,4))
        changed=x.clone();changed[:,3]+=100
        torch.testing.assert_close(m(changed,mask=mask)[0][:,:3],out[:,:3])
        # -1e9 cannot be represented by float16; -inf supports the same causal mask.
        half=m.half(); out,_=half(x.half(),mask=mask)
        self.assertTrue(torch.isfinite(out).all())

    def test_positional_even_and_odd_dimensions(self):
        m=load('05_positional_encoding')
        for d in [1,3,7,8]:
            pe=m.generate_positional_encoding(8,d)
            for pos in range(8):
                for col in range(d):
                    angle=pos/10000**(2*(col//2)/d)
                    expected=math.sin(angle) if col%2==0 else math.cos(angle)
                    self.assertAlmostEqual(pe[pos,col].item(),expected,places=6)

    def test_layernorm_and_block_causality(self):
        m=load('06_transformer_block'); x=torch.randn(2,5,16)
        ln=m.ManualLayerNorm(16)
        with torch.no_grad(): ln.gamma.copy_(torch.randn(16));ln.beta.copy_(torch.randn(16))
        torch.testing.assert_close(ln(x),F.layer_norm(x,(16,),ln.gamma,ln.beta,ln.eps))
        self.assertTrue(torch.isfinite(ln(torch.zeros_like(x))).all())
        block=m.TransformerBlock(16,4);mask=torch.ones(5,5).tril()
        expected=block(x,mask);changed=x.clone();changed[:,4]+=100
        torch.testing.assert_close(block(changed,mask)[:,:4],expected[:,:4])

    def test_baby_gpt(self):
        m=load('07_baby_gpt');model=m.BabyGPT(m.vocab_size)
        self.assertEqual(m.vocab_size,57)
        self.assertEqual(sum(p.numel() for p in model.parameters()),67872)
        self.assertEqual(m.encode('床前明月光'),[23,10,27,32,8])
        idx=torch.tensor([m.encode(m.poems[0])]);x=idx[:,:-1];target=idx[:,1:]
        logits,loss=model(x,target)
        self.assertEqual(tuple(logits.shape),(1,23,57))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        changed=x.clone();changed[:,-1]=0
        torch.testing.assert_close(model(changed)[0][:,:-1],logits[:,:-1])
        self.assertEqual(len(model.generate('床前',2,temperature=0)),4)
        with self.assertRaises(ValueError): model.generate('')
        # CUDA generation is checked only if a physical GPU is available.
        if torch.cuda.is_available():
            model.cuda();self.assertEqual(len(model.generate('床前',2,temperature=0)),4)

if __name__=='__main__': unittest.main()
