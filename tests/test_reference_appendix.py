"""P28: independent hand arithmetic, immutable evidence and reading contracts."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BODY = (ROOT/'chapters/10_appendix.md').read_text()


class ReferenceAppendixTests(unittest.TestCase):
    def test_legacy_ids_preserved_without_duplicate_explicit_ids(self):
        old = json.loads((ROOT/'tests/fixtures/p28_legacy_anchors.json').read_text())
        self.assertEqual(len(old), 8)
        ids = re.findall(r'\{#([^}]+)\}', BODY)
        self.assertEqual(len(ids), len(set(ids)))
        for anchor in old:
            self.assertIn(anchor, ids)

    def test_shape_card_numbers_and_grouped_cache_payload(self):
        rows = re.findall(r'\| [^\n]+ \| `\[([0-9,]+)\]` \| ([0-9]+) \|', BODY)
        self.assertEqual(len(rows), 4)
        for dimensions, expected in rows:
            self.assertEqual(math.prod(map(int, dimensions.split(','))), int(expected))
        self.assertEqual(2*2*2*3*8*4, 768)
        self.assertEqual(2*2*4*3*8*4, 1536)
        self.assertIn('768', BODY)
        self.assertIn('1536 字节', BODY)
        self.assertIn('不能推出速度', BODY)

    def test_positive_period_for_both_frequency_signs(self):
        self.assertIn(r'T=2\pi/\lvert\omega\rvert', BODY)
        self.assertIn('$D=32=H_q d$', BODY)
        for omega in (2., -2., .5, -.5):
            period = 2*math.pi/abs(omega)
            self.assertGreater(period, 0)
            for t in (.1, 1., 2.3):
                self.assertAlmostEqual(math.sin(omega*(t+period)), math.sin(omega*t), places=13)

    def test_vector_card_and_zero_denominator_boundary(self):
        a,b = (3,4),(4,0)
        dot = sum(x*y for x,y in zip(a,b))
        norm = lambda v: math.sqrt(sum(x*x for x in v))
        self.assertEqual((norm(a),norm(b),dot), (5,4,12))
        self.assertEqual(dot/(norm(a)*norm(b)), .6)
        self.assertIn('余弦 $12/20=0.6$', BODY)
        self.assertIn('零向量时分母为零', BODY)

    def test_probability_card_shift_and_cross_entropy(self):
        values = [math.log(1),math.log(2),math.log(3)]
        def normalized(logits):
            e = [math.exp(x-max(logits)) for x in logits]
            return [x/sum(e) for x in e]
        for offset in (0,1000,-1000):
            p = normalized([x+offset for x in values])
            for actual,expected in zip(p,(1/6,1/3,1/2)):
                self.assertAlmostEqual(actual,expected,places=13)
            self.assertAlmostEqual(-math.log(p[2]), math.log(2), places=13)
        self.assertIn('[1/6,1/3,1/2]', BODY)
        self.assertIn('0.693147', BODY)
        with self.assertRaises(OverflowError): math.exp(1000)

    def test_gradient_card_independent_finite_difference_and_two_steps(self):
        loss = lambda w: (2*w)**2/2
        h = 1e-5
        finite_difference = (loss(1+h)-loss(1-h))/(2*h)
        self.assertAlmostEqual(finite_difference,4,places=9)
        small = Fraction(1)-Fraction(1,10)*4
        large = Fraction(1)-4
        self.assertEqual((small,loss(small)), (Fraction(3,5),Fraction(18,25)))
        self.assertEqual((large,loss(large)),(-3,18))
        self.assertIn("$w'=0.6$，新损失 0.72",BODY)
        self.assertIn("$w'=-3$、$L'=18$",BODY)

    def test_loss_weighting_and_mib_units(self):
        token_mean = (1*Fraction(2)+3*Fraction(4))/4
        document_mean = (Fraction(2)+Fraction(4))/2
        self.assertEqual((token_mean,document_mean),(Fraction(7,2),3))
        self.assertIn('3.5 与 3',BODY)
        mib = Fraction(1000000*4,1024**2)
        self.assertEqual(f'{float(mib):.6f}', '3.814697')
        self.assertIn('3.814697 MiB',BODY)

    def test_worked_news_case_hash_fields_and_literal_projection(self):
        manifest = json.loads((ROOT/'code/data/qwen3/manifest.json').read_text())
        record = manifest['models']['dense']
        raw = (ROOT/'code/data/qwen3/dense/config.json').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(len(raw),726)
        self.assertEqual(digest,'660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd')
        self.assertEqual(digest,record['files']['config.json']['sha256'])
        config = json.loads(raw)
        self.assertEqual((config['hidden_size'],config['num_attention_heads'],config['head_dim']),(1024,16,128))
        self.assertEqual(config['hidden_size']//config['num_attention_heads'],64)
        self.assertEqual(config['num_attention_heads']*config['head_dim'],2048)
        for text in (record['revision'],digest,'726 字节','2048','64'):
            self.assertIn(text,BODY)
        modified = raw.replace(b'"hidden_size": 1024',b'"hidden_size": 1025')
        self.assertEqual(len(raw),len(modified))
        self.assertNotEqual(hashlib.sha256(modified).hexdigest(),digest)

    def test_four_evidence_states_and_no_performance_claim(self):
        for status in ('已核实','作者报告','本书复现','待核实'):
            self.assertIn('| '+status+' |',BODY)
        for boundary in ('未运行','没有下载真实权重','实现','审核','失效链接','跨版本/平台','不保证'):
            self.assertIn(boundary,BODY)
        self.assertIn('第',BODY)
        self.assertEqual(len(re.findall(r'^\d\. ',BODY.split('## P28.F')[1].split('**答案。**')[0],re.M)),6)

    def test_original_reading_list_and_versioned_sources(self):
        for link in ('https://arxiv.org/abs/1706.03762v7',
                     'https://www.3blue1brown.com/lessons/attention/',
                     'https://github.com/karpathy/nanoGPT',
                     'https://docs.pytorch.org/docs/2.14/notes/randomness.html'):
            self.assertIn(link,BODY)
        self.assertIn('2026-10-20',BODY)
        self.assertNotIn(r'\(',BODY)
        self.assertNotIn(r'\[',BODY)

    def test_commands_and_source_links_resolve_locally(self):
        for file in ('code/26_qwen3_case.py','scripts/check_book.py','tests/test_reference_appendix.py'):
            self.assertTrue((ROOT/file).is_file())
        self.assertIn('test_reference_appendix.py',BODY)
        self.assertIn('npm run check:layout',BODY)
        for relative in re.findall(r'\]\(([^)#]+\.md)\)',BODY):
            self.assertTrue((ROOT/'chapters'/relative).resolve().is_file(),relative)


if __name__ == '__main__':
    unittest.main()
