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
    def assert_worked_prose_arithmetic(self, body):
        # Every extraction is anchored to its worked paragraph, not a common
        # numeral elsewhere in the book. Arithmetic below is independent.
        vector = re.search(r"\*\*向量。\*\*[^\n]+", body).group()
        # Read both input vectors from the prose before computing the oracle.
        # A changed input must not pass just because the old printed answer stays.
        a_text, b_text = re.search(r"a=\(([^)]+)\).*?b=\(([^)]+)\)", vector).groups()
        a = tuple(float(x) for x in a_text.split(','))
        b = tuple(float(x) for x in b_text.split(','))
        self.assertEqual(len(a), len(b))
        number = r'[-+]?[0-9]+(?:\.[0-9]+)?'
        norm_a, norm_b = map(float, re.search(
            rf'长度分别 ({number}) 和 ({number})', vector).groups())
        dot = float(re.search(rf"点积 ({number})", vector).group(1))
        numerator, denominator, cosine = map(float, re.search(
            rf'余弦 \$({number})/({number})=({number})\$', vector).groups())
        actual_a = math.sqrt(sum(x*x for x in a))
        actual_b = math.sqrt(sum(x*x for x in b))
        actual_dot = sum(x*y for x,y in zip(a,b))
        self.assertAlmostEqual(norm_a, actual_a)
        self.assertAlmostEqual(norm_b, actual_b)
        self.assertAlmostEqual(dot, actual_dot)
        self.assertAlmostEqual(numerator, actual_dot)
        self.assertAlmostEqual(denominator, actual_a*actual_b)
        self.assertAlmostEqual(cosine, actual_dot/(actual_a*actual_b))

        probability = re.search(r"\*\*概率。\*\*[^\n]+", body).group()
        target_word = re.search(r"目标是第(.)类", probability).group(1)
        target = {'一': 0, '二': 1, '三': 2}[target_word]
        weights = (1,2,3)
        ce = -math.log(weights[target]/sum(weights))
        printed = float(re.search(r"approx([0-9.]+)", probability).group(1))
        self.assertAlmostEqual(ce, printed, places=6)

        gradient = re.search(r"\*\*梯度。\*\*[^\n]+", body).group()
        printed_grad = int(re.search(r"\(z-y\)x=([0-9]+)", gradient).group(1))
        w,x,y = 1,2,0
        self.assertEqual(printed_grad, (w*x-y)*x)
        printed_z, printed_loss = map(int, re.search(r'则 \$z=(\d+),L=(\d+)\$', gradient).groups())
        self.assertEqual(printed_z, w*x)
        self.assertEqual(printed_loss, (w*x-y)**2/2)

        payload = re.search(r"100 万个 float32 参数约 ([0-9,]+) 字节", body)
        self.assertIsNotNone(payload)
        self.assertEqual(int(payload.group(1).replace(',', '')), 10**6*4)

        claim = re.search(r"1024/16=(\d+)，与显式 (\d+) (不同|相同)", body)
        self.assertIsNotNone(claim)
        quotient,explicit = map(int,claim.group(1,2))
        config = json.loads((ROOT/'code/data/qwen3/dense/config.json').read_text())
        self.assertEqual(quotient, config['hidden_size']//config['num_attention_heads'])
        self.assertEqual(explicit, config['head_dim'])
        self.assertEqual(claim.group(3), '不同' if quotient != explicit else '相同')

    def test_six_worked_prose_claims_match_independent_arithmetic(self):
        self.assert_worked_prose_arithmetic(BODY)

    def test_actual_prose_mutations_fail_the_same_arithmetic_contract(self):
        self.assert_worked_prose_arithmetic(BODY)  # valid baseline first
        changes = [('a=(3,4)', 'a=(2,4)'),
                   ('b=(4,0)', 'b=(4,1)'),
                   ('点积 12', '点积 13'),
                   ('目标是第三类', '目标是第二类'),
                   ('(z-y)x=4', '(z-y)x=2'),
                   ('z=2,L=2', 'z=2,L=4'),
                   ('4,000,000 字节', '4,194,304 字节'),
                   ('1024/16=64，与显式 128 不同', '1024/16=128，与显式 128 相同')]
        for old,new in changes:
            with self.subTest(claim=old):
                self.assertEqual(BODY.count(old), 1)
                changed = BODY.replace(old,new,1)
                with self.assertRaises(AssertionError):
                    self.assert_worked_prose_arithmetic(changed)

    def assert_all_selfcheck_answers(self, body):
        line = re.search(r"\*\*答案。\*\* ([^\n]+)", body).group(1)
        answers = dict((int(n),text.strip()) for n,text in
                       re.findall(r"([1-6])）(.*?)(?=[1-6]）|$)", line))
        self.assertEqual(set(answers), set(range(1,7)))
        # Independent arithmetic plus precise local wording: duplicate numbers
        # elsewhere in the appendix cannot satisfy an incorrect answer.
        cache_bytes = 2*2*4*3*8*4
        self.assertEqual(answers[1], f'{cache_bytes} 字节，只是 payload 翻倍，不能推出速度。')
        logits = [math.log(n) for n in (1,2,3)]
        stable = lambda zs: [math.exp(z-max(zs))/sum(math.exp(t-max(zs)) for t in zs) for z in zs]
        for a,b in zip(stable(logits), stable([z+1000 for z in logits])):
            self.assertAlmostEqual(a,b,places=13)
        self.assertEqual(answers[2], '相同；减最大值后指数安全，直接指数可能溢出。')
        weighted = (1*2+3*4)/4
        unweighted = (2+4)/2
        self.assertEqual(answers[3], f'{weighted:g} 与 {unweighted:g}。')
        new_w = 1-1*((1*2-0)*2)
        new_loss = (new_w*2-0)**2/2
        self.assertEqual(answers[4], f"$w'={new_w}$、$L'={new_loss:g}$，大步长使本例更差。")
        cfg = json.loads((ROOT/'code/data/qwen3/dense/config.json').read_text())
        self.assertNotEqual(cfg['head_dim'], cfg['hidden_size']//cfg['num_attention_heads'])
        self.assertEqual(answers[5], f"该固定实现使用显式 {cfg['head_dim']}，需核对代码；等长不等于等字节，仍需 hash。")
        self.assertEqual(answers[6], '对本书而言待核实；若官方论文确有该结果，可另标作者报告，必须给硬件、软件、输入长度/batch、比较基线与计时方法，不能写本书复现。')

    def test_all_six_selfcheck_answers_match_independent_contracts(self):
        self.assert_all_selfcheck_answers(BODY)

    def test_selfcheck_answer_mutations_fail_without_shared_numeral_matches(self):
        self.assert_all_selfcheck_answers(BODY)
        changes = [('1）1536 字节', '1）768 字节'),
                   ('2）相同；', '2）不同；'),
                   ('3）3.5 与 3。', '3）3 与 3。'),
                   ("4）$w'=-3$、$L'=18$", "4）$w'=-3$、$L'=9$"),
                   ('5）该固定实现使用显式 128', '5）该固定实现使用显式 64'),
                   ('6）对本书而言待核实', '6）对本书而言已核实')]
        for old,new in changes:
            with self.subTest(answer=old):
                self.assertEqual(BODY.count(old),1)
                with self.assertRaises(AssertionError):
                    self.assert_all_selfcheck_answers(BODY.replace(old,new,1))

    def test_config_source_pin_is_not_replaceable_by_main(self):
        expected = 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json'
        def assert_pin(body):
            match = re.search(r'^\[qwen-config\]: (\S+)$',body,re.M)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1),expected)
        assert_pin(BODY)
        changed = BODY.replace(expected, expected.replace('c1899de289a04d12100db370d81485cdf75e47ca','main'))
        self.assertNotEqual(changed,BODY)
        with self.assertRaises(AssertionError): assert_pin(changed)

    def test_legacy_ids_preserved_without_duplicate_explicit_ids(self):
        old = json.loads((ROOT/'tests/fixtures/p28_legacy_anchors.json').read_text())
        self.assertEqual(len(old), 8)
        ids = re.findall(r'\{#([^}]+)\}', BODY)
        self.assertEqual(len(ids), len(set(ids)))
        for anchor in old + ['p28.d-运行与硬件先做小实验再决定是否扩展',
                             'p28.e-看到一条新模型新闻怎样不被标题带着走',
                             'p28.f-自测校验与最后一步', 'p28.g-本页原始来源与版本']:
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
        # A coherent different example also passes: this is extraction, not a
        # pinned old-vector literal masquerading as an independent computation.
        original = re.search(r"\*\*向量。\*\*[^\n]+", BODY).group()
        revised = "**向量。** $a=(-3,4)$、$b=(4,0)$：长度分别 5 和 4，点积 -12，余弦 $-12/20=-0.6$。"
        self.assert_worked_prose_arithmetic(BODY.replace(original, revised, 1))
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
        self.assertEqual(len(re.findall(r'^\d\. ',BODY.split('## 附录 F：')[1].split('**答案。**')[0],re.M)),6)

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
