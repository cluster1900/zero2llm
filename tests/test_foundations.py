"""P00–P02 标准库回归测试：python3 -m unittest discover -s tests -v。"""
import importlib.util
from pathlib import Path
import unittest
from importlib.metadata import version, PackageNotFoundError
import random

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'code' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = load('01_vector_similarity')
bpe = load('08_byte_bpe')


class VectorTests(unittest.TestCase):
    def test_dot_and_norm(self):
        self.assertEqual(vectors.dot_product([3, 4], [1, 2]), 11)
        self.assertEqual(vectors.vector_magnitude([3, 4]), 5)

    def test_cosine_scaling(self):
        self.assertAlmostEqual(vectors.cosine_similarity([3, 4], [6, 8]), 1)
        self.assertEqual(vectors.dot_product([3, 4], [6, 8]), 50)
        self.assertAlmostEqual(vectors.cosine_similarity([3, 4], [-3, -4]), -1)

    def test_zero_vector(self):
        with self.assertRaises(ValueError):
            vectors.cosine_similarity([0, 0], [1, 2])

    def test_dimension_mismatch(self):
        for func in [vectors.dot_product, vectors.euclidean_distance]:
            with self.assertRaises(ValueError):
                func([1, 2], [1])

    def test_known_fruit_values(self):
        self.assertAlmostEqual(vectors.euclidean_distance([8, 9, 5], [7.8, 8.5, 4.8]), 0.57445626465)


class BPETests(unittest.TestCase):
    def setUp(self):
        self.model = bpe.ByteBPE().train(['banana banana', 'banana bandana', '你好，你好！'], 20)

    def test_round_trip(self):
        for text in ['', 'banana', '你好', '新词🙂 café', 'e\u0301', '\n\t  ', '👩🏽\u200d💻', '\x00']:
            with self.subTest(text=text):
                self.assertEqual(self.model.decode(self.model.encode(text)), text)

    def test_decode_iterator(self):
        self.assertEqual(self.model.decode(iter(self.model.encode("你好🙂"))), "你好🙂")

    def test_deterministic(self):
        again = bpe.ByteBPE().train(['banana banana', 'banana bandana', '你好，你好！'], 20)
        self.assertEqual(self.model.merges, again.merges)
        self.assertEqual(self.model.vocab, again.vocab)

    def test_base_alphabet(self):
        model = bpe.ByteBPE()
        self.assertEqual(len(model.vocab), 256)
        self.assertEqual(model.encode('你'), [228, 189, 160])
        self.assertEqual(model.decode(model.encode('你')), '你')

    def test_no_document_boundary_merge(self):
        model = bpe.ByteBPE().train(['a', 'b'], 10)
        self.assertEqual(model.merges, {})

    def test_tie_and_nonoverlap(self):
        model = bpe.ByteBPE().train(['abab'], 1)
        self.assertEqual(model.encode('abab'), [256, 256])
        self.assertEqual(model.vocab[256], b'ab')
        self.assertEqual(bpe.merge_pair([97, 97, 97], (97, 97), 256), [256, 97])
        tie = bpe.ByteBPE().train(['bc', 'ab'], 1)
        self.assertEqual(next(iter(tie.merges)), (97, 98))

    def test_train_resets(self):
        self.model.train([], 20)
        self.assertEqual(len(self.model.vocab), 256)
        self.assertEqual(self.model.merges, {})

    def test_invalid_ids(self):
        for ids in [[-1], [999999], [True], ['1']]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model.decode(ids)
        with self.assertRaises(UnicodeDecodeError):
            self.model.decode([255])

    def test_invalid_training_arguments(self):
        for n in [-1, 0.5, True]:
            with self.assertRaises(ValueError):
                self.model.train(['a'], n)
        with self.assertRaises(TypeError):
            self.model.train('not a list')

    def test_ranked_reference(self):
        # 独立参考：每轮只挑当前相邻对中训练排名最早的一对。
        rng = random.Random(42)
        for _ in range(100):
            text = ''.join(rng.choice('banana你🙂 ') for _ in range(30))
            ids = list(text.encode('utf-8'))
            while True:
                pairs = set(zip(ids, ids[1:])) & self.model.merges.keys()
                if not pairs:
                    break
                pair = min(pairs, key=self.model.merges.get)
                ids = bpe.merge_pair(ids, pair, self.model.merges[pair])
            self.assertEqual(self.model.encode(text), ids)
            self.assertEqual(self.model.decode(ids), text)

    def test_tokens_never_exceed_bytes(self):
        for text in ['banana banana', '你好，你好！', 'unseen🙂']:
            self.assertLessEqual(len(self.model.encode(text)), len(text.encode('utf-8')))
        self.assertLess(len(self.model.encode('banana banana')), len('banana banana'))


try:
    HF_VERSION = version('tokenizers')
except PackageNotFoundError:
    HF_VERSION = '未安装'


@unittest.skipUnless(
    importlib.util.find_spec('tokenizers') and HF_VERSION == '0.21.4',
    f'选修：需要 tokenizers==0.21.4，当前 {HF_VERSION}',
)
class HFTests(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = load('09_hf_tokenizer').build_tokenizer()
        for text in ['', '新词🙂 café', 'e\u0301', '👩🏽\u200d💻', '\n\t  ']:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text).ids), text)


if __name__ == '__main__':
    unittest.main()
