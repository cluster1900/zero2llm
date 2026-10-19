import importlib.util
from pathlib import Path
import unittest
import random
import math
import re

spec = importlib.util.spec_from_file_location(
    "generation", Path(__file__).resolve().parents[1] / "code/10_generation_loop.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class GenerationTests(unittest.TestCase):
    def test_label_shift(self):
        full = ["<BOS>", "我", "爱", "学", "习", "<EOS>"]
        x, labels = generation.shifted_pairs(full)
        self.assertEqual(x, ["<BOS>", "我", "爱", "学", "习"])
        self.assertEqual(labels, ["我", "爱", "学", "习", "<EOS>"])
        self.assertEqual(x[1:], labels[:-1])
        self.assertNotEqual(x, labels)

    def test_mask_three_stages(self):
        raw, masked, weights = generation.mask_walkthrough()
        expected = [[100,0,0,0],[13.01,86.99,0,0],[9.06,27.23,63.71,0],[4.07,8.19,27.20,60.54]]
        for i in range(4):
            self.assertAlmostEqual(sum(weights[i]), 1)
            for j in range(4):
                self.assertAlmostEqual(weights[i][j]*100, expected[i][j], delta=.005)
                if j > i:
                    self.assertEqual(masked[i][j], -math.inf)
                    self.assertEqual(weights[i][j], 0)
                else:
                    self.assertEqual(masked[i][j], raw[i][j])

    def test_temperature_values_and_low_limit(self):
        expected = [[86.68,11.73,1.59],[66.52,24.47,9.00],[50.65,30.72,18.63]]
        for temp, row in zip([.5,1.,2.],expected):
            for actual, target in zip(generation.probabilities([2.,1.,0.],temp),row):
                self.assertAlmostEqual(actual*100,target,delta=.005)
        self.assertEqual(generation.choose_token([2.,1.,0.]),0)
        self.assertEqual(generation.choose_token([2.,1.,0.],1e-6,random.Random(42)),0)
        self.assertEqual(generation.choose_token([2.,2.,0.]),0)  # explicit tie policy

    def test_eos_and_length_stop(self):
        tokens, trace, reason = generation.generate(generation.scripted_logits,["床","前"],generation.VOCAB,"<EOS>",8)
        self.assertEqual(tokens,["床","前","明","月","光","<EOS>"])
        self.assertEqual(len(trace),4)
        self.assertEqual(reason,"eos")
        tokens, trace, reason = generation.generate(generation.scripted_logits,["床","前"],generation.VOCAB,"<EOS>",2)
        self.assertEqual(tokens,["床","前","明","月"])
        self.assertEqual(reason,"max_new_tokens")
        self.assertEqual(len(trace),2)

    def test_zero_budget_and_existing_eos(self):
        def never(_):
            self.fail("model must not be called")
        self.assertEqual(generation.generate(never,["床"],generation.VOCAB,"<EOS>",0),(["床"],[],"max_new_tokens"))
        self.assertEqual(generation.generate(never,["<EOS>"],generation.VOCAB,"<EOS>",9),(["<EOS>"],[],"eos"))

    def test_sampling_reproducible_and_prefix_immutable(self):
        prefix=["床","前"]
        args=(generation.scripted_logits,prefix,generation.VOCAB,"<EOS>",5)
        self.assertEqual(generation.generate(*args,temperature=2.,seed=8),
                         generation.generate(*args,temperature=2.,seed=8))
        self.assertEqual(prefix,["床","前"])

    def test_cross_attention_example(self):
        weights = [.2, .3, .5]
        values = [[1.,0.],[0.,1.],[1.,1.]]
        output = [sum(w*v[col] for w,v in zip(weights,values)) for col in range(2)]
        self.assertEqual(output,[.7,.8])
        # Two queries each compare against three keys, not a 2x2 self-attention table.
        queries = [[1.,0.,0.,0.],[0.,1.,0.,0.]]
        keys = [[1.,0.,0.,0.],[0.,1.,0.,0.],[1.,1.,0.,0.]]
        scores = [[sum(a*b for a,b in zip(q,k)) / 2 for k in keys] for q in queries]
        self.assertEqual((len(scores),len(scores[0])),(2,3))

    def test_trace_probabilities_match_chapter(self):
        _, trace, _ = generation.generate(generation.scripted_logits,["床","前"],generation.VOCAB,"<EOS>",8)
        expected=[[93.41,4.65,1.71,.23],[1.71,93.41,4.65,.23],
                  [1.71,4.65,93.41,.23],[1.74,1.74,1.74,94.79]]
        for event,row in zip(trace,expected):
            for actual,target in zip(event["probs_at_t1"],row):
                self.assertAlmostEqual(actual*100,target,delta=.005)

    def test_special_token_labels_are_literal_markdown(self):
        chapter = (Path(__file__).resolve().parents[1] / "chapters/07_encoder_decoder.md").read_text()
        prose = "\n".join(chapter.split("`" * 3)[::2])
        prose = re.sub(r"`[^`]*`", "", prose)
        self.assertNotIn("<EOS>", prose)
        self.assertNotIn("<BOS>", prose)

    def test_invalid_inputs(self):
        for temp in [0,-1,math.inf,math.nan]:
            with self.assertRaises(ValueError):
                generation.probabilities([1.,2.],temp)
        for logits in [[],[math.inf],[math.nan]]:
            with self.assertRaises(ValueError):
                generation.choose_token(logits)
        with self.assertRaises(ValueError):
            generation.shifted_pairs(["only"])
        with self.assertRaises(ValueError):
            generation.generate(lambda _: [1],["床"],generation.VOCAB,"<EOS>",2)
