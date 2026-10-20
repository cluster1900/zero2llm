"""P24: independent scalar oracles, real semantic mutations and SSE boundaries."""
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/25_quantized_serving.py"
spec = importlib.util.spec_from_file_location("p24_serving", SCRIPT)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def exhaustive_oracle(values, scale, zero, low, high):
    """Nearest-grid search in exact rational arithmetic; no round/clip helper."""
    s = Fraction(scale)
    return [min(range(low, high + 1),
                key=lambda q: (abs(Fraction(x) - (q - zero) * s), (q - zero) % 2))
            for x in values]


def independent_attention(query, keys, values):
    scores = []
    for key in keys:
        score = 0.
        for j in range(len(query)):
            score += query[j] * key[j]
        scores.append(score / len(query) ** .5)
    # Small fixture: ordinary exp is safe and independent of production max shift.
    denom = sum(math.exp(score) for score in scores)
    result = [0.] * len(values[0])
    for i, score in enumerate(scores):
        for d in range(len(result)):
            result[d] += math.exp(score) / denom * values[i][d]
    return result


def event(data):
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\r\n\r\n").encode()


def stream_fixture():
    return (b": keepalive\r\n\r\n" + event({"choices": [{"delta": {"role": "assistant"}}]})
            + event({"choices": [{"delta": {"content": "春"}}]})
            + event({"choices": [{"delta": {"content": "风🌱"}, "finish_reason": "stop"}]})
            + event({"choices": [], "usage": {"completion_tokens": 4}})
            + b"data: [DONE]\r\n\r\n")


def load_mutant(old, new, suffix):
    """Compile a real altered implementation, never merely mutate expected data."""
    source = SCRIPT.read_text()
    if source.count(old) != 1:
        raise AssertionError(f"mutation target must occur once: {old}")
    module = types.ModuleType("p24_mutant_" + suffix)
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new, 1), str(SCRIPT), "exec"), module.__dict__)
    return module


class QuantizationTests(unittest.TestCase):
    def test_symmetric_hand_table_ties_and_clipping(self):
        params = p.QParams(.25, 0)
        values = [-40., -.625, -.375, -.125, 0., .125, .375, .625, 40.]
        codes = p.quantize(values, params)
        self.assertEqual(codes, [-127, -2, -2, 0, 0, 0, 2, 2, 127])
        self.assertEqual(p.dequantize(codes, params), [-31.75, -.5, -.5, 0., 0., 0., .5, .5, 31.75])
        self.assertEqual(codes, exhaustive_oracle(values, .25, 0, -127, 127))

    def test_affine_hand_table_include_zero_and_constant_ranges(self):
        params = p.fit_params([-1., 2.], symmetric=False)
        self.assertAlmostEqual(params.scale, 1 / 85)
        self.assertEqual(params.zero_point, -43)
        self.assertEqual(p.quantize([-1., 0., 1., 2.], params), [-128, -43, 42, 127])
        self.assertEqual(p.dequantize([-128, -43, 42, 127], params), [-1., 0., 1., 2.])
        for values, symmetric in [([0., 0.], True), ([0., 0.], False),
                                   ([3., 3.], True), ([3., 3.], False),
                                   ([-3., -3.], True), ([-3., -3.], False)]:
            with self.subTest(values=values, symmetric=symmetric):
                pp = p.fit_params(values, symmetric=symmetric)
                self.assertGreater(pp.scale, 0)
                for actual, expected in zip(p.dequantize(p.quantize(values, pp), pp), values):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(p.dequantize(p.quantize([0.], pp), pp), [0.])

    def test_all_negative_affine_ranges_include_zero_independent_oracle(self):
        # Literal scale/code/reconstruction oracles, not a second fit_params call.
        # Constant-only roundtrip can pass with a wrong scale, so pin metadata too.
        cases = [
            ([-3., -3.], 1 / 85, [-128, -128, 127], [-3., -3., 0.]),
            ([-255., -170., -85.], 1., [-128, -43, 42, 127], [-255., -170., -85., 0.]),
            ([-1., -2.], 2 / 255, [-1, -128, 127], [-256 / 255, -2., 0.]),
        ]
        for calibration, scale, expected_codes, expected_values in cases:
            with self.subTest(calibration=calibration):
                params = p.fit_params(calibration, symmetric=False)
                self.assertAlmostEqual(params.scale, scale, places=15)
                self.assertEqual((params.zero_point, params.qmin, params.qmax), (127, -128, 127))
                codes = p.quantize(calibration + [0.], params)
                self.assertEqual(codes, expected_codes)
                for actual, expected in zip(p.dequantize(codes, params), expected_values):
                    self.assertAlmostEqual(actual, expected, places=14)

    def test_all_zero_scale_is_one_in_both_modes(self):
        # Zero roundtrip alone cannot distinguish scale=1 from any positive scale.
        for symmetric, expected_zero, expected_low in [(True, 0, -127), (False, -128, -128)]:
            with self.subTest(symmetric=symmetric):
                params = p.fit_params([0., 0., 0.], symmetric=symmetric)
                self.assertEqual(params.scale, 1.)
                self.assertEqual((params.zero_point, params.qmin, params.qmax),
                                 (expected_zero, expected_low, 127))
                codes = p.quantize([0., 0., 0.], params)
                self.assertEqual(codes, [expected_zero] * 3)
                self.assertEqual(p.dequantize(codes, params), [0., 0., 0.])

    def test_error_metrics_uses_absolute_magnitude_for_negative_errors(self):
        # Signed errors are [-2,-5]; max(errors)=-2 is NOT max absolute error=5.
        self.assertEqual(p.error_metrics([0., 0.], [2., 5.]),
                         {"max_abs": 5., "mse": 29 / 2})
        # Also cover mixed signs with the largest magnitude on the negative side.
        self.assertEqual(p.error_metrics([0., 0.], [5., -1.]),
                         {"max_abs": 5., "mse": 13.})

    def test_zero_point_added_after_ties_even_rounding(self):
        params = p.QParams(.25, 3, -128, 127)
        xs = [-.375, -.125, .125, .375]
        self.assertEqual(p.quantize(xs, params), [1, 3, 3, 5])
        self.assertEqual(p.quantize(xs, params), exhaustive_oracle(xs, .25, 3, -128, 127))
        # round(x/s + z) is different when odd z moves a half-way tie.
        self.assertNotEqual([round(x / .25 + 3) for x in xs], [1, 3, 3, 5])

    def test_seeded_exhaustive_independent_scalar_grid(self):
        rng = random.Random(24)
        for scale, zero, low in [(1., 0, -127), (.25, 3, -128), (.5, -71, -128)]:
            values = [rng.randint(-1024, 1024) / 8 for _ in range(80)]
            params = p.QParams(scale, zero, low, 127)
            self.assertEqual(p.quantize(values, params), exhaustive_oracle(values, scale, zero, low, 127))
            for x in values:
                if (low - zero) * scale <= x <= (127 - zero) * scale:
                    y = p.dequantize(p.quantize([x], params), params)[0]
                    self.assertLessEqual(abs(x - y), scale / 2)

    def test_scale_underflow_division_overflow_and_no_integer_wrap(self):
        with self.assertRaises(ValueError):
            p.fit_params([5e-324])
        with self.assertRaises(ValueError):
            p.fit_params([5e-324], symmetric=False)
        params = p.QParams(1e-300, 0)
        self.assertEqual(p.quantize([-1e300, 1e300], params), [-127, 127])
        params = p.fit_params([-1e308, 1e308], symmetric=False)
        self.assertTrue(math.isfinite(params.scale))
        self.assertEqual(p.quantize([-1e308, 1e308], params),
                         exhaustive_oracle([-1e308, 1e308], params.scale, params.zero_point, -128, 127))
        accumulator, result = p.quantized_dot([127] * 256, [127] * 256, p.QParams(1., 0), p.QParams(1., 0))
        self.assertEqual(accumulator, 4129024)
        self.assertEqual(result, 4129024.)
        self.assertNotEqual(accumulator, ((accumulator + 128) % 256) - 128)

    def test_centered_integer_dot_is_not_raw_int8_dot(self):
        a, value = p.quantized_dot([-2, 1], [9, 3], p.QParams(.5, -3), p.QParams(.25, 7))
        self.assertEqual(a, -14)
        self.assertEqual(value, -1.75)
        self.assertAlmostEqual(value, .5 * .5 + 2. * -1.)
        with self.assertRaises(ValueError):
            p.quantized_dot([1], [1, 2], p.QParams(1., 0), p.QParams(1., 0))

    def test_groups_reset_at_rows_keep_tail_and_count_scales(self):
        rows = [[.1, -.1, 12.], [3., -3., .2]]
        restored, groups = p.grouped_roundtrip(rows, 2)
        self.assertEqual([len(g["codes"]) for g in groups], [2, 1, 2, 1])
        self.assertEqual([g["scale"] for g in groups], [.1 / 127, 12 / 127, 3 / 127, .2 / 127])
        for row, actual in zip(rows, restored):
            for x, y in zip(row, actual):
                self.assertAlmostEqual(x, y)
        self.assertEqual(p.packed_weight_bytes(2, 3, 2), 22)
        self.assertEqual(p.packed_weight_bytes(2, 3, 99), 14)
        self.assertEqual(p.packed_weight_bytes(1, 8, 2), 24)

    def test_experiment_tables_with_independent_error_and_attention(self):
        result = p.experiment()
        self.assertAlmostEqual(result["group_8"]["max_abs"], .022047244094488216)
        self.assertAlmostEqual(result["group_8"]["mse"], .00015949531899063838)
        self.assertAlmostEqual(result["group_4"]["mse"], .000121675243350487)
        self.assertEqual(result["group_2"]["mse"], 0.)
        self.assertEqual([result[f"group_{n}"]["packed_bytes"] for n in (8, 4, 2)], [12, 16, 24])
        self.assertEqual(result["calibration_shift"]["codes"], [-127, -64, 0, 64, 127])
        self.assertAlmostEqual(result["calibration_shift"]["mse"], (4 + (1 / 254) ** 2 + (1 / 254) ** 2 + 4) / 5)
        keys, values = [[.12, -.7], [1.3, .04], [-.31, .9]], [[.2, 2.], [1.2, -.4], [-.7, .05]]
        expected = independent_attention([.8, -.3], keys, values)
        for x, y in zip(result["kv_cache"]["baseline"], expected):
            self.assertAlmostEqual(x, y, places=14)
        error = [x - y for x, y in zip(result["kv_cache"]["baseline"], result["kv_cache"]["restored"])]
        self.assertAlmostEqual(result["kv_cache"]["mse"], sum(x * x for x in error) / 2)
        self.assertGreater(result["kv_cache"]["mse"], 0)
        # Constant values make attention return that same vector, regardless of K.
        self.assertEqual(p.attention([0., 0.], keys, [[2., -1.]] * 3), [2., -1.])

    def test_invalid_numerical_inputs(self):
        for values in ([], [float("nan")], [float("inf")], [True], ["1"], [10 ** 400]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                p.fit_params(values)
        for scale in (0., -1., float("inf"), float("nan"), True, "1"):
            with self.assertRaises(ValueError):
                p.QParams(scale, 0)
        for zero in (128, -129, .5, True):
            with self.assertRaises(ValueError):
                p.QParams(1., zero)
        for codes in ([], [128], [-128], [1.5], [True]):
            with self.assertRaises(ValueError):
                p.dequantize(codes, p.QParams(1., 0))
        for rows in ([], [[]], [[1], [2, 3]]):
            with self.assertRaises(ValueError):
                p.grouped_roundtrip(rows, 2)
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                p.grouped_roundtrip([[1.]], size)
            with self.assertRaises(ValueError):
                p.packed_weight_bytes(1, 2, size)
        with self.assertRaises(ValueError):
            p.error_metrics([1], [1, 2])
        with self.assertRaises(ValueError):
            p.attention([1], [[1, 2]], [[3]])
        with self.assertRaisesRegex(ValueError, "finite"):
            p.attention([1e308], [[1e308], [1.]], [[1.], [2.]])
        with self.assertRaisesRegex(ValueError, "finite"):
            p.error_metrics([1e308], [-1e308])

    def test_exercises_budget_and_true_token_metrics(self):
        self.assertEqual(p.request_budget(7, 3, 10), 0)
        with self.assertRaises(ValueError):
            p.request_budget(8, 3, 10)
        for args in [(-1, 2, 10), (0, 0, 10), (True, 2, 10), (0, 2, 0)]:
            with self.assertRaises(ValueError):
                p.request_budget(*args)
        self.assertEqual(p.token_latencies(10., [10.2, 10.3, 10.6]),
                         {"ttft": 10.2 - 10., "tpot": (10.6 - 10.2) / 2, "tokens": 3})
        self.assertEqual(p.token_latencies(10., []), {"ttft": None, "tpot": None, "tokens": 0})
        self.assertIsNone(p.token_latencies(10., [10.2])["tpot"])
        with self.assertRaises(ValueError):
            p.token_latencies(10., [9.])


class StreamTests(unittest.TestCase):
    def test_utf8_and_crlf_every_boundary_single_bytes_and_coalesced_events(self):
        raw = stream_fixture()
        expected = {"text": "春风🌱", "finish_reason": "stop", "content_events": 2}
        self.assertEqual(p.collect_completion([raw]), expected)
        self.assertEqual(p.collect_completion(bytes([b]) for b in raw), expected)
        for cut in range(len(raw) + 1):
            self.assertEqual(p.collect_completion([raw[:cut], raw[cut:]]), expected)

    def test_multi_data_lines_completion_text_usage_and_real_llama_capture(self):
        raw = b'data: {"choices":\ndata: [{"text":"Hi","finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        self.assertEqual(p.collect_completion([raw])["text"], "Hi")
        raw = (ROOT / "code/fixtures/p24_llama_completion.sse").read_bytes()
        captured = p.collect_completion(raw[i:i + 7] for i in range(0, len(raw), 7))
        self.assertEqual(captured, {"text": "\x11" * 4, "finish_reason": "length", "content_events": 4})

    def test_multiline_sse_payload_preserves_newlines(self):
        # JSON ignores separator whitespace; inspect SSE payloads directly instead.
        for raw, expected in [
            (b"data: first\ndata: second\n\n", "first\nsecond"),
            (b"data: first\r\ndata:\r\ndata: third\r\n\r\n", "first\n\nthird"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(list(p.sse_data([raw])), [expected])
                self.assertEqual(list(p.sse_data(bytes([byte]) for byte in raw)), [expected])

    def test_incomplete_event_at_eof_is_rejected_at_sse_layer(self):
        prefix = b"data: complete\n\n"
        self.assertEqual(list(p.sse_data([prefix])), ["complete"])
        # Exercise buffer-only, data-only and both-pending EOF states. Calling the
        # SSE layer avoids collect_completion's missing-[DONE] check masking this.
        for tail in [b"data: pending", b"data: pending\n", b"data: first\ndata: pending"]:
            with self.subTest(tail=tail):
                reader = p.sse_data([prefix, tail])
                self.assertEqual(next(reader), "complete")
                with self.assertRaisesRegex(ValueError, "^truncated SSE event at EOF$"):
                    list(reader)

    def test_tools_and_error_objects_are_rejected_before_completion(self):
        # Every fixture can otherwise complete normally: a specific guard, not a
        # later "incomplete response" error, must reject it.
        for key in ("tool_calls", "function_call"):
            value = {"choices": [{"delta": {key: []}, "finish_reason": "stop"}]}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "^only text deltas"):
                p.collect_completion([event(value), b"data: [DONE]\n\n"])
        error = {"error": {"message": "busy"},
                 "choices": [{"text": "not a valid answer", "finish_reason": "stop"}]}
        with self.assertRaisesRegex(ValueError, "^server error"):
            p.collect_completion([event(error), b"data: [DONE]\n\n"])

    def test_empty_text_is_not_a_token_and_stream_finish_is_explicit(self):
        raw = event({"choices": [{"delta": {"content": None}, "finish_reason": "stop"}]}) + b"data: [DONE]\n\n"
        self.assertEqual(p.collect_completion([raw]), {"text": "", "finish_reason": "stop", "content_events": 0})
        for raw in [b"", b"data: [DONE]\n\n", stream_fixture().replace(b"data: [DONE]\r\n\r\n", b""),
                    stream_fixture()[:-1], b"data: {\n\n", b"data: \xff\n\n",
                    stream_fixture() + b"data: [DONE]\n\n"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                p.collect_completion([raw])

    def test_malformed_choices_and_post_finish_rejected(self):
        for value in [[], {"choices": []},
                      {"choices": [{"delta": {"content": 12}}]},
                      {"choices": [{"text": "ok", "finish_reason": 2}]}]:
            with self.assertRaises(ValueError):
                p.collect_completion([event(value), b"data: [DONE]\n\n"])
        with self.assertRaisesRegex(ValueError, "after finish_reason"):
            p.collect_completion([event({"choices": [{"text": "a", "finish_reason": "stop"}]}),
                                  event({"choices": [{"text": "b"}]}), b"data: [DONE]\n\n"])
        with self.assertRaises(ValueError):
            list(p.sse_data(["not bytes"]))


class ActualMutationTests(unittest.TestCase):
    def test_wrong_dequant_zero_point_is_killed(self):
        baseline = lambda module: module.dequantize([-128, -43, 42, 127], module.QParams(1 / 85, -43, -128, 127))
        self.assertEqual(baseline(p), [-1., 0., 1., 2.])
        mutant = load_mutant("params.scale * (q - params.zero_point)", "params.scale * q", "zero")
        self.assertNotEqual(baseline(mutant), [-1., 0., 1., 2.])

    def test_truncation_rounding_mutant_is_killed(self):
        mutant = load_mutant("return [round(min(params.qmax", "return [int(min(params.qmax", "truncate")
        self.assertEqual(p.quantize([.375], p.QParams(.25, 0)), [2])
        self.assertNotEqual(mutant.quantize([.375], mutant.QParams(.25, 0)), [2])

    def test_raw_integer_dot_mutant_is_killed(self):
        mutant = load_mutant("(x - x_params.zero_point) * (w - w_params.zero_point)", "x * w", "dot")
        args = ([-2, 1], [9, 3], p.QParams(.5, -3), p.QParams(.25, 7))
        self.assertEqual(p.quantized_dot(*args)[1], -1.75)
        self.assertNotEqual(mutant.quantized_dot(*args)[1], -1.75)

    def test_group_tail_drop_mutant_is_killed(self):
        mutant = load_mutant("range(0, len(row), group_size)", "range(0, len(row) - group_size + 1, group_size)", "tail")
        self.assertEqual(len(p.grouped_roundtrip([[1., 2., 3.]], 2)[0][0]), 3)
        self.assertNotEqual(len(mutant.grouped_roundtrip([[1., 2., 3.]], 2)[0][0]), 3)

    def test_context_off_by_one_and_missing_done_mutants_are_killed(self):
        mutant = load_mutant("prompt_tokens + max_new_tokens > context_size", "prompt_tokens + max_new_tokens >= context_size", "budget")
        self.assertEqual(p.request_budget(7, 3, 10), 0)
        with self.assertRaises(ValueError):
            mutant.request_budget(7, 3, 10)
        mutant = load_mutant("if not done or finish is None:", "if finish is None:", "done")
        raw = event({"choices": [{"text": "a", "finish_reason": "stop"}]})
        with self.assertRaises(ValueError):
            p.collect_completion([raw])
        self.assertEqual(mutant.collect_completion([raw])["text"], "a")

    def test_chunk_local_utf8_decode_mutant_is_killed(self):
        mutant = load_mutant("buffer += decoder.decode(chunk)", "buffer += chunk.decode('utf-8')", "utf8")
        raw = stream_fixture()
        self.assertEqual(p.collect_completion(bytes([b]) for b in raw)["text"], "春风🌱")
        with self.assertRaises(UnicodeDecodeError):
            mutant.collect_completion(bytes([b]) for b in raw)


    def test_affine_hi_excluding_zero_mutant_is_killed(self):
        values = [-255., -170., -85.]
        expected = [-255., -170., -85., 0.]

        def roundtrip(module):
            params = module.fit_params(values, symmetric=False)
            return module.dequantize(module.quantize(values + [0.], params), params)

        self.assertEqual(roundtrip(p), expected)
        mutant = load_mutant("max(0., max(values))", "max(values)", "affine_excludes_zero")
        wrong = roundtrip(mutant)  # must run successfully, not fail by exception
        self.assertAlmostEqual(wrong[0], -170., places=12)  # wrongly clips -255 to about -170
        self.assertNotEqual(wrong, expected)
        correct_constant = p.fit_params([-3., -3.], symmetric=False)
        self.assertEqual((correct_constant.scale, correct_constant.zero_point), (1 / 85, 127))
        wrong_constant = mutant.fit_params([-3., -3.], symmetric=False)
        self.assertNotEqual((wrong_constant.scale, wrong_constant.zero_point), (1 / 85, 127))

    def test_all_zero_default_scale_mutants_are_killed(self):
        cases = [
            (True, "bound / 127 if bound else 1.", "bound / 127 if bound else 2.", "symmetric_zero_scale"),
            (False, "hi / 255 - lo / 255 if hi != lo else 1.",
             "hi / 255 - lo / 255 if hi != lo else 2.", "affine_zero_scale"),
        ]
        for symmetric, old, new, name in cases:
            with self.subTest(symmetric=symmetric):
                self.assertEqual(p.fit_params([0., 0.], symmetric=symmetric).scale, 1.)
                mutant = load_mutant(old, new, name)
                wrong = mutant.fit_params([0., 0.], symmetric=symmetric)
                self.assertEqual(wrong.scale, 2.)
                self.assertNotEqual(wrong.scale, 1.)
                # The old roundtrip-only check would accept this wrong contract.
                self.assertEqual(mutant.dequantize(mutant.quantize([0., 0.], wrong), wrong), [0., 0.])

    def test_signed_maximum_error_mutant_is_killed(self):
        expected = {"max_abs": 5., "mse": 29 / 2}
        self.assertEqual(p.error_metrics([0., 0.], [2., 5.]), expected)
        mutant = load_mutant('"max_abs": max(abs(x) for x in errors)',
                             '"max_abs": max(errors)', "signed_max_error")
        wrong = mutant.error_metrics([0., 0.], [2., 5.])
        self.assertEqual(wrong, {"max_abs": -2., "mse": 29 / 2})
        self.assertNotEqual(wrong, expected)

    def test_incomplete_sse_event_drop_mutant_is_killed(self):
        prefix = b"data: complete\n\n"
        tails = [b"data: pending", b"data: pending\n", b"data: first\ndata: pending"]
        for tail in tails:
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, "^truncated SSE event at EOF$"):
                list(p.sse_data([prefix, tail]))
        mutant = load_mutant("if buffer or data:", "if False and (buffer or data):", "discard_incomplete_event")
        for tail in tails:
            with self.subTest(tail=tail):
                # The mutant silently succeeds and drops the pending event.
                self.assertEqual(list(mutant.sse_data([prefix, tail])), ["complete"])

    def test_multiline_sse_separator_mutant_is_killed(self):
        raw = b"data: first\ndata:\ndata: third\n\n"
        self.assertEqual(list(p.sse_data([raw])), ["first\n\nthird"])
        mutant = load_mutant('yield "\\n".join(data)', 'yield "".join(data)', "missing_data_newline")
        self.assertEqual(list(mutant.sse_data([raw])), ["firstthird"])
        self.assertNotEqual(list(mutant.sse_data([raw])), ["first\n\nthird"])

    def test_tool_calls_and_error_guards_mutants_are_killed(self):
        cases = [
            (' or "tool_calls" in delta', "tool_calls_guard", "^only text deltas",
             {"choices": [{"delta": {"tool_calls": []}, "finish_reason": "stop"}]},
             {"text": "", "finish_reason": "stop", "content_events": 0}),
            (' or "error" in event', "server_error_guard", "^server error",
             {"error": {"message": "busy"},
              "choices": [{"text": "not a valid answer", "finish_reason": "stop"}]},
             {"text": "not a valid answer", "finish_reason": "stop", "content_events": 1}),
        ]
        for old, name, reason, value, incorrectly_accepted in cases:
            with self.subTest(name=name):
                raw = event(value) + b"data: [DONE]\n\n"
                with self.assertRaisesRegex(ValueError, reason):
                    p.collect_completion([raw])
                mutant = load_mutant(old, "", name)
                # With the guard removed this must actually accept the stream,
                # proving neither malformed JSON nor missing finish masks it.
                self.assertEqual(mutant.collect_completion([raw]), incorrectly_accepted)


class FixtureTests(unittest.TestCase):
    def test_deterministic_tiny_gguf_header_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            report = p.write_tiny_gguf(path)
            raw = path.read_bytes()
            self.assertEqual(report, {"bytes": 81760, "tensors": 11, "parameters": 18624, "trained": False})
            self.assertEqual(raw[:4], b"GGUF")
            version, tensors, keys = struct.unpack_from("<IQQ", raw, 4)
            self.assertEqual((version, tensors), (3, 11))
            self.assertGreater(keys, 10)
            self.assertIn(b"tokenizer.ggml.tokens", raw)
            self.assertIn(b"blk.0.ffn_down.weight", raw)
            expected = json.loads((ROOT / "code/fixtures/p24_llama_smoke.json").read_text())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["f32_sha256"])

    def test_non_aligned_tensor_payload_preserves_gguf_offsets(self):
        # Add a three-float format-only probe before normal tensors. This is not
        # a new runnable Llama architecture; production smoke remains unchanged.
        original = SCRIPT.read_text()
        shape_line = 'shapes = [("token_embd.weight", [32, 259]), ("output_norm.weight", [32])]'
        self.assertEqual(original.count(shape_line), 1)
        fixture = original.replace(shape_line,
            'shapes = [("alignment_probe", [3]), ("token_embd.weight", [32, 259]), ("output_norm.weight", [32])]')
        def read_directory(raw):
            position = 24
            def take(fmt):
                nonlocal position
                value = struct.unpack_from(fmt, raw, position)
                position += struct.calcsize(fmt)
                return value[0] if len(value) == 1 else value
            def string():
                nonlocal position
                size = take('<Q'); value = raw[position:position+size]
                position += size
                return value.decode()
            def skip(kind):
                if kind == 8: string()
                elif kind == 9:
                    subtype, count = take('<IQ')
                    for _ in range(count): skip(subtype)
                else: take({4:'<I', 5:'<i', 6:'<f', 7:'<?'}[kind])
            tensors, keys = struct.unpack_from('<QQ', raw, 8)
            for _ in range(keys): string(); skip(take('<I'))
            directory = []
            for _ in range(tensors):
                name = string(); ndim = take('<I')
                dims = [take('<Q') for _ in range(ndim)]
                kind, offset = take('<IQ')
                directory.append((name, dims, kind, offset))
            start = ((position + 31)//32)*32
            return directory, start
        def verify(source, name):
            module = types.ModuleType(name); sys.modules[name] = module
            exec(compile(source, str(SCRIPT), 'exec'), module.__dict__)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'format-only.gguf'; module.write_tiny_gguf(path)
                raw = path.read_bytes()
            entries, start = read_directory(raw)
            self.assertEqual(entries[0], ('alignment_probe', [3], 0, 0))
            self.assertEqual(entries[1][3], 32)  # 12 data bytes + 20 padding bytes
            self.assertEqual(raw[start+12:start+32], bytes(20))
            self.assertTrue(all(offset % 32 == 0 for _,_,_,offset in entries))
        verify(fixture, 'p24_alignment_baseline')
        target = 'padded = blob + b"\\0" * (-len(blob) % 32)'
        self.assertEqual(fixture.count(target), 1)
        with self.assertRaises(AssertionError):
            verify(fixture.replace(target, 'padded = blob'), 'p24_alignment_mutant')

    def test_default_command_is_offline_stdlib_and_finite(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=20)
        data = json.loads(result.stdout)
        self.assertEqual(data["group_2"]["mse"], 0)
        self.assertNotIn("NaN", result.stdout)

    @unittest.skipUnless(os.environ.get("P24_LLAMA_BIN"), "optional real b5046 binaries not provided")
    def test_optional_real_llama_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = p.llama_smoke(os.environ["P24_LLAMA_BIN"], tmp)
            self.assertEqual(result["usage"]["completion_tokens"], 4)
            self.assertEqual(result["stream"]["finish_reason"], "length")
            self.assertEqual(result["stream"]["text"], "\x11" * 4)
            self.assertLess(result["q8_bytes"], result["f32_bytes"])
            self.assertFalse(result["quality_tested"])
            self.assertFalse(result["performance_comparison_tested"])


if __name__ == "__main__":
    unittest.main()
