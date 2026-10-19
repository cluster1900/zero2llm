"""P24: stdlib-only quantization, cache error and SSE boundary experiments.

This is scalar teaching code, NOT an optimized int8 GEMM or a GGUF quantizer.
The optional tiny GGUF is untrained and only tests llama.cpp's execution path.
"""
from __future__ import annotations

import argparse
import codecs
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import hashlib
import os
import socket
import subprocess
import time
import urllib.request
import struct
from typing import Iterable


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(values):
    data = list(values)
    if not data or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in data):
        raise ValueError("expected nonempty finite real values")
    try:
        data = [float(x) for x in data]
    except OverflowError as exc:
        raise ValueError("value outside float range") from exc
    if not all(math.isfinite(x) for x in data):
        raise ValueError("expected finite values")
    return data


@dataclass(frozen=True)
class QParams:
    scale: float
    zero_point: int
    qmin: int = -127
    qmax: int = 127

    def __post_init__(self):
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if any(isinstance(x, bool) or not isinstance(x, int)
               for x in (self.zero_point, self.qmin, self.qmax)):
            raise ValueError("integer zero-point and bounds required")
        if not -128 <= self.qmin < self.qmax <= 127 or not self.qmin <= self.zero_point <= self.qmax:
            raise ValueError("invalid signed int8 bounds or zero-point")


def fit_params(values, *, symmetric=True):
    """Min/max calibration; affine range includes 0, including constant tensors.

    Symmetric uses [-127,127], affine uses [-128,127]. All-zero scale is 1.
    Subnormal ranges whose scale underflows are explicitly rejected, not erased.
    """
    values = _finite(values)
    if not isinstance(symmetric, bool):
        raise ValueError("symmetric must be boolean")
    if symmetric:
        bound = max(abs(x) for x in values)
        return QParams(bound / 127 if bound else 1., 0)
    lo, hi = min(0., min(values)), max(0., max(values))
    # hi - lo might overflow even when both endpoints are finite.
    scale = hi / 255 - lo / 255 if hi != lo else 1.
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("calibrated scale underflow/overflow")
    zero = min(127, max(-128, -128 - round(lo / scale)))
    return QParams(scale, zero, -128, 127)


def quantize(values, params):
    """q = clip(round_ties_even(x/s) + z); saturate BEFORE int conversion.

    Division may reach +/-inf for a finite outlier and a tiny calibration scale.
    Clamping to integer thresholds first is equivalent and prevents overflow.
    """
    data = _finite(values)
    return [round(min(params.qmax - params.zero_point,
                      max(params.qmin - params.zero_point, x / params.scale)))
            + params.zero_point for x in data]


def dequantize(codes, params):
    data = list(codes)
    if not data or any(isinstance(q, bool) or not isinstance(q, int)
                       or not params.qmin <= q <= params.qmax for q in data):
        raise ValueError("codes must be nonempty integers inside quantizer range")
    return _finite([params.scale * (q - params.zero_point) for q in data])


def error_metrics(reference, reconstructed):
    a, b = _finite(reference), _finite(reconstructed)
    if len(a) != len(b):
        raise ValueError("error arrays must have equal length")
    errors = _finite([x - y for x, y in zip(a, b)])
    squared = _finite([x * x for x in errors])
    # Divide before summation to avoid needless sum overflow.
    return {"max_abs": max(abs(x) for x in errors),
            "mse": math.fsum(x / len(errors) for x in squared)}


def _matrix(rows):
    matrix = [_finite(row) for row in rows]
    if not matrix or len({len(row) for row in matrix}) != 1:
        raise ValueError("matrix must be nonempty and rectangular")
    return matrix


def grouped_roundtrip(rows, group_size):
    """Per-row contiguous groups; final partial group is NOT padded or dropped."""
    rows = _matrix(rows)
    _positive_int(group_size, "group_size")
    result, groups = [], []
    for row in rows:
        restored = []
        for start in range(0, len(row), group_size):
            original = row[start:start + group_size]
            params = fit_params(original)
            codes = quantize(original, params)
            restored.extend(dequantize(codes, params))
            groups.append({"scale": params.scale, "codes": codes})
        result.append(restored)
    return result, groups


def packed_weight_bytes(rows, columns, group_size):
    """Ideal packed symmetric int8 codes + one float32 scale per row-group.

    Excludes file metadata, alignment, workspaces and Python object overhead.
    """
    for name, n in (("rows", rows), ("columns", columns), ("group_size", group_size)):
        _positive_int(n, name)
    count = rows * ((columns + group_size - 1) // group_size)
    return rows * columns + 4 * count


def quantized_dot(x_codes, w_codes, x_params, w_params):
    """Integer accumulator is Python int (unbounded), NOT overflowing int8.

    A production kernel needs a sufficiently wide accumulator (often int32).
    This scalar reference does not claim kernel speed or support arbitrary dims.
    """
    dequantize(x_codes, x_params)  # validate codes without using float dot
    dequantize(w_codes, w_params)
    if len(x_codes) != len(w_codes):
        raise ValueError("dot vectors must have equal length")
    accum = sum((x - x_params.zero_point) * (w - w_params.zero_point)
                for x, w in zip(x_codes, w_codes))
    return accum, accum * x_params.scale * w_params.scale


def attention(query, keys, values):
    """One query over an already-visible prefix: no future tokens in this input."""
    query, keys, values = _finite(query), _matrix(keys), _matrix(values)
    if len(query) != len(keys[0]) or len(keys) != len(values):
        raise ValueError("attention dimensions disagree")
    scores = [math.fsum(a * b for a, b in zip(query, key)) / math.sqrt(len(query))
              for key in keys]
    scores = _finite(scores)  # finite inputs can still overflow their dot products
    weights = [math.exp(s - max(scores)) for s in scores]
    total = math.fsum(weights)
    return [math.fsum(w * row[j] for w, row in zip(weights, values)) / total
            for j in range(len(values[0]))]


def request_budget(prompt_tokens, max_new_tokens, context_size):
    """Count AFTER tokenizer, BOS and chat template; no silent truncation."""
    _positive_int(context_size, "context_size")
    _positive_int(max_new_tokens, "max_new_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a nonnegative integer")
    if prompt_tokens + max_new_tokens > context_size:
        raise ValueError("prompt plus generation exceeds context budget")
    return context_size - prompt_tokens - max_new_tokens


def sse_data(chunks: Iterable[bytes]):
    """Small SSE subset: UTF-8, LF/CRLF, comments, multi-line data, blank events.

    No reconnection, IDs, retries or event dispatch. A pending event at EOF is
    rejected. Network read boundaries are not UTF-8, JSON, or token boundaries.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer, data = "", []
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise ValueError("stream chunks must be bytes")
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
                continue
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if field == "data" and sep:
                data.append(value[1:] if value.startswith(" ") else value)
    buffer += decoder.decode(b"", final=True)
    if buffer or data:
        raise ValueError("truncated SSE event at EOF")


def collect_completion(chunks):
    """Single-choice chat/completion text only, not tool calls or logprobs.

    Require finish_reason AND [DONE]; missing either is an incomplete response.
    A usage-only empty-choices event is allowed. No automatic request replay.
    """
    parts, finish, done = [], None, False
    for payload in sse_data(chunks):
        if done:
            raise ValueError("data after [DONE]")
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("server error or invalid event")
        choices = event.get("choices")
        if choices == [] and "usage" in event:
            continue
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("expected exactly one choice")
        choice = choices[0]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict) or "tool_calls" in delta or "function_call" in delta:
            raise ValueError("only text deltas are supported")
        text = choice.get("text", delta.get("content", ""))
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("content must be text")
        if finish is not None:
            raise ValueError("choice after finish_reason")
        if text:
            parts.append(text)
        reason = choice.get("finish_reason")
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                raise ValueError("invalid finish_reason")
            finish = reason
    if not done or finish is None:
        raise ValueError("incomplete response: finish_reason and [DONE] required")
    return {"text": "".join(parts), "finish_reason": finish, "content_events": len(parts)}


def token_latencies(start, token_times):
    """Only for true one-timestamp-per-output-token data, NOT SSE chunk times."""
    times = _finite([start] + list(token_times))
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be monotonic")
    tokens = times[1:]
    return {"ttft": tokens[0] - times[0] if tokens else None,
            "tpot": (tokens[-1] - tokens[0]) / (len(tokens) - 1) if len(tokens) > 1 else None,
            "tokens": len(tokens)}


def experiment():
    weights = [[.1, -.1, .2, -.2, 12., -12., .4, -.4]]
    reference = weights[0]
    output = {}
    for group in (8, 4, 2):
        restored, scales = grouped_roundtrip(weights, group)
        output[f"group_{group}"] = {**error_metrics(reference, restored[0]),
                                     "groups": len(scales),
                                     "packed_bytes": packed_weight_bytes(1, 8, group)}
    calibration, evaluation = [-1., 0., 1.], [-3., -.5, 0., .5, 3.]
    p = fit_params(calibration)
    q = quantize(evaluation, p)
    output["calibration_shift"] = {"codes": q, "restored": dequantize(q, p),
                                   **error_metrics(evaluation, dequantize(q, p))}
    keys = [[.12, -.7], [1.3, .04], [-.31, .9]]
    values = [[.2, 2.], [1.2, -.4], [-.7, .05]]
    qk, _ = grouped_roundtrip(keys, 2)
    qv, _ = grouped_roundtrip(values, 2)
    baseline = attention([.8, -.3], keys, values)
    output["kv_cache"] = {"baseline": baseline,
                           "restored": attention([.8, -.3], qk, qv),
                           **error_metrics(baseline, attention([.8, -.3], qk, qv))}
    return output


def write_tiny_gguf(path):
    """Deterministic untrained one-layer Llama GGUF v3, float32, byte vocabulary.

    Narrow fixture writer, NOT a general HF converter. Format/model keys match
    llama.cpp b5046 / 74d4f5b041ad837153b0e90fc864b8290e01d8d5.
    The 259-token vocabulary has unk/BOS/EOS plus all 256 byte-fallback tokens.
    """
    def string(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def scalar(kind, value):
        return string(value) if kind == 8 else struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[kind], value)

    meta = []

    def add(key, kind, value):
        meta.append(string(key) + struct.pack("<I", kind) + scalar(kind, value))

    def array(key, kind, values):
        meta.append(string(key) + struct.pack("<IIQ", 9, kind, len(values))
                    + b"".join(scalar(kind, x) for x in values))

    add("general.architecture", 8, "llama")
    add("general.name", 8, "P24 untrained tiny smoke, NOT a language model benchmark")
    add("general.file_type", 4, 0)
    for key, value in (("context_length", 128), ("embedding_length", 32),
                       ("block_count", 1), ("feed_forward_length", 64),
                       ("attention.head_count", 4), ("attention.head_count_kv", 4),
                       ("rope.dimension_count", 8)):
        add("llama." + key, 4, value)
    add("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)
    add("tokenizer.ggml.model", 8, "llama")
    array("tokenizer.ggml.tokens", 8, ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)])
    array("tokenizer.ggml.scores", 6, [0.] * 259)
    array("tokenizer.ggml.token_type", 5, [2, 3, 3] + [6] * 256)
    for key, value in (("unknown_token_id", 0), ("bos_token_id", 1), ("eos_token_id", 2)):
        add("tokenizer.ggml." + key, 4, value)
    add("tokenizer.ggml.add_bos_token", 7, True)
    add("tokenizer.ggml.add_space_prefix", 7, False)
    rng = random.Random(24)
    shapes = [("token_embd.weight", [32, 259]), ("output_norm.weight", [32])]
    shapes += [(f"blk.0.{name}.weight", dims) for name, dims in (
        ("attn_norm", [32]), ("attn_q", [32, 32]), ("attn_k", [32, 32]),
        ("attn_v", [32, 32]), ("attn_output", [32, 32]), ("ffn_norm", [32]),
        ("ffn_gate", [32, 64]), ("ffn_down", [64, 32]), ("ffn_up", [32, 64]))]
    offset, info, blobs = 0, [], []
    for name, dims in shapes:
        values = [1.] * math.prod(dims) if len(dims) == 1 else [rng.uniform(-.1, .1) for _ in range(math.prod(dims))]
        blob = struct.pack("<" + "f" * len(values), *values)
        info.append(string(name) + struct.pack("<I", len(dims))
                    + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", 0, offset))
        padded = blob + b"\0" * (-len(blob) % 32)
        blobs.append(padded)
        offset += len(padded)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), len(meta)) + b"".join(meta) + b"".join(info)
    data = header + b"\0" * (-len(header) % 32) + b"".join(blobs)
    Path(path).write_bytes(data)
    return {"bytes": len(data), "tensors": len(shapes), "parameters": sum(math.prod(s) for _, s in shapes),
            "trained": False}


def llama_smoke(bin_dir, work_dir):
    """Opt-in real CPU smoke with existing official b5046 binaries, no downloads.

    Child-only LD_LIBRARY_PATH, loopback-only ephemeral server, always terminate.
    Does not install packages, publish an API or claim meaningful model quality.
    """
    binary, work = Path(bin_dir).resolve(), Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(binary))
    commands = []

    def run(name, args):
        command = [str(binary / name), *map(str, args)]
        commands.append(command)
        result = subprocess.run(command, env=env, capture_output=True, timeout=60, check=True)
        (work / (name + ".stdout")).write_bytes(result.stdout)
        (work / (name + ".stderr")).write_bytes(result.stderr)
        return result.stdout + result.stderr

    version = run("llama-cli", ["--version"]).decode()
    if "5046 (74d4f5b0)" not in version:
        raise ValueError("this smoke is pinned to llama.cpp b5046 (74d4f5b0)")
    f32, q8 = work / "tiny-f32.gguf", work / "tiny-q8_0.gguf"
    fixture = write_tiny_gguf(f32)
    run("llama-quantize", [f32, q8, "Q8_0"])
    run("llama-cli", ["-m", q8, "-p", "Hi", "-n", "4", "-c", "128", "-t", "1",
                      "-ngl", "0", "--seed", "24", "--temp", "0", "--no-display-prompt"])
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(binary / "llama-server"), "-m", str(q8), "-c", "128", "-t", "1", "-ngl", "0",
               "--host", "127.0.0.1", "--port", str(port), "--alias", "p24-tiny"]
    commands.append(command)
    # Ignore outbound proxy settings for this process-local loopback request only.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (work / "server.log").open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited before becoming ready; inspect server.log")
                try:
                    with opener.open(base + "/health", timeout=1) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("loopback server did not become ready")
                time.sleep(.05)
            payload = {"model": "p24-tiny", "prompt": "Hi", "max_tokens": 4,
                       "temperature": 0, "seed": 24, "stream": True}
            results = []
            for stream in (True, False):
                payload["stream"] = stream
                request = urllib.request.Request(base + "/v1/completions", json.dumps(payload).encode(),
                                                  {"Content-Type": "application/json"})
                with opener.open(request, timeout=30) as response:
                    raw = response.read()
                    expected_type = "text/event-stream" if stream else "application/json"
                    if response.status != 200 or expected_type not in response.headers.get("Content-Type", ""):
                        raise RuntimeError("unexpected HTTP status/content type")
                (work / ("completion.sse" if stream else "completion.json")).write_bytes(raw)
                results.append(collect_completion(raw[i:i + 7] for i in range(0, len(raw), 7))
                               if stream else json.loads(raw))
            streamed, ordinary = results
            if streamed["text"] != ordinary["choices"][0]["text"] or streamed["finish_reason"] != "length":
                raise AssertionError("stream/nonstream result mismatch")
            if ordinary["usage"]["completion_tokens"] != 4:
                raise AssertionError("expected four generated tokens")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    report = {"source_commit": "74d4f5b041ad837153b0e90fc864b8290e01d8d5", "version": version.strip(),
              "fixture": fixture, "f32_bytes": f32.stat().st_size, "q8_bytes": q8.stat().st_size,
              "f32_sha256": hashlib.sha256(f32.read_bytes()).hexdigest(),
              "q8_sha256": hashlib.sha256(q8.read_bytes()).hexdigest(),
              "stream": streamed, "usage": ordinary["usage"], "commands": commands,
              "quality_tested": False, "performance_comparison_tested": False}
    (work / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-tiny-gguf", type=Path, help="write untrained local smoke fixture, no downloads")
    parser.add_argument("--read-sse", type=Path, help="validate saved single-choice SSE bytes, no network")
    parser.add_argument("--llama-bin", type=Path, help="opt-in real b5046 CPU smoke; binaries already installed")
    parser.add_argument("--work-dir", type=Path, help="output directory for optional llama.cpp smoke")
    args = parser.parse_args()
    if args.llama_bin:
        if not args.work_dir:
            parser.error("--llama-bin requires --work-dir")
        result = llama_smoke(args.llama_bin, args.work_dir)
    elif args.write_tiny_gguf:
        result = write_tiny_gguf(args.write_tiny_gguf)
    elif args.read_sse:
        raw = args.read_sse.read_bytes()
        result = collect_completion(raw[i:i + 7] for i in range(0, len(raw), 7))
    else:
        result = experiment()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
