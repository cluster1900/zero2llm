"""P15: exact online attention, paged KV ownership, and serving metric arithmetic.

The mandatory experiment uses only the Python standard library, on CPU.
It does not change S4 or implement FlashAttention/CUDA/vLLM. Python containers,
optional dense masks and reconstruction copies are NOT GPU memory measurements.
Run: python code/16_attention_systems.py
Optional, separately requested: python code/16_attention_systems.py --gpu-sdpa
"""
import argparse
import heapq
import math
import random
import sys


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def matrix(value, name):
    """Copy a finite, nonempty rectangular list/tuple matrix to Python floats."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    width = None
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row or (width is not None and len(row) != width):
            raise ValueError(f"{name} must be rectangular with nonempty rows")
        width = len(row)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
            raise ValueError(f"{name} must contain finite real numbers, not booleans")
        result.append([float(x) for x in row])
    return result


def attention_inputs(q, k, v, allowed):
    q, k, v = matrix(q, "q"), matrix(k, "k"), matrix(v, "v")
    if len(q[0]) != len(k[0]) or len(k) != len(v):
        raise ValueError("Q/K widths and K/V sequence lengths must match")
    if allowed is not None:
        if (not isinstance(allowed, (list, tuple)) or len(allowed) != len(q)
                or any(not isinstance(row, (list, tuple)) or len(row) != len(k)
                       or any(type(x) is not bool for x in row) for row in allowed)):
            raise ValueError("allowed must be boolean [query_length,key_length]; True means visible")
    return q, k, v


def causal_mask(query_length, key_length, offset=0):
    """Absolute positions: query i is at offset+i, not always at i."""
    positive_int(query_length, "query_length")
    positive_int(key_length, "key_length")
    if type(offset) is not int or offset < 0 or offset + query_length > key_length:
        raise ValueError("query positions must fit in the supplied key sequence")
    return [[j <= offset + i for j in range(key_length)] for i in range(query_length)]


def dense_attention(q, k, v, allowed=None):
    """Readable full score matrix reference, one batch/head; all-masked -> zeros."""
    q, k, v = attention_inputs(q, k, v, allowed)
    scale = math.sqrt(len(q[0]))
    scores = [[math.fsum(a * b for a, b in zip(row, key)) / scale for key in k] for row in q]
    if any(not math.isfinite(x) for row in scores for x in row):
        raise ValueError("dot products overflowed; inputs must produce finite scores")
    result = []
    for i, row in enumerate(scores):
        visible = [j for j in range(len(k)) if allowed is None or allowed[i][j]]
        if not visible:
            result.append([0.0] * len(v[0]))
            continue
        peak = max(row[j] for j in visible)
        weights = [math.exp(row[j] - peak) for j in visible]
        total = math.fsum(weights)
        result.append([math.fsum(w * v[j][c] for w, j in zip(weights, visible)) / total
                       for c in range(len(v[0]))])
    return result


def blocked_attention(q, k, v, allowed=None, query_block_size=2, key_block_size=3):
    """Online stable softmax, preserving every permitted QK interaction.

    Q:[Nq,d], K:[Nk,d], V:[Nk,dv] -> [Nq,dv]. No dropout/backward.
    Only per-row current key-block scores are materialized. This scalar Python
    loop teaches the recurrence, not the GPU tiling/scheduling of FlashAttention.
    """
    positive_int(query_block_size, "query_block_size")
    positive_int(key_block_size, "key_block_size")
    q, k, v = attention_inputs(q, k, v, allowed)
    scale = math.sqrt(len(q[0]))
    result = []
    for query_start in range(0, len(q), query_block_size):
        query_end = min(query_start + query_block_size, len(q))
        for i in range(query_start, query_end):
            m, ell = -math.inf, 0.0
            acc = [0.0] * len(v[0])
            for key_start in range(0, len(k), key_block_size):
                key_end = min(key_start + key_block_size, len(k))
                visible = [j for j in range(key_start, key_end)
                           if allowed is None or allowed[i][j]]
                if not visible:
                    continue  # Never evaluate exp(-inf - -inf).
                scores = [math.fsum(a * b for a, b in zip(q[i], k[j])) / scale for j in visible]
                if any(not math.isfinite(x) for x in scores):
                    raise ValueError("dot products overflowed; inputs must produce finite scores")
                next_m = max(m, max(scores))
                alpha = 0.0 if ell == 0.0 else math.exp(m - next_m)
                weights = [math.exp(s - next_m) for s in scores]
                ell = alpha * ell + math.fsum(weights)
                acc = [alpha * acc[c] + math.fsum(w * v[j][c] for w, j in zip(weights, visible))
                       for c in range(len(acc))]
                m = next_m
            result.append([x / ell for x in acc] if ell else [0.0] * len(acc))
    return result


class PagedKVCache:
    """A one-head, fixed-width KV ownership simulation, NOT a paged GPU kernel.

    Physical blocks hold immutable (key_tuple,value_tuple) token payloads.
    Requests hold ordered block IDs. Refcounts count live request references.
    Full blocks may be shared; fork also shares a partial tail until append does
    copy-on-write. share_prefix accepts only complete blocks and assumes the
    caller already verified identical token history/model/positions. No APC
    hash lookup, eviction, tenant isolation, concurrency, or tensor allocator.
    """
    def __init__(self, block_size=2, key_dim=1, value_dim=1, max_blocks=16):
        for value, name in ((block_size, "block_size"), (key_dim, "key_dim"),
                            (value_dim, "value_dim"), (max_blocks, "max_blocks")):
            positive_int(value, name)
        self.block_size, self.key_dim, self.value_dim = block_size, key_dim, value_dim
        self.max_blocks = max_blocks
        self._blocks, self._refs, self._tables, self._lengths = {}, {}, {}, {}
        self._free = list(range(max_blocks))

    def _new_name(self, request):
        if not isinstance(request, str) or not request or request in self._tables:
            raise ValueError("request ID must be a new nonempty string")

    def _existing(self, request):
        if not isinstance(request, str) or request not in self._tables:
            raise ValueError("unknown request ID")

    def create(self, request):
        self._new_name(request)
        self._tables[request], self._lengths[request] = [], 0

    def _allocate(self, payload=()):
        if not self._free:
            raise MemoryError("KV block pool exhausted; no request state changed")
        block = heapq.heappop(self._free)
        self._blocks[block], self._refs[block] = list(payload), 1
        return block

    def append(self, request, key, value):
        """Append ONE token atomically; allocate before changing ownership."""
        self._existing(request)
        k, v = matrix([key], "key")[0], matrix([value], "value")[0]
        if len(k) != self.key_dim or len(v) != self.value_dim:
            raise ValueError("key/value widths do not match this pool")
        table = self._tables[request]
        if self._lengths[request] % self.block_size == 0:
            table.append(self._allocate())  # Full previous blocks remain shared.
        elif self._refs[table[-1]] > 1:
            old = table[-1]
            new = self._allocate(self._blocks[old])  # Copy ONLY the partial tail.
            self._refs[old] -= 1
            table[-1] = new
        self._blocks[table[-1]].append((tuple(k), tuple(v)))
        self._lengths[request] += 1

    def fork(self, source, target):
        """Two continuations start from identical entire history, including tail."""
        self._existing(source)
        self._new_name(target)
        self._tables[target] = list(self._tables[source])
        self._lengths[target] = self._lengths[source]
        for block in self._tables[target]:
            self._refs[block] += 1

    def share_prefix(self, source, target, tokens):
        """Create target with a known-compatible whole-block prefix of source."""
        self._existing(source)
        self._new_name(target)
        if (type(tokens) is not int or tokens < 0 or tokens > self._lengths[source]
                or tokens % self.block_size):
            raise ValueError("prefix must fit the source and end on a complete block")
        self._tables[target] = list(self._tables[source][:tokens // self.block_size])
        self._lengths[target] = tokens
        for block in self._tables[target]:
            self._refs[block] += 1

    def release(self, request):
        self._existing(request)
        for block in self._tables[request]:
            self._refs[block] -= 1
            if self._refs[block] == 0:
                del self._refs[block], self._blocks[block]
                heapq.heappush(self._free, block)
        del self._tables[request], self._lengths[request]

    def block_table(self, request):
        self._existing(request)
        return tuple(self._tables[request])

    def refcounts(self):
        return dict(self._refs)

    def read(self, request):
        """Logical order, never physical-ID order; returns fresh lists."""
        self._existing(request)
        payload = [item for block in self._tables[request] for item in self._blocks[block]]
        payload = payload[:self._lengths[request]]
        return [list(k) for k, _ in payload], [list(v) for _, v in payload]

    def stats(self):
        """Slot counts, not Python/GPU bytes; count each physical block once."""
        used = sum(len(payload) for payload in self._blocks.values())
        allocated = len(self._blocks) * self.block_size
        return {"logical_tokens": sum(self._lengths.values()), "physical_tokens": used,
                "allocated_slots": allocated, "unused_slots": allocated - used,
                "physical_blocks": len(self._blocks)}

    def check_invariants(self):
        counted = {block: 0 for block in self._blocks}
        for request, table in self._tables.items():
            length = self._lengths[request]
            assert len(table) == (length + self.block_size - 1) // self.block_size
            assert len(set(table)) == len(table)
            for i, block in enumerate(table):
                assert block in self._blocks
                expected = min(self.block_size, length - i * self.block_size)
                assert len(self._blocks[block]) == expected
                counted[block] += 1
        assert counted == self._refs
        assert all(ref > 0 for ref in self._refs.values())
        assert set(self._blocks).isdisjoint(self._free)
        assert len(self._free) == len(set(self._free))
        assert set(self._blocks) | set(self._free) == set(range(self.max_blocks))
        return True


def request_metrics(arrival_ms, token_times_ms):
    """Client-observed timestamps. TPOT excludes first-token delay; 1 token -> None."""
    if (type(arrival_ms) not in (int, float) or not math.isfinite(arrival_ms)
            or arrival_ms < 0 or not isinstance(token_times_ms, (list, tuple)) or not token_times_ms
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in token_times_ms)
            or token_times_ms[0] < arrival_ms
            or any(b < a for a, b in zip(token_times_ms, token_times_ms[1:]))):
        raise ValueError("finite nonnegative arrival and ordered nonempty token timestamps required")
    count = len(token_times_ms)
    return {"ttft_ms": token_times_ms[0] - arrival_ms,
            "tpot_ms": (token_times_ms[-1] - token_times_ms[0]) / (count - 1) if count > 1 else None,
            "e2e_ms": token_times_ms[-1] - arrival_ms, "output_tokens": count}


def output_throughput(output_tokens, window_start_ms, window_end_ms):
    """Completed output tokens per second within the explicitly chosen window."""
    if (type(output_tokens) is not int or output_tokens < 0
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in (window_start_ms, window_end_ms))
            or window_start_ms < 0 or window_end_ms <= window_start_ms):
        raise ValueError("nonnegative count and a finite positive observation window required")
    return 1000 * output_tokens / (window_end_ms - window_start_ms)


def scheduling_demo(continuous=True):
    """Fixed-cost DECODE-only slot toy: prompts were prefilled elsewhere.

    A/B ready at step 0 need 2/4 outputs, C ready at step 1 needs 2.
    Each iteration has 2 slots and costs 10 ms by assumption, not measurement.
    Static batching cannot refill a vacant slot until the whole batch finishes.
    """
    if type(continuous) is not bool:
        raise ValueError("continuous must be a boolean")
    pending = [(0, "A", 2), (0, "B", 4), (1, "C", 2)]
    slots, emissions, timeline, step = [None, None], {"A": [], "B": [], "C": []}, [], 0
    remaining = {}
    while pending or any(slots):
        if continuous or not any(slots):
            for i in range(2):
                if slots[i] is None and pending and pending[0][0] <= step:
                    _, name, tokens = pending.pop(0)
                    slots[i], remaining[name] = name, tokens
        timeline.append(tuple(slots))
        for i, name in enumerate(slots):
            if name is not None:
                emissions[name].append((step + 1) * 10)
                remaining[name] -= 1
                if remaining[name] == 0:
                    slots[i] = None
        step += 1
    return {"timeline": timeline, "token_times_ms": emissions, "duration_ms": step * 10}


def gpu_sdpa_demo():
    """Optional smoke, never a serving benchmark. Returns False on explicit skip."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("SKIP GPU SDPA: PyTorch is not installed; CPU lesson needs no torch. "
              "For GPU, install an official CUDA-compatible PyTorch build.")
        return False
    if not torch.cuda.is_available():
        print(f"SKIP GPU SDPA: torch={torch.__version__}, CUDA unavailable; GPU kernel not run.")
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        print(f"SKIP GPU SDPA: torch={torch.__version__} lacks the required backend API.")
        return False
    torch.manual_seed(15)
    q, k, v = [torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16) for _ in range(3)]
    reference = dense_attention(q[0, 0].tolist(), k[0, 0].tolist(), v[0, 0].tolist(), causal_mask(8, 8))
    try:
        with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                torch.cuda.synchronize()
    except RuntimeError as exc:
        print(f"SKIP GPU SDPA: forced Flash backend unavailable or failed: {exc}")
        return False
    expected = torch.tensor(reference, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(actual[0, 0].float(), expected, atol=3e-3, rtol=3e-3)
    events = sorted({event.key for event in profile.key_averages() if "attention" in event.key.lower()
                     or "flash" in event.key.lower()})
    print(f"GPU SDPA smoke: torch={torch.__version__}, CUDA={torch.version.cuda}, "
          f"device={torch.cuda.get_device_name()}, shape=(1,1,8,16), dtype=float16")
    print(f"Forced FLASH_ATTENTION; observed profiler events={events}; no latency/throughput claim.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-sdpa", action="store_true", help="optional GPU smoke; prints SKIP if unsupported")
    args = parser.parse_args(argv)
    print(f"P15 CPU teaching reference: Python {sys.version.split()[0]}, standard library, seed=15")
    q, k, v = [[1.0]], [[0.0], [math.log(2)], [math.log(4)]], [[2.0], [4.0], [8.0]]
    hand = blocked_attention(q, k, v, key_block_size=2)[0][0]
    assert math.isclose(hand, 6.0, abs_tol=1e-12)
    print(f"Online hand example: output={hand:.12f}, expected=6; blocks=[2,1]")
    rng = random.Random(15)
    q = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(5)]
    k = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(7)]
    v = [[rng.uniform(-2, 2) for _ in range(2)] for _ in range(7)]
    allowed = causal_mask(5, 7, offset=2)
    expected, actual = dense_attention(q, k, v, allowed), blocked_attention(q, k, v, allowed)
    error = max(abs(a - b) for row_a, row_b in zip(actual, expected) for a, b in zip(row_a, row_b))
    assert error < 1e-12
    print(f"Uneven blocks Q=5,K=7,d=3,dv=2: max_error={error:.3g}; tolerance=1e-12")
    masked = blocked_attention([[1]], [[1000], [1001], [1002]], [[2], [4], [8]], [[False] * 3])
    assert masked == [[0.0]]
    print(f"All-masked safety: {masked}")
    pool = PagedKVCache()
    pool.create("A")
    for token in (10, 20, 30):
        pool.append("A", [token], [token])
    pool.fork("A", "B")
    print(f"Before COW: A={pool.block_table('A')}, B={pool.block_table('B')}, refs={pool.refcounts()}")
    pool.append("B", [40], [40])
    pool.check_invariants()
    print(f"After COW: A={pool.block_table('A')}, B={pool.block_table('B')}, refs={pool.refcounts()}")
    print(f"Slots after COW: {pool.stats()}")
    for name, arrival, times in (("A", 0, [100, 140, 190]), ("B", 50, [150, 210])):
        print(f"Metric arithmetic {name}: {request_metrics(arrival, times)}")
    print(f"Output throughput: {output_throughput(5, 0, 210):.6f} tokens/s (invented timeline, not measured)")
    for continuous in (False, True):
        data = scheduling_demo(continuous)
        print(f"{'Continuous' if continuous else 'Static'} decode toy: {data}")
    if args.gpu_sdpa:
        gpu_sdpa_demo()
    else:
        print("GPU SDPA: not requested, not run. This CPU script is not a performance benchmark.")


if __name__ == "__main__":
    main()
