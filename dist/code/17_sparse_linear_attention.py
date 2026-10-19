"""P16: explicit sparse masks and normalized causal linear attention on CPU.

Run: OMP_NUM_THREADS=1 python code/17_sparse_linear_attention.py
Independent teaching experiment, NOT an S4 replacement or a fast kernel.
Window W includes the current position. float32/64 only; 1..512 tokens.
"""
import math
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P16 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

MAX_TOKENS = 512


def _length(value, name):
    if type(value) is not int or not 1 <= value <= MAX_TOKENS:
        raise ValueError(f"{name} must be an integer in 1..{MAX_TOKENS}")


def causal_window_mask(length, window, offset=0):
    """[C,P+C] bool mask; W counts current plus at most W-1 preceding keys.

    All prefix keys are supplied. This function does not implement ring storage.
    """
    _length(length, "length")
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    _length(offset + length, "total length")
    query_pos = torch.arange(offset, offset + length)[:, None]
    key_pos = torch.arange(offset + length)[None, :]
    return (key_pos <= query_pos) & (key_pos > query_pos - window)


def global_local_mask(length, window, global_positions=()):
    """Causal local + selected global query/key positions, not Longformer itself."""
    local = causal_window_mask(length, window)
    if not isinstance(global_positions, (tuple, list)):
        raise ValueError("global_positions must be a tuple/list of positions")
    if any(type(p) is not int or not 0 <= p < length for p in global_positions):
        raise ValueError("global position must be an integer inside the sequence")
    global_flags = torch.zeros(length, dtype=torch.bool)
    global_flags[list(global_positions)] = True
    causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    return (local | global_flags[:, None] | global_flags[None, :]) & causal


def receptive_positions(length, window, layers):
    """Structural reachability only: no claim that trained weights retain facts."""
    mask = causal_window_mask(length, window)
    if type(layers) is not int or layers < 0:
        raise ValueError("layers must be a nonnegative integer")
    reachable = [{i} for i in range(length)]
    for _ in range(layers):
        reachable = [set().union(*(reachable[j] for j in range(length) if mask[i, j]))
                     for i in range(length)]
    return tuple(tuple(sorted(row)) for row in reachable)


def _validate_qkv(q, k, v, equal_lengths=False):
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU autocast is outside this lesson; use float32 or float64 directly")
    tensors = (q, k, v)
    if any(not isinstance(x, torch.Tensor) or x.ndim != 3 for x in tensors):
        raise ValueError("q/k/v must be rank-three [B,T,width] tensors")
    if any(x.dtype not in (torch.float32, torch.float64) or x.device.type != "cpu"
           for x in tensors):
        raise ValueError("this CPU lesson supports only float32 and float64")
    if any(x.dtype != q.dtype for x in tensors):
        raise ValueError("q/k/v dtype must match")
    if any(min(x.shape) <= 0 for x in tensors):
        raise ValueError("q/k/v dimensions must be nonempty")
    _length(q.shape[1], "query length")
    _length(k.shape[1], "key length")
    if k.shape[0] != q.shape[0] or v.shape[:2] != k.shape[:2] or q.shape[2] != k.shape[2]:
        raise ValueError("batch, key/value length and query/key width must match")
    if equal_lengths and q.shape[1] != k.shape[1]:
        raise ValueError("linear lesson requires equal q/k/v lengths")
    if any(not torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("q/k/v must contain only finite values")


def masked_softmax_attention(q, k, v, mask):
    """Dense reference using a sparse mask; NOT a sparse execution kernel."""
    _validate_qkv(q, k, v)
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.device.type != "cpu" or tuple(mask.shape) != (q.shape[1], k.shape[1])):
        raise ValueError("mask must be a CPU bool [query length,key length] tensor")
    if not mask.any(dim=-1).all().item():
        raise ValueError("all-masked query rows are undefined in this lesson")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all().item():
        raise ValueError("scores overflowed; use smaller values or float64")
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    output = weights @ v
    if not torch.isfinite(output).all().item():
        raise ValueError("attention output is not finite")
    return output


def positive_features(x):
    """ELU(x)+1: x+1 for x>=0, exp(x) otherwise; no subtract/add cancellation.

    clamp prevents overflow in the unselected exp branch of torch.where.
    Extremely negative finite inputs can still underflow to zero.
    """
    return torch.where(x >= 0, x + 1, torch.exp(torch.clamp(x, max=0)))


def _normalized(numerator, denominator):
    if (not torch.isfinite(denominator).all().item()
            or not (denominator > 0).all().item()):
        raise ValueError("linear denominator is nonpositive or nonfinite; feature underflow/overflow")
    output = numerator / denominator
    if not torch.isfinite(output).all().item():
        raise ValueError("linear output is not finite; state or division overflow")
    return output


def linear_parallel(q, k, v):
    """Explicit causal kernel matrix oracle target, O(T^2), NOT a fast scan.

    The feature kernel replaces exp(q.k/sqrt(d)); it is not equal to softmax.
    """
    _validate_qkv(q, k, v, equal_lengths=True)
    qf, kf = positive_features(q), positive_features(k)
    kernel = qf @ kf.transpose(-2, -1)
    causal = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril()
    weights = kernel.masked_fill(~causal, 0)
    return _normalized(weights @ v, weights.sum(dim=-1, keepdim=True))


def linear_recurrent(q, k, v):
    """Return (all_outputs, final_S, final_z), without retaining per-token K/V.

    S:[B,r,dv], z:[B,r]. Constant-sized *inference state*, not constant total
    memory: inputs, outputs and autograd's saved history can grow with T.
    """
    _validate_qkv(q, k, v, equal_lengths=True)
    batch, length, width = q.shape
    state = q.new_zeros(batch, width, v.shape[-1])
    normalizer = q.new_zeros(batch, width)
    outputs = []
    for t in range(length):
        qt, kt = positive_features(q[:, t]), positive_features(k[:, t])
        state = state + kt[:, :, None] * v[:, t, None, :]
        normalizer = normalizer + kt
        numerator = (qt[:, :, None] * state).sum(dim=1)
        denominator = (qt * normalizer).sum(dim=1, keepdim=True)
        outputs.append(_normalized(numerator, denominator))
    return torch.stack(outputs, dim=1), state, normalizer


def retrieval_experiment(lengths=(1, 8, 64, 256)):
    """Same hand-set q/k/v, no training: target first, opposite-key distractors.

    Target value=1, other values=0. Success is |last output-1|<=0.1.
    This is kernel selectivity/interference, not a theorem about all recurrence.
    """
    rows = []
    for length in lengths:
        _length(length, "retrieval length")
        q = torch.tensor([4.0, -4.0], dtype=torch.float64).repeat(1, length, 1)
        k = -q.clone()
        k[:, 0] = q[:, 0]
        v = torch.zeros(1, length, 1, dtype=torch.float64)
        v[:, 0] = 1
        dense = masked_softmax_attention(q, k, v, causal_window_mask(length, length))
        local = masked_softmax_attention(q, k, v, causal_window_mask(length, 4))
        parallel = linear_parallel(q, k, v)
        recurrent, state, normalizer = linear_recurrent(q, k, v)
        torch.testing.assert_close(parallel, recurrent, atol=2e-14, rtol=2e-14)
        rows.append({"length": length, "softmax": dense[0, -1, 0].item(),
                     "window4": local[0, -1, 0].item(), "linear": recurrent[0, -1, 0].item(),
                     "state_bytes": sum(x.numel() * x.element_size() for x in (state, normalizer))})
    return rows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(16)
    print(f"P16 CPU | torch={torch.__version__} | seed=16 | float64")
    print("W=3 includes current; mask rows:", causal_window_mask(6, 3).int().tolist())
    print("position 5 after 1/2/3 layers:", [receptive_positions(6, 3, n)[5] for n in (1, 2, 3)])
    print("global key 0, query 5:", global_local_mask(6, 3, (0,))[5].int().tolist())
    q, k = [torch.randn(2, 7, 3, dtype=torch.float64) for _ in range(2)]
    v = torch.randn(2, 7, 2, dtype=torch.float64)
    parallel = linear_parallel(q, k, v)
    recurrent, state, normalizer = linear_recurrent(q, k, v)
    torch.testing.assert_close(parallel, recurrent, atol=2e-14, rtol=2e-14)
    print(f"linear parallel/recurrent max_error={(parallel-recurrent).abs().max().item():.2e}")
    print("state S/z shapes:", tuple(state.shape), tuple(normalizer.shape))
    print("retrieval: T softmax window4 linear state_bytes (success: absolute error <= 0.1)")
    with torch.no_grad():
        for row in retrieval_experiment():
            print(f"{row['length']:3d} {row['softmax']:.6f} {row['window4']:.6f} "
                  f"{row['linear']:.6f} {row['state_bytes']}")
    print("No model training, sparse kernel, GPU benchmark or S4 replacement was performed.")


if __name__ == "__main__":
    main()
