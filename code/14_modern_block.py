"""P13 / S3: change normalization and FFN, retain S2 RoPE and cache semantics.

CPU teaching experiment, no model download. Run: python code/14_modern_block.py
An explicit bias-free control separates parameter policy from the two ablations.
"""
import importlib.util
import math
from pathlib import Path
import sys
from dataclasses import dataclass, fields, replace

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "torch":
        print("P13 requires PyTorch. Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2) from None
    raise

_NAME = "zero2llm_s2"
if _NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(_NAME, Path(__file__).with_name("13_rope.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
s2 = sys.modules[_NAME]


def positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def positive_eps(eps):
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


class RMSNorm(nn.Module):
    """Normalize only the last feature axis; gamma only, no centering or beta.

    fp16/bf16 statistics and multiply use fp32; fp64 stays fp64. Return input
    dtype. This reduces low-precision overflow but is not arbitrary-range proof.
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        positive_int("d_model", d_model)
        positive_eps(eps)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or
                x.ndim < 1 or x.shape[-1] != self.weight.numel()):
            raise ValueError("RMSNorm expects floating [..., d_model] input")
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        inv_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return (work * inv_rms * self.weight.to(work.dtype)).to(x.dtype)


class SwiGLU(nn.Module):
    """Three projections: down(silu(gate(x)) * up(x)); * is elementwise."""
    def __init__(self, d_model, intermediate_size, bias=False):
        super().__init__()
        positive_int("d_model", d_model)
        positive_int("intermediate_size", intermediate_size)
        if type(bias) is not bool:
            raise ValueError("bias must be boolean")
        self.gate = nn.Linear(d_model, intermediate_size, bias=bias)
        self.up = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down = nn.Linear(intermediate_size, d_model, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def ffn_parameter_count(d_model, width, kind, bias=False):
    positive_int("d_model", d_model)
    positive_int("width", width)
    if kind not in ("gelu", "swiglu") or type(bias) is not bool:
        raise ValueError("need gelu/swiglu kind and boolean bias")
    if kind == "gelu":
        return 2 * d_model * width + (width + d_model if bias else 0)
    return 3 * d_model * width + (2 * width + d_model if bias else 0)


def matched_swiglu_width(d_model, gelu_width=None, bias=False, multiple_of=1):
    """Nearest allowed positive width to GELU parameter budget; ties go smaller.

    No GPU alignment rounding by default. Recompute the actual count after any
    multiple-of constraint; rounded widths need not have equal parameter counts.
    """
    positive_int("d_model", d_model)
    positive_int("multiple_of", multiple_of)
    if gelu_width is None:
        gelu_width = 4 * d_model
    target = ffn_parameter_count(d_model, gelu_width, "gelu", bias)
    slope = 3 * d_model + (2 if bias else 0)
    offset = d_model if bias else 0
    lower = max(1, (target - offset) // (slope * multiple_of))
    candidates = [lower * multiple_of, (lower + 1) * multiple_of]
    return min(candidates, key=lambda width: (abs(ffn_parameter_count(
        d_model, width, "swiglu", bias) - target), width))


@dataclass(frozen=True)
class ModernConfig(s2.RopeConfig):
    norm_kind: str = "rmsnorm"
    ffn_kind: str = "swiglu"
    norm_eps: float = 1e-5
    intermediate_size: int | None = None
    norm_bias: bool = False
    ffn_bias: bool = False

    def __post_init__(self):
        # Validate the base independently even if its validation enumerates fields.
        s2.RopeConfig(**{field.name: getattr(self, field.name) for field in fields(s2.RopeConfig)})
        if self.norm_kind not in ("layernorm", "rmsnorm"):
            raise ValueError("norm_kind must be layernorm or rmsnorm")
        if self.ffn_kind not in ("gelu", "swiglu"):
            raise ValueError("ffn_kind must be gelu or swiglu")
        positive_eps(self.norm_eps)
        if self.intermediate_size is not None:
            positive_int("intermediate_size", self.intermediate_size)
        if type(self.norm_bias) is not bool or type(self.ffn_bias) is not bool:
            raise ValueError("bias options must be boolean")
        if self.norm_kind == "rmsnorm" and self.norm_bias:
            raise ValueError("this RMSNorm has gamma only; norm_bias must be False")

    @property
    def ffn_width(self):
        if self.intermediate_size is not None:
            return self.intermediate_size
        return (4 * self.d_model if self.ffn_kind == "gelu" else
                matched_swiglu_width(self.d_model, bias=self.ffn_bias))


class ModernGPT(s2.RopeGPT):
    """S3 leaves S2 attention and forward/cache paths intact.

    Choosing LN + GELU, both biases, eps=1e-5 and width=4D exactly reproduces S2
    initialization with the same seed. S3 defaults use RMSNorm + bias-free SwiGLU.
    """
    def __init__(self, config):
        if not isinstance(config, ModernConfig):
            raise ValueError("ModernGPT requires ModernConfig")
        super().__init__(config)
        def norm(old):
            if config.norm_kind == "layernorm":
                result = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.norm_bias)
            else:
                result = RMSNorm(config.d_model, eps=config.norm_eps)
            with torch.no_grad():
                result.weight.copy_(old.weight)
                if getattr(result, "bias", None) is not None:
                    result.bias.copy_(old.bias)
            return result
        for block in self.blocks:
            block.ln1 = norm(block.ln1)
            block.ln2 = norm(block.ln2)
            if config.ffn_kind == "swiglu":
                block.mlp = SwiGLU(config.d_model, config.ffn_width, config.ffn_bias)
            elif config.ffn_width != 4 * config.d_model or not config.ffn_bias:
                old = block.mlp
                block.mlp = nn.Sequential(
                    nn.Linear(config.d_model, config.ffn_width, bias=config.ffn_bias),
                    nn.GELU(),
                    nn.Linear(config.ffn_width, config.d_model, bias=config.ffn_bias))
                # Keep old weights when merely removing biases, not redraw them.
                if config.ffn_width == 4 * config.d_model:
                    with torch.no_grad():
                        block.mlp[0].weight.copy_(old[0].weight)
                        block.mlp[2].weight.copy_(old[2].weight)
        self.ln_f = norm(self.ln_f)


# Cache code calls block.ln1/ln2/mlp, so it uses the actual replacement modules.
cached_forward = s2.cached_forward


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _is_ffn(name):
    return ".mlp." in name


def _is_norm(name):
    return name.startswith("ln_f.") or ".ln1." in name or ".ln2." in name


@torch.no_grad()
def copy_unchanged(source, target, changed):
    """Copy, then assert equality of every tensor outside the declared change.

    This is an experiment setup operation, NOT a trained-checkpoint conversion.
    Mismatched modules are newly initialized; no claim of functional equivalence.
    """
    if changed not in ("norm", "ffn"):
        raise ValueError("changed must be norm or ffn")
    excluded = _is_norm if changed == "norm" else _is_ffn
    old, new = source.state_dict(), target.state_dict()
    old_names = {name for name in old if not excluded(name)}
    new_names = {name for name in new if not excluded(name)}
    if old_names != new_names:
        raise ValueError("unchanged state keys differ")
    for name in sorted(old_names):
        if old[name].shape != new[name].shape:
            raise ValueError(f"unchanged tensor shape differs: {name}")
        new[name].copy_(old[name])
        if not torch.equal(new[name], old[name]):
            raise AssertionError(f"copy failed: {name}")
    # LN and RMSNorm have corresponding gamma. Hold this constant too.
    if changed == "norm":
        for name in old.keys() & new.keys():
            if _is_norm(name) and old[name].shape == new[name].shape:
                new[name].copy_(old[name])
    return tuple(sorted(old_names))


def make_ablation_models(config=None, seed=42):
    """A -> B changes normalization only; B -> C changes FFN family/budget.

    All controls are bias-free. Width adjustment belongs to the FFN treatment;
    this is not an activation-only experiment. Uses isolated CPU RNG state.
    """
    if config is None:
        config = ModernConfig(vocab_size=57)
    common = replace(config, norm_kind="layernorm", ffn_kind="gelu",
                     norm_bias=False, ffn_bias=False, intermediate_size=4 * config.d_model)
    configs = (common, replace(common, norm_kind="rmsnorm"),
               replace(common, norm_kind="rmsnorm", ffn_kind="swiglu",
                       intermediate_size=matched_swiglu_width(config.d_model)))
    models = []
    with torch.random.fork_rng(devices=[]):
        for item in configs:
            torch.manual_seed(seed)
            models.append(ModernGPT(item))
    copy_unchanged(models[0], models[1], "norm")
    copy_unchanged(models[1], models[2], "ffn")
    return dict(zip(("A_LN_GELU", "B_RMS_GELU", "C_RMS_SwiGLU"), models))


def run_ablation(config=None, seed=42, steps=8):
    """Tiny deterministic repeated-batch train smoke, NOT a quality benchmark."""
    positive_int("steps", steps)
    models = make_ablation_models(config, seed)
    config = next(iter(models.values())).config
    # Generate a single causal next-token batch, reuse unchanged for every model.
    length = min(12, config.max_seq_len)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    stream = torch.randint(config.vocab_size, (2, length + 1), generator=generator)
    inputs, targets = stream[:, :-1], stream[:, 1:]
    records = []
    for name, model in models.items():
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        before = model(inputs, targets)[1].item()
        finite = True
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            finite = finite and bool(torch.isfinite(loss)) and all(
                p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if not finite:
                raise AssertionError("nonfinite loss/gradient in ablation")
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full, loss = model(inputs, targets)
            cache, pieces = None, []
            for chunk in inputs.split(5, dim=1):
                logits, cache = cached_forward(model, chunk, cache)
                pieces.append(logits)
            error = (full - torch.cat(pieces, dim=1)).abs().max().item()
            torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=3e-6, rtol=1e-5)
        records.append(dict(name=name, width=model.config.ffn_width,
                            ffn_params=parameter_count(model.blocks[0].mlp),
                            params=parameter_count(model), before=before, after=loss.item(),
                            finite_backward=finite, cache_error=error))
    return records


def norm_walkthrough():
    """Independent two-coordinate hand table; eps=0.5 only for easy arithmetic."""
    x = torch.tensor([[1., 3.], [3., 5.]], dtype=torch.float64)
    eps = 0.5
    ln = nn.LayerNorm(2, eps=eps, bias=False).double()
    rms = RMSNorm(2, eps=eps).double()
    return dict(x=x, mean=x.mean(-1), variance=x.var(-1, correction=0),
                mean_square=x.square().mean(-1), ln=ln(x), rms=rms(x))


def gate_walkthrough():
    """x=[1,2], two projections and identity down: all rows hand-checkable."""
    layer = SwiGLU(2, 2).double()
    with torch.no_grad():
        layer.gate.weight.copy_(torch.tensor([[1., -1.], [0., 1.]]))
        layer.up.weight.copy_(torch.tensor([[1., 1.], [2., 1.]]))
        layer.down.weight.copy_(torch.eye(2))
    x = torch.tensor([1., 2.], dtype=torch.float64)
    gate, up = layer.gate(x), layer.up(x)
    return dict(x=x, gate=gate, sigmoid=gate.sigmoid(), silu=F.silu(gate),
                up=up, product=F.silu(gate) * up, output=layer(x))


def main():
    torch.set_num_threads(1)
    fmt = lambda values: "[" + ", ".join(f"{value:.6f}" for value in values) + "]"
    table = norm_walkthrough()
    for row in range(2):
        print(f"norm x={fmt(table['x'][row])} mean={table['mean'][row]:.6f} "
              f"var={table['variance'][row]:.6f} mean_square={table['mean_square'][row]:.6f} "
              f"LN={fmt(table['ln'][row])} RMS={fmt(table['rms'][row])}")
    gate = gate_walkthrough()
    for key in ("gate", "sigmoid", "silu", "up", "product", "output"):
        print(f"gate {key}={fmt(gate[key])}")
    print("FFN D=48: GELU h=192 no_bias=18432; SwiGLU m=128 no_bias=18432")
    print("CPU torch=" + torch.__version__ + ", float32, threads=1, seed=42, B=2, T=12, steps=8")
    print("same next-token batch; fresh AdamW lr=.001 betas=(.9,.999) eps=1e-8 wd=0; no dropout")
    print("control A is bias-free, not unchanged S2; copy unchanged tensors before training")
    for row in run_ablation():
        print(f"{row['name']}: width={row['width']} ffn={row['ffn_params']} total={row['params']} "
              f"loss={row['before']:.6f}->{row['after']:.6f} "
              f"finite_backward={row['finite_backward']} cache_error={row['cache_error']:.3g}")
    print("Training smoke only: one seed, one repeated synthetic batch, no held-out quality or speed claim.")


if __name__ == "__main__":
    main()
