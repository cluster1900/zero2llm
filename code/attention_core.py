"""CPU teaching primitives. Boolean masks use True=allowed, not nn.MultiheadAttention's convention."""
import math
import torch


def masked_softmax(scores, allowed=None):
    """Normalize the last axis; fully blocked rows have zero weights, not NaN.

    Finite floating scores are required. This explicit teaching policy is not a
    claim that every backend/API handles all-masked rows identically.
    """
    if not scores.is_floating_point() or scores.shape[-1] == 0:
        raise ValueError("scores must be floating with a nonempty last axis")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite before masking")
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if allowed.dtype != torch.bool:
        raise TypeError("allowed must be boolean: True means visible")
    try:
        allowed = torch.broadcast_to(allowed, scores.shape)
    except RuntimeError as exc:
        raise ValueError("mask must broadcast to scores") from exc
    # Replace fully blocked rows BEFORE softmax so neither forward nor backward sees NaN.
    any_allowed = allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(any_allowed, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=-1)
    return weights.masked_fill(~allowed, 0.0)


def self_attention_mask(valid, causal=True):
    """valid [B,T] -> allowed [B,1,T,T]; exclude padded queries AND keys."""
    if valid.ndim != 2 or valid.dtype != torch.bool or valid.shape[1] == 0:
        raise ValueError("valid must be nonempty boolean [B,T]")
    allowed = valid[:, None, :, None] & valid[:, None, None, :]
    if causal:
        allowed = allowed & torch.ones(
            valid.shape[1], valid.shape[1], dtype=torch.bool, device=valid.device).tril()
    return allowed


def scaled_attention(q, k, v, allowed=None):
    """Matching batch/head axes; Q [...,T,d], K [...,S,d], V [...,S,dv]."""
    if q.ndim < 2 or q.ndim != k.ndim or k.ndim != v.ndim:
        raise ValueError("Q/K/V must have matching ranks >= 2")
    if (q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]
            or q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]
            or q.shape[-1] == 0 or k.shape[-2] == 0):
        raise ValueError("incompatible Q/K/V dimensions")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = masked_softmax(scores, allowed)
    return weights @ v, weights
