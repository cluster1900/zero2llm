"""P22: offline DPO/GRPO arithmetic and genuine tiny CPU parameter updates.

Default: standard library only. --torch: prompt-conditioned three-action policy.
This policy chooses a WHOLE canned answer in one action, not an autoregressive
LLM. Token-level helpers separately verify shifting, length masks and reduction.
No TRL execution, pretrained weights, downloads, or external data are involved.
"""
from dataclasses import dataclass
import argparse
import json
import math


def _positive(name, value, allow_zero=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def dpo_pair_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.2):
    """One sequence-summed log-probability pair; natural logarithms."""
    _positive("beta", beta)
    if not all(math.isfinite(x) for x in (chosen, rejected, ref_chosen, ref_rejected)):
        raise ValueError("log probabilities must be finite")
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    z = beta * margin
    return max(0.0, -z) + math.log1p(math.exp(-abs(z)))


def group_advantages(rewards, prompt_ids, epsilon=1e-8):
    """Population std (divide by G), per prompt, plus epsilon OUTSIDE sqrt.

    Group membership uses prompt IDs, not an assumed contiguous reshape.
    Each group needs G>=2. An all-equal group is exactly zero, never NaN.
    """
    _positive("epsilon", epsilon)
    if len(rewards) != len(prompt_ids) or not rewards:
        raise ValueError("rewards and prompt_ids must have equal nonzero lengths")
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError("rewards must be finite")
    groups = {}
    for index, prompt_id in enumerate(prompt_ids):
        groups.setdefault(prompt_id, []).append(index)
    result = [0.0] * len(rewards)
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("each prompt needs at least two sampled answers")
        mean = sum(rewards[i] for i in indices) / len(indices)
        variance = sum((rewards[i] - mean) ** 2 for i in indices) / len(indices)
        denominator = math.sqrt(variance) + epsilon
        for i in indices:
            result[i] = (rewards[i] - mean) / denominator
    return result


def exact_kl(probabilities, reference):
    """Exact D_KL(p || q) over every action; p,q must have full support."""
    if len(probabilities) != len(reference) or not probabilities:
        raise ValueError("distributions must have equal nonzero lengths")
    for row in (probabilities, reference):
        if any(not math.isfinite(x) or x <= 0 for x in row):
            raise ValueError("strictly positive finite probabilities required")
        if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to one")
    return sum(p * math.log(p / q) for p, q in zip(probabilities, reference))


@dataclass(frozen=True)
class Rollout:
    sample_id: str
    prompt_id: int
    action_id: int
    completion: str


@dataclass(frozen=True)
class ScoredReward:
    sample_id: str
    prompt_id: int
    value: float


PROMPTS = ("1+1=?", "2+1=?")
ANSWERS = ("2", "我猜是 0", "3")
GOLD = ("2", "3")


def score_rollouts(rollouts):
    """Deliberately strict exact-match verifier for TWO synthetic questions."""
    return [ScoredReward(r.sample_id, r.prompt_id, float(r.completion == GOLD[r.prompt_id]))
            for r in rollouts]


def aligned_rewards(rollouts, rewards):
    """Reject positional mismatches before constructing tensors, no silent zip."""
    if len(rollouts) != len(rewards) or not rollouts:
        raise ValueError("rollout/reward count mismatch")
    seen = set()
    values, prompt_ids = [], []
    for row, reward in zip(rollouts, rewards):
        if row.sample_id in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row.sample_id)
        if row.sample_id != reward.sample_id or row.prompt_id != reward.prompt_id:
            raise ValueError("sample/prompt/reward alignment mismatch")
        if type(row.prompt_id) is not int or not 0 <= row.prompt_id < len(PROMPTS):
            raise ValueError("unknown prompt")
        if type(row.action_id) is not int or not 0 <= row.action_id < len(ANSWERS):
            raise ValueError("unknown action")
        if row.completion != ANSWERS[row.action_id]:
            raise ValueError("action/completion mismatch")
        if not math.isfinite(reward.value):
            raise ValueError("reward must be finite")
        values.append(reward.value)
        prompt_ids.append(row.prompt_id)
    return values, prompt_ids


def hand_demo():
    # Every row is a separate prompt with chosen/rejected/ref probabilities.
    pairs = [(0.6, 0.2, 0.4, 0.4), (0.2, 0.5, 0.1, 0.5)]
    losses = [dpo_pair_loss(*(math.log(p) for p in row), beta=0.5) for row in pairs]
    return {
        "dpo_beta": 0.5, "dpo_pair_losses": losses,
        "dpo_batch_mean": sum(losses) / len(losses),
        "group_rewards": [0, 1, 1, 2],
        "group_population_std": math.sqrt(0.5),
        "group_advantages": group_advantages([0, 1, 1, 2], [0] * 4),
        "all_equal_advantages": group_advantages([2, 2, 2, 2], [1] * 4),
        "exact_kl_example": exact_kl([0.6, 0.3, 0.1], [0.2, 0.5, 0.3]),
    }


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("P22 torch SKIP: PyTorch not installed; standard-library demo still works.") from exc
    return torch


def _check_token_batch(logits, input_ids, attention_mask, prompt_lengths):
    torch = _torch()
    if logits.ndim != 3 or input_ids.ndim != 2 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("expected logits [B,T,V] and IDs [B,T]")
    b, t = input_ids.shape
    if b == 0 or t < 2 or logits.shape[-1] < 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("empty/short batch or invalid mask shape")
    if input_ids.dtype != torch.long or torch.any(input_ids < 0) or torch.any(input_ids >= logits.shape[-1]):
        raise ValueError("all IDs, including PAD IDs, must be valid vocabulary indices")
    if not bool(torch.isfinite(logits).all()) or len(prompt_lengths) != b:
        raise ValueError("invalid logits or prompt lengths")
    for i, length in enumerate(prompt_lengths):
        if type(length) is not int or not 1 <= length < t:
            raise ValueError("each prompt needs at least one token and a completion")
        row = attention_mask[i].tolist()
        if any(x not in (0, 1, False, True) for x in row):
            raise ValueError("attention mask must be binary")
        real = sum(row)
        if row != [1] * real + [0] * (t - real) or real <= length:
            raise ValueError("right padding and a nonempty completion required")


def completion_log_probs(logits, input_ids, attention_mask, prompt_lengths):
    """Causal shift exactly once; exclude prompt/PAD, include actual EOS.

    Returns sequence SUMS [B] and completion-only mask [B,T-1]. Not means:
    changing a sequence log probability to a token average changes DPO's loss.
    """
    torch = _torch()
    _check_token_batch(logits, input_ids, attention_mask, prompt_lengths)
    next_logps = logits[:, :-1].log_softmax(dim=-1)
    targets = input_ids[:, 1:]
    selected = next_logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, input_ids.shape[1], device=logits.device)[None, :]
    starts = torch.tensor(prompt_lengths, device=logits.device)[:, None]
    mask = attention_mask[:, 1:].bool() & (positions >= starts)
    return (selected * mask).sum(dim=-1), mask


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.2):
    torch = _torch()
    _positive("beta", beta)
    rows = (chosen, rejected, ref_chosen, ref_rejected)
    if chosen.ndim != 1 or chosen.numel() == 0 or any(r.shape != chosen.shape for r in rows):
        raise ValueError("DPO expects equally sized nonempty vectors")
    if not all(bool(torch.isfinite(r).all()) for r in rows):
        raise ValueError("DPO requires finite log probabilities")
    margin = (chosen - rejected) - (ref_chosen.detach() - ref_rejected.detach())
    return torch.nn.functional.softplus(-beta * margin).mean()


def categorical_kl(logits, reference_logits):
    """Exact KL(pi_theta || pi_ref), summing ALL vocabulary/actions, not samples.

    At a fixed prefix this is a conditional next-token KL. Averaging it over
    sampled prefixes is NOT the full joint sequence-distribution KL. For the
    demo's one-step finite-action policy, it IS the full answer-distribution KL.
    """
    torch = _torch()
    if logits.shape != reference_logits.shape or logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("same shape and at least two actions required")
    if not bool(torch.isfinite(logits).all() & torch.isfinite(reference_logits).all()):
        raise ValueError("finite logits required")
    logp = logits.log_softmax(dim=-1)
    logq = reference_logits.detach().log_softmax(dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def grpo_loss(logits, targets, old_logps, reference_logits, mask, advantages,
              beta=0.04, clip_epsilon=0.2):
    """Educational token-wise clipped objective, exact conditional KL variant.

    logits/ref [N,L,V], targets/old/mask [N,L], advantages [N]. Each answer
    averages its valid tokens; then all answers have equal weight. old/ref and
    advantages are stop-gradient. Unlike original DeepSeekMath, the tiny vocab
    permits exact conditional KL rather than a sampled KL estimator.
    """
    torch = _torch()
    _positive("beta", beta, allow_zero=True)
    _positive("clip_epsilon", clip_epsilon)
    if clip_epsilon >= 1:
        raise ValueError("clip_epsilon must be below one")
    if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be nonempty [N,L,V]")
    expected = logits.shape[:2]
    if any(x.shape != expected for x in (targets, old_logps, mask)) or advantages.shape != (logits.shape[0],):
        raise ValueError("token/answer shape mismatch")
    if targets.dtype != torch.long or torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ValueError("targets must be vocabulary IDs")
    if mask.dtype != torch.bool or not bool(mask.any(dim=-1).all()):
        raise ValueError("binary bool mask and at least one token per answer required")
    if not bool(torch.isfinite(old_logps).all() & torch.isfinite(advantages).all()):
        raise ValueError("old logps/advantages must be finite")
    selected = logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ratio = (selected - old_logps.detach()).exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("probability ratio overflow: policy drift exceeds this dtype range")
    advantage = advantages.detach()[:, None]
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    token_loss = -torch.minimum(unclipped, clipped) + beta * categorical_kl(logits, reference_logits)
    if not bool(torch.isfinite(token_loss).all()):
        raise ValueError("GRPO token loss exceeds finite numerical range")
    loss = ((token_loss * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("GRPO reduction exceeds finite numerical range")
    return loss


def sample_rollouts(logits, group_size=8, seed=22):
    """Sample WITH replacement from the old policy, temperature=1, no top-k."""
    torch = _torch()
    if logits.shape != (len(PROMPTS), len(ANSWERS)) or type(group_size) is not int or group_size < 2:
        raise ValueError("expected [2,3] logits and G>=2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions = torch.multinomial(logits.detach().cpu().softmax(-1), group_size,
                               replacement=True, generator=generator)
    return [Rollout(f"{p}:{i}", p, int(action), ANSWERS[int(action)])
            for p, row in enumerate(actions) for i, action in enumerate(row)]


def torch_demo(seed=22):
    """Real CPU SGD, preserving the caller's CPU RNG and thread count.

    The CPU-only RNG scope and finally cleanup also apply if a training step
    raises. No CUDA generator is seeded or initialized for this CPU example.
    """
    torch = _torch()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            # Independent parameters for each prompt: no learned arithmetic/generalization.
            initial = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
            sft_optimizer = torch.optim.SGD([initial], lr=0.4)
            gold_ids = torch.tensor([0, 2])
            for _ in range(4):
                sft_optimizer.zero_grad()
                sft_loss = -initial.log_softmax(-1)[torch.arange(2), gold_ids].mean()
                sft_loss.backward()
                sft_optimizer.step()
            reference = initial.detach().clone()  # fixed SFT anchor, NEVER optimizer-owned
            reference_before = reference.clone()  # independent snapshot before ANY preference updates
            dpo_policy = torch.nn.Parameter(reference.clone())
            grpo_policy = torch.nn.Parameter(reference.clone())
            rows, chosen, rejected = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([1, 1])
            reference_logps = reference.log_softmax(-1)
            dpo_optimizer = torch.optim.SGD([dpo_policy], lr=0.4)
            dpo_start = None
            for _ in range(6):
                dpo_optimizer.zero_grad()
                lp = dpo_policy.log_softmax(-1)
                loss = dpo_loss(lp[rows, chosen], lp[rows, rejected],
                                reference_logps[rows, chosen], reference_logps[rows, rejected], beta=0.5)
                if dpo_start is None:
                    dpo_start = loss.item()
                loss.backward()
                dpo_optimizer.step()
            final_lp = dpo_policy.log_softmax(-1)
            dpo_end = dpo_loss(final_lp[rows, chosen], final_lp[rows, rejected],
                               reference_logps[rows, chosen], reference_logps[rows, rejected], beta=0.5)

            old = grpo_policy.detach().clone()  # one sampled batch, three optimizer steps
            rollouts = sample_rollouts(old, group_size=8, seed=seed)
            rewards, prompt_ids = aligned_rewards(rollouts, score_rollouts(rollouts))
            advantage = torch.tensor(group_advantages(rewards, prompt_ids), dtype=torch.float64)
            prompts = torch.tensor(prompt_ids)
            targets = torch.tensor([r.action_id for r in rollouts])[:, None]
            old_logps = old.log_softmax(-1)[prompts].gather(1, targets).detach()
            mask = torch.ones_like(targets, dtype=torch.bool)
            grpo_optimizer = torch.optim.SGD([grpo_policy], lr=0.2)
            for _ in range(3):
                grpo_optimizer.zero_grad()
                loss = grpo_loss(grpo_policy[prompts, None, :], targets, old_logps,
                                 reference[prompts, None, :], mask, advantage)
                loss.backward()
                grpo_optimizer.step()
            expected_reward = lambda logits: logits.softmax(-1)[rows, chosen].mean().item()
            return {
                "status": "EXECUTED: CPU finite-action toy; NOT TRL or LLM training",
                "torch": torch.__version__, "seed": seed, "parameters": initial.numel(),
                "sft_steps": 4, "dpo_steps": 6, "grpo_steps_one_batch": 3,
                "sft_logits": reference_before.tolist(), "dpo_logits": dpo_policy.detach().tolist(),
                "grpo_logits": grpo_policy.detach().tolist(),
                "sft_probabilities": reference_before.softmax(-1).tolist(),
                "dpo_probabilities": dpo_policy.detach().softmax(-1).tolist(),
                "grpo_probabilities": grpo_policy.detach().softmax(-1).tolist(),
                "dpo_loss_before_after": [dpo_start, dpo_end.item()],
                "exact_kl_to_sft_before": categorical_kl(reference_before, reference_before).tolist(),
                "exact_kl_to_sft_after_dpo": categorical_kl(dpo_policy, reference).detach().tolist(),
                "exact_kl_to_sft_after_grpo": categorical_kl(grpo_policy, reference).detach().tolist(),
                "expected_reward_before_after_dpo_grpo": [expected_reward(reference_before), expected_reward(dpo_policy), expected_reward(grpo_policy)],
                "sampled_action_ids": [r.action_id for r in rollouts], "sampled_rewards": rewards,
                "advantages": advantage.tolist(), "reference_unchanged": torch.equal(reference, reference_before),
            }

    finally:
        torch.set_num_threads(previous_threads)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="run genuine six-parameter CPU training")
    args = parser.parse_args()
    print(json.dumps({"standard_library": hand_demo()}, ensure_ascii=False, indent=2))
    if args.torch:
        try:
            report = torch_demo()
        except RuntimeError as exc:
            if str(exc).startswith("P22 torch SKIP:"):
                print(str(exc))
                return 2
            raise
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
