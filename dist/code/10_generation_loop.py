"""P08: standard-library generation trace. Scripted logits, not a trained model."""
import math
import random

VOCAB = ["明", "月", "光", "<EOS>"]
MASK_SCORES = [
    [2.15, 1.80, 0.95, 0.40],
    [1.30, 3.20, 2.10, 1.05],
    [0.85, 1.95, 2.80, 1.70],
    [0.40, 1.10, 2.30, 3.10],
]


def shifted_pairs(tokens):
    """Full sequence including BOS/EOS -> aligned decoder inputs and next-token labels."""
    if len(tokens) < 2:
        raise ValueError("need at least two tokens")
    return list(tokens[:-1]), list(tokens[1:])


def probabilities(logits, temperature=1.0):
    """Finite logits, positive finite temperature. Subtract max before division."""
    if not logits or not all(math.isfinite(v) for v in logits):
        raise ValueError("logits must be a nonempty finite list")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    maximum = max(logits)
    exps = [math.exp((v - maximum) / temperature) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def choose_token(logits, temperature=0.0, rng=None):
    """temperature=0: greedy (first maximum); positive: categorical sampling."""
    if temperature == 0:
        probabilities(logits)  # validate even in greedy mode
        return max(range(len(logits)), key=logits.__getitem__)
    probs = probabilities(logits, temperature)
    rng = rng if rng is not None else random.Random()
    draw = rng.random()
    running = 0.0
    for i, p in enumerate(probs):
        running += p
        if draw < running:
            return i
    return len(probs) - 1  # final rounding guard


def generate(logits_fn, prefix, vocab, eos, max_new_tokens, temperature=0.0, seed=42):
    """Toy full-prefix recomputation; no KV cache, batching, or trained model.

    Return full tokens (EOS retained), per-step trace, reason. max_new_tokens
    counts only newly appended tokens, including EOS, not prompt length.
    """
    if not prefix or not vocab or len(set(vocab)) != len(vocab) or eos not in vocab:
        raise ValueError("nonempty prefix, unique vocab containing EOS required")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    tokens, trace, rng = list(prefix), [], random.Random(seed)
    if tokens[-1] == eos:
        return tokens, trace, "eos"
    for step in range(max_new_tokens):
        logits = list(logits_fn(list(tokens)))
        if len(logits) != len(vocab):
            raise ValueError("one logit per vocabulary token required")
        chosen = vocab[choose_token(logits, temperature, rng)]
        trace.append({"step": step + 1, "prefix": list(tokens),
                      "probs_at_t1": probabilities(logits), "chosen": chosen})
        tokens.append(chosen)
        if chosen == eos:
            return tokens, trace, "eos"
    return tokens, trace, "max_new_tokens"


def scripted_logits(prefix):
    """Fixed finite-state teaching oracle: the prefix length picks a known row."""
    step = len(prefix) - 2
    rows = [[4., 1., 0., -2.], [0., 4., 1., -2.],
            [0., 1., 4., -2.], [0., 0., 0., 4.]]
    return rows[min(max(step, 0), 3)]


def mask_walkthrough():
    """Return raw, actual -inf causal scores, and exact zero future weights."""
    masked, weights = [], []
    for i, row in enumerate(MASK_SCORES):
        masked.append([v if j <= i else -math.inf for j, v in enumerate(row)])
        visible = probabilities(row[:i + 1])
        weights.append(visible + [0.] * (len(row) - i - 1))
    return [row[:] for row in MASK_SCORES], masked, weights


def main():
    x, labels = shifted_pairs(["<BOS>", "我", "爱", "学", "习", "<EOS>"])
    print("decoder input:", x)
    print("labels:", labels)
    for title, rows in zip(["raw", "causal (-inf)", "weights %"], mask_walkthrough()):
        print(title)
        for row in rows:
            print([round(v * 100, 2) for v in row] if title == "weights %" else row)
    print("logits [2,1,0], greedy index:", choose_token([2., 1., 0.]))
    for temp in [.5, 1., 2.]:
        print(f"T={temp}:", [round(v * 100, 2) for v in probabilities([2., 1., 0.], temp)])
    for limit in [8, 2]:
        tokens, trace, reason = generate(scripted_logits, ["床", "前"], VOCAB, "<EOS>", limit)
        for row in trace:
            print(row["step"], row["prefix"],
                  [round(v * 100, 2) for v in row["probs_at_t1"]], row["chosen"])
        print("stop:", reason, "new:", len(trace), "tokens:", tokens)


if __name__ == "__main__":
    main()
