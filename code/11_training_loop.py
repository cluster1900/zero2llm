"""P09: scalar reference first; optional, pinned PyTorch CPU training second.

python code/11_training_loop.py --scalar-only
python code/11_training_loop.py --steps 40 --seeds 7 7 11
No downloads, no files written, no training or torch import at module import.
"""
import argparse
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
import sys


def torch_eligible(present, installed_version):
    return bool(present and re.fullmatch(r'2\.14\.\d+(?:\+[A-Za-z0-9.]+)?', installed_version))


def require_torch():
    try:
        installed = version('torch')
    except PackageNotFoundError:
        installed = 'not installed'
    if not torch_eligible(importlib.util.find_spec('torch') is not None, installed):
        raise RuntimeError(f'CPU tensor demo requires stable torch 2.14.x; found {installed}')
    import torch
    return torch


def cross_entropy(logits, target):
    """Stable scalar log-sum-exp; input is scores, not probabilities."""
    top = max(logits)
    return top + math.log(sum(math.exp(x - top) for x in logits)) - logits[target]


def hand_calculation():
    # Two positions, two classes. log(3) makes probabilities exactly 3/4, 1/4.
    scores = [[math.log(3), 0.0], [0.0, math.log(3)]]
    targets = [0, 0]
    losses = [cross_entropy(z, y) for z, y in zip(scores, targets)]
    # Separate one-position classifier with two trainable logits, target class 0.
    gradient = [-.25, .25]  # softmax(z) - one_hot(0)
    updated = [scores[0][i] - .1 * gradient[i] for i in range(2)]
    return dict(logits=scores, targets=targets, losses=losses,
                mean_loss=sum(losses) / 2, gradient=gradient,
                updated_logits=updated, updated_loss=cross_entropy(updated, 0))


def shifted_pair(ids, start, length):
    if length < 1 or start < 0 or start + length >= len(ids):
        raise ValueError('Window must contain length+1 tokens within one document')
    return ids[start:start + length], ids[start + 1:start + length + 1]


def assert_shift(ids, start, length, x, y):
    expected_x, expected_y = shifted_pair(ids, start, length)
    if list(x) != expected_x or list(y) != expected_y:
        raise ValueError('Labels must be the next token, never a copy of inputs')


def split_documents(documents):
    """Whole-document split, not a random split of overlapping windows."""
    if len(documents) != 3 or len(set(documents)) != 3:
        raise ValueError('This teaching split requires three distinct documents')
    train, heldout = documents[:2], documents[2:]
    assert_disjoint(train, heldout)
    return train, heldout


def assert_disjoint(train, heldout):
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError('Train and heldout document identities must be disjoint')


def copy_cheat_loss(inputs, labels, vocab_size):
    """No learning: assign score 12 to the input token, 0 to every other token."""
    losses = []
    for token, target in zip(inputs, labels):
        scores = [0.] * vocab_size
        scores[token] = 12.
        losses.append(cross_entropy(scores, target))
    return sum(losses) / len(losses)


def load_baseline():
    spec = importlib.util.spec_from_file_location('p09_baby', Path(__file__).with_name('07_baby_gpt.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)  # backward adds; do not mix batches accidentally
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), tuple(logits.shape)


def evaluate(model, documents, encode):
    """Token-weighted full-document CE; no gradient, no optimizer, restore mode."""
    torch = require_torch()
    previous = model.training
    model.eval()
    total, count = 0., 0
    try:
        with torch.no_grad():
            for document in documents:
                ids = encode(document)
                x = torch.tensor([ids[:-1]], dtype=torch.long)
                y = torch.tensor([ids[1:]], dtype=torch.long)
                _, loss = model(x, y)
                total += float(loss) * y.numel()
                count += y.numel()
    finally:
        model.train(previous)
    return total / count


def run_cpu(seed=7, steps=40):
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    torch = require_torch()
    base = load_baseline()
    # Local RNG keeps batch sampling independent of other callers' Python RNG.
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, heldout = split_documents(base.poems)
    # Deliberately reuse ALL three poems' character vocabulary, not heldout labels.
    model = base.BabyGPT(base.vocab_size, d_model=16, num_heads=4,
                         num_layers=1, max_seq_len=32).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    before = dict(train=evaluate(model, train, base.encode),
                  heldout=evaluate(model, heldout, base.encode))
    first_loss, shape = None, None
    for step in range(steps):
        xs, ys = [], []
        for _ in range(4):
            ids = base.encode(rng.choice(train))
            start = rng.randrange(len(ids) - 8)
            x, y = shifted_pair(ids, start, 8)
            assert_shift(ids, start, 8, x, y)
            xs.append(x)
            ys.append(y)
        loss, shape = train_step(model, optimizer, torch.tensor(xs), torch.tensor(ys))
        if step == 0:
            first_loss = loss
    after = dict(train=evaluate(model, train, base.encode),
                 heldout=evaluate(model, heldout, base.encode))
    return dict(seed=seed, steps=steps, device='cpu', torch=torch.__version__,
                threads=1, batch_size=4, sequence_length=8, first_loss=first_loss,
                logits_shape=shape, before=before, after=after,
                train_documents=2, heldout_documents=1,
                caveat='Full-corpus vocabulary is transductive; one heldout poem is noisy, not general language ability')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar-only', action='store_true')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 7, 11])
    args = parser.parse_args()
    print(json.dumps(hand_calculation(), indent=2))
    print(json.dumps(dict(copy_labels=copy_cheat_loss([0, 1, 2], [0, 1, 2], 4),
                         next_labels=copy_cheat_loss([0, 1, 2], [1, 2, 3], 4))))
    if not args.scalar_only:
        try:
            for seed in args.seeds:
                print(json.dumps(run_cpu(seed, args.steps), ensure_ascii=False))
        except RuntimeError as error:
            parser.exit(2, str(error) + '\nUse --scalar-only for the no-dependency experiment.\n')


if __name__ == '__main__':
    main()
