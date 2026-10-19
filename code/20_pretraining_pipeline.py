"""P19: generated data -> isolated packed documents -> exact CPU resume.

python code/20_pretraining_pipeline.py --data-only  # standard library only
python code/20_pretraining_pipeline.py --steps 12 --interrupt 3
No network, pretrained weights, private data or persistent output by default.
The tiny packed LM reuses S3 RMSNorm/SwiGLU, not its cache/attention interface.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import unicodedata

PAD, BOS, EOS, UNK, IGNORE = 0, 1, 2, 3, -100
# A fixed teaching alphabet, NOT fitted on train/validation/test text.
ALPHABET = "春风月明山水花开日光云雨星河"
VOCAB = {symbol: i + 4 for i, symbol in enumerate(ALPHABET)}
VOCAB_SIZE = 4 + len(VOCAB)
PIPELINE_VERSION = "p19-nfc-whitespace-exact-docpack-v1"


@dataclass(frozen=True)
class Document:
    uid: str
    text: str
    source: str = "generated/nature"
    provenance: str = "generated-for-this-book"


def synthetic_documents():
    """Original tiny strings generated for this lesson; no scraped text or PII."""
    texts = ["春风", "月明山水", "山水", "花开日", "光", "云雨", "星河月明",
             "风春", "明月光", "水山"]
    docs = [Document(f"d{i:02}", text, "generated/nature" if i < 7 else "generated/reversal")
            for i, text in enumerate(texts)]
    return docs + [Document("duplicate", " 春\n风 "), Document("blank", " \n ")]


def clean_text(text):
    """Chinese toy policy: NFC then remove ALL whitespace; not universal cleaning."""
    if not isinstance(text, str):
        raise ValueError("document text must be a string")
    return "".join(unicodedata.normalize("NFC", text).split())


def clean_deduplicate(documents):
    """Global exact canonical-text dedup BEFORE split; first occurrence wins.

    This demo accepts only its declared generated provenance. It is not a license
    verifier or privacy detector and does not remove paraphrases/near duplicates.
    """
    kept, seen, ids = [], {}, set()
    report = {"raw": 0, "empty": [], "duplicates": []}
    for doc in documents:
        report["raw"] += 1
        if not isinstance(doc, Document) or doc.provenance != "generated-for-this-book":
            raise ValueError("demo accepts only declared generated-for-this-book data")
        if not doc.uid or doc.uid in ids:
            raise ValueError("document IDs must be nonempty and unique")
        ids.add(doc.uid)
        text = clean_text(doc.text)
        if not text:
            report["empty"].append(doc.uid)
        elif text in seen:
            report["duplicates"].append([doc.uid, seen[text]])
        else:
            seen[text] = doc.uid
            kept.append(Document(doc.uid, text, doc.source, doc.provenance))
    report["kept"] = len(kept)
    return kept, report


def split_documents(documents, seed=19):
    """Tiny 60/20/remainder split, whole documents, local deterministic shuffle."""
    docs = list(documents)
    if len(docs) < 5 or any(doc.text != clean_text(doc.text) or not doc.text for doc in docs):
        raise ValueError("need at least five nonempty cleaned documents")
    if len({doc.text for doc in docs}) != len(docs) or len({doc.uid for doc in docs}) != len(docs):
        raise ValueError("deduplicate before splitting")
    random.Random(seed).shuffle(docs)
    train_end, valid_end = 3 * len(docs) // 5, 4 * len(docs) // 5
    result = dict(train=docs[:train_end], validation=docs[train_end:valid_end], test=docs[valid_end:])
    assert_no_contamination(result)
    return result


def assert_no_contamination(splits):
    """Exact clean-text and ID checks only; NOT semantic/broad benchmark decontamination."""
    seen_text, seen_ids = set(), set()
    for docs in splits.values():
        for doc in docs:
            text = clean_text(doc.text)
            if text in seen_text or doc.uid in seen_ids:
                raise ValueError("duplicate document identity/content across or within splits")
            seen_text.add(text)
            seen_ids.add(doc.uid)


def encode(text):
    return [VOCAB.get(char, UNK) for char in text]


def attention_mask(segments):
    """True means allowed: same non-padding document AND causal j <= i."""
    return [[sid >= 0 and sid == other and j <= i for j, other in enumerate(segments)]
            for i, sid in enumerate(segments)]


def pack_documents(documents, block_size=12):
    """Whole-document greedy packing; reject long documents, never silently drop.

    Keep EOS as an input with ignored target. Each BOS predicts its own first
    token, last text token predicts EOS; EOS never predicts the next document.
    Position IDs restart at BOS. Empty input makes one all-padding row for tests.
    """
    if type(block_size) is not int or block_size < 3:
        raise ValueError("block_size must be an integer >= 3")
    rows, current = [], dict(inputs=[], labels=[], segments=[], positions=[], document_ids=[])

    def flush():
        nonlocal current
        count = block_size - len(current["inputs"])
        for key, pad in (("inputs", PAD), ("labels", IGNORE), ("segments", -1), ("positions", 0)):
            current[key] += [pad] * count
        current["allowed"] = attention_mask(current["segments"])
        rows.append(current)
        current = dict(inputs=[], labels=[], segments=[], positions=[], document_ids=[])

    for sid, doc in enumerate(documents):
        if not doc.text or doc.text != clean_text(doc.text):
            raise ValueError("packing expects nonempty cleaned text")
        tokens = [BOS] + encode(doc.text) + [EOS]
        if len(tokens) > block_size:
            raise ValueError(f"document {doc.uid} exceeds block_size; explicit chunking policy required")
        if len(current["inputs"]) + len(tokens) > block_size:
            flush()
        current["inputs"] += tokens
        current["labels"] += tokens[1:] + [IGNORE]
        current["segments"] += [sid] * len(tokens)
        current["positions"] += list(range(len(tokens)))
        current["document_ids"].append(doc.uid)
    if current["inputs"] or not rows:
        flush()
    return rows


def prepare_data(documents=None, block_size=12, seed=19):
    raw = synthetic_documents() if documents is None else list(documents)
    kept, report = clean_deduplicate(raw)
    splits = split_documents(kept, seed)
    rows = {name: pack_documents(docs, block_size) for name, docs in splits.items()}
    manifest = dict(pipeline=PIPELINE_VERSION, vocabulary=VOCAB, special=[PAD, BOS, EOS, UNK, IGNORE],
                    block_size=block_size, split_seed=seed, raw=[asdict(doc) for doc in raw],
                    splits={name: [asdict(doc) for doc in docs] for name, docs in splits.items()},
                    packed=rows)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return dict(splits=splits, rows=rows, cleaning=report, manifest=manifest,
                data_version=hashlib.sha256(payload.encode()).hexdigest())


def data_summary(data):
    result = dict(data_version=data["data_version"], cleaning=data["cleaning"], splits={})
    for name, docs in data["splits"].items():
        rows = data["rows"][name]
        counts = {}
        for doc in docs:
            counts[doc.source] = counts.get(doc.source, 0) + len(encode(doc.text)) + 1
        total = sum(counts.values())
        result["splits"][name] = dict(documents=[doc.uid for doc in docs], blocks=len(rows),
            valid_targets=total, input_slots=len(rows) * data["manifest"]["block_size"],
            source_target_tokens=counts,
            source_token_fractions={key: value / total for key, value in counts.items()})
    return result


def scalar_loss(logits, labels):
    """Independent stable scalar CE; sum/count/mean. Zero count has no mean."""
    if len(logits) != len(labels):
        raise ValueError("one label per logit row required")
    losses = []
    for scores, label in zip(logits, labels):
        if label == IGNORE:
            continue
        if not scores or type(label) is not int or not 0 <= label < len(scores):
            raise ValueError("invalid class index")
        top = max(scores)
        losses.append(top + math.log(sum(math.exp(z - top) for z in scores)) - scores[label])
    total, count = math.fsum(losses), len(losses)
    return dict(loss_sum=total, count=count, mean=total / count if count else None)


def hand_table():
    """Two docs 春风 / 月, block=8; full 18-class logits with chosen p(target)."""
    row = pack_documents([Document("a", "春风"), Document("b", "月")], 8)[0]
    probabilities = [.5, .25, .5, None, .125, .25, None, None]
    scores = []
    for target, probability in zip(row["labels"], probabilities):
        logits = [0.] * VOCAB_SIZE
        if target != IGNORE:
            # exp(z_target)/(exp(z_target)+V-1) = probability.
            logits[target] = math.log(probability * (VOCAB_SIZE - 1) / (1 - probability))
        scores.append(logits)
    return dict(row=row, logits=scores, probabilities=probabilities,
                loss=scalar_loss(scores, row["labels"]))


def require_torch():
    try:
        installed = version("torch")
    except PackageNotFoundError:
        installed = "not installed"
    if not re.fullmatch(r"2\.14\.\d+(?:\+[A-Za-z0-9.]+)?", installed):
        raise RuntimeError(f"P19 CPU training requires stable torch 2.14.x; found {installed}. "
                           "Install requirements.txt, or use --data-only.")
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError("P19 CPU training requires PyTorch; use --data-only or install requirements.txt.") from None
        raise
    return torch


def tensor_attention(q, k, v, allowed):
    """[B,T,D] single-head attention; all-forbidden rows return exact zero."""
    torch = require_torch()
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    visible = allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    safe_scores = torch.where(visible, scores, torch.zeros_like(scores))
    weights = safe_scores.softmax(dim=-1) * allowed.to(scores.dtype)
    return weights @ v


def tensor_loss(logits, labels):
    """Caller supplies ALREADY shifted labels; do not shift a second time."""
    torch = require_torch()
    count = int((labels != IGNORE).sum())
    if not count:
        # Differentiable zero, but callers must skip optimizer.step (AdamW state!).
        return logits.sum() * 0., 0
    total = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1), ignore_index=IGNORE, reduction="sum")
    return total, count


def make_model(block_size, width=12, dropout=.2):
    """Small inspection model, not the S3 cache API or a new mainline architecture."""
    torch = require_torch()
    name = "zero2llm_p19_s3"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("14_modern_block.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    s3 = sys.modules[name]

    class PackedLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token = torch.nn.Embedding(VOCAB_SIZE, width, padding_idx=PAD)
            self.position = torch.nn.Embedding(block_size, width)
            self.norm1, self.norm2 = s3.RMSNorm(width), s3.RMSNorm(width)
            self.qkv = torch.nn.Linear(width, 3 * width, bias=False)
            self.out = torch.nn.Linear(width, width, bias=False)
            self.ffn = s3.SwiGLU(width, 2 * width)
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(width, VOCAB_SIZE, bias=False)

        def forward(self, inputs, positions, allowed, real):
            h = self.token(inputs) + self.position(positions)
            q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
            h = h + self.drop(self.out(tensor_attention(q, k, v, allowed)))
            h = h + self.drop(self.ffn(self.norm2(h)))
            return self.head(h) * real.unsqueeze(-1).to(h.dtype)

    return PackedLM().cpu()


def tensor_row(row):
    torch = require_torch()
    return dict(inputs=torch.tensor([row["inputs"]], dtype=torch.long),
                positions=torch.tensor([row["positions"]], dtype=torch.long),
                allowed=torch.tensor([row["allowed"]], dtype=torch.bool),
                real=torch.tensor([[sid >= 0 for sid in row["segments"]]], dtype=torch.bool)), \
           torch.tensor([row["labels"]], dtype=torch.long)


class Trainer:
    """Single CPU process; checkpoint only AFTER a completed optimizer step.

    No dataloader workers, gradients-in-flight, AMP, CUDA or distributed state.
    Local Python shuffle RNG and stored CPU torch RNG isolate concurrent callers.
    """
    def __init__(self, data, seed=7, horizon=12):
        if type(horizon) is not int or horizon < 3:
            raise ValueError("horizon must be an integer >= 3")
        self.torch = torch = require_torch()
        self.data = data
        self.config = dict(seed=seed, horizon=horizon, block_size=data["manifest"]["block_size"],
                           width=12, dropout=.2, peak_lr=.003, warmup=2)
        self.rows = data["rows"]["train"]
        if not self.rows or not any(y != IGNORE for row in self.rows for y in row["labels"]):
            raise ValueError("training needs at least one valid target")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = make_model(self.config["block_size"], self.config["width"], self.config["dropout"])
            self.torch_rng = torch.get_rng_state().clone()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["peak_lr"],
                                           betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.shuffle_rng = random.Random(seed + 1)
        self.order = list(range(len(self.rows)))
        self.shuffle_rng.shuffle(self.order)
        self.cursor = self.epoch = self.step = self.tokens = 0
        self.history = []

    def learning_rate(self):
        s, c = self.step, self.config
        if s < c["warmup"]:
            return c["peak_lr"] * (s + 1) / c["warmup"]
        fraction = (s - c["warmup"]) / max(1, c["horizon"] - c["warmup"] - 1)
        return c["peak_lr"] * (1 - .9 * min(fraction, 1))

    def next_row(self):
        if self.cursor == len(self.order):
            self.shuffle_rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        index = self.order[self.cursor]
        self.cursor += 1
        return index, self.rows[index]

    def train_row(self, row):
        torch = self.torch
        inputs, labels = tensor_row(row)
        # No dropout draw, optimizer momentum/decay, scheduler or counter change.
        if not bool((labels != IGNORE).any()):
            return None
        self.model.train()
        rate = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.torch_rng)
            logits = self.model(**inputs)
            total, count = tensor_loss(logits, labels)
            loss = total / count
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
                raise RuntimeError("nonfinite gradient")
            self.optimizer.step()
            self.torch_rng = torch.get_rng_state().clone()
        self.step += 1
        self.tokens += count
        return dict(step=self.step, loss=float(loss.detach()), count=count, lr=rate)

    def run_steps(self, number):
        if type(number) is not int or number < 0 or self.step + number > self.config["horizon"]:
            raise ValueError("requested steps must fit the fixed schedule horizon")
        for _ in range(number):
            index, row = self.next_row()
            record = self.train_row(row)
            if record is None:
                raise ValueError("prepared training rows must have valid targets")
            self.history.append(dict(row=index, **record))
        return copy.deepcopy(self.history)

    def evaluate(self, split="validation"):
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown split")
        previous = self.model.training
        total, count = 0., 0
        self.model.eval()
        try:
            with self.torch.no_grad():
                for row in self.data["rows"][split]:
                    inputs, labels = tensor_row(row)
                    value, size = tensor_loss(self.model(**inputs), labels)
                    total += float(value)
                    count += size
        finally:
            self.model.train(previous)
        return dict(loss_sum=total, count=count, mean=total / count if count else None)

    def state_dict(self):
        # deepcopy is essential: state_dict tensors otherwise alias live model.
        return copy.deepcopy(dict(schema=1, torch_version=str(self.torch.__version__),
            data_version=self.data["data_version"], config=self.config,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            torch_rng=self.torch_rng, shuffle_rng=self.shuffle_rng.getstate(),
            order=self.order, cursor=self.cursor, epoch=self.epoch,
            step=self.step, tokens=self.tokens, history=self.history))

    def load_state_dict(self, saved):
        required = set(self.state_dict())
        if set(saved) != required:
            raise ValueError("checkpoint schema fields differ")
        for key, value in (("schema", 1), ("torch_version", str(self.torch.__version__)),
                           ("data_version", self.data["data_version"]), ("config", self.config)):
            if saved[key] != value:
                raise ValueError(f"checkpoint {key} mismatch")
        if sorted(saved["order"]) != list(range(len(self.rows))) or not 0 <= saved["cursor"] <= len(self.rows):
            raise ValueError("checkpoint data order/cursor invalid")
        if not 0 <= saved["step"] <= self.config["horizon"] or len(saved["history"]) != saved["step"]:
            raise ValueError("checkpoint step/history invalid")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        self.torch_rng = saved["torch_rng"].clone()
        self.shuffle_rng.setstate(saved["shuffle_rng"])
        for key in ("order", "cursor", "epoch", "step", "tokens", "history"):
            setattr(self, key, copy.deepcopy(saved[key]))

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        self.torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    def load(self, path):
        # Accept our local tensor/primitive checkpoint, never arbitrary pickled models.
        self.load_state_dict(self.torch.load(path, map_location="cpu", weights_only=True))


def exact_equal(left, right):
    """Recursive state equality including optimizer tensors, not only weights."""
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "shape"):
        torch = require_torch()
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    return left == right


def resume_experiment(data=None, steps=12, interrupt=3):
    if type(interrupt) is not int or not 0 < interrupt < steps:
        raise ValueError("interrupt must be between zero and total steps")
    data = prepare_data() if data is None else data
    torch = require_torch()
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        full = Trainer(data, horizon=steps)
        before = full.evaluate()
        full.run_steps(steps)
        partial = Trainer(data, horizon=steps)
        partial.run_steps(interrupt)
        with tempfile.TemporaryDirectory(prefix="p19-resume-") as directory:
            path = Path(directory) / "checkpoint.pt"
            partial.save(path)
            resumed = Trainer(data, horizon=steps)
            resumed.load(path)
            resumed.run_steps(steps - interrupt)
        matched = exact_equal(full.state_dict(), resumed.state_dict())
        if not matched:
            raise AssertionError("interrupted run differs from uninterrupted run")
        return dict(torch=str(torch.__version__), device="cpu", dtype="float32", threads=1,
                    seed=7, total_steps=steps, interrupt_after=interrupt, exact_resume=matched,
                    tokens=full.tokens, validation_before=before, validation_after=full.evaluate(),
                    history=full.history, test_evaluated=False)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--interrupt", type=int, default=3)
    args = parser.parse_args(argv)
    data = prepare_data()
    print(json.dumps(data_summary(data), ensure_ascii=False, indent=2))
    print("hand_table=" + json.dumps(hand_table(), ensure_ascii=False))
    if args.data_only:
        return 0
    try:
        result = resume_experiment(data, args.steps, args.interrupt)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("resume=" + json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
