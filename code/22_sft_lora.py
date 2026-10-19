"""P21: offline teaching chat template, assistant-only SFT and ordinary LoRA.

python code/22_sft_lora.py          # standard-library template / hand arithmetic
python code/22_sft_lora.py --train  # real tiny CPU updates; requires PyTorch
No pretrained checkpoint, HF/PEFT integration, QLoRA, model download or GPU run.
"""
from dataclasses import dataclass
import argparse
import copy
import json
import math
import sys

PAD, BOS, SYSTEM, USER, ASSISTANT, END, EOS, UNK = range(8)
IGNORE = -100
SPECIAL = ("<pad>", "<bos>", "<system>", "<user>", "<assistant>", "<end>", "<eos>", "<unk>")
TOKENS = SPECIAL + tuple("只答颜色问红蓝绿黄")
VOCAB = {token: index for index, token in enumerate(TOKENS)}
ROLE_IDS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
TEMPLATE_VERSION = "p21-char-chat-v1"


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ChatRow:
    input_ids: tuple
    labels: tuple
    assistant_spans: tuple  # [start, stop), answer content AND assistant EOS.
    kept_turns: int
    dropped_turns: int


def _content_ids(text):
    if not isinstance(text, str):
        raise ValueError("message content must be a string")
    # Character encoding never interprets user-supplied spellings as special IDs.
    return [VOCAB.get(char, UNK) for char in text]


def _validate_messages(messages, training):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a nonempty sequence")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each message must have exactly role and content")
        if message["role"] not in ROLE_IDS or not isinstance(message["content"], str):
            raise ValueError("unsupported role or non-string content")
    offset = int(messages[0]["role"] == "system")
    rest = messages[offset:]
    if not rest:
        raise ValueError("at least one user message is required")
    for index, message in enumerate(rest):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError("after optional system, roles must alternate user/assistant")
    if training and len(rest) % 2:
        raise ValueError("training conversation must end with assistant")
    if not training and len(rest) % 2 == 0:
        raise ValueError("generation prompt must end with user")
    return offset


def _message_ids(message):
    role = message["role"]
    terminator = EOS if role == "assistant" else END
    return [ROLE_IDS[role]] + _content_ids(message["content"]) + [terminator]


def encode_chat(messages, max_length=None):
    """Encode completed turns. Truncate only between complete user/assistant pairs.

    A too-long first pair is rejected, not silently converted to a partial answer
    or an artificial EOS. This deliberately conservative policy keeps the system
    prefix and earliest complete pairs; it is not a universal truncation recipe.
    """
    offset = _validate_messages(messages, training=True)
    if max_length is not None:
        _positive_int("max_length", max_length)
    ids = [BOS]
    if offset:
        ids.extend(_message_ids(messages[0]))
    labels = [IGNORE] * len(ids)
    spans = []
    pairs = (len(messages) - offset) // 2
    for turn in range(pairs):
        user, assistant = messages[offset + 2 * turn:offset + 2 * turn + 2]
        user_ids, assistant_ids = _message_ids(user), _message_ids(assistant)
        if max_length is not None and len(ids) + len(user_ids) + len(assistant_ids) > max_length:
            break
        ids.extend(user_ids)
        labels.extend([IGNORE] * len(user_ids))
        start = len(ids) + 1  # The assistant ROLE header is context, not a target.
        ids.extend(assistant_ids)
        labels.extend([IGNORE] + assistant_ids[1:])
        spans.append((start, len(ids)))
    if not spans:
        raise ValueError("max_length leaves no complete supervised turn")
    return ChatRow(tuple(ids), tuple(labels), tuple(spans), len(spans), pairs - len(spans))


def generation_prompt(messages):
    """Same template prefix as training, ending immediately after assistant role."""
    _validate_messages(messages, training=False)
    ids = [BOS]
    for message in messages:
        ids.extend(_message_ids(message))
    ids.append(ASSISTANT)
    return tuple(ids)


def collate(rows, pad_to=None):
    if not rows:
        raise ValueError("cannot collate an empty batch")
    width = max(len(row.input_ids) for row in rows)
    if pad_to is not None:
        _positive_int("pad_to", pad_to)
        if pad_to < width:
            raise ValueError("pad_to cannot truncate a row")
        width = pad_to
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in rows:
        size = len(row.input_ids)
        if size != len(row.labels) or not any(y != IGNORE for y in row.labels[1:]):
            raise ValueError("each row needs aligned labels and at least one causal target")
        result["input_ids"].append(list(row.input_ids) + [PAD] * (width - size))
        result["labels"].append(list(row.labels) + [IGNORE] * (width - size))
        result["attention_mask"].append([1] * size + [0] * (width - size))
    return result


def scalar_causal_loss(logits, labels):
    """Independent standard-library next-token NLL, stable even for large logits."""
    if len(logits) != len(labels):
        raise ValueError("batch dimensions differ")
    total, count = 0.0, 0
    for scores, targets in zip(logits, labels):
        if len(scores) != len(targets):
            raise ValueError("sequence dimensions differ")
        for position in range(1, len(targets)):
            target = targets[position]
            if target == IGNORE:
                continue
            values = scores[position - 1]
            if not 0 <= target < len(values) or not all(math.isfinite(x) for x in values):
                raise ValueError("invalid target or nonfinite logits")
            largest = max(values)
            total += largest + math.log(sum(math.exp(x - largest) for x in values)) - values[target]
            count += 1
    if count == 0:
        raise ValueError("no supervised causal targets")
    return total / count, count


def lora_parameter_count(in_features, out_features, rank):
    for name, value in (("in_features", in_features), ("out_features", out_features), ("rank", rank)):
        _positive_int(name, value)
    return rank * (in_features + out_features)


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = nn = F = None


if torch is not None:
    def masked_causal_loss(logits, labels):
        """Labels are unshifted: only this function shifts once."""
        if logits.ndim != 3 or labels.shape != logits.shape[:2]:
            raise ValueError("expected logits [B,T,V] and labels [B,T]")
        targets = labels[:, 1:]
        count = int((targets != IGNORE).sum())
        if not count:
            raise ValueError("no supervised causal targets")
        total = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                targets.reshape(-1), ignore_index=IGNORE, reduction="sum")
        return total / count


    class LoRALinear(nn.Module):
        """Ordinary alpha/r LoRA with no bias adapter, dropout, or quantization."""
        def __init__(self, base, rank=2, alpha=4.0):
            super().__init__()
            _positive_int("rank", rank)
            if not isinstance(base, nn.Linear):
                raise ValueError("base must be nn.Linear")
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("alpha must be finite and positive")
            self.base = base
            self.base.requires_grad_(False)
            self.rank, self.alpha = rank, alpha
            self.scale = alpha / rank
            self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
            self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

        def merged(self):
            """Return a NEW plain layer; never mutate or double-merge this adapter."""
            result = copy.deepcopy(self.base)
            with torch.no_grad():
                result.weight.add_(self.scale * (self.B @ self.A))
            return result


    class TinyChatLM(nn.Module):
        """One-head causal Transformer, width 16, no dropout, no weight tying."""
        def __init__(self, width=16, max_length=64):
            super().__init__()
            self.width, self.max_length = width, max_length
            self.embedding = nn.Embedding(len(TOKENS), width)
            self.position = nn.Embedding(max_length, width)
            self.norm1 = nn.LayerNorm(width)
            self.q_proj = nn.Linear(width, width, bias=False)
            self.k_proj = nn.Linear(width, width, bias=False)
            self.v_proj = nn.Linear(width, width, bias=False)
            self.o_proj = nn.Linear(width, width, bias=False)
            self.norm2 = nn.LayerNorm(width)
            self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
            self.final_norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, len(TOKENS), bias=False)

        def forward(self, input_ids, attention_mask):
            length = input_ids.size(1)
            if length > self.max_length or attention_mask.shape != input_ids.shape:
                raise ValueError("invalid length or attention mask shape")
            if not bool(attention_mask[:, 0].all()):
                raise ValueError("this teaching model requires nonempty, right-padded rows")
            x = self.embedding(input_ids) + self.position(torch.arange(length, device=input_ids.device))
            h = self.norm1(x)
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.width)
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            allowed = causal.unsqueeze(0) & attention_mask[:, None, :].bool()
            scores = scores.masked_fill(~allowed, float("-inf"))
            x = x + self.o_proj(torch.softmax(scores, dim=-1) @ v)
            x = x + self.ff(self.norm2(x))
            return self.lm_head(self.final_norm(x))


def synthetic_conversations():
    """Four generated tasks; the second turn tests multi-turn masking, not generalization."""
    result = []
    for color in "红蓝绿黄":
        messages = [{"role": "system", "content": "只答颜色"},
                    {"role": "user", "content": "问" + color},
                    {"role": "assistant", "content": color}]
        if color == "红":
            messages += [{"role": "user", "content": "问蓝"},
                         {"role": "assistant", "content": "蓝"}]
        result.append(messages)
    return result


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is missing; run the standard-library demo without --train")


def _tensor_batch(rows):
    _require_torch()
    return {key: torch.tensor(value, dtype=torch.long) for key, value in collate(rows).items()}


def _forward_loss(model, batch):
    return masked_causal_loss(model(batch["input_ids"], batch["attention_mask"]), batch["labels"])


def _updates(model, batch, steps, lr):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise AssertionError("nonfinite training loss")
        loss.backward()
        optimizer.step()


def train_demo(full_steps=24, lora_steps=24, seed=21):
    """Actual CPU SGD. LoRA CONTINUES the same toy task; not a quality comparison."""
    _require_torch()
    _positive_int("full_steps", full_steps)
    _positive_int("lora_steps", lora_steps)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Do not change the caller's RNG stream when used from a test suite.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            rows = [encode_chat(messages) for messages in synthetic_conversations()]
            batch = _tensor_batch(rows)
            model = TinyChatLM()
            total_parameters = sum(p.numel() for p in model.parameters())
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            before = float(_forward_loss(model, batch).detach())
            logits = model(batch["input_ids"], batch["attention_mask"])
            reference, count = scalar_causal_loss(logits.detach().tolist(), batch["labels"].tolist())
            if abs(reference - before) > 1e-6:
                raise AssertionError("independent scalar loss differs")
            _updates(model, batch, full_steps, lr=0.15)
            after = float(_forward_loss(model, batch).detach())
            full_change = max(float((value - initial[key]).abs().max()) for key, value in model.state_dict().items())
            if not after < before or full_change == 0:
                raise AssertionError("full-parameter updates did not learn")
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False)
            model.q_proj = LoRALinear(model.q_proj, rank=2, alpha=4)
            model.v_proj = LoRALinear(model.v_proj, rank=2, alpha=4)
            adapters = {name: p for name, p in model.named_parameters() if p.requires_grad}
            frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
            if set(adapters) != {"q_proj.A", "q_proj.B", "v_proj.A", "v_proj.B"}:
                raise AssertionError("wrong trainable parameter set")
            adapter_before = {name: p.detach().clone() for name, p in adapters.items()}
            lora_before = float(_forward_loss(model, batch).detach())
            if abs(lora_before - after) > 1e-7:
                raise AssertionError("zero-B LoRA changed initial predictions")
            _updates(model, batch, lora_steps, lr=0.5)
            lora_after = float(_forward_loss(model, batch).detach())
            for name, parameter in model.named_parameters():
                if name in frozen and (parameter.grad is not None or not torch.equal(parameter, frozen[name])):
                    raise AssertionError("frozen base changed or received a gradient")
            adapter_change = max(float((p.detach() - adapter_before[name]).abs().max()) for name, p in adapters.items())
            if not lora_after < lora_before or adapter_change == 0:
                raise AssertionError("adapter updates did not learn")
            model.eval()
            merged = copy.deepcopy(model)
            merged.q_proj, merged.v_proj = model.q_proj.merged(), model.v_proj.merged()
            with torch.no_grad():
                expected = model(batch["input_ids"], batch["attention_mask"])
                actual = merged(batch["input_ids"], batch["attention_mask"])
            merge_error = float((expected - actual).abs().max())
            if not torch.allclose(expected, actual, atol=2e-6, rtol=2e-5):
                raise AssertionError("merged model differs from adapter path")
            return {"torch": torch.__version__, "device": "cpu", "seed": seed,
                    "template_version": TEMPLATE_VERSION, "data": "synthetic-only",
                    "batch_shape": list(batch["input_ids"].shape), "supervised_targets": count,
                    "base_parameters": total_parameters, "lora_trainable_parameters": sum(p.numel() for p in adapters.values()),
                    "full_steps": full_steps, "full_loss_before": before, "full_loss_after": after,
                    "full_parameter_max_change": full_change, "scalar_loss_difference": abs(reference - before),
                    "lora_steps": lora_steps, "lora_loss_before": lora_before, "lora_loss_after": lora_after,
                    "adapter_max_change": adapter_change, "frozen_base_unchanged": True,
                    "merged_logits_max_error": merge_error,
                    "heldout_quality_evaluated": False, "hf_peft_qlora_run": False}
    finally:
        torch.set_num_threads(previous_threads)


def hand_demo():
    messages = [{"role": "user", "content": "问红"}, {"role": "assistant", "content": "红"}]
    row = encode_chat(messages)
    return {"template_version": TEMPLATE_VERSION, "tokens": [TOKENS[i] for i in row.input_ids],
            "input_ids": row.input_ids, "labels": row.labels, "assistant_spans": row.assistant_spans,
            "two_target_loss": -(math.log(0.5) + math.log(0.25)) / 2,
            "lora_3_to_2_rank1_parameters": lora_parameter_count(3, 2, 1),
            "training": "NOT RUN; add --train for actual CPU SGD"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="run real CPU full-parameter and LoRA updates")
    args = parser.parse_args(argv)
    print(json.dumps(hand_demo(), ensure_ascii=False, indent=2))
    if args.train:
        if torch is None:
            print("SKIP: PyTorch is missing. Standard-library demo ran; training did NOT run.", file=sys.stderr)
            return 2
        print(json.dumps(train_demo(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
