"""P20: exact standard-library ZeRO state ledger + optional real CPU HF smoke.

No model/tokenizer downloads, DeepSpeed installation or GPU benchmark.
python code/21_training_frameworks.py             # standard library only
python code/21_training_frameworks.py --hf        # pinned optional packages
"""
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import argparse
import copy
import json
import math
import os
import sys
import tempfile


FRAMEWORK_PINS = {
    "torch": "2.14.1", "transformers": "4.51.3", "accelerate": "1.6.0",
    "tokenizers": "0.21.4", "huggingface-hub": "0.30.2", "safetensors": "0.5.3",
}


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def shard_sizes(parameters, world_size):
    """Balanced, unpadded ownership; first remainder ranks own one extra item."""
    _integer("parameters", parameters)
    _integer("world_size", world_size, 1)
    quotient, remainder = divmod(parameters, world_size)
    return tuple(quotient + (rank < remainder) for rank in range(world_size))


@dataclass(frozen=True)
class MemorySpec:
    parameters: int = 10
    world_size: int = 4
    parameter_bytes: int = 2
    gradient_bytes: int = 2
    moment_bytes: int = 4  # EACH of Adam's m and v, not both combined.
    master_bytes: int = 4  # Set 0 if the chosen optimizer has no master copy.

    def __post_init__(self):
        _integer("parameters", self.parameters, 1)
        _integer("world_size", self.world_size, 1)
        for name in ("parameter_bytes", "gradient_bytes", "moment_bytes"):
            _integer(name, getattr(self, name), 1)
        _integer("master_bytes", self.master_bytes)


@dataclass(frozen=True)
class RankState:
    rank: int
    owned_elements: int
    parameters: int
    gradients: int
    first_moment: int
    second_moment: int
    master_weights: int

    @property
    def optimizer(self):
        return self.first_moment + self.second_moment + self.master_weights

    @property
    def total(self):
        return self.parameters + self.gradients + self.optimizer


def zero_ledger(spec, stage):
    """Bytes of persistent model state, not peak GPU allocation.

    stage=0 represents replicated DDP state. Stages 1/2/3 additionally shard
    optimizer / gradients / parameters. Excludes padding, activations, buffers,
    communication and transient all-gathers, allocator metadata and offloading.
    """
    _integer("stage", stage)
    if stage > 3:
        raise ValueError("stage must be 0, 1, 2 or 3")
    result = []
    for rank, owned in enumerate(shard_sizes(spec.parameters, spec.world_size)):
        p_count = owned if stage >= 3 else spec.parameters
        g_count = owned if stage >= 2 else spec.parameters
        o_count = owned if stage >= 1 else spec.parameters
        result.append(RankState(
            rank, owned, p_count * spec.parameter_bytes,
            g_count * spec.gradient_bytes, o_count * spec.moment_bytes,
            o_count * spec.moment_bytes, o_count * spec.master_bytes,
        ))
    return tuple(result)


def byte_units(value):
    _integer("bytes", value)
    return {"bytes": value, "GB": value / 10**9, "GiB": value / 2**30}


def ledger_self_test():
    assert shard_sizes(10, 4) == (3, 3, 2, 2)
    assert shard_sizes(2, 4) == (1, 1, 0, 0)
    spec = MemorySpec()
    expected = [(160, 160, 160, 160), (76, 76, 64, 64),
                (62, 62, 48, 48), (48, 48, 32, 32)]
    for stage, totals in enumerate(expected):
        assert tuple(row.total for row in zero_ledger(spec, stage)) == totals
    assert sum(row.total for row in zero_ledger(spec, 3)) == 160
    assert byte_units(2**30)["GiB"] == 1
    assert byte_units(10**9)["GB"] == 1


def framework_status():
    """Metadata-only check: importing this file needs no third-party package.

    Exact optional pins are intentionally conservative: an unverified version
    is SKIPPED, never silently reported as an executed compatibility check.
    """
    versions, issues = {}, []
    for package, expected in FRAMEWORK_PINS.items():
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed"
        versions[package] = version
        normalized = version.split("+")[0] if package == "torch" else version
        if normalized != expected:
            issues.append(f"{package}={version} (need {expected})")
    reason = ("; ".join(issues) + ". Install requirements.txt and "
              "requirements-frameworks.txt in a separate environment; skipped is not passed.") if issues else ""
    return not issues, reason, versions


def _imports():
    ready, reason, _ = framework_status()
    if not ready:
        raise RuntimeError("P20 HF SKIP: " + reason)
    # There is no Hub access, tracking integration, model download or remote code.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
                              Trainer, TrainingArguments, default_data_collator)
    return (torch, Tokenizer, models, pre_tokenizers, processors, GPT2Config,
            GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments,
            default_data_collator)


def make_fixture(seed=20):
    """Local WordLevel tokenizer + RANDOM tiny GPT-2, NOT pretrained GPT-2 BPE."""
    (torch, Tokenizer, models, pre_tokenizers, processors, GPT2Config,
     GPT2LMHeadModel, PreTrainedTokenizerFast, *_rest) = _imports()
    torch.manual_seed(seed)
    vocabulary = {token: i for i, token in enumerate(
        ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "我", "爱", "学习", "模型", "预测", "下一个", "词", "。"])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]", model_max_length=16,
        model_input_names=["input_ids", "attention_mask"], padding_side="right")
    batch = tokenizer(["我 爱 学习 。", "模型 预测 词"], padding=True, return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100
    # v4.51.3 Trainer counts unshifted valid labels for num_items_in_batch.
    # Position zero has no previous-token prediction, so mask it explicitly.
    batch["labels"][:, 0] = -100
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=16, n_ctx=16, n_embd=16,
        n_layer=1, n_head=2, n_inner=32, resid_pdrop=0.0, embd_pdrop=0.0,
        attn_pdrop=0.0, bos_token_id=2, eos_token_id=3, pad_token_id=0,
        use_cache=False, attn_implementation="eager")
    model = GPT2LMHeadModel(config).cpu().float()
    return tokenizer, model, dict(batch)


def shifted_loss_reference(logits, labels):
    """Independent differentiable CE: logsumexp minus target logit, one shift.

    Model-internal loss may shift labels itself; never pass this shifted tensor
    into model(labels=...). Only non-ignored next-token targets enter the mean.
    """
    import torch
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")
    targets = labels[:, 1:]
    scores = logits[:, :-1, :]
    valid = targets != -100
    if not bool(valid.any()):
        raise ValueError("at least one supervised next-token target is required")
    selected = targets[valid]
    if bool(((selected < 0) | (selected >= logits.shape[-1])).any()):
        raise ValueError("label outside vocabulary")
    rows = scores[valid]
    return (torch.logsumexp(rows, dim=-1) - rows.gather(1, selected[:, None]).squeeze(1)).mean()


def _reference_steps(model, batch, steps, learning_rate):
    import torch
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = shifted_loss_reference(logits, batch["labels"])
        loss.backward()
        optimizer.step()


def _hf_smoke_in(directory, steps):
    (torch, *_unused, Trainer, TrainingArguments, collator) = _imports()
    tokenizer, model, batch = make_fixture()
    reference = copy.deepcopy(model)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.eval()
    with torch.no_grad():
        before = model(**batch)
        expected_loss = shifted_loss_reference(before.logits, batch["labels"])
    torch.testing.assert_close(before.loss, expected_loss, rtol=1e-6, atol=1e-6)
    assert tuple(before.logits.shape) == (2, 6, 12), "wrong [B,T,V]"
    assert int((batch["labels"][:, 1:] != -100).sum()) == 9
    dataset = [{key: value[row].clone() for key, value in batch.items()} for row in range(2)]
    # One whole batch per step; dropout/clipping/weight-decay/scheduler disabled.
    # This makes a hand-written SGD loop a meaningful numerical oracle.
    arguments = TrainingArguments(
        output_dir=str(directory / "trainer"), use_cpu=True, max_steps=steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=0.02, lr_scheduler_type="constant", optim="sgd",
        weight_decay=0.0, max_grad_norm=0.0, seed=20, data_seed=20,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        report_to=[], disable_tqdm=True, logging_strategy="no",
        save_strategy="steps", save_steps=steps, save_total_limit=1,
        save_only_model=False, push_to_hub=False,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset,
                      data_collator=collator, processing_class=tokenizer)
    trained = trainer.train()
    _reference_steps(reference, batch, steps, 0.02)
    assert trainer.state.global_step == steps, "Trainer did not execute requested updates"
    parameter_error = max((value - dict(reference.named_parameters())[name]).abs().max().item()
                          for name, value in model.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, dict(reference.named_parameters())[name], atol=2e-6, rtol=2e-5)
    delta = max((value.detach() - initial[name]).abs().max().item() for name, value in model.named_parameters())
    assert math.isfinite(delta) and delta > 1e-7, "parameters did not update"
    model.eval()
    with torch.no_grad():
        after = model(**batch)
    assert math.isfinite(after.loss.item()) and after.loss.item() < before.loss.item(), "fixture loss did not fall"
    checkpoint = directory / "trainer" / f"checkpoint-{steps}"
    expected_files = ("model.safetensors", "config.json", "optimizer.pt", "scheduler.pt",
                      "trainer_state.json", "rng_state.pth", "training_args.bin")
    for filename in expected_files:
        assert (checkpoint / filename).is_file(), f"missing training state: {filename}"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == steps
    # Separately demonstrate a portable inference export, not training resumption.
    export = directory / "export"
    model.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    (export / "experiment.json").write_text(json.dumps({
        "versions": framework_status()[2], "steps": steps,
        "seed": 20, "precision": "float32", "device": "cpu", "pretrained": False,
        "tokenizer": "local WordLevel fixture", "attention_backend": "eager", "resume_tested": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    restored_model = type(model).from_pretrained(
        export, local_files_only=True, attn_implementation="eager").eval()
    assert restored_model.config._attn_implementation == model.config._attn_implementation == "eager"
    restored_tokenizer = type(tokenizer).from_pretrained(export, local_files_only=True)
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert restored_tokenizer("我 爱 学习 。")["input_ids"] == [2, 4, 5, 6, 11, 3]
    with torch.no_grad():
        restored = restored_model(**batch)
    torch.testing.assert_close(after.logits, restored.logits, atol=0, rtol=0)
    torch.testing.assert_close(after.loss, restored.loss, atol=0, rtol=0)
    return {
        "status": "PASS", "device": "cpu", "dtype": "float32", "seed": 20,
        "versions": framework_status()[2], "shape": list(before.logits.shape),
        "supervised_targets": 9, "steps": trainer.state.global_step,
        "loss_before": before.loss.item(), "loss_after": after.loss.item(),
        "reference_loss_error": abs(before.loss.item() - expected_loss.item()),
        "reference_parameter_max_error": parameter_error, "parameter_max_delta": delta,
        "reload_logits_max_error": (after.logits - restored.logits).abs().max().item(),
        "training_loss": trained.training_loss, "checkpoint_files": list(expected_files),
        "resume_tested": False,
    }


def hf_smoke(output_dir=None, steps=3):
    """Run a real Trainer on CPU, compare with independent SGD, save/reload.

    A user-supplied output directory must be absent or empty. No overwrites.
    Otherwise all outputs live in a TemporaryDirectory and are cleaned up.
    """
    _integer("steps", steps, 1)
    ready, reason, _ = framework_status()
    if not ready:
        raise RuntimeError("P20 HF SKIP: " + reason)
    import torch
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if output_dir is None:
            with tempfile.TemporaryDirectory(prefix="zero2llm-p20-") as name:
                return _hf_smoke_in(Path(name), steps)
        directory = Path(output_dir)
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("output directory must be absent or empty; refusing to overwrite")
        directory.mkdir(parents=True, exist_ok=True)
        return _hf_smoke_in(directory, steps)
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", action="store_true", help="run the optional real CPU Trainer smoke")
    parser.add_argument("--output-dir", type=Path, help="keep new local training checkpoint + export")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output_dir is not None and not args.hf:
        parser.error("--output-dir requires --hf")
    if args.steps < 1:
        parser.error("--steps must be positive")
    ledger_self_test()
    print("P20 ZeRO 持久模型状态账本 | 标准库 | 非真实 GPU 峰值")
    print("P=10, D=4; 权重/梯度=2 bytes，Adam m/v/master 各=4 bytes")
    print("各 rank 拥有元素:", shard_sizes(10, 4))
    for stage in range(4):
        rows = zero_ledger(MemorySpec(), stage)
        print(f"stage {stage}: rank bytes={[row.total for row in rows]}, "
              f"peak={max(row.total for row in rows)}, aggregate={sum(row.total for row in rows)}")
    print("ledger self-test: PASS; 排除激活、缓冲、通信、临时聚合及分片对齐填充")
    if not args.hf:
        print("HF smoke: NOT RUN（使用 --hf；本行不是 HF 通过）")
        return 0
    ready, reason, _ = framework_status()
    if not ready:
        print("HF smoke: SKIP — " + reason, file=sys.stderr)
        return 2
    result = hf_smoke(args.output_dir, args.steps)
    print("HF smoke:", json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
