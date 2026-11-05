"""B11 G6 rulings (08i–08l, bot-hub#123): exact pins for every edit on #60."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, new text): each must appear exactly once.
PINS = [('chapters/08i_pretraining_pipeline.md', '混合精度是让适合的算子用较低精度、敏感运算保持较高精度，不是把所有张量都改成半精度（[PyTorch 2.14 AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html)）；本章的 CPU float32 实验没有使用 AMP。'), ('chapters/08i_pretraining_pipeline.md', '在附录 H 所列环境下（单线程、float32、seed=7）的本书实测，数据版本指纹也列在附录 H：'), ('chapters/10_appendix.md', '| 08i | 来源核验日 | 2026-10-19 |\n| 08i | 数据版本指纹（SHA-256） | `1b1a8d64d76a38761d8166a145550efdab89a852874994c19be8333947c75705` |\n'), ('chapters/08i_pretraining_pipeline.md', '| 最终测试集 | 默认未评分 | 不用测试分数指导本演示 |\n\n2,040 个参数可以这样复算（单个 block，词表 18、块长 12、宽度 12、SwiGLU 中间维 24）：token embedding 18×12=216，位置 embedding 12×12=144，两个 RMSNorm 2×12=24，QKV 12×36=432，输出投影 12×12=144，SwiGLU 三个投影 3×12×24=864，输出头 12×18=216，合计 2,040。\n'), ('chapters/08j_training_frameworks.md', '## P20.10 多 GPU 选修：本章未运行'), ('chapters/08j_training_frameworks.md', '本章没有安装 DeepSpeed / Megatron，也没有多 GPU 运行、通信量、显存峰值或加速比的实测。想动手时，可从固定版 [Megatron-LM core_v0.12.0](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.0) 的 examples 开始，例如 [run_simple_mcore_train_loop.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/examples/run_simple_mcore_train_loop.py)（TP=2，需恰好两张可见 CUDA GPU，本书未执行）；不能把本章强制 use_cpu=True 的 --hf 脚本塞进多卡启动器冒充多卡训练。动手时记录代码版本、硬件与完整启动配置，先用最小模型检查目标与梯度；对比 DDP 与 ZeRO-2 时只改变状态分片，并真正重启恢复一次 checkpoint。'), ('chapters/08j_training_frameworks.md', '4. 例如错移 CE 标签，会被独立 CE 参考发现；本章测试实际执行的语义变异清单见 `docs/testing.md`。'), ('docs/testing.md', '## 08j_training_frameworks.md\n\ntests/test_training_frameworks.py 实际执行八种语义变异：丢掉分片余数、提前切分梯度、提前切分参数、漏算 master weight、错移 CE 标签、恢复首位 BOS 计数、取消 optimizer.step、损坏恢复后的权重。前四种被独立手算账本发现；错移标签被独立 CE 参考识别；后三种被真实 Trainer 更新、非零变化和恢复 logits 断言识别。\n\n'), ('chapters/08k_sft_lora.md', '\\mathcal L=-\\frac{\\ln 0.5+\\ln 0.25}{2}\n=\\frac{(-\\ln 0.5)+(-\\ln 0.25)}{2}\n=\\frac{0.69314718'), ('chapters/08k_sft_lora.md', '\n第二阶段只再降约 0.0008：它从第一阶段终点 0.61146003 起步，只训练 q/v 投影上的 128 个 LoRA 参数（B 初始为零），参数最大变化约 0.036，远小于第一阶段的约 0.766；本章没有另做实验区分这几个因素各自的作用。\n\n训练 loss 的下降证明这个目标'), ('chapters/08k_sft_lora.md', '[TRL 的 SFTTrainer 源码](https://github.com/huggingface/trl/blob/v0.18.2/trl/trainer/sft_trainer.py) 展示了成熟训练入口如何处理数据和训练配置。本章只引用它作框架定位，不运行；固定版本及 P20–P22 为何引用不同 TRL 版本，见附录 H。'), ('chapters/10_appendix.md', '| 08k | 来源核验日期 | 2026-10-19 |\n| 08k | TRL 源码快照（只读，未执行） | 0.18.2 SFTTrainer；P20 引 0.17.0 官方总览，P22 引 0.16.1 GRPO 源码，分别固定资料快照，不是同一套环境，未作跨版本兼容验证 |\n'), ('chapters/08k_sft_lora.md', '矩阵 oracle 直接检查手算答案。'), ('chapters/08k_sft_lora.md', '手算一个 LoRA 层，并检查基础权重保持不变。'), ('chapters/08k_sft_lora.md', '手写模板称为 `apply_chat_template` 的实际运行结果。\n'), ('chapters/08k_sft_lora.md', '的普通低秩更新及 alpha/r 缩放形式。'), ('chapters/08l_preference_optimization.md', '这篇论文已在 [P04 3.3a](#p04-softmax-collapse) 介绍；其 §3 讨论软目标交叉熵梯度 $q-p$ 的抵消与修法，本书不展开。'), ('chapters/08l_preference_optimization.md', '本书不展开。\n\n## 本章小结\n\n本章可以带走的一句话是：**偏好、奖励、策略、参考和评估是不同角色，先把每个数字属于谁说清楚，再让优化器动手。** 下一章会把这一框架接到推理预算、MTP 和投机解码，并继续区分训练目标与推理时计算。\n'), ('chapters/08l_preference_optimization.md', '本节核对 TRL 的 [DPOConfig 源码](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/dpo_config.py) 与 [GRPOConfig 源码](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/grpo_config.py)，只读未运行；固定版本、核验日期，以及 P20–P22 为何引用不同 TRL 版本，见附录 H。'), ('chapters/10_appendix.md', '| 08l | TRL 源码核验日期 | 2026-10-19 |\n| 08l | TRL 源码快照（只读，未执行） | 0.16.1 DPOConfig / GRPOConfig / GRPOTrainer，作为固定的历史 GRPO 实现切片，不是推荐退回旧环境 |\n'), ('chapters/08l_preference_optimization.md', '| B | 0.2 / 0.5 | 0.1 / 0.5 | ln 2 ≈ 0.693147 | 0.346574 | 0.534800 |\n\n代码输入的是对数概率：A 为 ln 0.6≈−0.510826 / ln 0.2≈−1.609438，参考 ln 0.4≈−0.916291 / −0.916291；B 为 ln 0.2≈−1.609438 / ln 0.5≈−0.693147，参考 ln 0.1≈−2.302585 / −0.693147。相对 margin 是两组差之差，例如 A：(−0.510826+1.609438)−0=1.098612。\n')]

# (file, removed text): must appear zero times.
GONE = [('chapters/08i_pretraining_pipeline.md', '### 混合精度不等于'), ('chapters/08i_pretraining_pipeline.md', '1b1a8d64d76a38761d8166a145550efdab89a852874994c19be8333947c75705'), ('chapters/08i_pretraining_pipeline.md', 'GradScaler，其状态'), ('chapters/08i_pretraining_pipeline.md', '`model.half()`'), ('chapters/08j_training_frameworks.md', 'YOUR_GPU_TRAIN.py'), ('chapters/08j_training_frameworks.md', 'train_micro_batch_size_per_gpu'), ('chapters/08j_training_frameworks.md', '动手前把下列内容保存成自己的实验单'), ('chapters/08k_sft_lora.md', 'DoRA'), ('chapters/08k_sft_lora.md', 'rsLoRA'), ('chapters/08k_sft_lora.md', '偷偷更新'), ('chapters/08k_sft_lora.md', '暗中改写'), ('chapters/08k_sft_lora.md', '更不把其他版本的参数名'), ('chapters/08k_sft_lora.md', '版本阅读说明'), ('chapters/08l_preference_optimization.md', 'EXECUTED'), ('chapters/08l_preference_optimization.md', '`log1p`'), ('chapters/08l_preference_optimization.md', '不是一条逐章升级或降级的安装路线'), ('chapters/08l_preference_optimization.md', '不是对当前最新版的描述'), ('chapters/08l_preference_optimization.md', '为什么相邻章节引用的 TRL 版本不一样')]

CLOSING = '本章可以带走的一句话是：**偏好、奖励、策略、参考和评估是不同角色，先把每个数字属于谁说清楚，再让优化器动手。** 下一章会把这一框架接到推理预算、MTP 和投机解码，并继续区分训练目标与推理时计算。'


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


class B11G6Rulings(unittest.TestCase):
    def test_new_text_once(self):
        for rel, new in PINS:
            with self.subTest(file=rel, new=new[:30]):
                self.assertEqual(read(rel).count(new), 1)

    def test_removed_text_gone(self):
        for rel, old in GONE:
            with self.subTest(file=rel, old=old[:30]):
                self.assertEqual(read(rel).count(old), 0)

    def test_08l_closing_sentence_is_chapter_end(self):
        body = read("chapters/08l_preference_optimization.md")
        self.assertEqual(body.count(CLOSING), 1)
        self.assertTrue(body.rstrip("\n").endswith(CLOSING))

    def test_08i_2040_breakdown_sums(self):
        parts = [18 * 12, 12 * 12, 2 * 12, 12 * 36, 12 * 12, 3 * 12 * 24, 12 * 18]
        self.assertEqual(parts, [216, 144, 24, 432, 144, 864, 216])
        self.assertEqual(sum(parts), 2040)

    def test_08i_2040_matches_code_model(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipped is not passed")
        spec = importlib.util.spec_from_file_location("p19", ROOT / "code/20_pretraining_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.make_model(12)
        shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        self.assertEqual(shapes, {"token.weight": (18, 12), "position.weight": (12, 12), "norm1.weight": (12,),
                                  "norm2.weight": (12,), "qkv.weight": (36, 12), "out.weight": (12, 12),
                                  "ffn.gate.weight": (24, 12), "ffn.up.weight": (24, 12),
                                  "ffn.down.weight": (12, 24), "head.weight": (18, 12)})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2040)


if __name__ == "__main__":
    unittest.main()
