"""
07_baby_gpt.py - 完整可运行的极简微型大模型（Baby-GPT）
适合人群：想亲眼见证一个真正的语言模型从“胡言乱语”训练到“出口成章”的全过程

模型目标：
读一段古诗文本，学会字与字之间的前后因果规律，
并给定开头的提示词（Prompt），能够自动一个字一个字地接龙写下去！
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 趣味训练数据集与分词器（Tokenizer）
# =====================================================================
# 三首诗分开保存。训练时每首都不许接上下一首。
# 如果粘成一整条，模型会偷懒：它只记住“第 i 个座位的下一个字”，
# 提示词“白日”一旦被放到第 0 号座位，它仍会接着背《静夜思》。
poems = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
]
text_corpus = "".join(poems)

# 统计所有出现过的不重复字符（词表）
unique_chars = sorted(list(set(text_corpus)))
vocab_size = len(unique_chars)
char2idx = {ch: i for i, ch in enumerate(unique_chars)}
idx2char = {i: ch for i, ch in enumerate(unique_chars)}

def encode(text):
    """把汉字转换成数字 ID 列表"""
    return [char2idx[c] for c in text]

def decode(indices):
    """把数字 ID 列表还原成汉字"""
    return "".join([idx2char[i] for i in indices])

# =====================================================================
# 2. 微型 GPT 架构（Baby-GPT）
# =====================================================================
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力机制"""
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 注册因果下三角掩码缓冲区（不需要计算梯度）
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time(Seq_len), Channel(d_model)

        # 投影并切分多头
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 打分：(Q · K^T) / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 施加下三角因果掩码：把未来看不到的位置遮挡为负无穷大
        scores = scores.masked_fill(self.mask[:T, :T] == 0, -1e9)
        
        # Softmax 归一化
        weights = F.softmax(scores, dim=-1)
        
        # 加权输出
        out = torch.matmul(weights, V)
        # 拼回原始维度
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class BabyBlock(nn.Module):
    """单个 GPT 积木块：Pre-LN + Causal Attention + Pre-LN + MLP"""
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BabyGPT(nn.Module):
    """完整的自回归生成大语言模型"""
    def __init__(self, vocab_size, d_model=48, num_heads=4, num_layers=2, max_seq_len=128):
        super().__init__()
        self.max_seq_len = max_seq_len
        # 1. 词嵌入与可学习的位置表。
        # 这里的 pos_emb 不是第 5 章那个 sin/cos 公式。
        # 每个座位号对应一行向量，具体数字在训练中自己学。GPT-2 用的就是这种做法。
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # 2. 堆叠多层 Transformer 积木块
        self.blocks = nn.ModuleList([
            BabyBlock(d_model, num_heads, max_seq_len) for _ in range(num_layers)
        ])
        
        # 3. 最终层归一化与分类输出头
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 生成位置索引 [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # 词向量与位置向量直接相加
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # 逐层穿过积木块
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # 交叉熵损失函数：预测下一个词的概率
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_text, max_new_tokens=15, temperature=1.0):
        """
        自回归生成文本：
        一个字一个字往后预测，把新生成的字再追加到输入末尾！
        """
        self.eval()
        idx = torch.tensor([encode(prompt_text)], dtype=torch.long)

        for _ in range(max_new_tokens):
            # 截断不能超过最大序列长度
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            # 只取最后一个时间步的预测结果。
            # temperature <= 0：每次都选分数最高的字，结果可以重复。
            # temperature > 0：先把分数除以温度，再按概率掷骰子。
            logits_last = logits[:, -1, :]
            if temperature <= 0:
                next_idx = torch.argmax(logits_last, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits_last / temperature, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
            # 追加到已知序列末尾
            idx = torch.cat((idx, next_idx), dim=1)

        return decode(idx[0].tolist())

# =====================================================================
# 3. 训练与效果展示
# =====================================================================
def main():
    print("=" * 65)
    print("【实战开始：训练一个我们自己的微型诗歌大模型 Baby-GPT】")
    print("=" * 65)
    print(f"训练语料长度: {len(text_corpus)} 字符")
    print(f"词表大小: {vocab_size} 个独一无二的汉字/标点\n")

    print("部分汉字编号（sorted 按 Unicode 排，编号没有大小含义）：")
    for ch in ["。", "一", "上", "不", "光", "前", "地", "床", "明", "月", "霜", "，"]:
        print(f"  {ch} -> {char2idx[ch]}")
    print("句号“。”的编码靠前，全角逗号“，”的编码靠后，所以逗号排在词表末尾。\n")

    torch.manual_seed(42)
    model = BabyGPT(vocab_size=vocab_size, d_model=48, num_heads=4, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params}（每个参数都是一个会在训练中被微调的小数）")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # 每首诗单独做成“输入 = 除了最后一个字，目标 = 除了第一个字”。
    # 三首诗都从座位 0 开始，模型就不能只靠座位号背诵。
    encoded_poems = [torch.tensor(encode(poem), dtype=torch.long) for poem in poems]

    print("【未训练时，每次都选当前分数最高的字】:")
    for prompt in ["床前", "白日", "春眠"]:
        print(f"Prompt: '{prompt}' ==> 生成: '{model.generate(prompt, max_new_tokens=15, temperature=0)}'")
    print()

    print("--- 启动模型训练（三首诗分别计算损失，再取平均）---")
    for step in range(80):
        model.train()
        optimizer.zero_grad()
        losses = []
        for data in encoded_poems:
            _, loss = model(data[:-1].unsqueeze(0), data[1:].unsqueeze(0))
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0 or step == 0:
            print(f"轮次 Step {step+1:2d} | 损失值 Loss: {loss.item():.4f}")

    print("\n" + "=" * 65)
    print("【训练完成：它背下了三首课文的开头，还不会自由作诗】")
    print("=" * 65)

    print("用贪心解码（temperature = 0，每次选概率最高的字）：")
    for prompt in ["床前", "白日", "春眠"]:
        gen = model.generate(prompt, max_new_tokens=15, temperature=0)
        print(f"输入提示: '{prompt}'")
        print(f"续写结果: '{gen}'\n")

    stranger = model.generate("明月", max_new_tokens=15, temperature=0)
    print(f"没在课文开头出现过的提示 '明月' ==> '{stranger}'")
    print("如果这里接不回原诗，就说明它是在背三首短课文，不是在理解月光。\n")

    print("[复盘认知]：")
    print("Loss 下降，靠的是同一套手续：")
    print("1. 词向量加上可学习的座位向量")
    print("2. 带因果掩码的 Q、K、V 点积与加权")
    print("3. 前馈网络、残差连接和层归一化")
    print("4. 用交叉熵量出“下一个字猜得有多差”，再小步调整参数")
    print("聊天模型和这个小脚本用的是同类下一步预测。")
    print("差别在于数据量和参数量：这里只有三首诗，模型把它们背下来了。")

if __name__ == "__main__":
    main()
