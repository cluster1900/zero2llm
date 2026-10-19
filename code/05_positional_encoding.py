"""05_positional_encoding.py: CPU 上看位置编码、顺序、相加与长度边界。

不同频率像转速不同的指针，但不承诺有限精度下任意长位置都唯一可辨。
保留 generate_positional_encoding(max_seq_len, d_model)、ascii_visualize 和 main
入口。公式采用 Attention Is All You Need §3.5；奇数宽度保留最后一列 sine，
不补一个超出宽度的 cosine。本例不训练模型，也不测试长文本任务质量。
"""

import math

import torch
import torch.nn as nn


def generate_positional_encoding(max_seq_len, d_model):
    """返回 CPU float32 [max_seq_len, d_model]，位置从 0 开始。

    max_seq_len 是非负整数，d_model 是正整数（不接受 bool）。
    i 是第几对维度，不是已经乘过 2 的列号：
      PE(pos, 2i)   = sin(pos / 10000 ** (2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000 ** (2i / d_model))
    d_model=5 时填充 (0,1)、(2,3)、最后的第 4 列 sine。
    空序列返回 [0, d_model]；生成长度不等于模型支持的上下文长度。
    """
    for name, value in [('max_seq_len', max_seq_len), ('d_model', d_model)]:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f'{name} must be an integer, not bool')
    if max_seq_len < 0 or d_model <= 0:
        raise ValueError('max_seq_len must be >= 0 and d_model must be > 0')

    pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32, device='cpu')
    for pos in range(max_seq_len):
        # 向上取整，奇数宽度的最后一列也要填；k 是“第几对”。
        for k in range((d_model + 1) // 2):
            div_term = math.pow(10000.0, (2 * k) / d_model)
            pe[pos, 2 * k] = math.sin(pos / div_term)
            if 2 * k + 1 < d_model:
                pe[pos, 2 * k + 1] = math.cos(pos / div_term)
    return pe


def ascii_visualize(pe_matrix):
    """用字符表示 [-1, 1]：空格最小，@ 最大；这不是精确的数值表。"""
    chars = ' .:-=+*#%@'
    rows, cols = pe_matrix.shape
    print(f'\n【位置编码字符热力图：{rows} 个位置，{cols} 个维度】')
    print('低维列变化快，高维列变化慢；颜色/字符相同可能只是量化后相同。')
    print('-' * 55)
    print('Pos \\ Dim | ' + ' '.join(f'd{c}' for c in range(cols)))
    print('-' * 55)
    for r in range(rows):
        line = f'Pos {r:2d}    | '
        for c in range(cols):
            value = pe_matrix[r, c].item()
            index = max(0, min(9, int((value + 1.0) / 2.0 * 9.99)))
            line += f' {chars[index]} '
        print(line)
    print('-' * 55)


def position_cosines(pe_matrix):
    """本章表格：第 0 行与各行的 cosine；输入须为非零行的二维矩阵。"""
    if pe_matrix.ndim != 2 or pe_matrix.shape[0] == 0:
        raise ValueError('expected a nonempty [positions, dimensions] matrix')
    norms = torch.linalg.vector_norm(pe_matrix, dim=-1)
    if not torch.isfinite(pe_matrix).all() or (norms == 0).any():
        raise ValueError('cosine requires finite, nonzero position vectors')
    return (pe_matrix @ pe_matrix[0]) / (norms * norms[0])


def permutation_example():
    """Q=K=V=X、无 mask/dropout 的小实验；只换词，不换座位编码。"""
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])  # 小猫、吃、小鱼
    order = torch.tensor([2, 1, 0])

    def attention(inputs):
        scores = inputs @ inputs.T / math.sqrt(inputs.shape[-1])
        return torch.softmax(scores, dim=-1) @ inputs

    pe = generate_positional_encoding(3, 2)
    return {
        'order': order,
        'plain': attention(x),
        'plain_swapped': attention(x[order]),
        'positioned': attention(x + pe),
        'positioned_swapped': attention(x[order] + pe),
    }


def addition_concat_example():
    """同一份词向量和 PE：Add 保持 D，Concat 变成 2D。"""
    x = torch.tensor([[[.2, -.1, .4, .0], [.5, .3, -.2, .1]]])
    pe = generate_positional_encoding(2, 4).unsqueeze(0)  # [1, T, D]
    return x, pe, x + pe, torch.cat((x, pe.expand_as(x)), dim=-1)


def length_example():
    """只检查能否取到编码，不把成功前向当成长度泛化。"""
    table = nn.Embedding(8, 8)
    try:
        table(torch.tensor([8]))  # 表只有第 0 到第 7 行。
    except IndexError:
        table_rejects_position_8 = True
    else:
        table_rejects_position_8 = False
    short = generate_positional_encoding(8, 8)
    longer = generate_positional_encoding(12, 8)
    return {
        'learned_parameters': table.weight.numel(),
        'table_rejects_position_8': table_rejects_position_8,
        'prefix_equal': torch.equal(short, longer[:8]),
        'longer_shape': tuple(longer.shape),
    }


def markdown_tables():
    """重新生成正文表 5-2 / 5-3 的全部数据行，方便修改后核对。"""
    pe = generate_positional_encoding(8, 8)
    rows = ['表 5-2：d_model=8，保留两位小数']
    for pos, row in enumerate(pe):
        rows.append('| **位置 ' + str(pos) + '** | ' +
                    ' | '.join(f'{v.item():.2f}' for v in row) + ' |')
    rows.append('\n表 5-3：位置 0 与各位置，保留三位小数')
    for pos, value in enumerate(position_cosines(pe)):
        bar = '█' * max(0, int(value.item() * 25))
        rows.append(f'| **位置 0 与 位置 {pos}** | **{value.item():.3f}** | `{bar}` |')
    return '\n'.join(rows)


def main():
    torch.manual_seed(6)
    torch.set_num_threads(1)
    print(f'设备=cpu；PyTorch={torch.__version__}；seed=6；不训练模型')
    print('【不同频率的座位号】')
    print(markdown_tables())
    pe = generate_positional_encoding(8, 8)
    ascii_visualize(pe)
    print('\n每对维度的分母与周期（d_model=8）：')
    for k in range(4):
        denominator = 10000 ** (2 * k / 8)
        print(f'd{2*k}/d{2*k+1}: 分母={denominator:g}，周期={2*math.pi*denominator:.2f}')
    similarities = position_cosines(pe)
    print(f'cos(PE0, PE3)={similarities[3].item():.3f} < '
          f'cos(PE0, PE6)={similarities[6].item():.3f}：距离更远却更相似。')
    print(f'PE[0,2]={pe[0,2].item():.3f}；PE[6,2]={pe[6,2].item():.3f}：'
          '相似不等于同一个向量。')

    demo = permutation_example()
    order = demo['order']
    plain_error = (demo['plain_swapped'] - demo['plain'][order]).abs().max().item()
    position_error = (demo['positioned_swapped'] - demo['positioned'][order]).abs().max().item()
    print('\n【换词实验：无位置、无 mask 的基线满足置换等变】')
    print(f'不加位置，换序误差={plain_error:.6f}')
    print(f'固定座位编码，只换词，换序误差={position_error:.6f}')
    print('这只证明位置可以改变计算，不证明随机玩具模型懂了主语宾语。')

    x, position, added, concatenated = addition_concat_example()
    print('\n【相加与拼接】')
    print(f'X={tuple(x.shape)}；PE={tuple(position.shape)}')
    print(f'Add={tuple(added.shape)}；Concat={tuple(concatenated.shape)}')
    for pos, row in enumerate(added[0]):
        print(f'Add 位置 {pos}: [' + ', '.join(f'{v.item():.3f}' for v in row) + ']')
    print('D=4，无 bias 的 Q 投影若输出仍为 4：Add=16 个参数，Concat=32 个参数。')

    lengths = length_example()
    print('\n【长度检查，不是长度泛化测试】')
    print(f'可学习表参数={lengths["learned_parameters"]}；位置 8 越界={lengths["table_rejects_position_8"]}')
    print(f'正弦扩展形状={lengths["longer_shape"]}；前 8 行不变={lengths["prefix_equal"]}')
    print(f'奇数宽度 d_model=5：PE[1,4]={generate_positional_encoding(2,5)[1,4].item():.6f}')
    print('能生成位置 8 到 11 的数值，不代表训练到长度 8 的模型会用好它们。')


if __name__ == '__main__':
    main()
