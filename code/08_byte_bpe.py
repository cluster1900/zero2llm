"""P02：可读的字节级 BPE，纯标准库、CPU、无需下载模型。

教学取舍：不做正则预切分、不归一化、不处理特殊 token；只学习字节对合并。
每个训练字符串是一篇独立文档；合并不会跨文档。实现优先清晰，不用于大语料。
"""
from collections import Counter


def merge_pair(ids, pair, new_id):
    """从左到右合并不重叠的相邻对；例如 aaa 合并 aa 后得到 [aa, a]。"""
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class ByteBPE:
    def __init__(self):
        # 每个 byte 都有 ID；因此未见过的合法 UTF-8 文本仍有基础表示。
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # (左 ID, 右 ID) -> 新 ID；插入顺序就是合并优先级。

    def train(self, documents, num_merges=20):
        """从头训练。频数相同时按 ID 对排序，保证同输入得到相同规则。"""
        if type(num_merges) is not int or num_merges < 0:
            raise ValueError("num_merges 必须是非负整数")
        if isinstance(documents, str):
            raise TypeError("请传入字符串列表，不是一条字符串")
        sequences = [list(text.encode('utf-8')) for text in documents]
        self.__init__()
        for _ in range(num_merges):
            counts = Counter()
            for ids in sequences:
                counts.update(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            new_id = len(self.vocab)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            sequences = [merge_pair(ids, pair, new_id) for ids in sequences]
        return self

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        # 新 ID 按训练次序递增；从早到晚应用学到的规则，而不是重新统计频数。
        for pair, new_id in self.merges.items():
            ids = merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        ids = list(ids)
        for token_id in ids:
            if type(token_id) is not int or token_id not in self.vocab:
                raise ValueError(f"无效 token ID: {token_id!r}")
        # 先拼 bytes 再解码。单个 token 可能只是一个汉字的一部分。
        return b''.join(self.vocab[token_id] for token_id in ids).decode('utf-8')


def main():
    tokenizer = ByteBPE().train(['banana banana', 'banana bandana', '你好，你好！'], 20)
    print('前四条合并规则：')
    for pair, new_id in list(tokenizer.merges.items())[:4]:
        print(f'{pair} -> {new_id}: {tokenizer.vocab[new_id]!r}')
    print(f'词表大小：{len(tokenizer.vocab)}')
    for text in ['banana banana', '你好，你好！', '新词🙂 café', '', 'e\u0301']:
        ids = tokenizer.encode(text)
        recovered = tokenizer.decode(ids)
        assert recovered == text
        print(f'{text!r}: 字符={len(text)}, UTF-8字节={len(text.encode("utf-8"))}, token={len(ids)}, 往返={recovered == text}')
    print('这些 token 数来自微型训练集，不代表任何商用模型的分词或计费。')


if __name__ == '__main__':
    main()
