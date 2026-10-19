"""P02 可选对照：tokenizers==0.21.4；CPU、本地合成语料、无权重下载。"""
from importlib.metadata import version
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


def build_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=276, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(['banana banana', 'banana bandana', '你好，你好！'], trainer)
    return tokenizer


def main():
    actual = version('tokenizers')
    if actual != '0.21.4':
        raise RuntimeError(f'本章验证版本为 0.21.4，当前为 {actual}；请按 requirements-tokenizer.txt 安装')
    tokenizer = build_tokenizer()
    print(f'tokenizers={actual}; vocab={tokenizer.get_vocab_size()}')
    for text in ['banana banana', '你好，你好！', '新词🙂 café', '', 'e\u0301']:
        ids = tokenizer.encode(text).ids
        assert tokenizer.decode(ids) == text
        print(f'{text!r}: token={len(ids)}, 往返=True')
    print('对照 round-trip 和词表规模，不要求与教学实现的 ID/并列合并顺序相同。')


if __name__ == '__main__':
    main()
