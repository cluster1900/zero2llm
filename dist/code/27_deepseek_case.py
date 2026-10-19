"""P26: offline DeepSeek evidence matrix and tiny, dependency-free experiments.

python code/27_deepseek_case.py [--json | --matrix-only]
No weights, network, remote code, Torch, model quality or speed benchmark.
The source fixtures are bytes fetched from immutable official model revisions.
Literature rows are manually reviewed evidence, NOT machine-verified facts.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

DATA = Path(__file__).resolve().parent / 'data' / 'deepseek'


def positive_int(name, value, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if allow_zero else 1}')
    return value


def load_sources(root=DATA):
    """Check snapshot integrity and immutable URL identity; never fetch anything."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported manifest schema')
    configs = {}
    for name, record in manifest['files'].items():
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('fixture must be a local JSON basename')
        model = manifest['models'][record['model']]
        revision = model['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('model revision must be an immutable commit SHA')
        expected_url = (f"https://huggingface.co/{model['repo']}/resolve/"
                        f"{revision}/{record['remote_path']}")
        if record['url'] != expected_url:
            raise ValueError('source URL does not match pinned model revision')
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'fixture checksum mismatch: {name}')
        configs[name] = json.loads(raw)
    return manifest, configs


def extract_fields(config, paths):
    """Missing fields remain explicit unknowns, NEVER False/zero/default values."""
    result, missing = {}, []
    for path in paths:
        node = config
        for part in path.split('.'):
            if not isinstance(node, dict) or part not in node:
                missing.append(path)
                node = None
                break
            node = node[part]
        result[path] = node
    return result, missing


# Requested paths are explicit. No classifier infers a training recipe from a
# field called 'nextn'/'mtp', or an attention algorithm from num_key_value_heads.
CONFIG_ROWS = (
    ('V3 / MLA', 'P11/P14', 'v3.config.json',
     ('kv_lora_rank', 'qk_rope_head_dim', 'num_attention_heads'), '只核对潜变量与位置分支尺寸'),
    ('V3 / MoE', 'P18', 'v3.config.json',
     ('first_k_dense_replace', 'n_routed_experts', 'num_experts_per_tok', 'n_shared_experts', 'scoring_func'), '前三层是 dense；本实验不含分组/节点限制'),
    ('V3 / 预测层', 'P23', 'v3.config.json',
     ('num_nextn_predict_layers',), '模块数不能单独证明训练目标'),
    ('R1 / 主干', 'P21/P22/P23', 'r1.config.json',
     ('model_type', 'num_hidden_layers', 'kv_lora_rank'), '配置不能证明后训练步骤'),
    ('R1-Distill-Qwen-1.5B / 学生', 'P14/P25', 'distill.config.json',
     ('model_type', 'num_attention_heads', 'num_key_value_heads', 'use_sliding_window'), '学生架构不是教师架构的缩小复印件'),
    ('V4.1-Flash / CED 与 SWA', 'P16/P17', 'v41.config.json',
     ('text_config.num_hidden_layers', 'text_config.sliding_window'), '20+20 划分需报告和源码佐证'),
    ('V4.1-Flash / CSA2', 'P17', 'v41.inference_config.json',
     ('kv_source_layers', 'index_source_layers', 'compress_ratios'), '仅主干前 n_layers 项参与账本'),
    ('V4.1-Flash / MoE', 'P18', 'v41.config.json',
     ('text_config.n_routed_experts', 'text_config.num_experts_per_tok', 'text_config.scoring_func'), '不能沿用 V3 sigmoid 路由公式'),
    ('V4.1-Flash / DSpark 字段', 'P23', 'v41.inference_config.json',
     ('n_mtp_layers', 'dspark_block_size', 'dspark_target_layer_ids'), '字段名 mtp 不是 backbone MTP 训练证据'),
    ('V4.1-Flash / 图像入口', 'P26', 'v41.config.json',
     ('vision_config.hidden_size', 'vision_config.downsample_ratio', 'text_config.hidden_size'), '只验证重排坐标，不复现视觉理解'),
    ('V4.1-Flash / mHC、Engram', 'P26', 'v41.inference_config.json',
     ('hc_mult', 'engram_layer_ids'), '只作为阅读路标，未训练或运行官方模块'),
)

PAPER_ROWS = (
    ('V3 / 顺序 MTP', 'P23', 'v3-paper', '§2.2、§4.2',
     '顺序模块、共享 embedding/output head；额外预测深度 1', '教学独立线性未来头不是该实现'),
    ('R1 / 后训练', 'P21/P22/P23', 'r1-paper', '§2.3、§2.4',
     '冷启动、推理 RL、拒绝采样后 SFT、综合 RL；蒸馏学生只做 SFT', '原始 R1 报告，不涵盖后续型号'),
    ('V4.1-Flash / CED 来源', 'P17', 'v41-paper', '§2.2',
     'decoder global KV 来自最终 encoder 表示；SWA KV 保持逐层来源', '账本不计算 attention 或 bounded replay'),
    ('V4.1-Flash / 训练时序', 'P23', 'v41-paper', '§2.1、§2.4.3',
     'backbone 预训练省略 MTP；DSpark 在其后专门训练', '不从 n_mtp_layers=3 推出预训练用了 MTP'),
    ('V4.1-Flash / KV 精度', 'P24/P26', 'v41-paper', '§2.4.4',
     'main KV 使用 FP4；量化 scale 与其他缓存另计', '仅计算假想固定 bit payload，不执行 FP4 codec'),
    ('V4.1-Flash / 多模态对齐', 'P26', 'v41-vision', 'Aligner.forward',
     '右下补零后按 channel→局部行→局部列重排，再由 MLP 对齐', '只复现重排；无 ViT、GELU 或训练权重'),
    ('V4.1-Flash / 扩展阅读', 'P26', 'v41-paper', '§2.4.1–2.4.3',
     'Single-Pass mHC、Engram、DSpark 分别处理残差、条件记忆与草稿', '概念不同，不能合并成一个注意力模块'),
    ('V4.1-Flash / 参考推理', 'P24/P26', 'v41-inference-readme', 'Minimal inference / Self-test',
     '参考 generation 为普通自回归；自测未初始化权重，只查连接和形状', '不是生产调度器，也不是官方数值复现'),
)


def evidence_matrix(manifest, configs):
    rows = []
    for component, chapters, file, paths, boundary in CONFIG_ROWS:
        fields, missing = extract_fields(configs.get(file, {}), paths)
        rows.append(dict(component=component, chapters=chapters, support='config',
                         status='unknown' if missing else 'extracted', values=fields,
                         missing=missing, source=manifest['files'][file]['url'],
                         location=', '.join(paths), boundary=boundary))
    for component, chapters, ref, location, statement, boundary in PAPER_ROWS:
        rows.append(dict(component=component, chapters=chapters,
                         support=manifest['references'][ref]['kind'], status='reviewed',
                         statement=statement, source=manifest['references'][ref]['url'],
                         location=location, boundary=boundary))
    return rows


def csa2_schedule(config):
    """Resolve source identities, NOT a compressor, indexer, or serving engine."""
    n = positive_int('n_layers', config['n_layers'])
    ratios = config['compress_ratios']
    if not isinstance(ratios, list) or len(ratios) < n:
        raise ValueError('compress_ratios must cover every backbone layer')
    if any(type(r) is not int or r < 0 for r in ratios):
        raise ValueError('compression ratios must be nonnegative integers')
    sources = config['kv_source_layers']
    indexes = config['index_source_layers']
    for name, ids in (('KV sources', sources), ('index sources', indexes)):
        if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < n for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError(f'{name} must be sorted unique backbone layer IDs')
    if not set(sources) <= set(indexes):
        raise ValueError('each Full layer must also produce indices')
    rows, kv, index = [], None, None
    for layer in range(n):
        ratio = ratios[layer]
        if ratio == 0:
            if layer in sources or layer in indexes:
                raise ValueError('SWA-only layer cannot produce global KV/indices')
            rows.append(dict(layer=layer, mode='SWA', ratio=0, kv_source=None,
                             index_source=None, local_source=layer))
            continue
        if layer in sources:
            kv, index = layer, layer
            mode = 'Full'
        elif kv is None:
            raise ValueError('global layer has no preceding KV source')
        elif ratio != ratios[kv]:
            raise ValueError('reused KV must have the same compression ratio')
        elif layer in indexes:
            index, mode = layer, 'Reindex'
        else:
            mode = 'Reuse'
        rows.append(dict(layer=layer, mode=mode, ratio=ratio, kv_source=kv,
                         index_source=index, local_source=layer))
    return rows


def completed_entries(query_position, ratio):
    """Only groups whose LAST token has arrived are causally visible."""
    positive_int('query_position', query_position, allow_zero=True)
    positive_int('ratio', ratio)
    return (query_position + 1) // ratio


def cache_ledger(config, tokens=10, channels=4, bits=4):
    """Hypothetical main-latent BODY only, compact packing per source bank.

    Excludes scale bytes, indexer K, local SWA, incomplete-group state, alignment,
    allocator slack and all weights. Never a claim about actual HBM/SSD footprint.
    """
    positive_int('tokens', tokens, allow_zero=True)
    positive_int('channels', channels)
    positive_int('bits', bits)
    schedule = csa2_schedule(config)
    global_rows = [r for r in schedule if r['mode'] != 'SWA']
    banks = [r for r in global_rows if r['mode'] == 'Full']

    def count(rows):
        entries = [tokens // r['ratio'] for r in rows]
        return dict(banks=len(rows), entries=sum(entries),
                    payload_bytes=sum((n * channels * bits + 7) // 8 for n in entries))
    return dict(tokens=tokens, channels=channels, bits=bits,
                independent_per_layer=count(global_rows), shared_sources=count(banks),
                modes=dict(Counter(r['mode'] for r in schedule)))


def _matrix(value, name):
    if (not isinstance(value, (list, tuple)) or not value or
            any(not isinstance(r, (list, tuple)) or not r for r in value)):
        raise ValueError(f'{name} must be a nonempty rectangular matrix')
    width = len(value[0])
    if any(len(r) != width or any(type(v) not in (int, float) or
           not math.isfinite(v) for v in r) for r in value):
        raise ValueError(f'{name} must be rectangular and finite')
    return len(value), width


def matmul(a, b):
    _, inner = _matrix(a, 'a')
    rows, width = _matrix(b, 'b')
    if inner != rows:
        raise ValueError('matrix multiplication dimension mismatch')
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def mla_probe():
    """One toy head, already-rotated position branch; NOT the complete V3 MLA."""
    latent = [[1., 2.], [-1., 1.], [2., 0.]]
    key_up, value_up = [[1., 2.], [-1., 1.]], [[2., 0.], [1., 3.]]
    query, query_rope = [2., -1.], [1., 2.]
    key_rope = [[0., 1.], [1., 0.], [2., -1.]]
    keys, values = matmul(latent, key_up), matmul(latent, value_up)
    expanded_logits = [(sum(q*k for q, k in zip(query, key)) +
                        sum(q*k for q, k in zip(query_rope, pos))) / 2
                       for key, pos in zip(keys, key_rope)]
    absorbed_query = matmul([query], list(map(list, zip(*key_up))))[0]
    latent_logits = [(sum(q*c for q, c in zip(absorbed_query, row)) +
                      sum(q*k for q, k in zip(query_rope, pos))) / 2
                     for row, pos in zip(latent, key_rope)]

    def probabilities(logits):
        exp = [math.exp(x-max(logits)) for x in logits]
        return [x/sum(exp) for x in exp]
    expanded = matmul([probabilities(expanded_logits)], values)[0]
    absorbed = matmul(matmul([probabilities(latent_logits)], latent), value_up)[0]
    return dict(latent_shape=[3, 2], positional_key_shape=[3, 2],
                expanded_logits=expanded_logits, latent_logits=latent_logits,
                expanded_output=expanded, absorbed_output=absorbed,
                max_error=max(abs(a-b) for a, b in zip(expanded, absorbed)))


def route_experts(affinities, biases, outputs, shared, k, scale=1.):
    """V3-inspired scoring distinction, NOT grouped/node-limited V3 routing.

    Inputs are post-sigmoid affinities. Bias selects; original affinity weights.
    Ties break by expert ID only for deterministic teaching tests.
    """
    n, width = _matrix(outputs, 'expert outputs')
    positive_int('k', k)
    if k > n or len(affinities) != n or len(biases) != n or len(shared) != width:
        raise ValueError('expert routing dimensions or k mismatch')
    if any(type(s) not in (int, float) or not math.isfinite(s) or not 0 < s < 1
           for s in affinities):
        raise ValueError('affinities must be finite and strictly between 0 and 1')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*biases, *shared, scale]):
        raise ValueError('bias, shared output and scale must be finite')
    if scale <= 0:
        raise ValueError('scale must be positive')
    selected = sorted(range(n), key=lambda i: (-(affinities[i]+biases[i]), i))[:k]
    denominator = sum(affinities[i] for i in selected)
    weights = [affinities[i]/denominator for i in selected]
    output = [shared[d]+scale*sum(w*outputs[i][d] for i, w in zip(selected, weights))
              for d in range(width)]
    return dict(selected=selected, weights=weights, output=output)


def unshuffle_tokens(grid, ratio=3):
    """H×W×C -> ceil(H/r)*ceil(W/r) rows of C*r*r, right/bottom zero pad.

    Within a row: channel, local row, local column (official Aligner F.unfold
    ordering). This is a coordinate permutation/padding, NOT average pooling.
    """
    positive_int('ratio', ratio)
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError('grid must be nonempty H x W x C')
    width, channels = _matrix(grid[0], 'grid row')
    if any(_matrix(row, 'grid row') != (width, channels) for row in grid):
        raise ValueError('grid must be rectangular H x W x C')
    height, result = len(grid), []
    for top in range(0, height, ratio):
        for left in range(0, width, ratio):
            result.append([grid[top+dy][left+dx][c]
                           if top+dy < height and left+dx < width else 0
                           for c in range(channels)
                           for dy in range(ratio) for dx in range(ratio)])
    return result


def experiment_report(configs):
    grid = [[[100*y+10*x+c for c in range(2)] for x in range(6)] for y in range(6)]
    shuffled = unshuffle_tokens(grid)
    return dict(mla=mla_probe(),
                moe=route_experts([.75, .5, .25], [-1., 0., 1.],
                                  [[2., 0.], [0., 4.], [6., 2.]], [1., -1.], 2),
                csa2=cache_ledger(configs['v41.inference_config.json']),
                vision=dict(input_shape=[6, 6, 2], output_shape=[len(shuffled), len(shuffled[0])],
                            first_token=shuffled[0], input_scalars=72, output_scalars=72))


def render_matrix(rows):
    def cell(x):
        text = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(',', ':'))
        return text.replace('|', '\\|').replace('\n', ' ')
    lines = ['| 组件 | 本书章节 | 证据与状态 | 字段/主张 | 边界 |',
             '| --- | --- | --- | --- | --- |']
    for row in rows:
        values = row.get('values', row.get('statement'))
        lines.append('| ' + ' | '.join(map(cell, [row['component'], row['chapters'],
                     f"[{row['support']}:{row['status']}]({row['source']}) {row['location']}",
                     values, row['boundary']])) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--json', action='store_true', help='完整离线报告 JSON')
    mode.add_argument('--matrix-only', action='store_true', help='只输出 Markdown 证据矩阵')
    args = parser.parse_args(argv)
    manifest, configs = load_sources()
    rows = evidence_matrix(manifest, configs)
    report = dict(checked_date=manifest['checked_date'], models=manifest['models'],
                  matrix=rows, experiments=experiment_report(configs),
                  runtime='Python standard library / CPU / offline',
                  not_run=['official weights', 'training', 'GPU kernels', 'quality benchmarks', 'serving throughput'])
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_matrix(rows))
        if not args.matrix_only:
            print('\nCPU 教学实验（不是官方模型性能）:')
            print(json.dumps(report['experiments'], ensure_ascii=False, indent=2, allow_nan=False))
            print('\n未运行: ' + ', '.join(report['not_run']))


if __name__ == '__main__':
    main()
