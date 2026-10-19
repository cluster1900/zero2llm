"""P25: offline provenance, independent arithmetic/model oracle, semantic mutants."""
from copy import deepcopy
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

from torch_test_support import torch_status, torch_reason

SCRIPT = Path(__file__).resolve().parents[1] / 'code/26_qwen3_case.py'
spec = importlib.util.spec_from_file_location('qwen3_case_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
READY, VERSION, PRESENT = torch_status()
if READY:
    import torch


def mutant(old, new):
    source = SCRIPT.read_text(encoding='utf-8')
    if source.count(old) != 1:
        raise AssertionError(f'mutation target must occur once: {old!r}')
    name = 'qwen3_mutant_' + hashlib.sha256((old+new).encode()).hexdigest()[:12]
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPT)
    sys.modules[name] = module
    exec(compile(source.replace(old, new), str(SCRIPT), 'exec'), module.__dict__)
    return module


def scalar_linear(x, weight):
    return [sum(a*b for a, b in zip(x, row)) for row in weight.tolist()]


def scalar_norm(x, weight, eps):
    scale = math.sqrt(sum(v*v for v in x)/len(x) + eps)
    return [v/scale*float(w) for v, w in zip(x, weight.detach().tolist())]


def scalar_rope(x, position, theta):
    half = len(x)//2
    out = x[:]
    for j in range(half):
        angle = position/(theta ** (2*j/len(x)))
        out[j] = x[j]*math.cos(angle) - x[j+half]*math.sin(angle)
        out[j+half] = x[j+half]*math.cos(angle) + x[j]*math.sin(angle)
    return out


def scalar_ffn(x, mlp):
    gate, up = scalar_linear(x, mlp.gate_proj.weight), scalar_linear(x, mlp.up_proj.weight)
    return scalar_linear([a/(1+math.exp(-a))*b for a, b in zip(gate, up)], mlp.down_proj.weight)


def scalar_moe(x, mlp, k):
    logits = scalar_linear(x, mlp.gate.weight)
    chosen = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
    peak = max(logits)
    # Independent full softmax, then selected renormalization.
    p = [math.exp(v-peak) for v in logits]
    p = [v/sum(p) for v in p]
    denom = sum(p[i] for i in chosen)
    outputs = [scalar_ffn(x, mlp.experts[i]) for i in chosen]
    return [sum(p[i]/denom*y[j] for i, y in zip(chosen, outputs)) for j in range(len(x))]


def scalar_model(model, ids):
    """No production norm/RoPE/attention/FFN/forward helpers or tensor matmul.

    Returns full-context logits AND pre-expansion per-layer K/V for one row.
    """
    c = model.config
    D, d, H, K = (c[k] for k in ('hidden_size','head_dim','num_attention_heads','num_key_value_heads'))
    eps = c['rms_norm_eps']
    hidden = [model.embed_tokens.weight[t].tolist() for t in ids]
    caches = []
    for layer in model.layers:
        a = layer.self_attn
        projected = []
        for pos, row in enumerate(hidden):
            normalized = scalar_norm(row, layer.input_layernorm.weight, eps)
            qflat = scalar_linear(normalized, a.q_proj.weight)
            kflat = scalar_linear(normalized, a.k_proj.weight)
            vflat = scalar_linear(normalized, a.v_proj.weight)
            qs = [scalar_rope(scalar_norm(qflat[h*d:(h+1)*d], a.q_norm.weight, eps), pos, c['rope_theta']) for h in range(H)]
            ks = [scalar_rope(scalar_norm(kflat[h*d:(h+1)*d], a.k_norm.weight, eps), pos, c['rope_theta']) for h in range(K)]
            vs = [vflat[h*d:(h+1)*d] for h in range(K)]
            projected.append((qs, ks, vs))
        caches.append(tuple([[[projected[t][kind][h][j] for j in range(d)]
                               for t in range(len(ids))] for h in range(K)] for kind in (1,2)))
        new_hidden = []
        for pos, row in enumerate(hidden):
            joined = []
            for h in range(H):
                kv = h // (H//K)
                scores = [sum(projected[pos][0][h][j]*projected[t][1][kv][j] for j in range(d))/math.sqrt(d) for t in range(pos+1)]
                peak = max(scores)
                probs = [math.exp(s-peak) for s in scores]
                probs = [p/sum(probs) for p in probs]
                joined += [sum(probs[t]*projected[t][2][kv][j] for t in range(pos+1)) for j in range(d)]
            output = scalar_linear(joined, a.o_proj.weight)
            residual = [x+y for x, y in zip(row, output)]
            normed = scalar_norm(residual, layer.post_attention_layernorm.weight, eps)
            ffn = (scalar_moe(normed, layer.mlp, c['num_experts_per_tok'])
                   if hasattr(layer.mlp, 'experts') else scalar_ffn(normed, layer.mlp))
            new_hidden.append([x+y for x, y in zip(residual, ffn)])
        hidden = new_hidden
    return [scalar_linear(scalar_norm(x, model.norm.weight, eps), model.lm_head.weight) for x in hidden], caches


class MetadataTests(unittest.TestCase):
    def test_official_fixture_identity_hashes_license_and_sizes(self):
        for kind, (repo, rev) in m.PINS.items():
            record = m.load_checkpoint(kind)
            self.assertEqual(len(rev), 40)
            self.assertEqual(record['provenance']['repo_id'], repo)
            self.assertEqual(record['provenance']['license'], 'Apache-2.0')
            self.assertIn('Apache License', (m.DATA/kind/'LICENSE').read_text())
            self.assertEqual(record['implementation']['revision'], '5f4ecf2d9f867a1255131d2461d75793c0cf1db2')
            self.assertLess(sum(x['bytes'] for x in record['provenance']['files'].values()), 25000)
        self.assertEqual(m.load_checkpoint('dense')['provenance']['files']['config.json']['sha256'], '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd')
        self.assertEqual(m.load_checkpoint('moe')['provenance']['files']['config.json']['sha256'], '2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297')

    def test_corrupt_bytes_and_revision_fail_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'qwen3'
            shutil.copytree(m.DATA, root)
            target = root/'dense/config.json'
            target.write_bytes(target.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                m.load_checkpoint('dense', root)
            shutil.copyfile(m.DATA/'dense/config.json', target)
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['models']['dense']['revision'] = 'main'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'revision'):
                m.load_checkpoint('dense', root)

    def test_exact_real_parameter_totals_independent_hand_arithmetic(self):
        # All constants independently transcribed from pinned config/source shapes.
        dense = 151936*1024 + 28*(2*1024*16*128 + 2*1024*8*128 + 2*128 + 2*1024 + 3*1024*3072) + 1024
        moe_base = 2*151936*2048 + 48*(2*2048*32*128 + 2*2048*4*128 + 2*128 + 2*2048 + 2048*128) + 2048
        moe_expert = 48*128*3*2048*768
        self.assertEqual(dense, 596049920)
        self.assertEqual(moe_base+moe_expert, 30532122624)
        for kind, total, active, nonemb in (
                ('dense', dense, dense, 440467456),
                ('moe', moe_base+moe_expert, moe_base+48*8*3*2048*768, 29909792768)):
            budget = m.parameter_budget(m.load_checkpoint(kind)['config'])
            self.assertEqual(budget['unique_total'], total)
            self.assertEqual(budget['active_matrix_convention'], active)
            self.assertEqual(budget['non_embedding'], nonemb)
        self.assertEqual(moe_base+48*8*3*2048*768, 3353032704)

    def test_exercise_explicit_head_width_and_qk_norm_weights(self):
        for kind, D, H, K in (('dense',1024,16,8),('moe',2048,32,4)):
            c = m.load_checkpoint(kind)['config']
            inventory = {p.name:p.shape for p in m.parameter_inventory(c)}
            self.assertEqual(inventory['layers.0.self_attn.q_proj.weight'], (H*128,D))
            self.assertEqual(inventory['layers.0.self_attn.k_proj.weight'], (K*128,D))
            self.assertEqual(inventory['layers.0.self_attn.o_proj.weight'], (D,H*128))
            self.assertEqual(inventory['layers.0.self_attn.q_norm.weight'], (128,))
            self.assertNotEqual(H*128, D)

    def test_exercise_cache_bytes_and_head_ratio(self):
        for kind, expected, ratio in (('dense',469762048,2),('moe',402653184,8)):
            c = m.load_checkpoint(kind)['config']
            self.assertEqual(m.cache_bytes(c,1,4096), expected)
            self.assertEqual(m.cache_bytes(c,2,4096,4), expected*4)
            expanded = deepcopy(c)
            expanded['num_key_value_heads'] = expanded['num_attention_heads']
            self.assertEqual(m.cache_bytes(expanded,1,4096), expected*ratio)
            self.assertEqual(m.cache_bytes(c,1,0), 0)

    def test_mixed_sparse_layer_selection_no_unused_dense_mlp_count(self):
        c = m.tiny_config('moe')
        c.update(num_hidden_layers=4, decoder_sparse_step=2, mlp_only_layers=[1])
        self.assertEqual([m.is_moe_layer(c,i) for i in range(4)], [False,False,False,True])
        rows = m.parameter_inventory(c)
        self.assertEqual(sum(p.count for p in rows if p.component=='experts'), 4*3*8*6)
        self.assertEqual(sum(p.count for p in rows if p.component=='dense_ffn'), 3*3*8*12)

    def test_invalid_scope_and_numeric_values(self):
        c = m.tiny_config('moe')
        for key,value in [('head_dim',3),('num_key_value_heads',3),('attention_bias',True),
                          ('hidden_act','gelu'),('rope_scaling',{}),('use_sliding_window',True),
                          ('norm_topk_prob',False),('num_experts_per_tok',5),('rms_norm_eps',float('nan')),
                          ('rope_theta',True),('decoder_sparse_step',0),('mlp_only_layers',[-1]),
                          ('tie_word_embeddings',1),('hidden_size',True),('model_type','qwen3_next')]:
            bad=deepcopy(c); bad[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_config(bad)
        for batch,length,size in [(0,1,2),(True,1,2),(1,-1,2),(1,65,2),(1,1,0)]:
            with self.assertRaises(ValueError): m.cache_bytes(c,batch,length,size)

    def test_component_table_has_immutable_traceable_evidence(self):
        for kind in m.PINS:
            record=m.load_checkpoint(kind)
            table=m.component_table(record)
            self.assertEqual(len(table),6)
            for _,value,url in table:
                self.assertTrue(value)
                self.assertNotIn('/main/',url)
                self.assertTrue(record['provenance']['revision'] in url or record['implementation']['revision'] in url)

    def test_exercise_budget_counts_closure_and_missing_answer(self):
        self.assertEqual(m.completion_budget([151667,7,8,151668,9],6),
                         dict(reasoning_ids=3,closing_ids=1,answer_ids=1,generated_ids=5,remaining=1,thinking_closed=True))
        incomplete=m.completion_budget([151667,7,8],3)
        self.assertEqual((incomplete['answer_ids'],incomplete['thinking_closed']),(0,False))
        nonthinking=m.completion_budget([9,151645],2,False)
        self.assertEqual((nonthinking['answer_ids'],nonthinking['reasoning_ids']),(2,0))
        self.assertEqual(m.context_budget(12,8,20),0)
        with self.assertRaises(ValueError): m.context_budget(12,9,20)
        for ids,limit in [([1,2],1),([True],1),([-1],1),([],True)]:
            with self.assertRaises(ValueError): m.completion_budget(ids,limit)
        self.assertEqual(m.completion_budget([],0)['answer_ids'],0)

    def test_accounting_semantic_mutants_are_executed_and_killed(self):
        def assert_total(module):
            self.assertEqual(module.parameter_budget(module.load_checkpoint('dense')['config'])['unique_total'],596049920)
        for old,new in [
            ("if not c['tie_word_embeddings']:", 'if True:'),
            ("D, V, d = c['hidden_size'], c['vocab_size'], c['head_dim']", "D, V, d = c['hidden_size'], c['vocab_size'], c['hidden_size']//c['num_attention_heads']"),
            ("for name in ('q_norm', 'k_norm'):", 'for name in ():'),
        ]:
            with self.subTest(mutant=old), self.assertRaises(AssertionError): assert_total(mutant(old,new))
        bad=mutant("'generated_ids': len(ids),", "'generated_ids': len(ids)-control,")
        with self.assertRaises(AssertionError): self.assertEqual(bad.completion_budget([1,151668,2],3)['generated_ids'],3)
        bad=mutant('reasoning, control, answer, complete = len(ids), 0, 0, False', 'reasoning, control, answer, complete = 0, 0, len(ids), True')
        with self.assertRaises(AssertionError): self.assertEqual(bad.completion_budget([1,2],2)['answer_ids'],0)

    def test_cli_offline_stdlib_and_missing_optional_dependency(self):
        blocker="""
import importlib.abc, runpy, socket, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','jinja2','transformers','huggingface_hub'):
            raise ModuleNotFoundError('blocked '+fullname,name=fullname)
sys.meta_path.insert(0,Blocker())
def no_network(*args,**kwargs): raise AssertionError('network forbidden')
socket.socket.connect=no_network
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result=subprocess.run([sys.executable,'-c',blocker,str(SCRIPT)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('596,049,920',result.stdout)
        self.assertIn('30,532,122,624',result.stdout)
        self.assertIn('pretrained inference: NOT RUN',result.stdout)
        for blocked in ["('torch',)", "('jinja2',)", "('torch','jinja2')"]:
            one=blocker.replace("('torch','jinja2','transformers','huggingface_hub')",blocked)
            missing=subprocess.run([sys.executable,'-c',one,str(SCRIPT),'--torch'],capture_output=True,text=True,timeout=20)
            self.assertEqual(missing.returncode,2,missing.stderr)
            self.assertIn('pip install -r requirements.txt',missing.stderr)
            self.assertNotIn('Traceback',missing.stderr)


@unittest.skipUnless(importlib.util.find_spec('jinja2'), 'template execution requires requirements.txt Jinja2; skip is not pass')
class TemplateTests(unittest.TestCase):
    def test_exercise_official_template_not_naive_concatenation(self):
        plain='<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n'
        for kind in m.PINS:
            record=m.load_checkpoint(kind)
            self.assertEqual(m.render_prompt(record,'1+1?',True),plain)
            self.assertEqual(m.render_prompt(record,'1+1?',False),plain+'<think>\n\n</think>\n\n')
            self.assertIn('/no_think',m.render_prompt(record,'1+1? /no_think',True))
            self.assertFalse(m.render_prompt(record,'1+1? /no_think',True).endswith('</think>\n\n'))
            tokens=record['tokenizer_config']['added_tokens_decoder']
            self.assertEqual(tokens['151667']['content'],'<think>')
            self.assertEqual(tokens['151668']['content'],'</think>')
            self.assertFalse(tokens['151668']['special'])
            self.assertIsNone(record['tokenizer_config']['bos_token'])
            self.assertEqual(record['config']['bos_token_id'],151643)


@unittest.skipUnless(READY,torch_reason('P25 tiny CPU probe',VERSION,PRESENT))
class TinyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(250)

    def make(self,kind='dense',module=m):
        return module.build_tiny_model(module.tiny_config(kind)).double().eval()

    def assert_scalar_model(self,module,kind='dense'):
        model=self.make(kind,module)
        ids=torch.tensor([[1,2,3,4,5],[2,4,3,6,1]])
        actual,cache=model(ids)
        for batch,tokens in enumerate(ids.tolist()):
            expected,kv=scalar_model(model,tokens)
            torch.testing.assert_close(actual[batch],torch.tensor(expected,dtype=torch.float64),atol=1e-11,rtol=1e-11)
            for layer in range(len(kv)):
                for side in (0,1):
                    torch.testing.assert_close(cache.layers[layer][side][batch],torch.tensor(kv[layer][side],dtype=torch.float64),atol=1e-11,rtol=1e-11)
        return model

    def test_scalar_oracle_dense_and_moe_full_logits_and_compact_cache(self):
        for kind in m.PINS:
            with self.subTest(kind=kind): self.assert_scalar_model(m,kind)

    def test_exercise_tiny_actual_named_shapes_and_unique_counts(self):
        for kind,count in [('dense',1536),('moe',2312)]:
            model=self.make(kind)
            actual={name:tuple(p.shape) for name,p in model.named_parameters()}
            expected={p.name:p.shape for p in m.parameter_inventory(model.config)}
            self.assertEqual(actual,expected)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            self.assertEqual(m.parameter_budget(model.config)['unique_total'],count)
        with self.assertRaisesRegex(ValueError,'refuses'):
            m.build_tiny_model(m.load_checkpoint('dense')['config'])

    def test_exercise_qk_norm_and_split_half_rotation_hand_reference(self):
        x=torch.tensor([[[[1.,2.,3.,4.],[4.,3.,2.,1.]]]],dtype=torch.float64)
        gain=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        expected=[scalar_norm(v,gain,1e-6) for v in x[0,0].tolist()]
        norm=m.rms_norm(x,gain,1e-6)
        torch.testing.assert_close(norm[0,0],torch.tensor(expected,dtype=torch.float64))
        rotated=m.split_half_rope(norm,7,1000000)
        reference=[scalar_rope(v,7+i,1000000) for i,v in enumerate(expected)]
        torch.testing.assert_close(rotated[0,0],torch.tensor(reference,dtype=torch.float64),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(rotated.square().sum(-1),norm.square().sum(-1))

    def assert_chunks(self,module,kind='dense'):
        model=self.make(kind,module)
        ids=torch.tensor([[1,2,3,4,5]])
        reference,_=scalar_model(model,ids[0].tolist())
        pieces=[]; cache=None
        for left,right in [(0,1),(1,3),(3,5)]:
            logits,cache=model(ids[:,left:right],cache)
            pieces.append(logits)
            self.assertEqual(cache.length,right)
        actual=torch.cat(pieces,1)[0]
        torch.testing.assert_close(actual,torch.tensor(reference,dtype=torch.float64),atol=1e-11,rtol=1e-11)
        self.assertEqual(sum(t.numel()*t.element_size() for p in cache.layers for t in p),1280)
        return model,cache

    def test_arbitrary_chunks_match_independent_full_prefix_and_cache_bytes(self):
        for kind in m.PINS:
            model,cache=self.assert_chunks(m,kind)
            self.assertEqual(m.cache_bytes(model.config,1,5,8),1280)
            self.assertEqual(tuple(cache.layers[0][0].shape),(1,2,5,4))
            copy=[tuple(t.clone() for t in pair) for pair in cache.layers]
            model(torch.tensor([[6,7]]),cache)
            for before,pair in zip(copy,cache.layers):
                for a,b in zip(before,pair): self.assertTrue(torch.equal(a,b))

    def test_future_token_invariance_both_architectures(self):
        for kind in m.PINS:
            model=self.make(kind)
            x=torch.tensor([[1,2,3,4,5]])
            a,_=model(x)
            x[:,-2:]=torch.tensor([[7,8]])
            b,_=model(x)
            torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)

    def test_cache_owner_weights_shape_dtype_context_and_config_guards(self):
        model=self.make(); ids=torch.tensor([[1,2]])
        _,cache=model(ids)
        with self.assertRaisesRegex(ValueError,'another'): self.make()(ids,cache)
        malformed=m.TinyCache(((cache.layers[0][0][:,:1],cache.layers[0][1][:,:1]),)*2,model,cache.signature)
        with self.assertRaisesRegex(ValueError,'compact'): model(ids,malformed)
        malformed=m.TinyCache(tuple((a.float(),b.float()) for a,b in cache.layers),model,cache.signature)
        with self.assertRaisesRegex(ValueError,'compact'): model(ids,malformed)
        with self.assertRaisesRegex(ValueError,'context'): model(torch.ones((1,63),dtype=torch.long),cache)
        with torch.no_grad(): model.embed_tokens.weight[0,0]+=1
        with self.assertRaisesRegex(ValueError,'weights'): model(ids,cache)
        model.config['rope_theta']=1234
        with self.assertRaisesRegex(ValueError,'immutable'): model(ids)
        fresh=self.make().train()
        with self.assertRaisesRegex(ValueError,'eval'): fresh(ids)
        for invalid in [torch.tensor([[17]]),torch.tensor([[-1]]),torch.empty((1,0),dtype=torch.long),ids.float()]:
            with self.assertRaises(ValueError): self.make()(invalid)

    def test_sparse_moe_dispatch_and_selected_probability_hand_oracle(self):
        model=self.make('moe'); moe=model.layers[0].mlp
        x=torch.tensor([[[1.,-1.,2.,0.,.5,1.,-2.,.25]]],dtype=torch.float64)
        with torch.no_grad():
            moe.gate.weight.zero_()
            moe.gate.weight[:,0]=torch.tensor([2.,1.,-1.,0.])
        calls=[0]*4
        hooks=[e.register_forward_hook(lambda _m,_a,_o,i=i:calls.__setitem__(i,calls[i]+1)) for i,e in enumerate(moe.experts)]
        expected=scalar_moe(x[0,0].tolist(),moe,2)
        output=moe(x)
        for hook in hooks: hook.remove()
        self.assertEqual(calls,[1,1,0,0])
        torch.testing.assert_close(output[0,0],torch.tensor(expected,dtype=torch.float64),atol=1e-12,rtol=1e-12)

    def test_executable_numeric_semantic_mutants(self):
        mutations=[
            ('keys = k.repeat_interleave(repeats, dim=1)', 'keys = k.repeat(1, repeats, 1, 1)', 'dense',False),
            ('offset + torch.arange(q.shape[2], device=q.device)', '0*offset + torch.arange(q.shape[2], device=q.device)', 'dense',True),
            ('rotated = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)', 'rotated = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)', 'dense',False),
            ('q = self.q_norm(self.q_proj(x).view(B, T, H, d)).transpose(1, 2)', 'q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)', 'dense',False),
            ('weights = logits.gather(1, selected).softmax(-1)', 'weights = logits.softmax(-1).gather(1, selected)', 'moe',False),
            ("selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :c['num_experts_per_tok']]", "selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :c['num_experts']]", 'moe',False),
            ('k = torch.cat((old[0], k), 2)', "k = torch.cat((split_half_rope(old[0], 0, c['rope_theta']), k), 2)", 'dense',True),
        ]
        for old,new,kind,chunked in mutations:
            with self.subTest(mutant=old),self.assertRaises(AssertionError):
                bad=mutant(old,new)
                self.assert_chunks(bad,kind) if chunked else self.assert_scalar_model(bad,kind)

    @unittest.skipUnless(importlib.util.find_spec('jinja2'), 'optional CLI also requires Jinja2; skip is not pass')
    def test_cli_optional_probe_has_actual_shapes_and_counts(self):
        result=subprocess.run([sys.executable,str(SCRIPT),'--torch'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('tiny dense: params=1536; shape=(1, 5, 17); cache_bytes=1280;',result.stdout)
        self.assertIn('tiny moe: params=2312; shape=(1, 5, 17); cache_bytes=1280;',result.stdout)
        self.assertIn('pretrained inference: NOT RUN',result.stdout)


# Optional reference library is intentionally absent from baseline requirements.
try:
    HF_VERSION = importlib.metadata.version('transformers')
except importlib.metadata.PackageNotFoundError:
    HF_VERSION = None


@unittest.skipUnless(READY and HF_VERSION == '4.51.3',
                     'optional official comparator requires isolated Transformers 4.51.3 and stable torch 2.14.x; skip is not pass')
class OptionalOfficialTests(unittest.TestCase):
    def test_random_tiny_official_dense_and_moe_shapes_logits_and_cache(self):
        # Constructors only: no from_pretrained, network, tokenizer or weights.
        from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
        torch.set_num_threads(1)
        for kind, Config, Model in [('dense',Qwen3Config,Qwen3ForCausalLM),
                                     ('moe',Qwen3MoeConfig,Qwen3MoeForCausalLM)]:
            with self.subTest(kind=kind):
                torch.manual_seed(89)
                c=m.tiny_config(kind)
                toy=m.build_tiny_model(c).float().eval()
                cfg=Config(**c)
                cfg._attn_implementation='eager'
                official=Model(cfg).float().eval()
                mapped={(key if key.startswith('lm_head.') else 'model.'+key):value
                        for key,value in toy.state_dict().items()}
                official.load_state_dict(mapped,strict=True)
                self.assertEqual(sum(p.numel() for p in official.parameters()),
                                 1536 if kind=='dense' else 2312)
                ids=torch.tensor([[1,2,3,4,5],[5,2,6,1,8]])
                with torch.no_grad():
                    actual,cache=toy(ids)
                    reference=official(ids,use_cache=True)
                torch.testing.assert_close(actual,reference.logits,rtol=3e-6,atol=3e-6)
                for layer,pair in enumerate(cache.layers):
                    for side in (0,1):
                        torch.testing.assert_close(pair[side],reference.past_key_values[layer][side],
                                                   rtol=3e-6,atol=3e-6)


if __name__=='__main__':
    unittest.main()
