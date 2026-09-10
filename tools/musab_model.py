"""36,321,280 parameter MathCore; cached CPU/CUDA inference without a trainer import."""
import argparse
import contextlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from musab_data import MAX_SEQ_LEN, ANSWER_BUDGET, BOS, EOS, PAD, encode_prompt, decode, pack

MODEL_CONFIG = dict(vocab=259, layers=12, dim=512, heads=8, kv_heads=2, head_dim=64, ff=1536, context=256)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return x * scale.to(x.dtype) * self.weight

def rope(x, cos, sin):
    a, b = x[..., ::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h = cfg['dim'], cfg['head_dim']
        self.q = nn.Linear(d, cfg['heads'] * h, bias=False)
        self.k = nn.Linear(d, cfg['kv_heads'] * h, bias=False)
        self.v = nn.Linear(d, cfg['kv_heads'] * h, bias=False)
        self.o = nn.Linear(cfg['heads'] * h, d, bias=False)
    def forward(self, x, cos, sin, cache=None, use_cache=False):
        b, t, _ = x.shape
        c = self.cfg
        q = self.q(x).view(b, t, c['heads'], c['head_dim']).transpose(1, 2)
        k = self.k(x).view(b, t, c['kv_heads'], c['head_dim']).transpose(1, 2)
        v = self.v(x).view(b, t, c['kv_heads'], c['head_dim']).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if cache is not None:
            if t != 1:
                raise ValueError('Cached continuation accepts one token per sequence')
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        new_cache = (k, v) if use_cache else None
        rep = c['heads'] // c['kv_heads']
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                          is_causal=(cache is None))
        return self.o(y.transpose(1, 2).contiguous().view(b, t, c['dim'])), new_cache

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.g = nn.Linear(cfg['dim'], cfg['ff'], bias=False)
        self.u = nn.Linear(cfg['dim'], cfg['ff'], bias=False)
        self.d = nn.Linear(cfg['ff'], cfg['dim'], bias=False)
    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg['dim']), RMSNorm(cfg['dim'])
        self.attn, self.ff = Attention(cfg), FeedForward(cfg)
    def forward(self, x, c, s, cache=None, use_cache=False):
        y, new_cache = self.attn(self.n1(x), c, s, cache, use_cache)
        x = x + y
        return x + self.ff(self.n2(x)), new_cache

class MathCore(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(MODEL_CONFIG if cfg is None else cfg)
        c = self.cfg
        self.embed = nn.Embedding(c['vocab'], c['dim'])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c['layers'])])
        self.norm = RMSNorm(c['dim'])
        inv = 1.0 / (10000.0 ** (torch.arange(0, c['head_dim'], 2).float() / c['head_dim']))
        angles = torch.outer(torch.arange(c['context']).float(), inv)[None, None]
        self.register_buffer('rope_cos', angles.cos(), persistent=False)
        self.register_buffer('rope_sin', angles.sin(), persistent=False)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, std=0.02 / math.sqrt(2 * c['layers']))
            nn.init.normal_(block.ff.d.weight, std=0.02 / math.sqrt(2 * c['layers']))
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, ids, caches=None, use_cache=False):
        offset = 0 if caches is None else caches[0][0].shape[-2]
        end = offset + ids.shape[1]
        if end > self.cfg['context']:
            raise ValueError('Context exhausted')
        c, s = self.rope_cos[..., offset:end, :], self.rope_sin[..., offset:end, :]
        x, result = self.embed(ids), []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, c, s, None if caches is None else caches[i], use_cache)
            if use_cache:
                result.append(cache)
        # Inference computes logits only for the token being predicted.
        hidden = self.norm(x[:, -1:] if use_cache else x)
        logits = F.linear(hidden, self.embed.weight)
        return (logits, result) if use_cache else logits

def collate(pairs, device):
    # None is padding of the FINAL global batch, with exactly zero supervised tokens.
    items = [pack(*r) if r is not None else ([BOS, EOS], [-100, -100]) for r in pairs]
    length = max(len(x) for x, _ in items)
    x = torch.full((len(items), length), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (ids, labels) in enumerate(items):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)

def loss_stats(logits, labels):
    # Explicit FP32 CE also protects evaluation outside autocast.
    z, y = logits[:, :-1].float(), labels[:, 1:]
    loss = F.cross_entropy(z.reshape(-1, z.shape[-1]), y.reshape(-1), ignore_index=-100, reduction='sum')
    good = y != -100
    return loss, good.sum(), ((z.argmax(-1) == y) & good).sum()

@torch.inference_mode()
def generate_batch(model, questions, device, max_new=ANSWER_BUDGET):
    """Group equal-length prompts: no padding positions or attention-mask ambiguity."""
    model.eval()
    groups, result = {}, [None] * len(questions)
    for i, q in enumerate(questions):
        p = encode_prompt(q)
        groups.setdefault(len(p), []).append((i, p))
    for length, entries in groups.items():
        x = torch.tensor([p for _, p in entries], device=device)
        tokens = [[] for _ in entries]
        ended = [False] * len(entries)
        caches = None
        for _ in range(min(max_new, model.cfg['context'] - length)):
            amp = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()
            with amp:
                logits, caches = model(x, caches=caches, use_cache=True)
            # PAD/BOS cannot form a valid answer. EOS remains legal.
            logits[:, -1, PAD] = -float('inf')
            logits[:, -1, BOS] = -float('inf')
            next_ids = logits[:, -1].argmax(-1).tolist()
            for j, token in enumerate(next_ids):
                if not ended[j]:
                    if token == EOS:
                        ended[j] = True
                    else:
                        tokens[j].append(token)
            if all(ended):
                break
            x = torch.tensor([[EOS if ended[j] else token] for j, token in enumerate(next_ids)], device=device)
        for j, (i, _) in enumerate(entries):
            result[i] = dict(answer=decode(tokens[j]).strip(), ended=ended[j], tokens=len(tokens[j]))
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--int8', action='store_true', help='Optional dynamic CPU quantization; can change answers')
    a = p.parse_args()
    torch.set_num_threads(max(1, a.threads))
    ck = torch.load(a.weights, map_location='cpu', weights_only=True)
    if ck.get('format') != 'MUSAB_V12_INFERENCE':
        raise ValueError('Use laptop_fp16.pt, not the training resume checkpoint')
    model = MathCore(ck['model_config'])
    model.load_state_dict(ck['model'], strict=True)
    del ck
    model.eval()
    if a.int8:
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    r = generate_batch(model, [a.question], torch.device('cpu'))[0]
    print(r['answer'])
    if not r['ended']:
        print('[Generation reached its length limit without EOS]')

if __name__ == '__main__':
    main()
