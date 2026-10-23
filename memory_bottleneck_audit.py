#!/usr/bin/env python3
"""Read-only checkpoint audit for MiniGPTRecall (the supplied RNN(9) model).

Place beside config.py, miniGPT_model_RNN.py and gutenberg_qa_loader.py.
Run: python memory_bottleneck_audit.py --checkpoint checkpoints_gutenberg_qa/latest.pt
Use --help for paths and budgets. No training module is imported, no optimizer is
created, and no checkpoint is written. Requires your existing PyTorch/tokenizer
environment. Dataset input is the original Gutenberg QA JSONL schema, not the
memory_diagnostics JSONL log.

Reports: report.md, summary.json, per_sample.csv, writer_per_sample.jsonl,
and a few full pooling-weight snapshots in pooling_examples.json.

All standard recall metrics are teacher-forced, except greedy_exact. The latter
uses the gold answer LENGTH as a fixed generation budget (not an EOS test).
Interventions are diagnostic distribution shifts, not causal proof of capacity.
Slot permutation is an invariance sanity check for the supplied reader, not a
test of slot specialization. ZeroMemory differs from NoRead because LayerNorm
can have nonzero learned bias. Do not interpret high cosine alone as collapse.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', type=Path, default=Path.cwd())
    p.add_argument('--model-module', default='miniGPT_model_RNN', help='Module name or .py path, including filenames with parentheses.')
    p.add_argument('--loader-module', default='gutenberg_qa_loader')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, help='Validation JSONL file or directory; default config.GUTENBERG_QA_DIR/validation.')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--scan-records', type=int, default=10000, help='Reservoir sampling over this many records from shuffled files; 0 scans all.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, etc.')
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32', help='FP32 by default for stable diagnostics; use fp16 to match CUDA training AMP.')
    p.add_argument('--oracle-samples', type=int, default=32, help='Matched prompt-only/direct-evidence/evidence-only-memory tests; 0 disables.')
    p.add_argument('--greedy-samples', type=int, default=32, help='Free-running answer tests for Normal, NoRead, ShuffleHistory; 0 disables.')
    p.add_argument('--gradient-samples', type=int, default=1, help='Full recurrent backward probes, processed one example at a time; 0 disables.')
    p.add_argument('--recall-weight', type=float, help='Default config.RECALL_LOSS_WEIGHT, otherwise 10.')
    p.add_argument('--save-weight-examples', type=int, default=4)
    p.add_argument('--output', type=Path, help='New output directory; never overwrites an existing directory.')
    a = p.parse_args()
    for k in ('samples', 'batch_size'):
        if getattr(a, k) < 1:
            p.error(k + ' must be positive')
    for k in ('scan_records', 'oracle_samples', 'greedy_samples', 'gradient_samples', 'save_weight_examples'):
        if getattr(a, k) < 0:
            p.error(k + ' must be nonnegative')
    return a


def module_from(name, root):
    if name.endswith('.py'):
        path = Path(name)
        path = path if path.is_absolute() else root / path
        spec = importlib.util.spec_from_file_location('_audit_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name)


def resolve(path, root):
    return path if path.is_absolute() else root / path


def load_samples(path, loader, count, scan_limit, seed, width, vocab):
    """Sample raw records; deliberately never enable support-copy assistance.

    Call the installed loader's materialize_record for its exact W4 placement and
    answer offsets. A private seed per example avoids ordering/batching effects.
    One QA is chosen per sampled record, matching build_sequences semantics.
    """
    paths = sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('No QA JSONL files found at ' + str(path))
    rng = random.Random(seed)
    rng.shuffle(paths)
    reservoir, seen = [], 0
    for file in paths:
        with file.open(encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('schema_version') != 1 or r.get('block_size') != width or r.get('tokenizer') != 'gpt2':
                    raise ValueError(f'{file}:{line_no}: expected Gutenberg QA schema 1 / gpt2 / block_size={width}.')
                if len(r.get('tokens', [])) != 5 * width + 1 or not r.get('qas'):
                    raise ValueError(f'{file}:{line_no}: expected five windows and nonempty qas.')
                if r.get('_support_window') is not None:
                    raise ValueError('Dataset already carries assisted support metadata; use original unassisted records.')
                seen += 1
                item = (r, str(file), line_no)
                if len(reservoir) < count:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < count:
                        reservoir[j] = item
                if scan_limit and seen >= scan_limit:
                    break
        if scan_limit and seen >= scan_limit:
            break
    rng.shuffle(reservoir)
    result = []
    for i, (r, file, line_no) in enumerate(reservoir):
        tokens, start, end, q = loader.materialize_record(r, random.Random(seed + 1000003 * (i + 1)))
        source = q.get('source_window')
        lo, hi = q.get('local_start'), q.get('local_end')
        if type(source) is not int or not 0 <= source < 4:
            raise ValueError(f'{r.get("id")}: source_window must be 0..3 for historical evidence.')
        if q.get('support_window') is not None or '_support_tokens' in q:
            raise ValueError('Assisted example detected; use original records.')
        # Evidence coordinates are optional diagnostic metadata. Some dataset
        # versions use missing or cross-window spans. Do not guess their units,
        # clamp them, or reject otherwise valid QA examples.
        span_valid = type(lo) is int and type(hi) is int and 0 <= lo < hi <= width
        if len(tokens) != 5 * width + 1 or not 0 <= start < end <= width:
            raise ValueError('Invalid materialized sequence/answer span.')
        if min(tokens) < 0 or max(tokens) >= vocab:
            raise ValueError('Dataset token ID outside checkpoint vocabulary.')
        if tokens[4 * width + start + 1:4 * width + end + 1] != q['answer_tokens']:
            raise ValueError('Answer alignment mismatch between loader output and answer_tokens.')
        if tokens[:4 * width] != r['tokens'][:4 * width]:
            raise ValueError('Loader changed history; this audit requires unassisted, unmodified history.')
        result.append(dict(index=i, source_id=r.get('id'), file=file, line=line_no,
                           tokens=tokens, start=start, end=end, q=q, source=source,
                           lo=lo, hi=hi, span_valid=span_valid))
    if len(result) < 2:
        raise ValueError('At least two examples are required for a deranged history-shuffle control.')
    return result, dict(scanned_records=seen, selected_records=len(result), files=len(paths), scan_limit=scan_limit)


def scalar(x):
    return float(x.detach().cpu())


def vectors(x):
    """Diagnostics per example. Effective rank is of the uncentered slot matrix."""
    x = x.float()
    b, m, d = x.shape
    norms = x.norm(dim=-1)
    z = F.normalize(x, dim=-1)
    sim = z @ z.transpose(-1, -2)
    off = ~torch.eye(m, dtype=torch.bool, device=x.device)
    s = torch.linalg.svdvals(x)
    prob = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    rank = (-(prob * prob.clamp_min(1e-12).log()).sum(-1)).exp()
    rank = torch.where(s.sum(-1) > 1e-12, rank, torch.zeros_like(rank))
    out = []
    for i in range(b):
        out.append(dict(slot_norms=norms[i].tolist(), cosine_matrix=sim[i].tolist(),
                        mean_offdiag_cosine=scalar(sim[i][off].mean()) if m > 1 else None,
                        effective_rank=scalar(rank[i]), zero_slots=int((norms[i] < 1e-8).sum())))
    return out


def writer_rows(assign, weights, pooled, before, after, batch, window):
    a, w = assign.float(), weights.float()
    b, t, m = a.shape
    ent = -(a * a.clamp_min(1e-12).log()).sum(-1).mean(-1) / max(math.log(m), 1e-12)
    went = -(w * w.clamp_min(1e-12).log()).sum(1)
    eff = went.exp()
    pv, mv = vectors(pooled), vectors(after)
    out = []
    for i, sample in enumerate(batch):
        record = dict(index=sample['index'], window=window, source_window=sample['source'],
                      assignment_entropy_normalized=scalar(ent[i]),
                      assignment_mass=a[i].sum(0).tolist(),
                      effective_tokens_per_slot=eff[i].tolist(),
                      top_token_per_slot=w[i].argmax(0).tolist(),
                      top_token_weight_per_slot=w[i].max(0).values.tolist(),
                      pooled=pv[i], memory=mv[i],
                      relative_update_per_slot=((after[i]-before[i]).float().norm(dim=-1) /
                          before[i].float().norm(dim=-1).clamp_min(1e-8)).tolist())
        if sample['source'] == window and sample['span_valid']:
            mass = w[i, sample['lo']:sample['hi']].sum(0)
            baseline = (sample['hi']-sample['lo']) / t
            record.update(evidence_mass_per_slot=mass.tolist(), uniform_evidence_mass=baseline,
                          evidence_enrichment_per_slot=(mass / baseline).tolist())
        out.append(record)
    return out


def write_window(model, x, memory=None):
    # Exactly the supplied forward's independent writer path. Avoids unused
    # decoder logits during history-only probes. Does not detach gradients.
    emb = model.token_embedding_table(x)
    if memory is None:
        memory = model.init_memory(x.shape[0], emb.device, emb.dtype)
    features = model.memory_encoder(emb)
    pooled = model.memory_compressor(features, memory)
    return model.memory(pooled, memory), pooled, memory


@contextmanager
def reader_mode(model, mode):
    handles = []
    def hook(module, inputs, output):
        if mode == 'off':
            return torch.zeros_like(output)
        x, memory = inputs
        # Uniform weights in each head: concatenated mean(V) equals mean of
        # full projected values. Dropout is disabled by model.eval().
        return module.dropout_residual(module.proj(module.value(memory).mean(1, keepdim=True).expand(-1, x.shape[1], -1)))
    try:
        if mode != 'normal':
            for block in model.blocks:
                handles.append(block.memory_attention.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def batch_tensors(batch, width, device):
    t = torch.tensor([s['tokens'] for s in batch], dtype=torch.long, device=device)
    return t[:, :-1].reshape(-1, 5, width), t[:, 1:].reshape(-1, 5, width)


def scored_rows(model, x, y, memory, spans, mode='normal'):
    with reader_mode(model, mode):
        logits, _, _ = model(x, memory=memory, update_memory=False)
    rows = []
    for i, (start, end) in enumerate(spans):
        z = logits[i, start:end].float()
        target = y[i, start:end]
        logp = F.log_softmax(z, -1)
        correct = z.argmax(-1).eq(target)
        first_rank = int((z[0] > z[0, target[0]]).sum()) + 1
        rows.append(dict(tokens=end-start, ce_sum=scalar(-logp.gather(1, target[:, None]).sum()),
                         first_correct=int(correct[0]), exact_correct=int(correct.all()),
                         first_rank=first_rank, first_probability=scalar(logp[0, target[0]].exp()),
                         first_top5=int(first_rank <= 5)))
    return rows


def wilson(correct, n):
    if not n:
        return [None, None]
    p, z = correct / n, 1.96
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [mid-half, mid+half]


def summarize(rows):
    n = len(rows)
    tok = sum(r['tokens'] for r in rows)
    hit = sum(r['exact_correct'] for r in rows)
    return dict(samples=n, tokens=tok, ce=sum(r['ce_sum'] for r in rows)/tok,
                first_accuracy=sum(r['first_correct'] for r in rows)/n,
                exact_accuracy=hit/n, exact_wilson95=wilson(hit, n),
                first_top5=sum(r['first_top5'] for r in rows)/n,
                mean_first_rank=sum(r['first_rank'] for r in rows)/n,
                mean_first_probability=sum(r['first_probability'] for r in rows)/n)


def paired_delta(normal, other, seed):
    """Paired example bootstrap of exact accuracy difference; correlated books
    are not accounted for, so this interval is descriptive, not a formal claim.
    """
    a = {r['index']:r for r in normal}
    pairs = [(a[r['index']], r) for r in other if r['index'] in a]
    ds = [x['exact_correct']-y['exact_correct'] for x,y in pairs]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(ds, k=len(ds)))/len(ds) for _ in range(1000))
    return dict(samples=len(ds), normal_minus_condition_exact=sum(ds)/len(ds),
                paired_bootstrap95=[boot[24], boot[974]],
                normal_wins=sum(d > 0 for d in ds), condition_wins=sum(d < 0 for d in ds))


def greedy(model, prefix, answer, memory, mode='normal'):
    seq = prefix.clone()
    predicted = []
    with reader_mode(model, mode):
        for _ in answer:
            logits, _, _ = model(seq, memory=memory, update_memory=False)
            token = logits[:, -1].argmax(-1, keepdim=True)
            predicted.append(int(token[0, 0]))
            seq = torch.cat([seq, token], dim=1)
    return dict(predicted=predicted, target=answer, greedy_exact=int(predicted == answer))


def gradient_probe(model, sample, width, device, recall_weight):
    """Eval-mode FP32 gradient paths; no optimizer, no parameter .grad writes.
    State gradients include future recurrent dependence; pooled gradients probe
    each historical write. This tests THIS graph, not the user's training loop.
    """
    x, y = batch_tensors([sample], width, device)
    states, pooled, lm_terms = [], [], []
    memory = None
    captured = []
    h = model.memory_compressor.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        for w in range(5):
            logits, _, memory = model(x[:, w], memory=memory, update_memory=w < 4)
            lm_terms.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, w].reshape(-1)))
            if w < 4:
                states.append(memory)
        pooled = captured
        s,e = sample['start'],sample['end']
        recall = F.cross_entropy(logits[:,s:e].reshape(-1, logits.shape[-1]), y[:,-1,s:e].reshape(-1))
        lm = sum(lm_terms)/len(lm_terms)
        named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        targets = [p for _,p in named] + states + pooled
        rg = torch.autograd.grad(recall, targets, retain_graph=True, allow_unused=True)
        lg = torch.autograd.grad(lm, [p for _,p in named], allow_unused=True)
        groups = defaultdict(lambda: [0.,0.,0.,0])
        for (name,p),r,l in zip(named, rg, lg):
            if name.startswith('memory_encoder.'):
                group='encoder'
            elif name.startswith('memory_compressor.'):
                group='compressor'
            elif name.startswith('memory.'):
                group='gru'
            elif '.memory_attention.' in name or '.memory_ln.' in name or '.ln2.' in name:
                group='reader'
            elif name.startswith('token_embedding_table.') or name.startswith('lm_head.'):
                group='shared_embedding'
            else:
                group='decoder'
            g=groups[group]
            if r is not None: g[0] += scalar(r.float().square().sum())
            if l is not None: g[1] += scalar(l.float().square().sum())
            if r is not None and l is not None: g[2] += scalar((r.float()*l.float()).sum())
            g[3] += int(r is not None)
        result={}
        for group,(rr,ll,dot,connected) in groups.items():
            nr,nl=math.sqrt(rr),math.sqrt(ll)
            result[group]=dict(recall_grad_norm=nr, lm_grad_norm=nl,
                weighted_recall_grad_norm=recall_weight*nr,
                weighted_recall_to_lm_ratio=recall_weight*nr/nl if nl else None,
                cosine_recall_vs_lm=dot/(nr*nl) if nr and nl else None,
                recall_connected_parameter_tensors=connected)
        tail=rg[len(named):]
        norm=lambda g: scalar(g.float().norm()) if g is not None else None
        return dict(index=sample['index'], source_window=sample['source'], recall_ce=scalar(recall),
                    lm_ce=scalar(lm), groups=result,
                    state_gradient_norms=[norm(g) for g in tail[:4]],
                    pooled_write_gradient_norms=[norm(g) for g in tail[4:]])
    finally:
        h.remove()


def pooling_metadata(model, state):
    c = model.memory_compressor
    return dict(routing_passes=getattr(c, 'routing_iterations', 1),
                pooling_heads=getattr(c, 'pooling_heads', 1),
                residual_multihead=hasattr(c, 'multihead_out'),
                checkpoint_has_multihead=any(k.startswith('memory_compressor.multihead_') for k in state),
                encoder_blocks=len(model.memory_encoder.blocks) if hasattr(model.memory_encoder, 'blocks') else 1)


def main():
    args = arguments()
    global torch, F
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit('Run this script in the same Python environment as training (PyTorch is required).') from exc
    import torch.nn.functional as F
    root = args.project_dir.resolve()
    sys.path.insert(0, str(root))
    config = importlib.import_module('config')
    model_module = module_from(args.model_module, root)
    loader = module_from(args.loader_module, root)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.precision != 'fp32' and device.type != 'cuda':
        raise ValueError('Use --precision fp32 for CPU diagnostics.')
    amp = lambda: torch.autocast(device_type='cuda', dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16) if args.precision != 'fp32' else nullcontext()
    checkpoint = resolve(args.checkpoint, root).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must contain a parameter dictionary.')
    state = payload.get('model_state_dict', payload.get('state_dict', payload))
    if not isinstance(state, dict):
        raise ValueError('Expected state_dict, model_state_dict, or a plain parameter dictionary.')
    for prefix in ('module.', '_orig_mod.', 'module.'):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]:v for k,v in state.items()}
    vocab = state['token_embedding_table.weight'].shape[0]
    model = model_module.MiniGPTRecall(vocab_size=vocab)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError('Checkpoint/config/model mismatch. Use the original model and config, including slot count and width; no weights were skipped.\n' + str(exc)) from exc
    writer_architecture = pooling_metadata(model, state)
    model = model.to(device).float().eval()
    model.requires_grad_(True)
    width = int(config.block_size)
    m, dm = model.memory.memory_slots, model.memory.memory_dims
    dataset = resolve(args.dataset, root) if args.dataset else resolve(Path(getattr(config,'GUTENBERG_QA_DIR','./gutenberg_qa')),root)/'validation'
    samples, sampling = load_samples(dataset, loader, args.samples, args.scan_records, args.seed, width, vocab)
    output = args.output or (root / ('memory_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    n = len(samples)
    unsupported_spans = [dict(index=s['index'],source_id=s['source_id'],
        source_window=s['source'],local_start=s['lo'],local_end=s['hi'])
        for s in samples if not s['span_valid']]
    if unsupported_spans:
        print(f'Note: {len(unsupported_spans)}/{n} examples have unsupported evidence spans. '
              'Keeping them for recall/slot/gradient tests; skipping their span-based pooling and oracle checks. '
              f'First example: {unsupported_spans[0]}', flush=True)
    rows, writers, snapshots = [], [], []
    memories = []
    writer_forward_max_difference = None
    print(f'Auditing {n} examples | slots={m} x {dm} | {device} | {args.precision}', flush=True)

    def add(condition, batch, records):
        for sample, row in zip(batch, records):
            rows.append(dict(condition=condition,index=sample['index'],source_window=sample['source'],
                             answer_length=sample['end']-sample['start'],**row))

    with torch.no_grad():
        for begin in range(0,n,args.batch_size):
            batch = samples[begin:begin+args.batch_size]
            x,y = batch_tensors(batch,width,device)
            spans=[(s['start'],s['end']) for s in batch]
            memory=None
            stage=[]
            with amp():
                for w in range(4):
                    memory, pooled, previous = write_window(model,x[:,w],memory)
                    if begin == 0 and w == 0:
                        _, _, reference = model(x[:, w], memory=previous, update_memory=True)
                        writer_forward_max_difference = scalar((reference-memory).float().abs().max())
                        tol = 1e-5 if args.precision == 'fp32' else 1e-3
                        if not torch.allclose(reference, memory, atol=tol, rtol=tol):
                            raise RuntimeError('Writer-only shortcut differs from model.forward. This model version needs a different adapter.')
                    stage.append(memory.clone())
                    c=model.memory_compressor
                    writers.extend(writer_rows(c.last_assignments,c.last_weights,pooled,previous,memory,batch,w))
                    for i,s in enumerate(batch):
                        if s['index'] < args.save_weight_examples:
                            snapshots.append(dict(index=s['index'],window=w,token_ids=x[i,w].tolist(),
                                assignments=c.last_assignments[i].float().tolist(),
                                pooling_weights=c.last_weights[i].float().tolist(),
                                head_assignments=(c.last_head_assignments[i].float().tolist()
                                    if getattr(c, 'last_head_assignments', None) is not None else None),
                                head_pooling_weights=(c.last_head_weights[i].float().tolist()
                                    if getattr(c, 'last_head_weights', None) is not None else None)))
                    add(f'AfterW{w}',batch,scored_rows(model,x[:,-1],y[:,-1],memory,spans))
                memories.extend(memory.detach().cpu().unbind(0))
                add('Normal',batch,scored_rows(model,x[:,-1],y[:,-1],memory,spans))
                add('NoRead',batch,scored_rows(model,x[:,-1],y[:,-1],memory,spans,'off'))
                add('ZeroMemory',batch,scored_rows(model,x[:,-1],y[:,-1],torch.zeros_like(memory),spans))
                add('UniformRead',batch,scored_rows(model,x[:,-1],y[:,-1],memory,spans,'uniform'))
                mean=memory.mean(1,keepdim=True).expand_as(memory)
                add('MeanSlots',batch,scored_rows(model,x[:,-1],y[:,-1],mean,spans))
                add('PermuteSlots',batch,scored_rows(model,x[:,-1],y[:,-1],memory.flip(1),spans))
                for slot in range(m):
                    dropped=memory.clone(); dropped[:,slot]=0
                    add(f'ZeroSlot{slot}',batch,scored_rows(model,x[:,-1],y[:,-1],dropped,spans))
                    repeated=memory[:,slot:slot+1].expand_as(memory)
                    add(f'RepeatSlot{slot}',batch,scored_rows(model,x[:,-1],y[:,-1],repeated,spans))
                source_x=torch.stack([x[i,s['source']] for i,s in enumerate(batch)])
                source_mem,_,_=write_window(model,source_x)
                add('SourceWindowOnly',batch,scored_rows(model,x[:,-1],y[:,-1],source_mem,spans))
                immediate=torch.stack([stage[s['source']][i] for i,s in enumerate(batch)])
                add('AtEvidenceWindow',batch,scored_rows(model,x[:,-1],y[:,-1],immediate,spans))
            print(f'Writer / slot probes: {min(begin+len(batch),n)}/{n}',flush=True)

        # Global derangement, including a partial final batch: no example gets
        # its own memory. Other examples may nevertheless share a fact/history.
        order=list(range(n)); random.Random(args.seed+7).shuffle(order)
        donor={order[i]:order[(i+1)%n] for i in range(n)}
        for begin in range(0,n,args.batch_size):
            batch=samples[begin:begin+args.batch_size]
            x,y=batch_tensors(batch,width,device)
            memory=torch.stack([memories[donor[s['index']]] for s in batch]).to(device)
            with amp():
                add('ShuffleHistory',batch,scored_rows(model,x[:,-1],y[:,-1],memory,[(s['start'],s['end']) for s in batch]))

        greedy_rows=[]
        for s in samples[:args.greedy_samples]:
            x,y=batch_tensors([s],width,device)
            prefix=x[:,-1,:s['start']+1]
            answer=y[0,-1,s['start']:s['end']].tolist()
            with amp():
                for label,mem,mode in [('Normal',memories[s['index']],'normal'),('NoRead',memories[s['index']],'off'),('ShuffleHistory',memories[donor[s['index']]],'normal')]:
                    greedy_rows.append(dict(index=s['index'],condition=label,**greedy(model,prefix,answer,mem[None].to(device),mode)))
            print(f'Greedy probes: {s["index"]+1}/{min(n,args.greedy_samples)}',flush=True)

        # These matched controls remove final-window filler and use the SAME
        # q.prompt_tokens and gold targets. Direct context never includes the
        # gold answer except as normal teacher-forced preceding answer tokens.
        oracle_skipped=[]
        for s in samples[:args.oracle_samples]:
            q=s['q']; answer=q['answer_tokens']; prompt=q['prompt_tokens']
            if not s['span_valid']:
                oracle_skipped.append(dict(index=s['index'],reason='unsupported evidence coordinates',
                    local_start=s['lo'],local_end=s['hi']))
                continue
            evidence=s['tokens'][s['source']*width+s['lo']:s['source']*width+s['hi']]
            if len(evidence)+1+len(prompt)+len(answer)>width+1:
                oracle_skipped.append(dict(index=s['index'],reason='evidence + prompt + answer exceeds context; no truncation'))
                continue
            with amp():
                ex=torch.tensor([evidence],dtype=torch.long,device=device)
                em,_,_=write_window(model,ex)
                full=memories[s['index']][None].to(device)
                for label,context,mem,mode in [('OraclePromptOnly',prompt,full,'off'),
                    ('OracleDirectEvidence',evidence+[198]+prompt,full,'off'),
                    ('OracleEvidenceOnlyMemory',prompt,em,'normal'),
                    ('OracleFullMemory',prompt,full,'normal')]:
                    t=torch.tensor([context+answer],dtype=torch.long,device=device)
                    span=[(len(context)-1,len(context)+len(answer)-1)]
                    add(label,[s],scored_rows(model,t[:,:-1],t[:,1:],mem,span,mode))
            print(f'Evidence controls: {s["index"]+1}/{min(n,args.oracle_samples)}',flush=True)

    gradients=[]
    weight=args.recall_weight if args.recall_weight is not None else float(getattr(config,'RECALL_LOSS_WEIGHT',10.))
    for s in samples[:args.gradient_samples]:
        with torch.enable_grad():
            gradients.append(gradient_probe(model,s,width,device,weight))
        print(f'Gradient probes: {len(gradients)}/{min(n,args.gradient_samples)}',flush=True)

    grouped=defaultdict(list)
    for row in rows: grouped[row['condition']].append(row)
    conditions={k:summarize(v) for k,v in grouped.items()}
    paired={k:paired_delta(grouped['Normal'],v,args.seed) for k,v in grouped.items() if k!='Normal' and not k.startswith('Oracle')}
    by_source={str(w):{k:summarize([r for r in v if r['source_window']==w]) for k,v in grouped.items() if any(r['source_window']==w for r in v)} for w in range(4)}
    writer_summary={}
    for w in range(4):
        sub=[r for r in writers if r['window']==w]
        avg=lambda xs:sum(xs)/len(xs) if xs else None
        writer_summary[f'W{w}']=dict(assignment_entropy_normalized=avg([r['assignment_entropy_normalized'] for r in sub]),
            effective_tokens_per_slot=avg([v for r in sub for v in r['effective_tokens_per_slot']]),
            pooled_offdiag_cosine=avg([r['pooled']['mean_offdiag_cosine'] for r in sub if r['pooled']['mean_offdiag_cosine'] is not None]),
            memory_offdiag_cosine=avg([r['memory']['mean_offdiag_cosine'] for r in sub if r['memory']['mean_offdiag_cosine'] is not None]),
            memory_effective_rank=avg([r['memory']['effective_rank'] for r in sub]),
            max_evidence_enrichment=avg([max(r['evidence_enrichment_per_slot']) for r in sub if 'evidence_enrichment_per_slot' in r]))
    manifest=[dict(index=s['index'],source_id=s['source_id'],file=s['file'],line=s['line'],source_window=s['source'],
                   evidence_span=[s['lo'],s['hi']],evidence_span_supported=s['span_valid'],
                   question=s['q'].get('question'),answer=s['q'].get('answer'),
                   answer_span=[s['start'],s['end']],tokens_sha256=hashlib.sha256(json.dumps(s['tokens']).encode()).hexdigest(),
                   shuffle_donor=donor[s['index']]) for s in samples]
    warnings=[
        'Teacher-forced exact accuracy is not free-running exact accuracy.',
        'Stage probes reuse the same final QA window, including filler and teacher-forced earlier answer tokens.',
        'Evidence metadata marks the provided span; other history may also contain relevant facts.',
        'SourceWindowOnly changes history length/initial state; oracle evidence-only memory also changes window length.',
        'ZeroSlot and repeated/mean slots are distribution shifts, not equivalent to retraining with fewer slots.',
        'Bootstrap/Wilson intervals ignore within-book dependence and are descriptive.',
        'Gradient probes use a recreated uninterrupted FP32 eval graph; they do not verify detach/optimizer behavior in the training script.',
        'No capacity ceiling can be established without a matched retrained slot-count comparison.',
        'Assignment entropy, effective tokens, and evidence enrichment describe the original pooling branch only; pooled geometry describes the combined candidate. Per-head maps are saved in pooling_examples.json.',
        'Pooling diagnostics are computed before the GRU; memory diagnostics after it. Effective rank is uncentered.',
        'Dataset sampling uses raw validation records and deliberately ignores support-copy and evidence-window config filters.',
        'Deterministic algorithms use warn_only; backend nondeterminism may still be reported by PyTorch.',
    ]
    if args.scan_records and sampling['scanned_records']>=args.scan_records:
        warnings.append('Record scan cap reached; sample is representative only of the scanned prefix of shuffled files. Use --scan-records 0 for the full dataset.')
    if unsupported_spans:
        warnings.append(f'{len(unsupported_spans)} examples have unsupported evidence spans. They remain in general tests but lack evidence-enrichment and oracle checks. Retention timing uses source_window as supplied; cross-window evidence may not be complete at that point. See unsupported_evidence_spans for original coordinates.')
    permutation_error=max(abs(a['ce_sum']/a['tokens']-b['ce_sum']/b['tokens']) for a,b in zip(grouped['Normal'],grouped['PermuteSlots']))
    if permutation_error>0.001:
        warnings.append('Slot-permutation CE difference exceeds 0.001; investigate numerics, nondeterminism, or a changed reader implementation.')
    summary=dict(metadata=dict(checkpoint=str(checkpoint),model_module=args.model_module,torch_version=torch.__version__,
        device=str(device),precision=args.precision,seed=args.seed,block_size=width,memory_slots=m,memory_slot_dims=dm,
        writer_architecture=writer_architecture,checkpoint_tokens_seen=payload.get("tokens_seen"),
        recall_weight=weight,sampling=sampling,writer_forward_max_difference=writer_forward_max_difference,
        permutation_max_sample_ce_difference=permutation_error),
        conditions=conditions,paired_vs_normal=paired,by_source_window=by_source,writer_summary=writer_summary,
        greedy=greedy_rows,gradients=gradients,oracle_skipped=oracle_skipped,
        unsupported_evidence_spans=unsupported_spans,manifest=manifest,warnings=warnings)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (output/'pooling_examples.json').write_text(json.dumps(snapshots,allow_nan=False),encoding='utf-8')
    with (output/'writer_per_sample.jsonl').open('w',encoding='utf-8') as f:
        for row in writers:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (output/'per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=['# Memory bottleneck audit','',f'Checkpoint: `{checkpoint}`',f'Examples: {n}; memory: {m} × {dm}; precision: {args.precision}.','',
        '## Answer metrics','', '| Condition | n | First | Exact (teacher-forced) | Answer CE |','|---|---:|---:|---:|---:|']
    for k,v in conditions.items():report.append(f'| {k} | {v["samples"]} | {v["first_accuracy"]:.2%} | {v["exact_accuracy"]:.2%} | {v["ce"]:.4f} |')
    report+=['','## Writer diagnostics','', '| Window | Assignment entropy / log(M) | Effective tokens/slot | Pooled cosine | Memory cosine | Memory rank | Best evidence enrichment |', '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in writer_summary.items():report.append('| '+k+' | '+' | '.join('N/A' if x is None else f'{x:.3f}' for x in v.values())+' |')
    if greedy_rows:
        report+=['','## Greedy generation','', 'Fixed gold answer length; no gold answer tokens fed back.','']
        for label in ('Normal','NoRead','ShuffleHistory'):
            rr=[r for r in greedy_rows if r['condition']==label]
            report.append(f'- {label}: {sum(r["greedy_exact"] for r in rr)/len(rr):.2%} exact, n={len(rr)}.')
    report+=['','## Retention by evidence source','',
        'Same final QA input queried with memory immediately after the evidence window and after all history. Later text is not guaranteed irrelevant.','',
        '| Evidence window | n | At evidence: exact | Final: exact | Final minus immediate |',
        '|---|---:|---:|---:|---:|']
    for w, group in by_source.items():
        if 'Normal' in group:
            a,b=group['AtEvidenceWindow'],group['Normal']
            report.append(f'| W{w} | {b["samples"]} | {a["exact_accuracy"]:.2%} | {b["exact_accuracy"]:.2%} | {100*(b["exact_accuracy"]-a["exact_accuracy"]):+.2f} pp |')
    if gradients:
        report+=['','## Gradient probes','', 'FP32, evaluation mode, one example at a time; gradients are not optimizer updates.','']
        for probe in gradients:
            report += [f'### Example {probe["index"]} (evidence W{probe["source_window"]})','',
                '| Group | Recall norm | LM norm | Weighted recall / LM | Gradient cosine |',
                '|---|---:|---:|---:|---:|']
            for group,g in probe['groups'].items():
                values=[g[k] for k in ('recall_grad_norm','lm_grad_norm','weighted_recall_to_lm_ratio','cosine_recall_vs_lm')]
                report.append('| '+group+' | '+' | '.join('N/A' if v is None else f'{v:.4g}' for v in values)+' |')
            report+=['',f'W0..W3 state gradient norms: `{probe["state_gradient_norms"]}`.',
                f'W0..W3 pooled-write gradient norms: `{probe["pooled_write_gradient_norms"]}`.','']
    report+=['','## How to interpret','',
        '- Assignment entropy near 1 plus effective token count near block_size suggests diffuse pooling. Examine pooled vectors and functional interventions before concluding collapse.',
        '- High slot similarity alone is insufficient. MeanSlots, RepeatSlot, and UniformRead test whether distinctions affect answers.',
        '- Normal minus AtEvidenceWindow below zero indicates worse recall after subsequent writes; inspect source-window groups and paired confidence intervals in summary.json.',
        '- SourceWindowOnly improving over Normal suggests interference, but also changes the starting state.',
        '- Compare OracleDirectEvidence with OraclePromptOnly on the same subset. If direct evidence helps but evidence-only memory fails, investigate the write/read interface.',
        '- If OracleEvidenceOnlyMemory works better than OracleFullMemory, full-history compression/interference is implicated.',
        '- PermuteSlots should preserve answers: this reader has no slot-position signal. Invariance is expected even for specialized slots.',
        '- Gradient norms of None mean disconnected; zero means connected but locally zero. Small magnitudes alone do not prove vanishing gradients.',
        '- Negative recall/LM gradient cosine suggests local objective conflict. Inspect several examples before changing loss weights.',
        '- Compare 4 vs 16 slots with matched retraining only after examining writer and retention results.','',
        '## Limits / checks','']+['- '+w for w in warnings]
    report+=['',f'Oracle examples skipped (fit or unsupported coordinates): {len(oracle_skipped)}. Gradient examples: {len(gradients)}.',
             'Detailed per-source results, paired intervals, gradient groups, and sample provenance are in summary.json.']
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'+ '\n'.join(report[:8+len(conditions)]),flush=True)
    print(f'\nSaved reports to {output}',flush=True)


if __name__ == '__main__':
    main()
