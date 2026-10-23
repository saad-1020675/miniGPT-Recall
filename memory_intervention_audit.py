#!/usr/bin/env python3
"""Controlled routing / GRU / reading audit for MiniGPTRecall.

Place beside config.py, miniGPT_model_RNN.py and gutenberg_qa_loader.py.
python memory_intervention_audit.py --checkpoint checkpoints_gutenberg_qa/best.pt

Standalone: does not require memory_bottleneck_audit.py. Read-only checkpoint
loading; no training, optimizer or checkpoint writes. Raw validation QA records
are sampled WITHOUT assistance or source-window training filters. Metrics are
teacher-forced. Outputs: report.md, summary.json, per_sample.csv, manifest.json.
Use --continue-writes to additionally propagate reset interventions through
remaining history windows. Separate banks are reader-only probes, never fed to
the fixed-size writer. Evidence pooling uses privileged span annotations.
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

def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--project-dir', type=Path, default=Path.cwd())
    p.add_argument('--model-module', default='miniGPT_model_RNN')
    p.add_argument('--loader-module', default='gutenberg_qa_loader')
    p.add_argument('--dataset', type=Path)
    p.add_argument('--samples', type=int, default=256, help='Total records; source-window counts are reported, not forced equal.')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--scan-records', type=int, default=10000, help='0 scans all records; use this for broader book coverage.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--precision', choices=['fp32','fp16','bf16'], default='fp32')
    p.add_argument('--continue-writes', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.samples < 2 or a.batch_size < 1 or a.scan_records < 0:
        p.error('samples >= 2, batch-size >= 1, scan-records >= 0 required')
    return a


def compare_rows(baseline, intervention, seed):
    """Positive delta means the intervention improves over its matched baseline."""
    base = {r['index']:r for r in baseline}
    pairs = [(base[r['index']], r) for r in intervention if r['index'] in base]
    if not pairs:
        return None
    ds = [b['exact_correct']-a['exact_correct'] for a,b in pairs]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(ds, k=len(ds)))/len(ds) for _ in range(1000))
    tokens = sum(a['tokens'] for a,b in pairs)
    return dict(samples=len(ds), baseline_exact=sum(a['exact_correct'] for a,b in pairs)/len(ds),
                intervention_exact=sum(b['exact_correct'] for a,b in pairs)/len(ds),
                rescued=sum(d > 0 for d in ds), broken=sum(d < 0 for d in ds),
                delta_exact=sum(ds)/len(ds), bootstrap95=[boot[24],boot[974]],
                ce_improvement=sum(a['ce_sum']-b['ce_sum'] for a,b in pairs)/tokens)


def masked_pool(compressor, features, assignments, spans, head_assignments=None):
    """Keep full-window encoder features, but pool annotated evidence only."""
    positions = torch.arange(features.shape[1], device=features.device)[None, :]
    lo = torch.tensor([s['lo'] for s in spans], device=features.device)[:, None]
    hi = torch.tensor([s['hi'] for s in spans], device=features.device)[:, None]
    restricted = assignments * ((positions >= lo) & (positions < hi)).unsqueeze(-1)
    weights = restricted / restricted.sum(1, keepdim=True).clamp_min(1e-8)
    values = compressor.value(features)
    pooled = weights.to(values.dtype).transpose(1, 2) @ values
    if hasattr(compressor, 'multihead_out'):
        if head_assignments is None:
            raise RuntimeError('Missing captured multi-head routing for masked pooling')
        B, T, _ = features.shape
        H = compressor.pooling_heads
        M = assignments.shape[-1]
        mask = ((positions >= lo) & (positions < hi))[:, None, :, None]
        restricted_heads = head_assignments * mask
        head_weights = restricted_heads / restricted_heads.sum(-2, keepdim=True).clamp_min(1e-8)
        head_values = compressor.multihead_value(features).reshape(
            B, T, H, compressor.pooling_value_dims).transpose(1, 2)
        head_pooled = head_weights.to(head_values.dtype).transpose(-2, -1) @ head_values
        head_pooled = head_pooled.transpose(1, 2).contiguous().reshape(B, M, compressor.memory_dims)
        pooled = pooled + compressor.multihead_out(head_pooled)
    return pooled


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
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit('Run in your training environment: PyTorch is required.') from exc
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
    if device.type != 'cuda' and args.precision != 'fp32':
        raise ValueError('Use fp32 on CPU.')
    amp = lambda: (torch.autocast('cuda', dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16)
                   if args.precision != 'fp32' else nullcontext())
    checkpoint = resolve(args.checkpoint, root).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload.get('model_state_dict', payload.get('state_dict', payload))
    for prefix in ('module.', '_orig_mod.', 'module.'):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]:v for k,v in state.items()}
    vocab = state['token_embedding_table.weight'].shape[0]
    model = model_module.MiniGPTRecall(vocab_size=vocab)
    model.load_state_dict(state, strict=True)
    writer_architecture = pooling_metadata(model, state)
    model = model.to(device).float().eval()
    width = int(config.block_size)
    dataset = (resolve(args.dataset, root) if args.dataset else
               resolve(Path(getattr(config,'GUTENBERG_QA_DIR','./gutenberg_qa')),root)/'validation')
    samples, sampling = load_samples(dataset, loader, args.samples, args.scan_records, args.seed, width, vocab)
    output = resolve(args.output, root) if args.output else root / ('memory_intervention_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    manifest = [dict(index=s['index'], source_id=s['source_id'], file=s['file'], line=s['line'],
                     source_window=s['source'], question=s['q']['question'], answer=s['q']['answer'],
                     answer_span=[s['start'],s['end']], evidence_span=[s['lo'],s['hi']],
                     evidence_span_supported=s['span_valid'],
                     tokens_sha256=hashlib.sha256(json.dumps(s['tokens'],separators=(',',':')).encode()).hexdigest()) for s in samples]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows = []
    checks = dict(writer_max_abs_difference=0.0, duplicate_bank_max_ce_difference=0.0)
    tol = 1e-5 if args.precision == 'fp32' else 2e-3
    # Group by evidence window so every intervention in a batch occurs at the
    # same recurrent step. Original sample indices and sample seeds are retained.
    with torch.inference_mode():
        for source in range(4):
            group = [s for s in samples if s['source'] == source]
            for begin in range(0,len(group),args.batch_size):
                batch = group[begin:begin+args.batch_size]
                x,y = batch_tensors(batch,width,device)
                spans = [(s['start'],s['end']) for s in batch]
                def score(name, memory, chosen=None, mode='normal'):
                    ids = list(range(len(batch))) if chosen is None else chosen
                    records = scored_rows(model,x[ids,-1],y[ids,-1],memory,
                                          [spans[i] for i in ids],mode)
                    for i,row in zip(ids,records):
                        rows.append(dict(condition=name,index=batch[i]['index'],source_window=source,**row))
                    return records
                with amp():
                    h = model.init_memory(len(batch),device,model.token_embedding_table.weight.dtype)
                    for w in range(source):
                        h,_,_ = write_window(model,x[:,w],h)
                    zero = torch.zeros_like(h)
                    features = model.memory_encoder(model.token_embedding_table(x[:,source]))
                    c = model.memory_compressor
                    uh = c(features,h)
                    u0 = c(features,zero)
                    assignments0 = c.last_assignments.clone()
                    head_assignments0 = getattr(c, 'last_head_assignments', None)
                    head_assignments0 = head_assignments0.clone() if head_assignments0 is not None else None
                    # All-token reconstruction must reproduce BOTH pooling branches.
                    reconstructed = masked_pool(c, features, assignments0,
                        [{'lo': 0, 'hi': features.shape[1]} for _ in batch], head_assignments0)
                    if not torch.allclose(reconstructed, u0, atol=tol, rtol=tol):
                        raise RuntimeError('Masked-pooling adapter fails full-window reconstruction')
                    e = model.memory(u0,zero)
                    states = {'StopLaterWrites':model.memory(uh,h),
                              'ResetRouting':model.memory(u0,h),
                              'ResetGRU':model.memory(uh,zero),
                              'ResetBoth':e}
                    # Validate the shortcut for EVERY batch, including nonzero H.
                    _,_,reference = model(x[:,source],memory=h,update_memory=True)
                    diff = scalar((reference-states['StopLaterWrites']).abs().max())
                    checks['writer_max_abs_difference'] = max(checks['writer_max_abs_difference'],diff)
                    if not torch.allclose(reference,states['StopLaterWrites'],atol=tol,rtol=tol):
                        raise RuntimeError('Writer adapter does not match model.forward; use the supplied model architecture.')
                    normal = states['StopLaterWrites']
                    for w in range(source+1,4):
                        normal,_,_ = write_window(model,x[:,w],normal)
                    score('Normal',normal)
                    score('NoRead',normal,mode='off')
                    for name,state in states.items():
                        score(name,state)
                    score('SeparateBanks',torch.cat([h,e],dim=1))
                    isolated = score('IsolatedBank',e)
                    duplicate = score('DuplicateBank',torch.cat([e,e],dim=1))
                    cd = max(abs(a['ce_sum']/a['tokens']-b['ce_sum']/b['tokens']) for a,b in zip(isolated,duplicate))
                    checks['duplicate_bank_max_ce_difference'] = max(checks['duplicate_bank_max_ce_difference'],cd)
                    valid = [i for i,s in enumerate(batch) if s['span_valid']]
                    if valid:
                        pooled = masked_pool(c,features[valid],assignments0[valid],[batch[i] for i in valid],
                                             head_assignments0[valid] if head_assignments0 is not None else None)
                        score('EvidenceOnlyPooling',model.memory(pooled,zero[valid]),chosen=valid)
                    if args.continue_writes:
                        for name in ('ResetRouting','ResetGRU','ResetBoth'):
                            state = states[name]
                            for w in range(source+1,4):
                                state,_,_ = write_window(model,x[:,w],state)
                            score(name+'ThenContinue',state)
                print(f'W{source}: {min(begin+args.batch_size,len(group))}/{len(group)} done',flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['condition']].append(row)
    contrasts = [('LaterUpdates','Normal','StopLaterWrites'),
                 ('Routing','StopLaterWrites','ResetRouting'),
                 ('GRUIncorporation','StopLaterWrites','ResetGRU'),
                 ('RoutingAndGRU','StopLaterWrites','ResetBoth'),
                 ('SeparateStorage','StopLaterWrites','SeparateBanks'),
                 ('RemoveOldBank','SeparateBanks','IsolatedBank'),
                 ('DuplicateBankSanity','IsolatedBank','DuplicateBank'),
                 ('EvidenceSelection','IsolatedBank','EvidenceOnlyPooling')]
    if args.continue_writes:
        contrasts += [(name+'Continued','Normal',name+'ThenContinue') for name in ('ResetRouting','ResetGRU','ResetBoth')]
    comparisons = {}
    for label,base,intervention in contrasts:
        if not grouped[intervention]:
            continue
        comparisons[label] = dict(baseline=base,intervention=intervention,
            overall=compare_rows(grouped[base],grouped[intervention],args.seed),
            by_source_window={str(w):compare_rows([r for r in grouped[base] if r['source_window']==w],
                 [r for r in grouped[intervention] if r['source_window']==w],args.seed) for w in range(4)})
    warnings = [
        'Interventions measure sensitivity, not additive causal shares; resets and doubled banks change the state distribution.',
        'All main mechanism contrasts stop later writes; compare continued variants to Normal only.',
        'SeparateBanks doubles read capacity and changes attention normalization. DuplicateBank is a numerical control, not a capacity-matched learned baseline.',
        'EvidenceOnlyPooling masks and renormalizes the captured final-pass assignments in every pooling branch; intermediate routing iterations remain unmasked.',
        'EvidenceOnlyPooling uses privileged annotations and full-window contextual encoder features; it does not isolate encoder contamination.',
        'Unsupported evidence spans remain in main tests and are skipped only for EvidenceOnlyPooling.',
        'NoRead is a memory ablation, not a direct-evidence decoder ceiling test.',
        'The model may have failed to encode an answer even when later updates cause no further loss.',
        'Raw unassisted validation ignores training support copies and evidence-window filters.',
        'Bootstrap intervals resample examples and ignore within-book dependence; use descriptively.',
        'Samples are not stratified; inspect window counts and use a larger sample/full scan for coverage.',
        'Teacher-forced exact accuracy is not free-running generation accuracy.'
    ]
    if checks['duplicate_bank_max_ce_difference'] > (1e-4 if args.precision=='fp32' else 0.03):
        warnings.append('Duplicate-bank numerical invariance check exceeded tolerance; investigate reader implementation/precision before interpreting bank tests.')
    meta = dict(checkpoint=str(checkpoint), checkpoint_tokens_seen=payload.get('tokens_seen'),
                checkpoint_step=payload.get('step'), memory_slots=model.memory.memory_slots,
                memory_slot_dims=model.memory.memory_dims, block_size=width,
                seed=args.seed,precision=args.precision,device=str(device),torch_version=torch.__version__,
                sampling=sampling,continue_writes=args.continue_writes)
    meta['writer_architecture'] = writer_architecture
    meta['evidence_pooling_semantics'] = 'final_pass_mask_all_branches'
    summary = dict(metadata=meta,checks=checks,
                   conditions={k:summarize(v) for k,v in grouped.items() if v},
                   by_source_window={str(w):{k:summarize([r for r in v if r['source_window']==w])
                       for k,v in grouped.items() if any(r['source_window']==w for r in v)} for w in range(4)},
                   comparisons=comparisons,skipped_evidence_pooling=[s['index'] for s in samples if not s['span_valid']],warnings=warnings)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (output/'per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report = ['# Memory intervention audit','',f'Checkpoint: `{checkpoint}`',
              f"Tokens seen stored in checkpoint: {meta['checkpoint_tokens_seen']}",'',
              '## Condition results','', '| Condition | n | Exact | Answer CE |','|---|---:|---:|---:|']
    for name,metrics in summary['conditions'].items():
        report.append(f"| {name} | {metrics['samples']} | {100*metrics['exact_accuracy']:.2f}% | {metrics['ce']:.4f} |")
    report += ['', '## Intervention ranking', '',
               'Ranked by net paired accuracy benefit. Different baselines are explicit; benefits must not be added together. Evidence-selection comparisons include valid spans only.', '',
               '| Test | Baseline → intervention | n | Rescued | Broken | Net pp | 95% interval pp | CE improvement |',
               '|---|---|---:|---:|---:|---:|---|---:|']
    for label,v in sorted(comparisons.items(),key=lambda item:item[1]['overall']['delta_exact'],reverse=True):
        d=v['overall'];lo,hi=d['bootstrap95']
        report.append(f"| {label} | {v['baseline']} → {v['intervention']} | {d['samples']} | {d['rescued']} | {d['broken']} | {100*d['delta_exact']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {d['ce_improvement']:+.4f} |")
    for w,conditions in summary['by_source_window'].items():
        report += ['',f'## W{w}', '', '| Condition | n | Exact | Answer CE |','|---|---:|---:|---:|']
        for name,d in conditions.items():
            report.append(f"| {name} | {d['samples']} | {100*d['exact_accuracy']:.2f}% | {d['ce']:.4f} |")
    report += ['', '## Interpretation', '',
               'Routing rescue implicates memory-conditioned pooling. GRU reset rescue implicates incorporation of old state. Rescue only from resetting both suggests interaction. Removing H from [H,E] improves reading without changing E. Evidence-only pooling rescue implicates selection within the source window, not necessarily raw storage capacity.', '',
               'A large rescue is a candidate for a matched training experiment, not proof that this intervention is deployable or the sole failure source.', '',
               '## Limitations', ''] + ['- '+w for w in warnings]
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished. Read {output / "report.md"}',flush=True)


if __name__ == '__main__':
    main()
