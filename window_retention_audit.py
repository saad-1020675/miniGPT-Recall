#!/usr/bin/env python3
"""Read-only variable-history retention audit for MiniGPTRecall.

Place beside config.py, miniGPT_model_RNN.py and gutenberg_qa_loader.py, or
supply --root. Uses the installed model and loader; no other audit imports.
Example:
 python window_retention_audit.py --checkpoint checkpoints/best.pt --output audit32
Re-use examples for another checkpoint:
 python window_retention_audit.py --checkpoint other/best.pt --manifest audit32/manifest.json --output audit16

Use --history-windows 1..4 (excludes the question window). Otherwise use
the manifest horizon, then checkpoint sidecar, then MEMORY_WINDOWS_PER_SEQUENCE
from config (total windows, including the question; fallback 3).
All answer scores are teacher-forced. The final question window never writes. Defaults to
config's assistance mode; --assist none/unlabeled/labeled overrides it.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def module_from(name, root):
    if name.endswith('.py'):
        path = Path(name)
        path = path if path.is_absolute() else root/path
        spec = importlib.util.spec_from_file_location('_retention_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name)


def prepare_record(raw, width, assist, enc, evidence_windows=None, sequence_windows=3):
    if raw.get('schema_version') != 1 or raw.get('block_size') != width or raw.get('tokenizer') != 'gpt2':
        raise ValueError(f'{raw.get("id")}: expected schema 1, gpt2, block_size={width}.')
    if len(raw.get('tokens', [])) != 5*width+1 or not raw.get('qas'):
        raise ValueError(f'{raw.get("id")}: expected original five-window QA records.')
    if raw.get('_support_window') is not None:
        raise ValueError('Use the original raw dataset, not materialized assisted records.')
    record = dict(raw)
    record['_sequence_windows'] = sequence_windows
    eligible = []
    label = enc.encode_ordinary('\nSupporting evidence:\n')
    newline = enc.encode_ordinary('\n')
    for original in raw['qas']:
        if evidence_windows is not None and original.get('source_window') not in evidence_windows:
            continue
        q = dict(original)
        if not q.get('prompt_tokens') or not q.get('answer_tokens'):
            raise ValueError(f'{raw.get("id")}: empty prompt/answer tokens.')
        if assist != 'none':
            if not isinstance(q.get('evidence'), str) or not q['evidence'].strip():
                raise ValueError(f'{raw.get("id")}: missing evidence text for assistance.')
            evidence = enc.encode_ordinary(q['evidence'])
            labeled = label + evidence + newline
            # Match training's common eligibility across label modes.
            if len(labeled)+len(q['prompt_tokens'])+len(q['answer_tokens'])+1 > width:
                continue
            q['_support_tokens'] = labeled if assist == 'labeled' else evidence+newline
            q['_support_label_length'] = len(label) if assist == 'labeled' else 0
            q['_support_evidence_length'] = len(evidence)
        eligible.append(q)
    if not eligible:
        return None
    record['qas'] = eligible
    if assist != 'none':
        record['_support_window'] = 'source'
    return record


def validate_sample(s, width, vocab, sequence_windows=3):
    tokens = s['tokens']
    if len(tokens) != sequence_windows*width+1 or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise ValueError('Manifest contains invalid token IDs or sequence length.')
    if type(s['source']) is not int or not 0 <= s['source'] < sequence_windows-1:
        raise ValueError('Evidence source must be inside the retained history; use the variable-length loader.')
    start, end = s['start'], s['end']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= width:
        raise ValueError('Invalid final-window answer coordinates.')
    if tokens[(sequence_windows-1)*width+start+1:(sequence_windows-1)*width+end+1] != s['answer_tokens']:
        raise ValueError('Answer-token alignment mismatch.')
    if digest(tokens) != s['tokens_sha256']:
        raise ValueError('Manifest token fingerprint mismatch.')


def collect_samples(args, config, loader, enc, width, vocab, assist):
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('block_size') != width or manifest.get('sequence_windows') != args.sequence_windows:
            raise ValueError('Manifest block size or history length differs from the requested audit.')
        samples = manifest['samples']
        for s in samples:
            validate_sample(s, width, vocab, args.sequence_windows)
        if [s['index'] for s in samples] != list(range(len(samples))):
            raise ValueError('Manifest sample indices must be unique and sequential.')
        if not set(range(args.sequence_windows-1)).issubset({s['source'] for s in samples}):
            raise ValueError('Manifest must include questions from every history window.')
        return manifest
    path = args.data or Path(getattr(config, 'GUTENBERG_QA_DIR', './gutenberg_qa'))/'validation'
    path = path if path.is_absolute() else args.root/path
    paths = sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError(f'No QA JSONL files found at {path}.')
    excluded = set()
    if args.exclude_ids:
        excluded = {line.strip() for line in args.exclude_ids.read_text().splitlines() if line.strip()}
    filters = getattr(config, 'GUTENBERG_EVIDENCE_WINDOWS', None)
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    reservoir, scanned, eligible = [], 0, 0
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                scanned += 1
                if str(raw.get('id')) not in excluded:
                    record = prepare_record(raw, width, assist, enc, filters, args.sequence_windows)
                    if record is not None:
                        eligible += 1
                        item = (record, str(path), line_number)
                        if len(reservoir) < args.samples:
                            reservoir.append(item)
                        else:
                            index = rng.randrange(eligible)
                            if index < args.samples:
                                reservoir[index] = item
                if args.scan_limit and scanned >= args.scan_limit:
                    break
        if args.scan_limit and scanned >= args.scan_limit:
            break
    rng.shuffle(reservoir)
    samples = []
    for index, (record, path, line) in enumerate(reservoir):
        tokens, start, end, q = loader.materialize_record(record, random.Random(args.seed+1000003*(index+1)))
        source = q.get('source_window')
        lo = q.get('support_local_start', q.get('local_start'))
        hi = q.get('support_local_end', q.get('local_end'))
        span_valid = type(lo) is int and type(hi) is int and 0 <= lo < hi <= width
        sample = dict(index=index, source_id=record.get('id'), file=path, line=line,
            tokens=tokens, tokens_sha256=digest(tokens), start=start, end=end,
            source=source, original_source_window=q.get('original_source_window'),
            history_start_window=q.get('history_start_window'),
            question=q.get('question'), answer=q.get('answer'), answer_tokens=q['answer_tokens'],
            evidence_start=lo, evidence_end=hi, evidence_span_valid=span_valid)
        validate_sample(sample, width, vocab, args.sequence_windows)
        samples.append(sample)
    if not samples or not set(range(args.sequence_windows-1)).issubset({s['source'] for s in samples}):
        raise ValueError('Need questions from every history window. Increase --samples/--scan-limit or remove source filters.')
    return dict(schema_version=1, block_size=width, sequence_windows=args.sequence_windows, assistance=assist,
        sampling=dict(seed=args.seed, scanned_records=scanned, eligible_records=eligible,
                      files=len(paths), scan_limit=args.scan_limit,
                      excluded_ids=len(excluded), evidence_windows=filters), samples=samples)


def write_features(model, features, memory, chunk_size, reset_route=False, reset_gru=False):
    """Resets affect ONLY the first chunk at this window's incoming boundary."""
    states = []
    for index, chunk in enumerate(features.split(chunk_size, dim=1)):
        prior = memory
        routing_prior = torch.zeros_like(prior) if reset_route and index == 0 else prior
        candidate = model.memory_compressor(chunk, routing_prior)
        gru_prior = torch.zeros_like(prior) if reset_gru and index == 0 else prior
        memory = model.memory(candidate, gru_prior)
        states.append(memory)
    return memory, states


@contextmanager
def no_read(model, disabled):
    handles = []
    try:
        if disabled:
            for block in model.blocks:
                handles.append(block.memory_attention.register_forward_hook(
                    lambda module, inputs, output: torch.zeros_like(output)))
            if not handles:
                raise ValueError('No memory_attention modules found for NoRead.')
        yield
    finally:
        for handle in handles:
            handle.remove()


def score_batch(model, x, y, memory, samples, disabled=False):
    with no_read(model, disabled):
        logits, _, _ = model(x, memory=memory, update_memory=False)
    logits = logits.reshape(len(samples), x.shape[1], -1)
    rows = []
    for index, sample in enumerate(samples):
        start, end = sample['start'], sample['end']
        selected = logits[index, start:end].float()
        target = y[index, start:end]
        correct = selected.argmax(-1).eq(target)
        losses = torch.nn.functional.cross_entropy(selected, target, reduction='none')
        rows.append(dict(exact=int(correct.all().item()), first=int(correct[0].item()),
                         ce_sum=float(losses.sum().item()), answer_tokens=end-start))
    return rows


def aggregate(rows):
    if not rows:
        return dict(samples=0, exact_accuracy=None, first_accuracy=None, answer_ce=None)
    n = len(rows)
    return dict(samples=n, exact_accuracy=sum(r['exact'] for r in rows)/n,
        first_accuracy=sum(r['first'] for r in rows)/n,
        answer_ce=sum(r['ce_sum'] for r in rows)/sum(r['answer_tokens'] for r in rows))


def paired(rows, before, after, source, bootstrap, seed):
    selected = [r for r in rows if source is None or r['source'] == source]
    a = {r['index']: r for r in selected if r['condition'] == before}
    b = {r['index']: r for r in selected if r['condition'] == after}
    ids = sorted(a.keys() & b.keys())
    if not ids:
        return dict(samples=0)
    gained = sum(not a[i]['exact'] and b[i]['exact'] for i in ids)
    lost = sum(a[i]['exact'] and not b[i]['exact'] for i in ids)
    both_wrong = sum(not a[i]['exact'] and not b[i]['exact'] for i in ids)
    both_right = len(ids)-gained-lost-both_wrong
    result = dict(samples=len(ids), before=before, after=after, source_window=source,
        before_accuracy=sum(a[i]['exact'] for i in ids)/len(ids),
        after_accuracy=sum(b[i]['exact'] for i in ids)/len(ids),
        accuracy_delta_pp=100*(gained-lost)/len(ids),
        correct_to_incorrect=lost, incorrect_to_correct=gained,
        both_correct=both_right, both_incorrect=both_wrong,
        mean_example_ce_change=sum(b[i]['ce_sum']/b[i]['answer_tokens']-a[i]['ce_sum']/a[i]['answer_tokens'] for i in ids)/len(ids))
    if bootstrap:
        import numpy as np
        rng = np.random.default_rng(seed)
        # Equivalent paired bootstrap for {-1,0,+1} exact-score changes.
        draws = rng.multinomial(len(ids), [lost/len(ids), (len(ids)-lost-gained)/len(ids), gained/len(ids)], size=bootstrap)
        changes = 100*(draws[:, 2]-draws[:, 0])/len(ids)
        result['delta_ci95_pp'] = [float(x) for x in np.quantile(changes, [0.025, 0.975])]
    return result


def donor_indices(samples, width, seed, history_windows=2):
    hashes = [digest(s['tokens'][:history_windows*width]) for s in samples]
    if len(set(hashes)) < 2:
        raise ValueError('ShuffleHistory requires at least two distinct histories.')
    rng = random.Random(seed)
    result = []
    for i in range(len(samples)):
        j = rng.randrange(len(samples))
        while hashes[j] == hashes[i]:
            j = (j+1) % len(samples)
        result.append(j)
    return result


def write_report(output, summary):
    history = summary['metadata']['sequence_windows']-1
    sources = [f'W{i}' for i in range(history)]
    lines = ['# Window retention audit', '',
        'Scores are teacher-forced. The final question window never writes to memory.', '',
        '| Condition | ' + ' | '.join(sources) + ' |',
        '|---|' + '---:|'*history]
    for name, groups in summary['conditions'].items():
        values = [groups.get(source, {}).get('exact_accuracy') for source in sources]
        lines.append('| '+name+' | '+' | '.join('—' if v is None else f'{100*v:.2f}%' for v in values)+' |')
    lines += ['', '| Paired comparison | Delta | Lost | Rescued |', '|---|---:|---:|---:|']
    for name, comparison in summary['paired'].items():
        if comparison['samples']:
            lines.append(f'| {name} | {comparison["accuracy_delta_pp"]:+.2f} pp | '
                         f'{comparison["correct_to_incorrect"]} | {comparison["incorrect_to_correct"]} |')
    lines += ['', '## Interpretation', '',
        '- Wi_LaterWrites compares the same questions immediately after their source window with the final memory. Losses and rescues may cancel.',
        '- Wi_PrecedingHistory compares writing Wi alone with its state after the natural prefix through Wi. It isolates sensitivity to preceding writes, excluding later writes.',
        '- Trajectory comparisons follow each subsequent write/chunk for evidence already encountered.',
        '- Reset interventions clear routing input, GRU input, or both on the FIRST CHUNK of the named window; all later writes proceed normally. ResetBoth is checked against replaying the suffix from zero.',
        '- SourceOnly contains only the source window written from zero; FinalFromSource retains its natural suffix but omits the prefix.',
        '- EvidenceFirst/Last stably move the source window to the first/last history position; other windows retain relative order and the question is fixed. Reordered prose may change discourse meaning.',
        '- A state before its evidence window is an absent-evidence control. Intermediate/reset/reordered states may be outside the training distribution.',
        '- Full-window bidirectional encoding precedes chunking; chunk states are not strictly prefix-only representations.',
        '- Native source groups contain different questions. Use paired interventions to assess positional effects.',
        '- Bootstrap intervals are descriptive example-level intervals, not book-cluster intervals. Source-ID exclusions apply only when supplied.',
        '- Reuse manifests only at the SAME history length and assistance mode. Different lengths create different inputs and are not a paired horizon comparison.',
        '- Original dataset records contain four history windows plus a question window, limiting this script to 1–4 history windows.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--model-module', default='miniGPT_model_RNN')
    parser.add_argument('--loader-module', default='gutenberg_qa_loader')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, default=Path('window_retention_results'))
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--scan-limit', type=int, default=10000, help='0 scans the entire split.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--assist', choices=['config', 'none', 'unlabeled', 'labeled'], default='config')
    parser.add_argument('--history-windows', type=int, choices=range(1, 5), help='Number of history windows, excluding the final question window (1..4).')
    parser.add_argument('--chunk-size', type=int, help='Explicit runtime override; recorded in metadata.')
    parser.add_argument('--routing-passes', type=int)
    parser.add_argument('--pooling-heads', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--exclude-ids', type=Path, help='Text file: one raw source ID per line.')
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 1 or args.scan_limit < 0 or args.bootstrap < 0:
        parser.error('Need samples>=2, batch-size>=1, scan-limit>=0 and bootstrap>=0.')
    args.root = args.root.resolve()
    for key in ['checkpoint', 'data', 'manifest', 'exclude_ids', 'output']:
        value = getattr(args, key)
        if value is not None and not value.is_absolute():
            setattr(args, key, args.root/value)
    return args


def resolve_windows(requested_history, manifest_windows, saved_windows, config_windows):
    total = (requested_history+1 if requested_history is not None else
             manifest_windows if manifest_windows is not None else
             saved_windows if saved_windows is not None else config_windows)
    if type(total) is not int or not 2 <= total <= 5:
        raise ValueError('Need 1..4 history windows plus one question window.')
    if manifest_windows is not None and total != manifest_windows:
        raise ValueError('Requested horizon differs from manifest; create a new manifest for each horizon.')
    return total


def evidence_order(history, source, last):
    others = [w for w in range(history) if w != source]
    return others+[source] if last else [source]+others


def summarize_rows(rows, history, chunks, bootstrap, seed):
    conditions = {}
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group = [r for r in rows if r['condition'] == condition]
        conditions[condition] = dict(all=aggregate(group), **{
            f'W{source}': aggregate([r for r in group if r['source'] == source]) for source in range(history)})
    comparisons = {'MatchedOrder_All': ('EvidenceFirst', 'EvidenceLast', None)}
    for source in range(history):
        comparisons[f'W{source}_LaterWrites'] = (f'AfterW{source}', 'Normal', source)
        comparisons[f'W{source}_PrecedingHistory'] = ('SourceOnly', f'AfterW{source}', source)
        comparisons[f'MatchedOrder_W{source}'] = ('EvidenceFirst', 'EvidenceLast', source)
        comparisons[f'W{source}_PrefixEffectFinal'] = ('FinalFromSource', 'Normal', source)
        for w in range(1, history):
            for prefix in ['ResetRouting', 'ResetGRU', 'ResetBoth']:
                name = f'{prefix}W{w}'
                comparisons[f'W{source}_{name}'] = ('Normal', name, source)
        trajectory = [f'AfterW{source}']
        for w in range(source+1, history):
            trajectory.extend(f'AfterW{w}Chunk{i}' for i in range(1, chunks))
            trajectory.append(f'AfterW{w}' if w < history-1 else 'Normal')
        for before, after in zip(trajectory, trajectory[1:]):
            comparisons[f'W{source}_Trajectory_{before}_to_{after}'] = (before, after, source)
    pairs = {name: paired(rows, before, after, source, bootstrap, seed+i)
             for i, (name, (before, after, source)) in enumerate(comparisons.items())}
    return conditions, pairs


def main():
    global torch
    args = parse_args()
    import torch
    import tiktoken
    sys.path.insert(0, str(args.root))
    config = importlib.import_module('config')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = payload.get('model_state_dict', payload.get('state_dict', payload))
    for _ in range(3):
        prefix = next((p for p in ['module.', '_orig_mod.'] if state and all(k.startswith(p) for k in state)), None)
        if prefix is None:
            break
        state = {k[len(prefix):]: v for k, v in state.items()}
    slots, slot_width = state['memory_compressor.slot_embeddings'].shape
    config.memory_slots, config.memory_slot_dims = slots, slot_width
    sidecar = args.checkpoint.with_name(args.checkpoint.name+'.writer.json')
    saved = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    writer = saved.get('writer_config', {})
    for name, field, override in [('memory_write_chunk_size', 'write_chunk_size', args.chunk_size),
                                  ('memory_routing_passes', 'routing_passes', args.routing_passes),
                                  ('memory_pooling_heads', 'pooling_heads', args.pooling_heads)]:
        if override is not None:
            setattr(config, name, override)
        elif field in writer:
            setattr(config, name, writer[field])
    width = int(config.block_size)
    if 'block_size' in writer and writer['block_size'] != width:
        raise ValueError('Checkpoint sidecar block_size differs from config.')
    chunk = getattr(config, 'memory_write_chunk_size', width)
    if type(chunk) is not int or chunk < 1 or width % chunk:
        raise ValueError('Write chunk size must be a positive divisor of block_size.')
    if not sidecar.exists():
        print('No writer sidecar: routing passes, pooling heads and chunk size come from config/CLI. Verify them against training.', flush=True)
    assist = args.assist
    if args.manifest:
        manifest_mode = json.loads(args.manifest.read_text(encoding='utf-8')).get('assistance')
        if manifest_mode not in ('none', 'unlabeled', 'labeled'):
            raise ValueError('Manifest has invalid assistance metadata.')
        if assist != 'config' and assist != manifest_mode:
            raise ValueError('Requested assistance differs from the fixed manifest inputs.')
        assist = manifest_mode
    if assist == 'config':
        support = getattr(config, 'GUTENBERG_SUPPORT_WINDOW', None)
        if support not in (None, 'source'):
            raise ValueError('This audit requires no assist or source-following assists; use --assist explicitly.')
        assist = ('none' if support is None else
                  'labeled' if getattr(config, 'GUTENBERG_SUPPORT_LABEL', True) else 'unlabeled')
    manifest_windows = (json.loads(args.manifest.read_text(encoding='utf-8')).get('sequence_windows')
                        if args.manifest else None)
    args.sequence_windows = resolve_windows(args.history_windows, manifest_windows,
        writer.get('sequence_windows'), getattr(config, 'MEMORY_WINDOWS_PER_SEQUENCE', 3))
    history_windows = args.sequence_windows-1
    config.MEMORY_WINDOWS_PER_SEQUENCE = args.sequence_windows
    loader = module_from(args.loader_module, args.root)
    model_module = module_from(args.model_module, args.root)
    vocab = state['token_embedding_table.weight'].shape[0]
    model = model_module.MiniGPTRecall(vocab_size=vocab)
    model.load_state_dict(state, strict=True)
    model = model.to(device).float().eval()
    actual_chunk = getattr(model, 'memory_write_chunk_size', width)
    if actual_chunk != chunk:
        raise ValueError('Loaded model does not honor configured chunk size.')
    enc = tiktoken.get_encoding('gpt2')
    manifest = collect_samples(args, config, loader, enc, width, vocab, assist)
    samples = manifest['samples']
    if len(samples) < 2:
        raise ValueError('Need at least two examples.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/name).exists() for name in ['summary.json', 'manifest.json', 'per_sample.csv', 'report.md']):
        raise FileExistsError('Output already contains audit results; use a new --output directory.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    donors = donor_indices(samples, width, args.seed+9, history_windows)
    rows = []
    checks = dict(writer_max_abs_difference=0.0, reset_both_max_abs_difference=0.0)
    def encode(x):
        return model.memory_encoder(model.token_embedding_table(x))
    with torch.inference_mode():
        for begin in range(0, len(samples), args.batch_size):
            batch = samples[begin:begin+args.batch_size]
            tokens = torch.tensor([s['tokens'] for s in batch], dtype=torch.long, device=device)
            x = tokens[:, :-1].reshape(len(batch), args.sequence_windows, width)
            y = tokens[:, 1:].reshape(len(batch), args.sequence_windows, width)
            zero = model.init_memory(len(batch), device, model.token_embedding_table.weight.dtype)
            features = [encode(x[:, w]) for w in range(history_windows)]
            states, traces = [], []
            memory = zero
            for w, features_w in enumerate(features):
                prior = memory
                memory, trace = write_features(model, features_w, prior, chunk)
                states.append(memory)
                traces.append(trace)
                if begin == 0:
                    _, _, reference = model(x[:, w], memory=prior, update_memory=True)
                    difference = float((reference-memory).abs().max().item())
                    checks['writer_max_abs_difference'] = max(checks['writer_max_abs_difference'], difference)
                    if not torch.allclose(reference, memory, atol=1e-5, rtol=1e-5):
                        raise RuntimeError('Audit writer does not reproduce model.forward. Do not interpret these results.')
            def score(name, memory, disabled=False):
                records = score_batch(model, x[:, -1], y[:, -1], memory, batch, disabled)
                for sample, record in zip(batch, records):
                    rows.append(dict(index=sample['index'], source_id=sample['source_id'],
                                     source=sample['source'], condition=name, **record))
            def select_source(memories):
                selected = zero
                for w, value in enumerate(memories):
                    mask = torch.tensor([s['source'] == w for s in batch], device=device)[:, None, None]
                    selected = torch.where(mask, value, selected)
                return selected
            def replay(order, initial):
                result = initial
                for w in order:
                    result, _ = write_features(model, features[w], result, chunk)
                return result
            for w, state_w in enumerate(states):
                score(f'AfterW{w}', state_w)
                for i, partial in enumerate(traces[w][:-1], 1):
                    score(f'AfterW{w}Chunk{i}', partial)
            score('Normal', states[-1])
            only = [write_features(model, f, zero, chunk)[0] for f in features]
            # Legacy W1Only name is retained; other horizons get each WiOnly.
            for w, state_w in enumerate(only):
                score(f'W{w}Only', state_w)
            score('SourceOnly', select_source(only))
            score('AtSource', select_source(states))
            suffixes = [replay(range(w+1, history_windows), only[w]) for w in range(history_windows)]
            score('FinalFromSource', select_source(suffixes))
            score('NoRead', states[-1], True)
            for w in range(1, history_windows):
                for prefix, route, gru in [('ResetRouting', True, False), ('ResetGRU', False, True), ('ResetBoth', True, True)]:
                    altered, _ = write_features(model, features[w], states[w-1], chunk, route, gru)
                    if prefix == 'ResetBoth':
                        difference = float((altered-only[w]).abs().max().item())
                        checks['reset_both_max_abs_difference'] = max(checks['reset_both_max_abs_difference'], difference)
                        if not torch.allclose(altered, only[w], atol=1e-5, rtol=1e-5):
                            raise RuntimeError(f'ResetBoth must reproduce W{w}Only at its boundary.')
                    altered = replay(range(w+1, history_windows), altered)
                    score(f'{prefix}W{w}', altered)
            orders_first = [replay(evidence_order(history_windows, w, False), zero) for w in range(history_windows)]
            orders_last = [replay(evidence_order(history_windows, w, True), zero) for w in range(history_windows)]
            score('EvidenceFirst', select_source(orders_first))
            score('EvidenceLast', select_source(orders_last))
            donor_batch = [samples[donors[i]] for i in range(begin, begin+len(batch))]
            donor_tokens = torch.tensor([s['tokens'][:history_windows*width] for s in donor_batch], dtype=torch.long, device=device)
            wrong = zero
            for w in range(history_windows):
                wrong, _ = write_features(model, encode(donor_tokens[:, w*width:(w+1)*width]), wrong, chunk)
            score('ShuffleHistory', wrong)
            print(f'Scored {begin+len(batch)}/{len(samples)} examples', flush=True)
    conditions, pairs = summarize_rows(rows, history_windows, width//chunk, args.bootstrap, args.seed)
    partitions = {}
    for source in range(history_windows):
        comparison = pairs[f'W{source}_LaterWrites']
        partitions[f'W{source}'] = dict(
            became_incorrect_after_source=comparison['correct_to_incorrect'],
            incorrect_at_both_states=comparison['both_incorrect'])
    summary = dict(metadata=dict(checkpoint=str(args.checkpoint), device=str(device), precision='fp32',
        memory_slots=slots, memory_slot_dims=slot_width, write_chunk_size=chunk,
        routing_passes=getattr(model.memory_compressor, 'routing_iterations', None),
        pooling_heads=getattr(model.memory_compressor, 'pooling_heads', None),
        sequence_windows=args.sequence_windows, history_windows=history_windows, checkpoint_sidecar=saved,
        assistance=manifest['assistance'], sampling=manifest['sampling'],
        manifest_sha256=digest(manifest), checkpoint_tokens_seen=payload.get('tokens_seen'),
        runtime_overrides=dict(history_windows=args.history_windows, chunk_size=args.chunk_size, routing_passes=args.routing_passes, pooling_heads=args.pooling_heads),
        torch_version=str(torch.__version__), bootstrap=args.bootstrap),
        checks=checks, conditions=conditions, paired=pairs, final_error_partitions=partitions,
        w0_final_error_partition=dict(
            became_incorrect_after_later_writes=pairs['W0_LaterWrites']['correct_to_incorrect'],
            incorrect_at_both_states=pairs['W0_LaterWrites']['both_incorrect']))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output/'per_sample.csv').open('w', newline='', encoding='utf-8') as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    write_report(args.output, summary)
    primary = pairs['W0_LaterWrites']
    print(f'W0 after W0: {100*primary["before_accuracy"]:.2f}%; after final history: {100*primary["after_accuracy"]:.2f}%; '
          f'lost {primary["correct_to_incorrect"]}, rescued {primary["incorrect_to_correct"]}.')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
