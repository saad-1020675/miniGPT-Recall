"""Adapter for the uploaded training_RNN.py memory sequence interface."""
import json
from pathlib import Path
import random


def sequence_window_count(runtime_config):
    """Total windows, including one question window; raw dataset has five."""
    count = getattr(runtime_config, 'MEMORY_WINDOWS_PER_SEQUENCE', 5)
    if type(count) is not int or not 2 <= count <= 5:
        raise ValueError('MEMORY_WINDOWS_PER_SEQUENCE must be an integer from 2 to 5 '
                         '(1..4 history windows plus one question window).')
    return count


def dataset_files(root):
    root = Path(root)
    train = sorted((root/'train').glob('*.jsonl'))
    valid = sorted((root/'validation').glob('*.jsonl'))
    if not train or not valid:
        raise ValueError('Generate nonempty train/validation book files first.')
    return train, valid


def read_records(paths, block_size, max_records=None):
    # Loaded here so the standalone audit can still import materialize_record
    # without requiring the training project's config.py.
    try:
        import config as runtime_config
    except ModuleNotFoundError as exc:
        if exc.name != 'config':
            raise
        runtime_config = None
    sequence_windows = sequence_window_count(runtime_config)
    question_window = sequence_windows - 1
    include_label = getattr(runtime_config, 'GUTENBERG_SUPPORT_LABEL', True)
    if type(include_label) is not bool:
        raise ValueError('GUTENBERG_SUPPORT_LABEL must be True or False.')
    evidence_windows = getattr(runtime_config, 'GUTENBERG_EVIDENCE_WINDOWS', None)
    # None: no assist. Integer: local destination. "question": final window.
    # "source": follow the
    # selected QA's source_window (resolved after seeded QA selection).
    support_window = getattr(runtime_config, 'GUTENBERG_SUPPORT_WINDOW', None)
    if support_window == 'question':
        support_window = question_window
    if (support_window is not None and support_window != 'source'
            and (type(support_window) is not int or not 0 <= support_window <= question_window)):
        raise ValueError('GUTENBERG_SUPPORT_WINDOW must be None, "source", "question", '
                         f'or an integer from 0 through {question_window} for this stage.')
    if evidence_windows is not None:
        if (not isinstance(evidence_windows, (list, tuple)) or not evidence_windows
                or any(type(w) is not int or not 0 <= w < 4 for w in evidence_windows)):
            raise ValueError('GUTENBERG_EVIDENCE_WINDOWS must be None or a nonempty list/tuple of W0..W3 indices.')
        evidence_windows = set(evidence_windows)
    records = []
    skipped_for_fit = 0
    for path in paths:
        with Path(path).open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['schema_version'] != 1 or r['block_size'] != block_size or r['tokenizer'] != 'gpt2':
                    raise ValueError('Dataset schema/block_size/tokenizer mismatch; rebuild dataset.')
                if len(r['tokens']) != 5*block_size+1 or not r['qas']:
                    raise ValueError('Invalid passage-backed sequence.')
                # Keep raw records intact until a QA has been selected. The
                # materializer then chooses a contiguous crop containing it.
                r['_sequence_windows'] = sequence_windows
                if evidence_windows is not None:
                    r['qas'] = [q for q in r['qas']
                                if q.get('source_window') in evidence_windows]
                    if not r['qas']:
                        continue
                if support_window is not None:
                    # Use the SAME eligible candidates for W4 and W3 experiments.
                    # Never truncate evidence, question, or answer to fit.
                    import tiktoken
                    enc = tiktoken.get_encoding('gpt2')
                    eligible = []
                    for original in r['qas']:
                        if not isinstance(original.get('evidence'), str) or not original['evidence'].strip():
                            raise ValueError(f"Missing supporting evidence in {r['id']}")
                        q = dict(original)
                        label = enc.encode_ordinary('\nSupporting evidence:\n')
                        evidence = enc.encode_ordinary(q['evidence'])
                        labeled_support = label + evidence + enc.encode_ordinary('\n')
                        # Same eligibility with labels enabled and disabled.
                        if len(labeled_support) + len(q['prompt_tokens']) + len(q['answer_tokens']) + 1 > block_size:
                            skipped_for_fit += 1
                            continue
                        q['_support_tokens'] = (labeled_support if include_label
                                                else evidence + enc.encode_ordinary('\n'))
                        q['_support_label_length'] = len(label) if include_label else 0
                        q['_support_evidence_length'] = len(evidence)
                        eligible.append(q)
                    if not eligible:
                        continue
                    r['qas'] = eligible
                    r['_support_window'] = support_window
                    # q = r["qas"][0]
                    # print("TRAINING PROMPT:", repr(enc.decode(q["prompt_tokens"])))
                    # print("TRAINING ANSWER:", repr(enc.decode(q["answer_tokens"])))
                records.append(r)
                if max_records is not None and len(records) >= max_records:
                    if support_window is not None:
                        destination = 'selected evidence window' if support_window == 'source' else f'W{support_window}'
                        print(f'Support copy in {destination}: {len(records):,} histories retained; {skipped_for_fit:,} oversized QAs skipped.', flush=True)
                    return records
    if support_window is not None:
        destination = 'selected evidence window' if support_window == 'source' else f'W{support_window}'
        print(f'Support copy in {destination}: {len(records):,} histories retained; {skipped_for_fit:,} oversized QAs skipped.', flush=True)
    return records


def _crop_for_question(r, original_q, rng):
    """Preserve a contiguous history containing the target and remap its index.

    The following natural window supplies question filler. No evidence is
    relocated, and the input record/QA are never modified. Raw audit records
    without _sequence_windows retain their original five-window layout.
    """
    count = r.get('_sequence_windows', 5)
    if type(count) is not int or not 2 <= count <= 5:
        raise ValueError('Sequence length must be 2..5 windows.')
    source = original_q.get('source_window')
    if type(source) is not int or not 0 <= source < 4:
        raise ValueError(f'{r.get("id")}: source_window must be in W0..W3.')
    history_count = count - 1
    first = max(0, source - history_count + 1)
    last = min(source, 4 - history_count)
    start = first if first == last else rng.randint(first, last)
    width = r['block_size']
    cropped = dict(r)
    cropped['tokens'] = r['tokens'][start*width:(start+count)*width+1]
    q = dict(original_q)
    q['original_source_window'] = source
    q['source_window'] = source - start
    q['history_start_window'] = start
    q['question_window'] = history_count
    return cropped, q


def materialize_record(r, rng):
    """Return flat sequence and final-question-window Y answer coordinates."""
    B = r['block_size']
    q = rng.choice(r['qas'])
    r, q = _crop_for_question(r, q, rng)
    question_window = q['question_window']
    if r.get('_support_window') is not None:
        # Placement consumes a separate stream so switching W4 -> W3 keeps
        # question choices and final sample ordering identical for a pool seed.
        return _materialize_with_support(r, q, random.Random(rng.getrandbits(64)))
    prompt, answer = q['prompt_tokens'], q['answer_tokens']
    length = len(prompt) + len(answer) + 1
    if length > B or not prompt or not answer:
        raise ValueError('QA does not fit the final question window.')
    # Keep a random-length prefix of the final natural filler window.
    # GPT-2 newline is token 198. Cropped history is unchanged without assists.
    final = r['tokens'][question_window*B:(question_window+1)*B+1].copy()
    # GPT-2 tokens can split a UTF-8 character. Insert only on character-safe
    # boundaries on BOTH sides of the removed filler span.
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    offsets = list(range(B-length+1))
    rng.shuffle(offsets)
    for offset in offsets:
        try:
            enc.decode(final[:offset], errors='strict')
            enc.decode(final[offset+length:B], errors='strict')
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('No character-safe QA insertion offset.')
    final[offset:offset+length] = prompt + answer + [198]
    tokens = r['tokens'][:question_window*B] + final
    start = offset + len(prompt) - 1  # logits at preceding input predict answer
    return tokens, start, start+len(answer), q


def _insert_support_chunk(tokens, window, B, chunk, rng):
    """Replace filler at complete UTF-8 boundaries, keeping sequence length."""
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    if len(chunk) > B:
        raise ValueError('Support/QA chunk exceeds window width.')
    origin = window * B
    local = tokens[origin:origin+B]
    offsets = list(range(B-len(chunk)+1))
    rng.shuffle(offsets)
    for offset in offsets:
        try:
            enc.decode(local[:offset], errors='strict')
            enc.decode(local[offset+len(chunk):], errors='strict')
        except UnicodeDecodeError:
            continue
        tokens[origin+offset:origin+offset+len(chunk)] = chunk
        return offset
    raise ValueError('No character-safe support/QA insertion offset.')


def _materialize_with_support(r, original_q, rng):
    B = r['block_size']
    q = dict(original_q)
    question_window = q.get('question_window', 4)
    window = r['_support_window']
    if window == 'source':
        window = q.get('source_window')
        if type(window) is not int or not 0 <= window < question_window:
            raise ValueError(f'{r.get("id")}: source-following assist requires a source inside the retained history.')
    tokens = r['tokens'].copy()
    support = q['_support_tokens']
    qa = q['prompt_tokens'] + q['answer_tokens'] + [198]
    if window == question_window:
        support_offset = _insert_support_chunk(tokens, question_window, B, support + qa, rng)
        qa_offset = support_offset + len(support)
    else:
        support_offset = _insert_support_chunk(tokens, window, B, support, rng)
        qa_offset = _insert_support_chunk(tokens, question_window, B, qa, rng)
    q['support_window'] = window
    q['support_local_start'] = support_offset + q['_support_label_length']
    q['support_local_end'] = q['support_local_start'] + q['_support_evidence_length']
    start = qa_offset + len(q['prompt_tokens']) - 1
    end = start + len(q['answer_tokens'])
    if not 0 <= start < end <= B:
        raise ValueError('Injected support produced invalid answer coordinates.')
    return tokens, start, end, q


def build_sequences(documents, memories, block_size, pool_seed):
    import torch
    rng = random.Random(pool_seed)
    samples = []
    for r in documents:
        if r['block_size'] != block_size:
            raise ValueError('Record block_size differs from builder block_size.')
        seq, start, end, q = materialize_record(r, rng)
        count = q['question_window'] + 1
        t = torch.tensor(seq, dtype=torch.int32, device='cpu')
        x = t[:-1].reshape(count,block_size).clone()
        y = t[1:].reshape(count,block_size).clone()
        samples.append({'x':x,'y':y,'answer_start':start,'answer_end':end,
            'association_count':1,'all_facts_w0_single_qa':False,
            'memory':{'question':q['question'],'answer':q['answer']},
            'placements':[{'kind':'fact','is_target':True,
                           'window':q.get('support_window', q['source_window']),
                           'local_start':q.get('support_local_start', q['local_start']),
                           'local_end':q.get('support_local_end', q['local_end'])}],
            'source_id':r['id'],
            'source_window':q.get('support_window', q['source_window']),
            'original_source_window':q['original_source_window'],
            'history_start_window':q['history_start_window'],
            'sequence_windows':count,
            'question_window':q['question_window']})
    rng.shuffle(samples)
    return samples
