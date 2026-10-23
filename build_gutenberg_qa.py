"""Build a NEW deterministic, book-split cloze dataset for the existing loader.

This is a release baseline, not the missing historical QA-generation pipeline.
It uses complete sentences within token windows and a simple word sampler.
No LLM, external API, or question-aware memory labeling is used.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import random
import re

from dataset_utils import canonical_book_id, book_split

INSTRUCTION = 'Earlier in the passage, what word completes this sentence?'
STOP = set('about after again against among before being could every first great little might other shall should their there these thing those through under until where which while whole would'.split())


def window_questions(tokens, enc, width, rng):
    qas = []
    for window in range(4):
        ids = tokens[window*width:(window+1)*width]
        raw = b''.join(enc.decode_single_token_bytes(t) for t in ids)
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # A window may begin/end inside a multi-byte character. This simple
            # baseline skips such windows instead of corrupting the evidence.
            continue
        offsets = [0]
        for t in ids:
            offsets.append(offsets[-1] + len(enc.decode_single_token_bytes(t)))
        candidates = list(re.finditer(r'[^.!?\n]+[.!?]', text))
        rng.shuffle(candidates)
        for match in candidates:
            byte_start = len(text[:match.start()].encode('utf-8'))
            byte_end = len(text[:match.end()].encode('utf-8'))
            start = bisect.bisect_right(offsets, byte_start)-1
            end = bisect.bisect_left(offsets, byte_end)
            if not 0 <= start < end <= width:
                continue
            try:
                evidence = enc.decode(ids[start:end], errors='strict')
            except UnicodeDecodeError:
                continue
            words = [m for m in re.finditer(r'\b[A-Za-z]{4,16}\b', evidence)
                     if m.group().lower() not in STOP]
            if not words:
                continue
            word = rng.choice(words)
            answer = word.group()
            sentence = (evidence[:word.start()]+'[BLANK]'+evidence[word.end():]).strip()
            question = INSTRUCTION+'\n'+sentence
            prompt = enc.encode_ordinary('\nQuestion: '+question+'\nAnswer:')
            answer_ids = enc.encode_ordinary(' '+answer)
            # Same eligibility whether later training enables support labels
            # or not; avoids silently changing populations for assist ablations.
            labeled = enc.encode_ordinary('\nSupporting evidence:\n'+evidence+'\n')
            if len(labeled)+len(prompt)+len(answer_ids)+1 > width:
                continue
            qas.append(dict(question=question, answer=answer, source_window=window,
                            local_start=start, local_end=end, evidence=evidence,
                            prompt_tokens=prompt, answer_tokens=answer_ids))
            break
    return qas


def main():
    import tiktoken
    from config import DATA_DIR, GUTENBERG_QA_DIR, block_size
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(DATA_DIR))
    p.add_argument('--output', type=Path, default=Path(GUTENBERG_QA_DIR))
    p.add_argument('--block-size', type=int, default=block_size)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--max-books', type=int)
    p.add_argument('--max-records-per-book', type=int)
    p.add_argument('--holdout-manifest', action='append', type=Path, default=[],
                   help='Force books in a frozen audit manifest into validation.')
    args = p.parse_args()
    if args.block_size < 32 or not 0 < args.validation_fraction < 1:
        p.error('block size must be >=32 and validation fraction in (0,1)')
    if any(v is not None and v < 1 for v in (args.max_books, args.max_records_per_book)):
        p.error('limits must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('QA output must be empty; never mix rebuilt datasets.')
    holdout = set()
    for mf in args.holdout_manifest:
        for sample in json.loads(mf.read_text(encoding='utf-8'))['samples']:
            holdout.add(canonical_book_id(sample['source_id']))
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):
        (args.output/split).mkdir(exist_ok=True)
    enc = tiktoken.get_encoding('gpt2')
    seen_ids, seen_hashes, inventory = set(), set(), []
    files = sorted(args.input.rglob('*.txt'))
    if not files:
        raise ValueError('No input text files found.')
    count = 0
    for path in files:
        book = canonical_book_id(path.name)
        if book in seen_ids:
            continue  # one deterministic edition per Gutenberg ID
        raw = path.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        if content_hash in seen_hashes:
            continue  # also suppress identical content under different IDs
        seen_ids.add(book); seen_hashes.add(content_hash)
        split = ('validation' if book in holdout else
                 book_split(book, args.seed, args.validation_fraction))
        ids = enc.encode_ordinary(raw.decode('utf-8'))
        B = args.block_size
        record_count = 0
        destination = args.output/split/f'{book}.jsonl'
        with destination.open('w', encoding='utf-8') as out:
            for offset in range(0, len(ids)-5*B, 5*B):
                tokens = ids[offset:offset+5*B+1]
                rng = random.Random(f'{args.seed}:{book}:{offset}')
                qas = window_questions(tokens, enc, B, rng)
                if not qas:
                    continue
                record = dict(schema_version=1, id=f'{book}:{offset}', book_id=book,
                              block_size=B, tokenizer='gpt2', tokens=tokens, qas=qas)
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
                record_count += 1; count += 1
                if args.max_records_per_book and record_count >= args.max_records_per_book:
                    break
        if not record_count:
            destination.unlink()
        inventory.append(dict(book_id=book, split=split, source_file=path.relative_to(args.input).as_posix(),
                              source_sha256=content_hash, records=record_count))
        if len(inventory) % 100 == 0:
            print(f'Processed {len(inventory):,} books | {count:,} records', flush=True)
        if args.max_books and len(inventory) >= args.max_books:
            break
    metadata = dict(builder='release-baseline-v1; not historical builder', schema_version=1,
        block_size=args.block_size, seed=args.seed, validation_fraction=args.validation_fraction,
        forced_validation_book_ids=sorted(holdout), tokenizer='gpt2', tokenizer_version=tiktoken.__version__,
        records=count, books=inventory)
    source_manifest = args.input/'download_manifest.json'
    if source_manifest.exists():
        source = json.loads(source_manifest.read_text())
        metadata['source_dataset'] = {k:source[k] for k in ('dataset','revision','split')}
    (args.output/'build_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    counts = {s:sum(b['records'] for b in inventory if b['split']==s) for s in ('train','validation')}
    print(f'Built {count:,} records: {counts}. Metadata saved.', flush=True)
    if not all(counts.values()):
        print('WARNING: one split is empty; increase book limit or supply more books.', flush=True)


if __name__ == '__main__':
    main()
