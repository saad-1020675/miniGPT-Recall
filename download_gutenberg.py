"""Export a pinned English Gutenberg snapshot; retain IDs and content hashes.

Historical runs used the same dataset name, but their revision was not saved.
This downloader records the resolved revision for NEW reproducible runs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from config import DATA_DIR, HF_TOKEN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(DATA_DIR))
    p.add_argument('--revision', default='main', help='Commit SHA recommended; main is resolved and recorded.')
    p.add_argument('--cache-dir', default='./train_data/hf_cache')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'download_manifest.json').exists():
        raise FileExistsError('Output already has download_manifest.json; use a new directory to avoid mixing snapshots.')
    repo = 'manu/project_gutenberg'
    revision = HfApi(token=HF_TOKEN).dataset_info(repo, revision=args.revision).sha
    print(f'Loading {repo} @ {revision}, split=en', flush=True)
    dataset = load_dataset(repo, split='en', revision=revision,
                          cache_dir=args.cache_dir, token=HF_TOKEN)
    books = []
    seen_books, seen_content = set(), set()
    from dataset_utils import canonical_book_id
    for index, book in enumerate(dataset):
        identifier, text = str(book['id']), book['text']
        if not text:
            continue
        # An upstream ID is metadata, never an arbitrary local path.
        safe_id = ''.join(c for c in identifier if c.isalnum() or c in '-_')
        if not safe_id:
            raise ValueError('Empty sanitized book ID')
        name = f'{index:06d}_{safe_id}.txt'
        raw = text.encode('utf-8')
        canonical = canonical_book_id(safe_id)
        fingerprint = hashlib.sha256(raw).hexdigest()
        if canonical in seen_books or fingerprint in seen_content:
            continue
        seen_books.add(canonical); seen_content.add(fingerprint)
        target = args.output / name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f'Existing file differs: {name}; use an empty output directory.')
        target.write_bytes(raw)
        books.append(dict(source_id=identifier, file=name, sha256=hashlib.sha256(raw).hexdigest()))
        if index % 1000 == 0:
            print(f'Exported {len(books):,} nonempty rows of {len(dataset):,}', flush=True)
    manifest = dict(schema_version=1, dataset=repo, revision=revision, split='en',
                    text_processing='UTF-8 export; first edition per canonical book ID; identical-content deduplication; no header/footer stripping', books=books)
    (args.output / 'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Exported {len(books):,} files. Revision and hashes saved.', flush=True)


if __name__ == '__main__':
    main()
