"""Portable book identity and deterministic book-level splits for new datasets."""
import hashlib
from pathlib import Path
import re


def canonical_book_id(value):
    text = str(value)
    # Raw record ID NUMBER:token_offset, plain ID, or exported row_ID.txt.
    if re.fullmatch(r'\d+:\d+', text):
        return str(int(text.split(':', 1)[0]))
    stem = Path(text.replace('\\', '/')).stem
    stem = re.sub(r'^\d{6}_', '', stem)
    match = re.fullmatch(r'(\d+)(?:[-_].*)?', stem)
    if not match:
        raise ValueError(f'Cannot identify Gutenberg book from {value!r}')
    return str(int(match.group(1)))


def book_split(book_id, seed=42, validation_fraction=.1):
    value = int.from_bytes(hashlib.sha256(f'{seed}:{book_id}'.encode()).digest()[:8], 'big') / 2**64
    return 'validation' if value < validation_fraction else 'train'
