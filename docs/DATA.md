# Data, tokenization and rebuilding

## Source

The provided downloader originally exported `manu/project_gutenberg`, English
split `en`, into row-prefixed TXT files. The original dataset commit was not
recorded. The release downloader resolves a revision to a SHA, records UTF-8
content hashes, selects one edition per canonical Gutenberg book ID, and skips
identical content duplicates. Keep the manifest and pin its SHA for subsequent
runs. Use a new output directory when changing revision.

Source: <https://huggingface.co/datasets/manu/project_gutenberg>.
Text terms: <https://www.gutenberg.org/policy/license.html>.
This code does not determine public-domain status for a reader's jurisdiction.
Headers, footers, edition artifacts and OCR errors can remain in this corpus.

The raw corpus and generated QA dataset are intentionally excluded from the
release. The two small audit bundles retain tokenized evaluation excerpts. GPT-2
IDs can be decoded back to text; tokenization is not anonymization.

## Raw QA schema

`gutenberg_qa/train/*.jsonl` and `gutenberg_qa/validation/*.jsonl` contain one JSON
record per line. The loader expects:

| Field | Meaning |
|---|---|
| `schema_version` | 1 |
| `id` | Canonical Gutenberg book ID plus token offset, `NUMBER:offset` |
| `block_size` / `tokenizer` | Window length and literal `"gpt2"` |
| `tokens` | Exactly `5 * block_size + 1` GPT-2 token IDs |
| `qas` | Nonempty list of candidate questions |
| QA `source_window` | Original evidence window, integer 0–3 |
| QA `local_start`, `local_end` | Evidence interval in source-window input-token coordinates, end exclusive |
| QA `evidence` | Supporting sentence as a string |
| QA `question`, `answer` | Human-readable metadata |
| QA `prompt_tokens` | Full prompt through `Answer:`, without the answer |
| QA `answer_tokens` | Tokenization of a leading space followed by the gold answer |

The extra final token supplies shifted next-token targets. Answer start/end
coordinates produced by materialization refer to **Y/logit positions**, not the
first answer's input position. The first answer input token is one position
after `start`. This is why replay includes `prefix[:start+1]` before decoding.

The new builder samples one eligible sentence/word question per original history
window, producing up to four candidate QAs. It uses deterministic seeds, complete
sentences, strict UTF-8 handling, no question model or paid API, and labeled-assist
fit checks for consistent eligibility. It skips windows whose boundary splits a
UTF-8 character and sentences too long to fit. It is intentionally simple and
is **not an exact reconstruction of the historical question distribution**.

## Materialization and assistance

The loader chooses a QA with a seeded RNG and inserts its prompt, gold answer
and newline into the final natural-prose filler window, replacing an equal
number of tokens. Without assists the history tokens are unchanged. The decoder
can see question tokens and earlier gold answer tokens during teacher-forced
training; it cannot causally see future answer tokens when predicting them.

`GUTENBERG_SUPPORT_WINDOW=None` adds no evidence copy. `"source"` inserts an
additional copy of the selected evidence in that QA's source window. The
original evidence may also remain, subject to insertion replacing a prose span;
this is added training assistance, not just a metadata field. `GUTENBERG_SUPPORT_LABEL`
controls whether that copy is prefixed by `Supporting evidence:`. Setting it
False removes the label; it does not remove the evidence copy. Fixed integer
support windows and `"question"` are also supported by the loader; see its
validation rules before using them in an audit.

A 2–5 total-window curriculum crops contiguous history around the selected
source, remaps the source index, and retains a following question window. It
does not randomly crop away the selected evidence window. `original_source_window`
and `history_start_window` preserve provenance. The raw JSONL stays five-window;
a block-size change requires rebuilding it.

## Splits and leakage

For newly built data, splitting uses a stable hash of book ID and seed, not a
file's absolute location. Editions with the same canonical ID remain together;
identical content is deduplicated. Both supplied audit manifests can be passed
as holdouts to force their book IDs into validation. The new standard loader
honors the QA build manifest when present; otherwise it hashes Gutenberg IDs.
Mixed training additionally aligns prose book IDs with the actual QA split.
Generic, non-Gutenberg prose filenames fall back to the legacy seeded file split.

Build the QA dataset before prose pretraining if you want its forced holdouts
respected there. Do not train on a standalone prose export using one split and
later evaluate recall from those same books under another split. Edition-level
or content-level relatedness can still extend beyond numeric IDs; these checks
are not a comprehensive semantic duplicate detector.

The new sorted file order and book-grouped standard split differ from the
historical file-level prose split. Use fresh output directories for a new release
run. Old checkpoints are not guaranteed to resume with an altered dataset order.
