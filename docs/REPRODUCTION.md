# Reproduction status

## Three distinct levels

1. **Inspect the reported results now.** All 22 supplied diagnostic logs and both
   audit reports are included. They can be read without a GPU or weights.
   `tools/summarize_runs.py` rebuilds the log index without PyTorch.
2. **Rerun evaluation of the historical checkpoint.** The materialized audit
   manifests preserve original token sequences and target spans. This requires
   the matching checkpoint, which was not included, plus its matching runtime
   settings. A current random model cannot reproduce its accuracy.
3. **Train a new instance from raw Gutenberg.** The pinned downloader and new
   baseline builder make this executable, but the resulting corpus/questions and
   split are not the original dataset. Similar performance is an experiment,
   not a guaranteed reproduction of the historical numbers.

## Missing historical artifacts

| Artifact | Status / consequence |
|---|---|
| Model checkpoints | Not uploaded; cannot rerun historical predictions here |
| Original full TXT and QA corpora | Excluded because of size |
| Original cloze dataset builder | Absent; replacement builder is explicitly new |
| Dataset commit SHA / cleaning specification | Unknown for historical runs |
| Complete checkpoint ancestry and stage configs | Not present for every diagnostic log |
| Complete package lockfile / GPU specification | Not present; audit summaries retain recorded Torch version |
| Exact evaluation sequences | Included in both audit manifests |
| Historical training metrics | Included, with original schemas and counters |

The audit summaries record `torch_version=2.13.0+cu132` and FP32 scoring. These
are supplied metadata, not an environment reconstructed during packaging.
Release verification uses a separate CPU environment described in
[validation notes](VALIDATION.md). Compatibility ranges in requirements.txt are
not a substitute for the historical lockfile.

## What to save for future experiments

Keep a config snapshot, command, startup seed, dataset commit, raw/build
manifests, train/validation book IDs, exact evaluation manifest, dependency
versions, GPU/precision settings, checkpoint ancestry and writer sidecar. Record
whether metrics use teacher forcing, greedy generation, answer-only or
answer-plus-terminator loss. Store latest and metric-selected checkpoints if you
intend to compare both objectives.

A weights-only checkpoint is smaller than a full AdamW resume checkpoint because
it omits moment estimates, stream cursors, RNG state and data-order metadata.
The included exporter writes a separate file and allowlists scalar model/writer
metadata. Do not overwrite the private resumable original. A future release can
host those exports separately; this repository does not publish or upload them
automatically.

## Sharing without the full dataset

Share the source/build manifests, book splits, tokenizer identity, build command
and evaluation fixtures. Readers can regenerate a NEW dataset using the recorded
revision, or reproduce the exact corpus if the historical generation pipeline
is later supplied. Avoid implying that matching block size and tokenizer alone
recreates the old QA distribution.
