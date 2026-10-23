# MiniGPT Recall

MiniGPT Recall is an experimental Transformer–GRU language model that learns to
carry information beyond its immediate decoder context. Its objective is to
read prose **before knowing the question**, compress that prose into a fixed
set of persistent memory slots, and later answer a cloze question using memory
and the question window.

The small configuration has **24,509,440 unique parameters**, including tied
input/output embeddings, and an **8 × 256** persistent state. It is trained on
English prose from **Project Gutenberg**, with passage-backed fill-in-the-blank
questions. This repository collects the current model, training loops,
controlled audits, and diagnostics from its development.

This is a research prototype. The demonstrated task is cloze recall, usually
one missing word, not unrestricted factual question answering. Coherent story
generation and transfer to questions about the model's own generated prose
remain separate objectives under investigation.

Currently exploring pre-trained models to generate open-ended style questions
instead of one-word cloze style questions from the Gutenberg dataset in 5
context window sequences. Limiting recall to 4 context windows is a design
choice, not necessarily a limitation that has been established.

Also investigating using a MoE architecture pattern to seperate prose
generation from recall objectives due to the fact that they seem to 
compete as training objectives.



## Measured results

The supplied files support the following results. “Exact” in this table means
**teacher-forced answer-token exact accuracy**: every answer token must be the
argmax when conditioned on the gold preceding answer tokens. It does not test
free-generation stopping or story quality.

| Experiment / population | Examples | Exact | First token | Notes |
|---|---:|---:|---:|---|
| No-assist training, best logged validation | 256 | 85.94% | 89.84% peak separately | Peaks can be different checkpoints |
| No-assist final checkpoint, trainer validation replay | 256 | 84.77% | 89.45% | 2,200,002,560 tokens; fixed trainer examples |
| Same no-assist checkpoint, broader audit | 1,024 | 69.34% | 72.85% | Reservoir sample from 10,000 scanned validation records |
| Mixed 50/50 updates, best logged validation | 256 | 86.33% | 91.80% peak separately | Uses answer-plus-newline training; 50/50 is updates |

The first two audit populations differ substantially. The smaller validation
result must not be presented as accuracy on the full validation distribution.
The broader sample also is not an exhaustive or independent-book benchmark.

A three-history-window stage reached 91.41% exact on its 256 validation samples;
it is a shorter task than the four-history-window result. Historical assisted,
exclusive-window, and synthetic experiments are not interchangeable with the
no-assist result above. See [results](docs/RESULTS.md), the
[training index](results/training_index.json), and the raw diagnostic files.

## Architecture

A sequence normally contains five windows of 256 GPT-2 tokens:

- **W0–W3:** history prose. Each completed window is encoded bidirectionally,
  pooled into candidate slots, and integrated into persistent memory.
- **W4:** natural filler plus the cloze prompt and answer during training.
  The decoder reads the accumulated memory; W4 never writes to it.

The decoder contains four causal Transformer blocks, each with cross-attention
to memory. The memory encoder has four bidirectional Transformer blocks. The
compressor uses learned slot identities, competitive attention pooling, three
routing passes, and an additional multi-head pooling branch. Each persistent
slot has its own GRUCell with separate learned weights. A shared, temporary GRU
inside the compressor refines routing between passes; it is distinct from the
persistent per-slot GRUs.

The encoder and decoder share the token embedding table, and the decoder output
projection is tied to that same table. RoPE supplies positions within windows.
There is no external retrieval index, document store, or mixture-of-experts
router in this implementation.

For the small configuration:

| Setting | Value |
|---|---:|
| Vocabulary | 50,257 GPT-2 tokens |
| Window size | 256 tokens |
| Encoder / decoder width | 256 / 256 |
| Encoder / decoder blocks | 4 / 4 |
| Transformer heads / pooling heads | 4 / 4 |
| Persistent slots | 8 × 256 features |
| Routing passes / writes per full window | 3 / 1 |
| Unique parameters | 24,509,440 |
| Persistent state alone, FP32, batch size 1 | 8 KiB |

That state size excludes model weights, activations, temporary routing tensors,
and optimizer state. The state is a lossy learned representation, not a verbatim
archive. The feature-count ratio is 32:1 for one window and 128:1 for four
windows at equal encoder/slot width; these ratios are not information-theoretic
bounds on what it can remember.

See [architecture details](docs/ARCHITECTURE.md).

## What is included, and what is missing

Included: the model, standard/recall/mixed training scripts, checkpoint migration
logic, generator and replay modes, 22 historical training diagnostic logs, two
complete audit bundles, and their materialized evaluation token sequences.

**Checkpoints and the full dataset are not included.** The original QA-building
script, complete historical environment lockfile, and exact dataset revision
were also absent from the supplied files. Consequently this release does not
yet provide exact end-to-end reproduction of the reported checkpoint.

A new deterministic baseline builder is included so someone can run the task
from source. Its word selection and split must not be assumed to match the
historical dataset. The frozen audit manifests preserve the original evaluation
inputs, so the matching checkpoint can be evaluated later without reconstructing
the entire corpus. See [reproduction status](docs/REPRODUCTION.md).

## Setup

Use Python 3.10 or newer. Run commands from the repository root. A CUDA-capable
GPU is useful for training; CPU is suitable for helper tests and small smoke
runs. Install the appropriate PyTorch build using the
[official installation instructions](https://pytorch.org/get-started/locally/).
Then install the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CPU or CUDA PyTorch build you selected, then:
python -m pip install -r requirements.txt
python tools/inspect_model.py
python -m unittest discover -v
```

The requirements are compatibility ranges, not a claim that every combination
has been tested. GPT-2 tokenization may download its vocabulary files on first
use. Dataset downloads require internet access. If Hugging Face authentication
is needed, use the `HF_TOKEN` environment variable; do not place a token in code.

## Rebuild a new dataset

The download uses `manu/project_gutenberg`, English split `en`. A moving revision
such as `main` is resolved to a commit SHA and recorded with exported file hashes.
Keep `download_manifest.json` with your experiment.

```bash
python download_gutenberg.py
python build_gutenberg_qa.py \
  --holdout-manifest results/audits/no_assist_1024/manifest.json \
  --holdout-manifest results/audits/trainer_validation_256/manifest.json
```

The second command produces `gutenberg_qa/train/*.jsonl`,
`gutenberg_qa/validation/*.jsonl`, and a build manifest. It groups examples by
Gutenberg book ID, selects one edition per ID, removes identical text duplicates,
and forces books used by the provided audit fixtures into validation. It builds
five-window records; the loader can crop these to 2–5 windows while preserving
the question's source window.

This baseline retains raw text, including any Gutenberg headers and footers;
it does not reproduce undocumented historical cleaning or question selection.
For an initial smoke run use `--max-books 100 --max-records-per-book 10`, check
that both splits are nonempty, and use a fresh output directory. Avoid treating
a reduced-data debug run as a benchmark.

See [data format and caveats](docs/DATA.md).

## Configuration and training

`config.py` keeps the existing macro names and explanatory comments. The default
architecture is the small 8 × 256 configuration; initialization paths default
to `None`. `configs/supplied_384_experiment.json` records the architecture and
mixed ratio in the uploaded config, which used a 384-wide backbone and 8 × 384
memory. That larger model is about 45.46M parameters and is **not** the sub-30M
configuration behind the 8 × 256 audits.

You may edit `config.py` directly, or pass a JSON override preset before the
model is imported. Unknown config names are rejected. The training launcher
seeds Python/PyTorch and selects one original loop:

```bash
# Independent prose objective, with memory reads/writes bypassed
python tools/run_training.py standard --decoder-only --config configs/baseline_8x256.json

# Recurrent cloze training; both memory and decoder train
python tools/run_training.py recall --config configs/baseline_8x256.json

# Mixed prose and recall objectives; separate data streams, shared model
python tools/run_training.py mixed --config configs/baseline_8x256.json
```

These commands are alternatives, not a full historical recipe. To transfer
weights between stages, set the destination stage's `*_INIT_CHECKPOINT` to the
previous checkpoint and use a **new checkpoint output directory**. To resume the
same stage, leave its `latest.pt` in its existing output directory and keep its
architecture/data configuration unchanged. Resuming restores optimizer,
scheduler progress, stream positions and RNG state as supported by that loop.
A weights-only export cannot resume optimizer history.

For exclusive recall training, the launcher also supports `recall --memory-only`
and `recall --decoder-only`. Source-window filtering and assist placement remain
config settings. A simple progression is short histories first, then longer
histories, with paired no-assist evaluation at each stage. The archive does not
establish a single optimal curriculum; historical notes are provided as notes,
not as a guaranteed route to the table above.

Key controls:

| Macro | Meaning |
|---|---|
| `MIXED_RECALL_RATIO` | Fraction of optimizer updates assigned to recall; default 0.3 |
| `MIXED_PROSE_BATCH_SIZE` | Prose training batch size only; recall/validation use `batch_size` |
| `MEMORY_WINDOWS_PER_SEQUENCE` | Total windows, including question window, from 2 through 5 |
| `GUTENBERG_SUPPORT_WINDOW` | `None` for no assist; `"source"` copies evidence into its source window |
| `GUTENBERG_SUPPORT_LABEL` | Whether an enabled assist includes its explicit label |
| `GUTENBERG_EVIDENCE_WINDOWS` | `None` or selected original history indices, e.g. `[2]` |
| `memory_write_chunk_size` | Set equal to `block_size` for one write per window |
| `RECALL_LOSS_WEIGHT` | Extra weight on recall targets; latest loops train answer plus newline |
| `TOKEN_STOP_COUNT` | Token stop target; latest checkpoint is saved before stopping |
| `POOL_SIZE` | Raw-file RAM-pool budget; not a hard cap on total process RAM |
| `MIXED_ACTIVATION_CHECKPOINTING` | Recompute recall activations in backward; retains cross-window gradients |
| `MIXED_VERIFY_DATA_HASH` | Verify content on resume; disabling reduces startup checks |

A 0.3 recall update ratio is **not** a 30/70 token split: with equal batch sizes
and five recall windows, its expected token share is approximately 68.18% recall
and 31.82% prose. A 0.5 update ratio gives approximately 83.33% recall tokens.
Read the logged `recall_tokens` and `prose_tokens` rather than inferring them from
the ratio. See [training guide](docs/TRAINING.md) for loss and resume details.

## Evaluate recall

With the appropriate checkpoint and matching 256-wide config:

```bash
python window_retention_audit.py \
  --checkpoint checkpoints/reference/no_assist_8x256.pt \
  --manifest results/audits/trainer_validation_256/manifest.json \
  --output run_outputs/trainer_replay --assist none --chunk-size 256

python window_retention_audit.py \
  --checkpoint checkpoints/reference/no_assist_8x256.pt \
  --manifest results/audits/no_assist_1024/manifest.json \
  --output run_outputs/broad_replay --assist none --chunk-size 256
```

Those checkpoint paths are examples, not bundled files. Without a sidecar, set
routing passes, pooling heads, backbone width and block size to match training;
these runtime choices cannot all be recovered from parameter shapes. Keep
writer metadata beside exported checkpoints. For a new dataset, omit
`--manifest` and use `--data gutenberg_qa/validation`; this creates a new audit
population, so do not compare its headline directly with a different manifest.

To test deployment-style answers, replay the exact token sequences with greedy
free generation. No answer tokens are supplied to the decoder:

```bash
python generate_RNN.py --mode replay \
  --checkpoint checkpoints/reference/no_assist_8x256.pt \
  --manifest results/audits/trainer_validation_256/manifest.json \
  --sample-index 0
# Omit --sample-index to replay every example; use --save-json to save answers.
```

The replay reports full generated-answer exact accuracy, including unwanted
extra text through its effect on the match. It separately reports whether the
generated token prefix matches the answer. The latter does not verify stopping.
The other two audit scripts provide additional interventions; see their
`--help` and [evaluation guide](docs/EVALUATION.md).

## Generate prose, then ask a cloze question

```bash
python generate_RNN.py --mode generate \
  --checkpoint checkpoints/reference/your_model.pt \
  --prompt "Eleanor arrived at the village carrying a locked wooden box." \
  --compare-memory --save-json run_outputs/story.json
```

After seeing the generated prose, enter only a sentence with its missing word
replaced by `[BLANK]`, for example:

```text
Eleanor arrived at the village carrying a locked [BLANK] box.
```

The default wrapper supplies the cloze instruction and `Answer:`. Do not also
paste the instruction unless using `--raw-question-prompt` for the entire exact
prompt. Type real line breaks, not literal backslash-n characters. Answer
decoding is already greedy and stops at newline/EOT or the token budget; prose
sampling uses `--temperature` and `--top-k`.

`--mode paste --history-file story.txt` instead ingests existing UTF-8 text,
requiring at least `history_windows × block_size` tokens and using exactly that
prefix. Manual question placement uses neutral filler, whereas training uses
natural prose filler. This, generated-text quality, source-window boundaries,
and question formatting can all cause distribution shifts. Exact manifest replay
is the first check when generation disagrees with training validation.

## Extending the experiments

Compare changes using the same starting checkpoint, dataset manifest, writer
settings, token budget, and evaluation population. Useful axes include slot
width versus slot count, encoder width/depth, pooling/routing variants, and
prose/recall objective balance. Wider states change both representational
capacity and recurrent parameter count; they are not a pure storage-size test.

The mixed trainer includes migration for several architecture changes, but
partial weight transfer does not preserve the original function. Save the
source config, use a new optimizer/output directory, and evaluate at step zero.
The chunk-writing experiments are retained for comparison; the project's current
preferred baseline is one full-window write, not more frequent chunk writes.

For sharing a checkpoint without its optimizer and local file-order metadata:

```bash
python tools/export_checkpoint.py checkpoints_mixed/latest.pt exports/model.pt
```

Retain your full checkpoint privately for resuming. Distribute the exported
weights, writer sidecar, matching config, dataset revision, and evaluation
manifest together. The exporter is useful when full optimizer checkpoints exceed
an upload limit; it does not create a missing checkpoint.

## Release notes and licensing

Local user paths in supplied diagnostic metadata have been replaced with
portable references. Evaluation token IDs and measured metrics are preserved.
See [release notes](docs/RELEASE_NOTES.md) for code changes and validation limits.

License under MIT License
