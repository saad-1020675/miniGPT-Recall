# Training and resume guide

## Entry points

`training_RNN_gutenberg.py` contains the current standard and passage-backed
recall loops. `MEMORY_TEST` selects them only when running that file directly.
`training_RNN_mixed.py` is standalone mixed training; `MEMORY_TEST` does not
select its mode. The seeded `tools/run_training.py` wrapper chooses these loops
without duplicating their code. `training_RNN.py` is a legacy synthetic FAQ
trainer, retained for history; its original synthetic dataset is not included.

A seed helps repeat initializations and sampling. It does not guarantee bitwise
agreement across PyTorch versions, GPU kernels, mixed precision or hardware.
For comparison, save the entire config, dependency versions, data manifests and
checkpoint writer sidecars.

## Objectives

- **Standard decoder-only:** independent prose windows; bypass memory reads and
  writes, freeze memory-specific components, optimize the language-model decoder.
- **Recall:** unroll history windows with memory connected through updates;
  optimize next-token prose loss and an additional weighted recall loss.
- **Mixed:** alternate recall batches and independent decoder-only prose batches
  using a deterministic update schedule; shared decoder/embeddings receive both
  objectives. Memory components receive the recall objective.

The latest passage-backed recall and mixed loops include the answer's newline
terminator in the extra recall loss. Accuracy/answer CE diagnostics still score
the answer-token span, excluding the terminator. A lower combined recall loss
therefore need not imply a lower answer-only CE or higher exact accuracy. Train
the terminator to learn stopping, and verify it with greedy free generation.

`Recall LM` is next-token LM loss over the recall sequence, not just answers and
not a measurement on the dedicated prose stream. `Prose LM` evaluates independent
prose windows with memory bypassed. `Selection loss` in mixed training is the
equal-weight mean of prose LM and recall combined loss. Thus `best.pt` is not
necessarily the best recall-accuracy checkpoint or the lowest prose-LM checkpoint.
Use both `best.pt` and `latest.pt` deliberately.

## Update ratios versus token ratios

Let r be the fraction of recall updates, R the recall batch size, P the prose
batch size, and N the total recall-window count. Both use block length T. The
expected recall fraction of processed target tokens is:

`r * R * N / (r * R * N + (1-r) * P)`.

For N=5 and R=P, r=0.5 gives about 83.33% recall tokens and r=0.3 gives about
68.18%. Increasing `MIXED_PROSE_BATCH_SIZE` changes this. Loss is averaged by the
loop; processed-token share is not identical to gradient influence. Compare
logged stream token counts and both validation objectives.

## Pools, validation and stops

`POOL_SIZE` budgets raw source-file bytes selected for each RAM pool, not all
Python/token/tensor allocations. Mixed training splits that budget between its
two streams. Compact CPU token buffers and carry handling reduce redundant
storage. One unusually large book and the separate validation cache can still
exceed the nominal budget.

`VALIDATION_BATCHES × batch_size` controls the usual upper number of evaluated
examples, not `MIXED_PROSE_BATCH_SIZE`. The supplied fixed recall population has
256 examples. Size-limit macros restrict files and are intended for debug only.
Validation caches use stable sampling; increasing validation coverage can reveal
population differences hidden by a small fixed subset.

`EVAL_INTERVAL_TOKENS`, `CHECKPOINT_INTERVAL_TOKENS`, `WARMUP_TOKENS`, `MAX_TOKENS`,
`MIN_LEARN_RATE`, `STOP_AFTER_POOL_CT` and `TOKEN_STOP_COUNT` retain their original
macro names. Mixed training schedules LR using total processed tokens across
both streams. `TOKEN_STOP_COUNT` is checked at a completed optimizer-update
boundary; latest is saved before stopping, so the total can overshoot. The loop
can also finish because its selected stream exhausts `num_epochs`. Set enough
epochs/data for the desired token budget.

## Stage changes and full resume

| Action | Configuration |
|---|---|
| Start from random weights | `*_INIT_CHECKPOINT=None`, new stage output directory |
| Transfer weights from another run | Set destination `*_INIT_CHECKPOINT`, new output directory |
| Resume interrupted same-stage training | Same output directory with its `latest.pt`; unchanged architecture/data |
| Change widths, slot count, depth or objectives | New output directory and weights-only initialization; step-zero evaluation |

Existing latest checkpoints take precedence over initialization in their stage.
Do not delete your only checkpoint to change stages; point the new stage at a
new directory. Mixed resumable checkpoints contain optimizer/scaler/RNG state,
per-stream cursors, counters and configuration guards. Weight-only initialization
starts new optimizer state. Keep checkpoint input and output directories distinct
when experimenting with migrations.

Width/slot/depth migration can copy compatible regions and initialize new ones;
it cannot guarantee the source model's function survives. Shrinkage discards
weights. RoPE caches are derived and rebuilt for a target block size, but a new
block size also needs a matching QA dataset. Head count, routing passes and
pooling-head runtime semantics must match training even if matrix shapes load.
Do not infer a valid experiment from `load_state_dict` success alone.

Training checkpoints use Python serialization; load full resumable files only
from a source you trust. Public release weights can be exported using the
allowlisted weights-only exporter, which drops file lists and optimizer state.

## Activation checkpointing

`MIXED_ACTIVATION_CHECKPOINTING=True` uses non-reentrant PyTorch recomputation in
the recall graph and preserves RNG state. Memory stays connected across history
windows, so this is not truncated BPTT or the discarded chunk-writing variant.
It trades extra computation for lower saved-activation VRAM. Actual peak memory
and speed depend on shapes, batch sizes, device and kernel dispatch. No GPU
memory benchmark was rerun when preparing this release.

## Historical curriculum

`historical_training_notes.txt` is retained verbatim apart from line endings.
It records earlier synthetic and assisted stages, including older loss weights
and manual checkpoint handling. It is not a complete executable recipe for the
final audited model. Development changed encoder depth, persistent GRUs, slot
width, routing, support labels, number of windows, and the terminator objective.
The included logs do not identify every upstream checkpoint/config transition.
