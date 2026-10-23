# Supplied training runs and audit results

This index is generated from the separate training-diagnostics archive. The JSONL
files preserve their original metric schemas and record order. Filenames are
historical labels, not complete experiment configurations. Missing architecture
or upstream checkpoint metadata is unknown; do not infer it from a filename.

Peak exact and peak first-token accuracy may occur at different checkpoints.
These are repeated evaluations, not independent test-set trials. Higher peaks
across a long series can also reflect selection over many checkpoints.

## Training log inventory

| Log | Records | Final logged tokens | Peak exact | Peak first | Best prose LM |
|---|---:|---:|---:|---:|---:|
| [memory_diagnostic_memslot32_per-slotGRU.jsonl](../results/training/memory_diagnostic_memslot32_per-slotGRU.jsonl) | 12 | 240,009,216 | 64.84% | 68.75% | — |
| [memory_diagnostic_width384_mem8_mixed30-70.jsonl](../results/training/memory_diagnostic_width384_mem8_mixed30-70.jsonl) | 17 | 320,008,192 | 80.86% | 86.72% | 4.7849 |
| [memory_diagnostics_1_chunk._sequence-3_no_label.jsonl](../results/training/memory_diagnostics_1_chunk._sequence-3_no_label.jsonl) | 12 | 240,009,216 | 62.50% | 66.02% | — |
| [memory_diagnostics_Decoder_Only_W4_Evidence.jsonl](../results/training/memory_diagnostics_Decoder_Only_W4_Evidence.jsonl) | 46 | 225,013,760 | 71.48% | 80.86% | — |
| [memory_diagnostics_W0_Evidence_weight10.jsonl](../results/training/memory_diagnostics_W0_Evidence_weight10.jsonl) | 253 | 3,680,010,240 | 57.42% | 65.62% | — |
| [memory_diagnostics_W3_Evidence_Memory_Only_Pt1.jsonl](../results/training/memory_diagnostics_W3_Evidence_Memory_Only_Pt1.jsonl) | 173 | 865,013,760 | 35.55% | 39.84% | — |
| [memory_diagnostics_chunks_sequence-3.jsonl](../results/training/memory_diagnostics_chunks_sequence-3.jsonl) | 25 | 500,011,008 | 50.39% | 53.52% | — |
| [memory_diagnostics_cloze_W3_evidence_weight20.jsonl](../results/training/memory_diagnostics_cloze_W3_evidence_weight20.jsonl) | 226 | 1,995,018,240 | 51.95% | 60.94% | — |
| [memory_diagnostics_cloze_baseline.jsonl](../results/training/memory_diagnostics_cloze_baseline.jsonl) | 160 | 800,010,240 | 93.75% | 95.70% | — |
| [memory_diagnostics_cloze_joint_W0.jsonl](../results/training/memory_diagnostics_cloze_joint_W0.jsonl) | 49 | 980,008,960 | 80.86% | 88.28% | — |
| [memory_diagnostics_iterative_slot_random_assist-labels.jsonl](../results/training/memory_diagnostics_iterative_slot_random_assist-labels.jsonl) | 55 | 1,100,001,280 | 85.94% | 91.02% | — |
| [memory_diagnostics_memslot16_W2_Frozen_Decoder.jsonl](../results/training/memory_diagnostics_memslot16_W2_Frozen_Decoder.jsonl) | 40 | 800,010,240 | 60.16% | 67.19% | — |
| [memory_diagnostics_memslot16_joint.jsonl](../results/training/memory_diagnostics_memslot16_joint.jsonl) | 22 | 440,012,800 | 7.03% | 7.81% | — |
| [memory_diagnostics_memslot16_random_evidence.jsonl](../results/training/memory_diagnostics_memslot16_random_evidence.jsonl) | 55 | 1,100,001,280 | 37.50% | 42.58% | — |
| [memory_diagnostics_per-slotGRU.jsonl](../results/training/memory_diagnostics_per-slotGRU.jsonl) | 55 | 1,100,001,280 | 63.28% | 69.53% | — |
| [memory_diagnostics_slot-width128_memslot16_sequence-3.jsonl](../results/training/memory_diagnostics_slot-width128_memslot16_sequence-3.jsonl) | 12 | 240,009,216 | 69.14% | 73.05% | — |
| [memory_diagnostics_width256_memslot8_sequence-3.jsonl](../results/training/memory_diagnostics_width256_memslot8_sequence-3.jsonl) | 13 | 500,011,008 | 89.84% | 92.58% | — |
| [memory_diagnostics_width256_memslot8_sequence-4.jsonl](../results/training/memory_diagnostics_width256_memslot8_sequence-4.jsonl) | 20 | 400,015,360 | 91.41% | 93.75% | — |
| [memory_diagnostics_width256_memslot8_terminator.jsonl](../results/training/memory_diagnostics_width256_memslot8_terminator.jsonl) | 110 | 2,200,002,560 | 82.42% | 87.89% | — |
| [memory_diagnostics_width256_sequence-5_no-assist.jsonl](../results/training/memory_diagnostics_width256_sequence-5_no-assist.jsonl) | 110 | 2,200,002,560 | 85.94% | 89.84% | — |
| [mixed_diagnostics_width256_memslot8_mixed30-70.jsonl](../results/training/mixed_diagnostics_width256_memslot8_mixed30-70.jsonl) | 69 | 1,340,014,592 | 81.25% | 88.28% | 4.6997 |
| [mixed_diagnostics_width256_memslot8_mixed50-50.jsonl](../results/training/mixed_diagnostics_width256_memslot8_mixed50-50.jsonl) | 112 | 2,200,014,848 | 86.33% | 91.80% | 4.7207 |

The 30/70 mixed log stops at about 1.34B logged tokens; the 50/50 log reaches
2.20B. This archive therefore does not support an equal-duration final comparison
of those two runs. A log can include resumes and repeated step-zero events;
`tokens_this_run` is not always the global counter. Dedicated prose loss appears
only where that objective was evaluated. Historical recall LM is not a direct
substitute for dedicated prose LM.

## No-assist audit comparison

Both audit summaries identify the same 8 × 256 no-assist checkpoint at
2,200,002,560 processed tokens. The checkpoint itself is not included.

| Metric | Trainer population (256) | Broader population (1,024) |
|---|---:|---:|
| Normal exact | 84.77% | 69.34% |
| Normal first token | 89.45% | 72.85% |
| Normal answer CE | 0.85185 | 1.87874 |
| NoRead exact | 6.25% | 4.98% |
| SourceOnly exact | 85.94% | 73.34% |
| AtSource exact | 84.38% | 70.61% |

Artifacts: [trainer replay report](../results/audits/trainer_validation_256/report.md)
and [broader audit report](../results/audits/no_assist_1024/report.md). Each directory
includes its full summary, per-sample CSV and materialized token manifest.

On the broader audit, 22 final errors were correct at source and later became
incorrect; 292 were already incorrect at source and remained incorrect. Thus most
final failures in that test cannot be attributed solely to later overwrites.
This does not isolate the compressor from the encoder, GRU or decoder. More
controlled component tests would be required for that claim.

## Rebuild the index

```bash
python tools/summarize_runs.py
```

The machine-readable [training index](../results/training_index.json) records
peak-token positions, final Normal/NoRead metrics, writer metadata where present,
and dedicated prose-loss minima. No new training results are added by the indexer.
