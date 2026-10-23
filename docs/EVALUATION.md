# Evaluation and interpretation

## Teacher forcing versus generation

At first-answer prediction, neither evaluation method supplies a gold answer
prefix. At later answer tokens, teacher forcing supplies the previous **gold**
tokens regardless of earlier predictions; greedy generation supplies its own
predictions. Parallel masked decoder logits implement teacher forcing in one
forward pass, so it need not be a literal iterative token loop.

First-token accuracy measures only the first GPT-2 subword token. Answer-token
exact requires all token argmax predictions to be right under gold prefixes. A
word can have multiple subword tokens, and a misspelling can reflect one incorrect
continuation. Teacher-forced exact omits answer termination. Greedy full-answer
exact instead generates through a newline/EOT/budget, strips outer whitespace,
and compares the complete resulting string with the target.

It is possible to get the correct answer prefix but continue with unwanted text.
The generator reports token-prefix match separately; do not call it full-answer
accuracy. An immediate newline yields an empty answer and is incorrect for a
nonempty gold answer. Changing temperature cannot fix a greedy first-token rank
in the thousands; investigate matching weights, runtime configuration, prompts,
memory ingestion and evaluation population first.

## Preferred window-retention audit

`window_retention_audit.py` supports variable history length and materialized
manifests. The supplied results include Normal, NoRead, source-only/at-source,
intermediate writes, reset-routing/reset-GRU interventions, reordered windows,
and paired error partitions. The final question input is held fixed across
conditions; it never writes memory. See `--help` for explicit runtime overrides.

A true NoRead control bypasses decoder cross-attention output. A zero memory
state is not equivalent: learned normalization/projection terms can still affect
outputs. Cross-attention ablation shows a dependence on the pathway; it alone
does not prove that every answer is retrieved faithfully.

`AtSource` decodes using the recurrent state just after the source window under
the normal preceding history. `SourceOnly` writes only the source window from
zero memory. These are different counterfactuals. If an answer is wrong at both
AtSource and final state, the audit cannot attribute that error to later writes;
possible causes still include encoding, compression, recurrent integration,
reader/decoder behavior, and question ambiguity. A source-only improvement is
not proof that a single component is defective.

`became_incorrect_after_source` counts correct-at-source answers that become
wrong in the final state. `incorrect_at_both_states` counts failures already
present at source. Rescues count wrong-to-right transitions. A small net accuracy
change can conceal both rescues and losses, so examine paired counts.
Bootstrap intervals resample examples, not independent books; dependence among
passages from one book can make them overconfident as generalization intervals.

`memory_bottleneck_audit.py` and `memory_intervention_audit.py` retain earlier
five-window experiments. They are additional tools, not substitutes for exact
trainer replay. For variable-length runs start with the window-retention audit.
Always inspect recorded runtime settings; this repository's model has a residual
multi-head pooling branch that a simplistic single-head reconstruction omits.

## Frozen fixtures

Both manifests store the actual sequence tokens, answer coordinates, answer
IDs and token fingerprints. They can be replayed without the original raw QA
files. `file` fields in the larger manifest are now portable provenance
references; they are not required local files. An overall manifest hash changed
when absolute paths were redacted; per-sample token hashes are unchanged and the
original aggregate hash is retained in the summary metadata.

Two identical manifests are retained in the trainer audit directory for backwards
compatibility: `manifest.json` and `trainer_validation_manifest.json`. Neither
requires putting a gold answer into an inference prefix. Exact replay constructs
that prefix only through the token immediately preceding the first gold answer.

## Comparing experiments

Use the same manifest, fixed runtime writer settings, comparable training data
and token budget. Report sample counts/source-window counts, assistance, history
length, backbone/slot widths, terminator training, checkpoint tokens and decoding
method. A shuffled-memory or reset-state intervention can be out of distribution;
its delta is evidence about sensitivity, not a universal causal decomposition.

The raw audit summaries support a large population difference: 84.77% exact on
the 256 trainer examples versus 69.34% on the broader 1,024-example sample. Neither
an architectural change nor a training gain should be judged from mismatched
populations. Add larger held-out books and greedy replay when claiming general
recall or deployment performance.
