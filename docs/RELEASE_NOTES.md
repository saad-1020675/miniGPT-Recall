# Release preparation notes

This package combines the supplied model-source archive and the separate
training-diagnostics archive. No checkpoints, training corpora, credentials or
local caches are bundled.

## Privacy changes

- Replaced absolute Windows user-directory paths in the larger audit manifest
  and both audit summaries with portable provenance references.
- Preserved model input tokens, token fingerprints, question/answer metadata,
  measured metrics, raw training log values and public Gutenberg record IDs.
- Refreshed the redacted manifest's aggregate hash; retained its original hash
  in audit metadata. This hash change reflects path redaction, not changed input.
- Added environment-based HF_TOKEN handling and ignore rules for credentials,
  generated corpora, checkpoints and local editor/cache files.
- Normalized ZIP entry timestamps and omitted local filesystem metadata.

Automated pattern checks cover common home-directory paths, email addresses and
recognizable credential formats. They are not proof that arbitrary prose or
future additions cannot contain personal information. Public corpus characters'
names and historical authors are not treated as the project author's identity.

## Packaging and behavior changes

- Kept the source's explanatory comments, correcting a few stale dimensional
  examples and adding release-specific documentation.
- Changed default widths from 384 to 256 to match the small architecture behind
  the supplied audits. Preserved the uploaded architecture/mixed ratio in
  `configs/supplied_384_experiment.json`. This does not convert a trained checkpoint.
- Default initialization paths are None, avoiding reliance on absent local
  checkpoints. Config JSON overrides support repeatable experiment settings.
- Fixed generator control flow: removed unconditional hard-coded manifest replay;
  introduced explicit replay mode, sample selection, JSON output, and complete
  sample iteration. Generate/paste modes are reachable again.
- Used the configured generation checkpoint path and restricted inference loading
  to weights-only deserialization. Full training resume loading remains in the
  original trusted-checkpoint path.
- Added a seeded launcher around the existing training scripts. It does not
  duplicate the loops or alter the model's learned equations.
- Added a revision-recording downloader and a NEW deterministic baseline cloze
  builder, with book grouping, text deduplication and optional fixture holdouts.
- Sorted prose discovery and grouped Gutenberg prose splits by book ID, honoring
  a new QA build manifest when present. This changes the historical prose split;
  start a new experiment rather than silently resuming with a new file order.
- Added a weights-only checkpoint exporter, parameter inspector, training log
  indexer, CPU verification tests, requirements, ignore rules and documentation.
- Fixed an outdated test fixture that lacked the audit's sequence_windows field.

`training_RNN.py` and the historical curriculum notes are retained for context.
Older synthetic datasets and their construction scripts are not supplied.
Chunked writes remain available as an experimental control, but documentation
uses the current preferred single-write-per-window baseline.

## Results and limitations

No historical accuracy was recalculated using trained weights, because those
weights were not present. README tables are derived from the supplied logs and
summaries. They distinguish assistance, sequence length, teacher forcing,
validation population, and peaks at different checkpoints. CPU smoke checks
verify mechanics; they do not establish the reported recall or prose quality.

The source code is licensed under the [MIT License](../LICENSE).

Project Gutenberg text and tokenized excerpts retain their applicable
copyright status and terms; they are not covered by this project's code license.
