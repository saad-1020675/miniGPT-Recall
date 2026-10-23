# Release verification

Verification was performed on CPU with Python 3.12, PyTorch 2.14.1+cpu,
tiktoken 0.14.0 and NumPy 2.3.5. This is a release smoke
environment, not the original training environment. No historical trained
checkpoint or GPU training run was available.

Completed checks:

- Compiled every Python source file.
- Instantiated the default model and verified **24,509,440** unique parameters;
  confirmed the token embedding and output weight are tied.
- Ran all **16** existing audit and new release tests successfully, including
  mocked downloader revision/deduplication and book-split checks.
- Checked frozen audit manifests: all 1,280 samples retain valid token hashes,
  shifted answer spans, vocabulary bounds and matching aggregate hashes.
- Built real GPT-2 cloze candidates, checked exact evidence spans, and materialized
  2–5-window sequences while preserving the selected source and answer positions.
- Compared tiny-model checkpointed and ordinary recurrent training: losses,
  states and gradients match within test tolerance; gradients reach the encoder,
  compressor and every persistent slot GRU.
- Built a temporary 24-book CPU smoke dataset with both splits, then exercised
  mixed recall/prose updates, checkpoint-before-stop, same-stage resume with no
  extra updates at an already reached stop count, weights-only export and replay.
- Exercised independent standard decoder-only and recall training entry points
  using the tiny configuration and bounded token stops.
- Ran the window-retention audit on eight temporary examples and exercised
  both generated-prose mode and manifest replay with exported tiny-model weights.
- Checked documentation links and scanned text files for common private-path,
  email and credential patterns; no remaining matches were found.

The temporary smoke corpus, random/untrained model weights and generated outputs
are not included or counted as research results. Training was deliberately far
too short to learn the task. The full raw Gutenberg download, long GPU runs,
historical checkpoint accuracy and cross-hardware numerical reproducibility
were not verified. The downloader's live API behavior must be checked when
rebuilding a full dataset; its resolved revision should be recorded.

The CI workflow repeats compilation and the unit tests on CPU. Test results do
not establish model quality or a performance guarantee for a newly built corpus.
