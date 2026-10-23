# Architecture and update semantics

For batch size B, window length T, encoder/decoder width C, slot count M and slot
width D, the embedding/encoder output is `[B,T,C]`. The compressor returns
`[B,M,D]`. Persistent GRU state and decoder-visible memory both have `[B,M,D]`
shape; there is no flattening to a single C-wide recurrent vector.

The encoder consumes completed windows bidirectionally. It can use all tokens in
the window, but not future windows. Current-window decoder logits read incoming
memory; the newly written state is available to the next window. The final
question window does not write. Consequently neither future prose nor the
question guides the earlier memory writes at inference time.

## Compressor

Working slot queries start from incoming memory plus learned slot identities.
Keys/values come from the encoder. For each token, scores compete across slots;
assignments are then normalized over tokens separately for each slot to produce
a weighted average. The additional multi-head branch performs its own slot
competition per head, pools feature subsets, concatenates them, and adds an
output projection to the original candidate.

Between routing passes, a **shared** temporary GRU and residual MLP update the
working queries using the previous pooled candidate. The last pass returns the
candidate; only then do the persistent GRUs perform their update. Increasing
routing passes therefore does not mean the persistent state is written that
many times. Temporary working state is recreated on each compressor call.

This is a custom iterative competitive pooling mechanism. It is related to Slot
Attention and attention pooling, but is not an exact implementation of the
original Slot Attention algorithm or standard Set Transformer PMA.

## Persistent state

There is one GRUCell per slot. For slot s, the incoming candidate x and previous
state h are D-dimensional. PyTorch's GRUCell uses:

```
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
h_next = (1 - z) * n + z * h
```

The gates are runtime vectors computed from inputs and learned weights. They
are not one learned scalar per slot. Each persistent cell has `6 D² + 6 D`
parameters when input and hidden widths are both D. At 8 × 256, the persistent
GRUs have 3,158,016 learned parameters in total, while their runtime memory is
only 2,048 values per example. This distinction matters when scaling width.

Per-slot weights permit different update behavior, but do not enforce a fixed
book, window, character or fact per slot. Retention and extraction are learned
jointly. Distinct slots do not imply lossless encoding or perfect retrieval.

## Decoder

Each causal block applies self-attention, memory cross-attention, and a feed
forward network with residual connections and layer normalization. Decoder
tokens form cross-attention queries; slots provide keys and values after learned
projections. The question influences retrieval through these queries, even
though it was absent when the history was written.

The last output projection shares the token embedding weight. Freezing only
one reference to the tied weight freezes the same Parameter through both names;
optimizers and parameter counts should not count it twice.

## Widths and compression

C and D are independent. A 256→384 projection can generate 384 coordinates from
a 256-dimensional vector, but a linear projection's rank remains at most 256.
It cannot recover information already discarded. Nonlinear processing across
tokens and recurrent updates make whole-system behavior more complex than that
single projection. Equal widths avoid that particular width change; they do
not avoid pooling many tokens into fewer slots.

The release baseline has C=D=256, M=8 and four history windows. Each full window
is encoded once and written once (`memory_write_chunk_size=block_size`). Chunked
writes remain configurable as a historical ablation, but are not the preferred
baseline. Activation checkpointing is a separate training-time optimization:
it recomputes activations without intentionally adding writes or detaching the
memory across windows.
