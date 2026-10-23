# Window retention audit

Scores are teacher-forced. The final question window never writes to memory.

| Condition | W0 | W1 | W2 | W3 |
|---|---:|---:|---:|---:|
| AfterW0 | 74.48% | 6.37% | 3.98% | 3.81% |
| AfterW1 | 73.78% | 75.70% | 1.20% | 3.81% |
| AfterW2 | 73.08% | 74.10% | 66.53% | 3.39% |
| AfterW3 | 73.78% | 72.11% | 65.74% | 64.83% |
| Normal | 73.78% | 72.11% | 65.74% | 64.83% |
| W0Only | 74.48% | 6.37% | 3.98% | 3.81% |
| W1Only | 5.24% | 69.72% | 1.99% | 3.81% |
| W2Only | 5.24% | 3.19% | 74.90% | 2.54% |
| W3Only | 6.29% | 4.38% | 2.39% | 74.15% |
| SourceOnly | 74.48% | 69.72% | 74.90% | 74.15% |
| AtSource | 74.48% | 75.70% | 66.53% | 64.83% |
| FinalFromSource | 73.78% | 70.52% | 75.30% | 74.15% |
| NoRead | 6.29% | 5.18% | 3.59% | 4.66% |
| ResetRoutingW1 | 40.21% | 70.52% | 71.71% | 68.64% |
| ResetGRUW1 | 3.85% | 71.71% | 74.10% | 68.22% |
| ResetBothW1 | 4.20% | 70.52% | 73.71% | 68.64% |
| ResetRoutingW2 | 40.21% | 66.93% | 73.31% | 68.22% |
| ResetGRUW2 | 4.20% | 2.79% | 67.73% | 72.88% |
| ResetBothW2 | 3.85% | 2.79% | 75.30% | 73.31% |
| ResetRoutingW3 | 42.66% | 69.72% | 58.17% | 72.03% |
| ResetGRUW3 | 4.90% | 4.78% | 3.98% | 69.07% |
| ResetBothW3 | 6.29% | 4.38% | 2.39% | 74.15% |
| EvidenceFirst | 73.78% | 69.72% | 74.10% | 71.61% |
| EvidenceLast | 70.63% | 66.14% | 67.33% | 64.83% |
| ShuffleHistory | 2.45% | 3.19% | 2.79% | 1.69% |

| Paired comparison | Delta | Lost | Rescued |
|---|---:|---:|---:|
| MatchedOrder_All | -4.98 pp | 91 | 40 |
| W0_LaterWrites | -0.70 pp | 9 | 7 |
| W0_PrecedingHistory | +0.00 pp | 0 | 0 |
| MatchedOrder_W0 | -3.15 pp | 23 | 14 |
| W0_PrefixEffectFinal | +0.00 pp | 0 | 0 |
| W0_ResetRoutingW1 | -33.57 pp | 98 | 2 |
| W0_ResetGRUW1 | -69.93 pp | 202 | 2 |
| W0_ResetBothW1 | -69.58 pp | 201 | 2 |
| W0_ResetRoutingW2 | -33.57 pp | 98 | 2 |
| W0_ResetGRUW2 | -69.58 pp | 203 | 4 |
| W0_ResetBothW2 | -69.93 pp | 202 | 2 |
| W0_ResetRoutingW3 | -31.12 pp | 90 | 1 |
| W0_ResetGRUW3 | -68.88 pp | 198 | 1 |
| W0_ResetBothW3 | -67.48 pp | 195 | 2 |
| W0_Trajectory_AfterW0_to_AfterW1 | -0.70 pp | 6 | 4 |
| W0_Trajectory_AfterW1_to_AfterW2 | -0.70 pp | 5 | 3 |
| W0_Trajectory_AfterW2_to_Normal | +0.70 pp | 1 | 3 |
| W1_LaterWrites | -3.59 pp | 10 | 1 |
| W1_PrecedingHistory | +5.98 pp | 6 | 21 |
| MatchedOrder_W1 | -3.59 pp | 21 | 12 |
| W1_PrefixEffectFinal | +1.59 pp | 9 | 13 |
| W1_ResetRoutingW1 | -1.59 pp | 12 | 8 |
| W1_ResetGRUW1 | -0.40 pp | 8 | 7 |
| W1_ResetBothW1 | -1.59 pp | 13 | 9 |
| W1_ResetRoutingW2 | -5.18 pp | 14 | 1 |
| W1_ResetGRUW2 | -69.32 pp | 174 | 0 |
| W1_ResetBothW2 | -69.32 pp | 174 | 0 |
| W1_ResetRoutingW3 | -2.39 pp | 10 | 4 |
| W1_ResetGRUW3 | -67.33 pp | 169 | 0 |
| W1_ResetBothW3 | -67.73 pp | 170 | 0 |
| W1_Trajectory_AfterW1_to_AfterW2 | -1.59 pp | 4 | 0 |
| W1_Trajectory_AfterW2_to_Normal | -1.99 pp | 6 | 1 |
| W2_LaterWrites | -0.80 pp | 3 | 1 |
| W2_PrecedingHistory | -8.37 pp | 29 | 8 |
| MatchedOrder_W2 | -6.77 pp | 26 | 9 |
| W2_PrefixEffectFinal | -9.56 pp | 29 | 5 |
| W2_ResetRoutingW1 | +5.98 pp | 11 | 26 |
| W2_ResetGRUW1 | +8.37 pp | 1 | 22 |
| W2_ResetBothW1 | +7.97 pp | 9 | 29 |
| W2_ResetRoutingW2 | +7.57 pp | 5 | 24 |
| W2_ResetGRUW2 | +1.99 pp | 3 | 8 |
| W2_ResetBothW2 | +9.56 pp | 5 | 29 |
| W2_ResetRoutingW3 | -7.57 pp | 22 | 3 |
| W2_ResetGRUW3 | -61.75 pp | 156 | 1 |
| W2_ResetBothW3 | -63.35 pp | 160 | 1 |
| W2_Trajectory_AfterW2_to_Normal | -0.80 pp | 3 | 1 |
| W3_LaterWrites | +0.00 pp | 0 | 0 |
| W3_PrecedingHistory | -9.32 pp | 24 | 2 |
| MatchedOrder_W3 | -6.78 pp | 21 | 5 |
| W3_PrefixEffectFinal | -9.32 pp | 24 | 2 |
| W3_ResetRoutingW1 | +3.81 pp | 3 | 12 |
| W3_ResetGRUW1 | +3.39 pp | 7 | 15 |
| W3_ResetBothW1 | +3.81 pp | 5 | 14 |
| W3_ResetRoutingW2 | +3.39 pp | 2 | 10 |
| W3_ResetGRUW2 | +8.05 pp | 7 | 26 |
| W3_ResetBothW2 | +8.47 pp | 4 | 24 |
| W3_ResetRoutingW3 | +7.20 pp | 7 | 24 |
| W3_ResetGRUW3 | +4.24 pp | 0 | 10 |
| W3_ResetBothW3 | +9.32 pp | 2 | 24 |

## Interpretation

- Wi_LaterWrites compares the same questions immediately after their source window with the final memory. Losses and rescues may cancel.
- Wi_PrecedingHistory compares writing Wi alone with its state after the natural prefix through Wi. It isolates sensitivity to preceding writes, excluding later writes.
- Trajectory comparisons follow each subsequent write/chunk for evidence already encountered.
- Reset interventions clear routing input, GRU input, or both on the FIRST CHUNK of the named window; all later writes proceed normally. ResetBoth is checked against replaying the suffix from zero.
- SourceOnly contains only the source window written from zero; FinalFromSource retains its natural suffix but omits the prefix.
- EvidenceFirst/Last stably move the source window to the first/last history position; other windows retain relative order and the question is fixed. Reordered prose may change discourse meaning.
- A state before its evidence window is an absent-evidence control. Intermediate/reset/reordered states may be outside the training distribution.
- Full-window bidirectional encoding precedes chunking; chunk states are not strictly prefix-only representations.
- Native source groups contain different questions. Use paired interventions to assess positional effects.
- Bootstrap intervals are descriptive example-level intervals, not book-cluster intervals. Source-ID exclusions apply only when supplied.
- Reuse manifests only at the SAME history length and assistance mode. Different lengths create different inputs and are not a paired horizon comparison.
- Original dataset records contain four history windows plus a question window, limiting this script to 1–4 history windows.
