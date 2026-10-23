# Window retention audit

Scores are teacher-forced. The final question window never writes to memory.

| Condition | W0 | W1 | W2 | W3 |
|---|---:|---:|---:|---:|
| AfterW0 | 77.78% | 3.23% | 14.29% | 0.00% |
| AfterW1 | 81.48% | 85.48% | 11.43% | 0.00% |
| AfterW2 | 81.48% | 85.48% | 91.43% | 0.00% |
| AfterW3 | 79.63% | 85.48% | 91.43% | 81.43% |
| Normal | 79.63% | 85.48% | 91.43% | 81.43% |
| W0Only | 77.78% | 3.23% | 14.29% | 0.00% |
| W1Only | 0.00% | 85.48% | 10.00% | 0.00% |
| W2Only | 1.85% | 4.84% | 91.43% | 1.43% |
| W3Only | 0.00% | 3.23% | 11.43% | 87.14% |
| SourceOnly | 77.78% | 85.48% | 91.43% | 87.14% |
| AtSource | 77.78% | 85.48% | 91.43% | 81.43% |
| FinalFromSource | 79.63% | 85.48% | 90.00% | 87.14% |
| NoRead | 0.00% | 4.84% | 17.14% | 1.43% |
| ResetRoutingW1 | 61.11% | 85.48% | 92.86% | 81.43% |
| ResetGRUW1 | 0.00% | 85.48% | 88.57% | 80.00% |
| ResetBothW1 | 0.00% | 85.48% | 91.43% | 77.14% |
| ResetRoutingW2 | 55.56% | 80.65% | 90.00% | 87.14% |
| ResetGRUW2 | 0.00% | 3.23% | 91.43% | 85.71% |
| ResetBothW2 | 1.85% | 3.23% | 90.00% | 85.71% |
| ResetRoutingW3 | 68.52% | 82.26% | 88.57% | 78.57% |
| ResetGRUW3 | 0.00% | 3.23% | 11.43% | 80.00% |
| ResetBothW3 | 0.00% | 3.23% | 11.43% | 87.14% |
| EvidenceFirst | 79.63% | 85.48% | 87.14% | 85.71% |
| EvidenceLast | 75.93% | 82.26% | 88.57% | 81.43% |
| ShuffleHistory | 1.85% | 0.00% | 5.71% | 0.00% |

| Paired comparison | Delta | Lost | Rescued |
|---|---:|---:|---:|
| MatchedOrder_All | -2.34 pp | 12 | 6 |
| W0_LaterWrites | +1.85 pp | 0 | 1 |
| W0_PrecedingHistory | +0.00 pp | 0 | 0 |
| MatchedOrder_W0 | -3.70 pp | 3 | 1 |
| W0_PrefixEffectFinal | +0.00 pp | 0 | 0 |
| W0_ResetRoutingW1 | -18.52 pp | 10 | 0 |
| W0_ResetGRUW1 | -79.63 pp | 43 | 0 |
| W0_ResetBothW1 | -79.63 pp | 43 | 0 |
| W0_ResetRoutingW2 | -24.07 pp | 13 | 0 |
| W0_ResetGRUW2 | -79.63 pp | 43 | 0 |
| W0_ResetBothW2 | -77.78 pp | 42 | 0 |
| W0_ResetRoutingW3 | -11.11 pp | 6 | 0 |
| W0_ResetGRUW3 | -79.63 pp | 43 | 0 |
| W0_ResetBothW3 | -79.63 pp | 43 | 0 |
| W0_Trajectory_AfterW0_to_AfterW1 | +3.70 pp | 0 | 2 |
| W0_Trajectory_AfterW1_to_AfterW2 | +0.00 pp | 0 | 0 |
| W0_Trajectory_AfterW2_to_Normal | -1.85 pp | 1 | 0 |
| W1_LaterWrites | +0.00 pp | 0 | 0 |
| W1_PrecedingHistory | +0.00 pp | 0 | 0 |
| MatchedOrder_W1 | -3.23 pp | 2 | 0 |
| W1_PrefixEffectFinal | +0.00 pp | 0 | 0 |
| W1_ResetRoutingW1 | +0.00 pp | 0 | 0 |
| W1_ResetGRUW1 | +0.00 pp | 0 | 0 |
| W1_ResetBothW1 | +0.00 pp | 0 | 0 |
| W1_ResetRoutingW2 | -4.84 pp | 3 | 0 |
| W1_ResetGRUW2 | -82.26 pp | 51 | 0 |
| W1_ResetBothW2 | -82.26 pp | 51 | 0 |
| W1_ResetRoutingW3 | -3.23 pp | 2 | 0 |
| W1_ResetGRUW3 | -82.26 pp | 51 | 0 |
| W1_ResetBothW3 | -82.26 pp | 51 | 0 |
| W1_Trajectory_AfterW1_to_AfterW2 | +0.00 pp | 0 | 0 |
| W1_Trajectory_AfterW2_to_Normal | +0.00 pp | 0 | 0 |
| W2_LaterWrites | +0.00 pp | 1 | 1 |
| W2_PrecedingHistory | +0.00 pp | 2 | 2 |
| MatchedOrder_W2 | +1.43 pp | 2 | 3 |
| W2_PrefixEffectFinal | +1.43 pp | 2 | 3 |
| W2_ResetRoutingW1 | +1.43 pp | 1 | 2 |
| W2_ResetGRUW1 | -2.86 pp | 2 | 0 |
| W2_ResetBothW1 | +0.00 pp | 1 | 1 |
| W2_ResetRoutingW2 | -1.43 pp | 1 | 0 |
| W2_ResetGRUW2 | +0.00 pp | 1 | 1 |
| W2_ResetBothW2 | -1.43 pp | 3 | 2 |
| W2_ResetRoutingW3 | -2.86 pp | 3 | 1 |
| W2_ResetGRUW3 | -80.00 pp | 56 | 0 |
| W2_ResetBothW3 | -80.00 pp | 56 | 0 |
| W2_Trajectory_AfterW2_to_Normal | +0.00 pp | 1 | 1 |
| W3_LaterWrites | +0.00 pp | 0 | 0 |
| W3_PrecedingHistory | -5.71 pp | 5 | 1 |
| MatchedOrder_W3 | -4.29 pp | 5 | 2 |
| W3_PrefixEffectFinal | -5.71 pp | 5 | 1 |
| W3_ResetRoutingW1 | +0.00 pp | 0 | 0 |
| W3_ResetGRUW1 | -1.43 pp | 1 | 0 |
| W3_ResetBothW1 | -4.29 pp | 3 | 0 |
| W3_ResetRoutingW2 | +5.71 pp | 0 | 4 |
| W3_ResetGRUW2 | +4.29 pp | 0 | 3 |
| W3_ResetBothW2 | +4.29 pp | 2 | 5 |
| W3_ResetRoutingW3 | -2.86 pp | 4 | 2 |
| W3_ResetGRUW3 | -1.43 pp | 2 | 1 |
| W3_ResetBothW3 | +5.71 pp | 1 | 5 |

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
