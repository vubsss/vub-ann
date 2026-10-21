# Ticket 06: Part 3 query-difficulty experiment

**Phase:** 5 | **Target:** Oct 22 | **Depends on:** 02, 05

## Tasks

- [x] Define a GloVe query-difficulty score using exact-neighbour distances, and specify how four bins are formed.
- [x] Write a short prediction and mechanism in `code/hypothesis.md`; **commit it before running this experiment**.
- [x] Measure per-bin recall for HNSW, IVF-Flat, and IVF-PQ at documented search settings on the held-out queries.
- [x] Create a figure or compact table and explain whether the hypothesis held.

## Acceptance

- [x] Git history proves the hypothesis commit predates result generation.
- [x] Four bins, their query counts, and each index's recall are recorded; the analysis notes any limitations of the difficulty score.

Completed 21 October 2026. The preregistered prediction and bin rule were
committed as `82d9cdc` before `scripts.run_query_difficulty` generated results.
The score is `(d10-d1)/d10` from true squared L2 distances to exact top-ten
neighbours of unit-normalized GloVe vectors. Sorting the 5,000 held-out queries
by score, with original row as tie-breaker, yields four 1,250-query bins.

At frozen tuning settings, bin-1 to bin-4 recall was HNSW 0.9150 to 0.9826,
IVF-Flat 0.9636 to 0.9865, and IVF-PQ 0.8147 to 0.9024. All three endpoint
gaps were positive, and IVF-PQ's was largest. IVF-Flat fell slightly from bin
3 to bin 4, so its predicted monotonic rise failed. Full bin values, score
ranges, settings, and matching tuning/final run IDs are in
`results/query_difficulty.csv`; the figure and limitations are in
`results/figures/glove_query_difficulty.*` and
`results/query_difficulty_analysis.md`. All 21 code tests passed.
