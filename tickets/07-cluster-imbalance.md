# Ticket 07: Part 4 synthetic cluster-imbalance experiment

**Phase:** 6 | **Target:** Oct 23 | **Depends on:** 03, 05

## Tasks

- [x] Specify a seeded generator with no more than 500,000 vectors and a separate query set.
- [x] Use balanced clusters as the control and at least three increasing imbalance settings. Change cluster population balance only; keep other generation and index settings fixed.
- [x] Write a short prediction and mechanism in `code/hypothesis.md`; **commit it before running this experiment**.
- [x] Compare at least HNSW and IVF-Flat using exact ground truth, recall, and single-query speed.
- [x] Produce one main figure and explain the observed mechanism, hypothesis outcome, and untested cases in at most one report page.

## Acceptance

- [x] The same seed reproduces all generated data and recall results; timed QPS varies with system load.
- [x] Git history proves the Part 4 hypothesis predates result generation.
- [x] Control plus three settings are present in the results CSV and main figure.

Completed 21 October 2026. The hypothesis was committed as `379f0d9` before
the synthetic data/results were generated. Seed 20261021 creates 96,000 base
vectors from 16 fixed Gaussian clusters and 1,024 separate queries. The four
population ratios are 1:1, 4:1, 16:1, and 64:1. The fixed queries give each
cluster equal weight. Both HNSW and IVF-Flat use fixed settings, exact Flat
ground truth, tie-aware recall@10, and single-query timing with one thread.

At the extreme setting, IVF-Flat recall fell by 0.0303 from the balanced
control, versus 0.0029 for HNSW, supporting the predicted relative decline.
The predicted failure on rare clusters was wrong: IVF-Flat retained 1.0000
recall for queries from the four smallest clusters, while its recall on the
four largest fell from 0.9262 to 0.7672. The one main figure is
`results/figures/cluster_imbalance.*`; the concise mechanism and limitations
are in `results/cluster_imbalance_analysis.md`. Eight settings each have three
repetition rows in `results/results.csv`, with source and exact-neighbour
hashes. A second full run reproduced all eight data hashes and recall values;
the final timing repetitions use 20 query passes each. All 24 code tests pass.
