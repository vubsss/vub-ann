# Ticket 07: Part 4 synthetic cluster-imbalance experiment

**Phase:** 6 | **Target:** Oct 23 | **Depends on:** 03, 05

## Tasks

- [ ] Specify a seeded generator with no more than 500,000 vectors and a separate query set.
- [ ] Use balanced clusters as the control and at least three increasing imbalance settings. Change cluster population balance only; keep other generation and index settings fixed.
- [ ] Write a short prediction and mechanism in `code/hypothesis.md`; **commit it before running this experiment**.
- [ ] Compare at least HNSW and IVF-Flat using exact ground truth, recall, and single-query speed.
- [ ] Produce one main figure and explain the observed mechanism, hypothesis outcome, and untested cases in at most one report page.

## Acceptance

- [ ] The same seed reproduces all generated data and results.
- [ ] Git history proves the Part 4 hypothesis predates result generation.
- [ ] Control plus three settings are present in the results CSV and main figure.
