# Ticket 02: Implement exact baseline and recall

**Phase:** 2 | **Target:** Oct 18 | **Depends on:** 01

## Tasks

- [ ] Implement recall@10 without a framework helper; accept ties at the tenth exact-neighbour distance.
- [ ] Add unit tests for perfect recall, missing neighbours, and boundary ties.
- [ ] Build FAISS exact indexes for SIFT and normalized GloVe.
- [ ] Compare exact neighbours/distances against each dataset's stored reference, accounting for valid ties.
- [ ] Measure one-query-at-a-time and all-query batch exact search; record the split, repetitions, median timing, and QPS.

## Acceptance

- [ ] Recall tests and exact/reference checks pass.
- [ ] Baseline rows exist in `results/results.csv`, with enough detail to reproduce the reported timings.
- [ ] The report notes why batch search and one-at-a-time search differ.
