# Ticket 02: Implement exact baseline and recall

**Phase:** 2 | **Target:** Oct 18 | **Depends on:** 01

## Tasks

- [x] Implement recall@10 without a framework helper; accept ties at the tenth exact-neighbour distance.
- [x] Add unit tests for perfect recall, missing neighbours, and boundary ties.
- [x] Build FAISS exact indexes for SIFT and normalized GloVe.
- [x] Compare exact neighbours/distances against each dataset's stored reference, accounting for valid ties.
- [x] Measure one-query-at-a-time and all-query batch exact search; record the split, repetitions, median timing, and QPS.

## Acceptance

- [x] Recall tests and exact/reference checks pass.
- [x] Baseline rows exist in `results/results.csv`, with enough detail to reproduce the reported timings.
- [x] The report draft notes why batch search and one-at-a-time search differ.

Completed 16 October 2026. Six tests passed. All 10,000 queries per dataset
passed the reference check; 65 SIFT and one GloVe top-10 ID sets differed only
at valid ties. Twelve CSV rows each contain one timing repetition and its group
median. The final
report PDF is assembled in Ticket 08 from `REPORT_DRAFT.md` and later results.
