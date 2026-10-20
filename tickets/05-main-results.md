# Ticket 05: Run and analyze the main benchmark

**Phase:** 4 | **Target:** Oct 20-21 | **Depends on:** 02, 04

## Tasks

- [x] Run final measurements on the last 5,000 queries for both datasets using settings chosen on the tuning half.
- [x] Plot recall@10 against one-query-at-a-time QPS for each dataset, showing each family's best measured trade-off and the exact baseline.
- [x] Create the QPS-at-recall-0.90/0.95 table, clearly marking unreachable thresholds.
- [x] Create the build-time, index-size, and bytes/vector table for configurations at recall 0.95.
- [x] Explain any failure to reach high recall using the index mechanism and measured evidence.

## Acceptance

- [x] Two dataset plots and both tables can be generated from `results/results.csv`.
- [x] Every plotted point and table entry traces to recorded benchmark runs.

Completed 20 October 2026. Full-size tuning used the first 1,000 queries of
the tuning half, two timing repetitions, and all base vectors. Frozen
selections in `results/selections/full_*.json` preceded final measurement.
Final runs used all 5,000 held-out queries, three repetitions, one FAISS
search thread, and the same tuned index builds cached between phases.

`results/results.csv` holds 144 tuning rows, 117 final ANN rows, and 12 exact
baseline rows. `code/scripts/analyze_main.py` regenerates the two figures in
`results/figures/`, the threshold and build/memory tables, plotted-point IDs,
and `results/main_analysis.md` using the CSV alone. A provenance audit traced
all 39 plotted points, 16 threshold rows, and six build/memory rows to run IDs
and checked each final ANN setting against its matching tuning run.

At the frozen 0.95 settings, SIFT HNSW measured 3,219 QPS at 0.9647 recall
and IVF-Flat 943 QPS at 0.9743. GloVe HNSW measured 652 QPS at 0.9598 and
IVF-Flat 142 QPS at 0.9803. The tested IVF-PQ grids reached only 0.8735
(SIFT) and 0.8630 (GloVe) on held-out queries. Their all-list-probed points
still missed 0.90; quantized distance ranking therefore remains a measured
source of error. These misses describe the tested eight-bit PQ settings, not
all possible IVF-PQ designs.
