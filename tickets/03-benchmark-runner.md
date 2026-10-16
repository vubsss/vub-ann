# Ticket 03: Build a reproducible benchmark runner

**Phase:** 3 | **Target:** Oct 19 | **Depends on:** 01, 02

## Tasks

- [x] Add configuration for dataset, index family, build settings, search settings, seed, and quick/full modes.
- [x] Force one thread for the reported ANN search timings and issue queries one at a time.
- [x] Repeat each timing a few times; store each run and calculate the median QPS.
- [x] Record recall@10, build time, serialized index size, bytes/vector, machine/library versions, and all settings.
- [x] Keep tuning outputs distinct from final held-out outputs.

## Acceptance

- [x] A small smoke run is repeatable and produces parseable, run-level CSV rows.
- [x] No setting is selected using the final 5,000 queries.

Completed 17 October 2026. `results/benchmark_smoke.csv` has three quick-mode
SIFT HNSW tuning rows and three held-out rows. The held-out invocation loaded
the frozen `results/quick_hnsw_sift_selection.json`, which records the tuning
run ID and dataset checksum. The CLI does not accept an arbitrary configuration
for held-out queries. Thirteen tests passed. Ticket 04 will add setting sweeps and
full-size candidate evaluation.
