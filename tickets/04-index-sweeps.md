# Ticket 04: Implement HNSW, IVF-Flat, and IVF-PQ sweeps

**Phase:** 3 | **Target:** Oct 19 | **Depends on:** 03

## Tasks

- [x] Implement construction and search for all three index families on both datasets.
- [x] Try several build settings per family (for example HNSW `M`, IVF `nlist`, and viable PQ code settings).
- [x] Sweep `efSearch` or `nprobe` from low recall toward at least 0.95 when feasible.
- [x] Use tuning queries to choose settings and preserve the chosen configuration before held-out measurement.
- [x] Log any index that cannot reach 0.90 or 0.95 and its highest observed recall.

## Acceptance

- [x] Each dataset/family has a recorded sweep with build and search parameters.
- [x] A small-subset run completes for all three families.

Completed 17 October 2026. The quick run used the first 20,000 base vectors
and 500 tuning queries per dataset. `results/sweeps_quick.csv` has 330 timed
rows (110 settings, three repetitions each). No held-out query was used for
these sweeps or selections. Frozen settings for 0.90 and 0.95 are in
`results/selections/`, and `results/sweep_summary_quick.json` records threshold
status and maxima:

| Dataset | HNSW highest | IVF-Flat highest | Tested IVF-PQ highest |
| --- | ---: | ---: | ---: |
| SIFT | 1.0000 | 1.0000 | 0.7932 |
| GloVe | 0.9952 | 1.0000 | 0.6344 |

HNSW and IVF-Flat reached 0.95 on both quick subsets. The tested IVF-PQ
configurations reached neither 0.90 nor 0.95. These maxima describe this
20,000-vector grid, not a claim that every possible PQ configuration fails.
Ticket 05 will tune full-size settings before held-out reporting.
