# Ticket 05: Run and analyze the main benchmark

**Phase:** 4 | **Target:** Oct 20-21 | **Depends on:** 02, 04

## Tasks

- [ ] Run final measurements on the last 5,000 queries for both datasets using settings chosen on the tuning half.
- [ ] Plot recall@10 against one-query-at-a-time QPS for each dataset, showing each family's best measured trade-off and the exact baseline.
- [ ] Create the QPS-at-recall-0.90/0.95 table, clearly marking unreachable thresholds.
- [ ] Create the build-time, index-size, and bytes/vector table for configurations at recall 0.95.
- [ ] Explain any failure to reach high recall using the index mechanism and measured evidence.

## Acceptance

- [ ] Two dataset plots and both tables can be generated from `results/results.csv`.
- [ ] Every plotted point and table entry traces to recorded benchmark runs.
