# Ticket 03: Build a reproducible benchmark runner

**Phase:** 3 | **Target:** Oct 19 | **Depends on:** 01, 02

## Tasks

- [ ] Add configuration for dataset, index family, build settings, search settings, seed, and quick/full modes.
- [ ] Force one thread for the reported ANN search timings and issue queries one at a time.
- [ ] Repeat each timing a few times; store each run and calculate the median QPS.
- [ ] Record recall@10, build time, serialized index size, bytes/vector, machine/library versions, and all settings.
- [ ] Keep tuning outputs distinct from final held-out outputs.

## Acceptance

- [ ] A small smoke run is repeatable and produces parseable, run-level CSV rows.
- [ ] No setting is selected using the final 5,000 queries.
