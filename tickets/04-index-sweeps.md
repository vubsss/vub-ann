# Ticket 04: Implement HNSW, IVF-Flat, and IVF-PQ sweeps

**Phase:** 3 | **Target:** Oct 19 | **Depends on:** 03

## Tasks

- [ ] Implement construction and search for all three index families on both datasets.
- [ ] Try several build settings per family (for example HNSW `M`, IVF `nlist`, and viable PQ code settings).
- [ ] Sweep `efSearch` or `nprobe` from low recall toward at least 0.95 when feasible.
- [ ] Use tuning queries to choose settings and preserve the chosen configuration before held-out measurement.
- [ ] Log any index that cannot reach 0.90 or 0.95 and its highest observed recall.

## Acceptance

- [ ] Each dataset/family has a recorded sweep with build and search parameters.
- [ ] A small-subset run completes for all three families.
