# Ticket 06: Part 3 query-difficulty experiment

**Phase:** 5 | **Target:** Oct 22 | **Depends on:** 02, 05

## Tasks

- [ ] Define a GloVe query-difficulty score using exact-neighbour distances, and specify how four bins are formed.
- [ ] Write a short prediction and mechanism in `code/hypothesis.md`; **commit it before running this experiment**.
- [ ] Measure per-bin recall for HNSW, IVF-Flat, and IVF-PQ at documented search settings on the held-out queries.
- [ ] Create a figure or compact table and explain whether the hypothesis held.

## Acceptance

- [ ] Git history proves the hypothesis commit predates result generation.
- [ ] Four bins, their query counts, and each index's recall are recorded; the analysis notes any limitations of the difficulty score.
