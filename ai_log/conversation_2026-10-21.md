# AI conversation log — 21 October 2026

## Ticket 06

### User prompt

> do ticket 6

### Assistant work and response

The assistant read Ticket 06, the assignment plan, the benchmark code, and the
frozen GloVe selections. It wrote a Part 3 hypothesis defining local contrast
as `(d10-d1)/d10`, four equal-count held-out query bins, and predicted recall
ordering. The hypothesis was committed as `82d9cdc` before any Part 3 result
generation. Because `.git` was protected by the workspace sandbox, the commit
used an approved escalation.

The assistant implemented `scripts.run_query_difficulty`, reusing the cached
full-size indexes and the frozen settings. It searched all 5,000 held-out
GloVe queries for HNSW, IVF-Flat, and IVF-PQ, computed tie-aware per-query
recall with true source-vector distances, checked each overall mean against
the existing final run, and generated a 12-row table, figure, and analysis.
The four bins each contain 1,250 queries. HNSW and IVF-PQ increased
monotonically; IVF-Flat dipped slightly in bin 4. IVF-PQ had the largest
bin-4 minus bin-1 gap. The assistant documented this partial match to the
hypothesis and the score's limitations, updated the README and ticket,
inspected the figure, and ran all 21 tests successfully.

## Ticket 07

### User prompt

> complete ticket 7

### Assistant work and response

The assistant read Ticket 07 and the Part 4 assignment text, then wrote the
synthetic-cluster hypothesis in `code/hypothesis.md` and committed it as
`379f0d9` before generating data or results. It fixed 96,000 base vectors,
16 Gaussian clusters, 1,024 separate queries, seed 20261021, and four
population ratios from balanced to 64:1. It implemented a generator that
keeps centres, noise, insertion order, and queries fixed, plus a runner that
compares HNSW and IVF-Flat with exact ground truth and one-query-at-a-time
timing.

The first timing pass was too short for stable QPS, so the assistant changed
each timing repetition to 20 passes over the query set and reran all eight
settings. It also recorded IVF centroid allocation and scanned-list sizes.
Source, query, and exact-neighbour hashes and all eight recall values matched
the first run. The main figure, summary CSV, and analysis were generated from
the run-level CSV. The predicted larger IVF-Flat recall decline was observed,
but rare-cluster queries did not fail; the loss came from large clusters.
The assistant documented this partial hypothesis outcome and limitations,
updated the README and ticket, inspected the figure, and ran all 24 tests.
