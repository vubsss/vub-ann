# AI conversation log — 17 October 2026

## Ticket 03

### User prompt

> do ticket 3

### Assistant work and responses

The assistant read Ticket 03, the plan, existing dataset and exact baseline
code, and the current CSV. It implemented a configurable single-thread FAISS
runner for HNSW, IVF-Flat, and IVF-PQ, with quick/full modes and separated
tuning and held-out commands. The held-out command consumes a frozen tuning
selection file. The assistant added tests and a quick SIFT HNSW smoke run.

The assistant told the user it would find the ticket, implement the runner, and
verify it; then reported the existing baseline/split, the completed quick
tuning run, and the pending held-out and test checks. The final status is
recorded by the corresponding Ticket 03 code, CSV, and ticket checklist.

## Ticket 04

### User prompt

> do ticket 4

### Assistant work and responses

The assistant extended the runner so each built index can be reused across a
search-parameter sweep. It added a two-dataset quick sweep plan, tuning-only
selection logic for recall targets 0.90 and 0.95, a CLI, tests, raw CSV output,
summary JSON, and frozen selection files. The first 100-query run showed
timing noise, so the assistant increased the quick plan to 500 tuning queries
and reran it. The final quick sweep produced 330 timing rows across 110
settings. HNSW and IVF-Flat reached 0.95 on both datasets; the tested IVF-PQ
settings peaked at 0.7932 on SIFT and 0.6344 on GloVe. No held-out query was
used to select a setting for this sweep.
