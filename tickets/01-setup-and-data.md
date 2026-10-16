# Ticket 01: Set up project and datasets

**Phase:** 1 | **Target:** Oct 16-17 | **Depends on:** none

## Tasks

- [x] Track `code/` in the project Git repository and choose a clear layout for loaders, benchmarks, experiments, tests, and plotting.
- [x] Pin dependencies in `requirements.txt`; record Python, FAISS, hnswlib (if used), NumPy, CPU, RAM, and thread settings.
- [x] Document download paths for the ann-benchmarks SIFT-1M and GloVe-100 files without adding datasets to Git or the final ZIP.
- [x] Load base vectors, queries, and stored neighbours; validate shapes and IDs.
- [x] Normalize GloVe base and query vectors and use Euclidean distance.
- [x] Reserve the first 5,000 queries for tuning and the last 5,000 for reporting.

## Acceptance

- [x] A small data-loading smoke run works for both datasets and prints validated shapes and metadata.
- [x] The README describes data placement and the query split.

Completed 16 October 2026. The code is tracked in the cwd repository; the
initial separate repository was removed at the user's request.
The full-data smoke command passed for both downloaded files; `pytest` passed (2 tests).
