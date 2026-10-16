# Ticket 01: Set up project and datasets

**Phase:** 1 | **Target:** Oct 16-17 | **Depends on:** none

## Tasks

- [ ] Create `code/` as a Git repository and choose a clear layout for loaders, benchmarks, experiments, tests, and plotting.
- [ ] Pin dependencies in `requirements.txt`; record Python, FAISS, hnswlib (if used), NumPy, CPU, RAM, and thread settings.
- [ ] Document download paths for the ann-benchmarks SIFT-1M and GloVe-100 files without adding datasets to Git or the final ZIP.
- [ ] Load base vectors, queries, and stored neighbours; validate shapes and IDs.
- [ ] Normalize GloVe base and query vectors and use Euclidean distance.
- [ ] Reserve the first 5,000 queries for tuning and the last 5,000 for reporting.

## Acceptance

- [ ] A small data-loading smoke run works for both datasets and prints validated shapes and metadata.
- [ ] The README describes data placement and the query split.
