# Assignment 3 report draft

## Part 1: Exact baseline

FAISS `IndexFlatL2` provides the exhaustive reference for both datasets. GloVe
base and query vectors are normalized before Euclidean search. SIFT's stored
Euclidean distances are compared with the square root of FAISS squared L2;
GloVe's stored angular distances are compared with half of squared L2. A
retrieved ID counts toward recall@10 if its true vector distance is at most the
exact tenth-neighbor distance, including boundary ties. The check uses all
10,000 queries; timing uses only held-out queries 5,001–10,000.

SIFT has 65 queries and GloVe has one query whose FAISS and stored top-10 ID
sets differ, but all differences are valid tenth-neighbor ties. The tie-aware
recall in both directions is 1.0 on both datasets.

| Dataset | Search calls | Median seconds for 5,000 queries | Median QPS |
| --- | --- | ---: | ---: |
| SIFT-1M | One query per call | 138.565 | 36.1 |
| SIFT-1M | All queries in one call | 42.890 | 116.6 |
| GloVe-100 | One query per call | 111.648 | 44.8 |
| GloVe-100 | All queries in one call | 42.880 | 116.6 |

Batch search can reuse blocks of base vectors and use more efficient matrix
operations across queries. It also pays Python-to-FAISS call overhead once,
whereas one-at-a-time search pays it 5,000 times. Both modes use one thread;
the difference comes from execution shape, not parallelism. Exact batch QPS
must therefore be identified separately from the required single-query ANN
comparison.

Each value above is the median of three repetitions. SIFT batch QPS is about
3.2 times the one-at-a-time QPS; GloVe batch QPS is about 2.6 times as high.
The per-repetition measurements and environment metadata are in
`results/results.csv`.
