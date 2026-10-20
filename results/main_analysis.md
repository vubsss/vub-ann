# Main benchmark results

Generated from `results.csv`; QPS uses one query per FAISS call and the median of repeated timings. All reported points use held-out queries 5,001–10,000.

## SIFT-1M

| Family | QPS at 0.90 | QPS at 0.95 | Highest recall |
| --- | ---: | ---: | ---: |
| HNSW | 4,493 | 3,219 | 0.9886 |
| IVF-Flat | 1,118 | 943 | 0.9991 |
| IVF-PQ | unreachable | unreachable | 0.8735 |
| Exact Flat | 36 | 36 | 1.0000 |

Exact Flat: 36.1 single-query QPS at recall 1.0 (run `90ed7c1e-2d6f-4ec5-bd0b-75103ac1a50f`).

| Family at 0.95 | Status | Build (s) | Index (MiB) | Bytes/vector |
| --- | --- | ---: | ---: | ---: |
| HNSW | reached | 208.60 | 625.9 | 656.3 |
| IVF-Flat | reached | 158.78 | 497.9 | 522.1 |
| IVF-PQ | unreachable | — | — | — |

IVF-PQ reached at most 0.8735 recall. With every coarse list probed, its best measured recall was 0.8735 at 72.7 serialized bytes/vector (run `4982a8cf-fd06-40f4-b7ed-626cb9be32b3`). Scanning every list removes missed coarse lists as the sole explanation; the 8-bit product-quantized distance estimates can still misrank candidates.

## GloVe-100

| Family | QPS at 0.90 | QPS at 0.95 | Highest recall |
| --- | ---: | ---: | ---: |
| HNSW | 1,328 | 652 | 0.9598 |
| IVF-Flat | 551 | 142 | 0.9803 |
| IVF-PQ | unreachable | unreachable | 0.8630 |
| Exact Flat | 45 | 45 | 1.0000 |

Exact Flat: 44.8 single-query QPS at recall 1.0 (run `25aa7c3a-2fb4-40f6-87f7-958d2de7732f`).

| Family at 0.95 | Status | Build (s) | Index (MiB) | Bytes/vector |
| --- | --- | ---: | ---: | ---: |
| HNSW | reached | 509.92 | 758.6 | 672.2 |
| IVF-Flat | reached | 17.41 | 460.9 | 408.4 |
| IVF-PQ | unreachable | — | — | — |

IVF-PQ reached at most 0.8630 recall. With every coarse list probed, its best measured recall was 0.8630 at 58.4 serialized bytes/vector (run `9fd64c5b-18e0-4b52-94fd-b10e835fe861`). Scanning every list removes missed coarse lists as the sole explanation; the 8-bit product-quantized distance estimates can still misrank candidates.

The two threshold tables and plotted point IDs are in `main_qps_thresholds.csv`, `main_build_memory.csv`, and `main_plot_points.csv`. Threshold QPS and memory use the frozen tuning selection, never a setting chosen from held-out performance. `unreachable` means no measured setting in the tuning-approved grid reached the threshold; `selected_below_target` means the frozen selection missed it although another measured setting reached it. Index size is serialized FAISS bytes; bytes/vector divides by the full base count.
