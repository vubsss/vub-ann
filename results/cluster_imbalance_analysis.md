# Part 4: Synthetic cluster imbalance

Pre-result prediction: `code/hypothesis.md`, commit `379f0d9`. Seed 20261021 generates 96,000 vectors in 32 dimensions from 16 Gaussian clusters (centre radius 10, spread 1). Only population counts change; base noise, insertion order, and 1,024 separate queries (64 per cluster) stay fixed. HNSW uses `M=16, efConstruction=100, efSearch=32`; IVF-Flat uses `nlist=64, nprobe=4`. Exact Flat top-ten results define tie-aware recall. QPS uses one query per call, one thread, 20 passes per repetition, and three timing repetitions.

| Ratio | Min–max cluster size | HNSW recall | HNSW QPS | IVF-Flat recall | IVF-Flat QPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1:1 | 6000–6000 | 0.9068 | 17,085 | 0.9721 | 10,754 |
| 4:1 | 2744–10977 | 0.8985 | 19,273 | 0.9611 | 12,740 |
| 16:1 | 1068–17089 | 0.9048 | 21,837 | 0.9396 | 11,022 |
| 64:1 | 368–23524 | 0.9039 | 25,378 | 0.9418 | 14,046 |

From 1:1 to 64:1, HNSW recall changed by -0.0029 and IVF-Flat by -0.0303. The predicted larger IVF-Flat decline occurred.

The predicted sparse-cluster failure did not occur: IVF-Flat recall on queries from the four smallest clusters stayed at 1.0000, while recall on the four largest fell from 0.9262 to 0.7672. Assigning learned IVF centroids to the nearest generating centre, the four smallest clusters had 11 then 6 centroids; the four largest had 22 then 45. Few lists can cover a sparse cluster, while fixed `nprobe=4` covers a smaller share of a large cluster. Mean vectors scanned for large-cluster queries rose from 5,625 to 6,644, so scan work alone did not preserve recall. Centroid assignment and scanned-list size support this interpretation but do not prove which lists held missed neighbours. HNSW graph paths were not inspected. The seed reproduced all data hashes and recall on a second run; QPS varied. Dimension, spread, query mix, and search effort were not varied. The summary CSV records settings and run IDs.
