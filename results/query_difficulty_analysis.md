# Part 3: GloVe query difficulty

Pre-result hypothesis: `code/hypothesis.md` (commit `82d9cdc`).
The 5,000 held-out GloVe queries are ordered by `(d10-d1)/d10`, where distances are true squared L2 to exact neighbours of unit-normalized vectors. Ties are broken by original query row. Four equal-count bins contain 1,250 queries each; bin 1 has the smallest local contrast.

| Bin | Count | Contrast range | HNSW | IVF-Flat | IVF-PQ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,250 | 0.0262–0.1051 | 0.9150 | 0.9636 | 0.8147 |
| 2 | 1,250 | 0.1051–0.1660 | 0.9628 | 0.9818 | 0.8545 |
| 3 | 1,250 | 0.1660–0.2804 | 0.9789 | 0.9894 | 0.8806 |
| 4 | 1,250 | 0.2804–0.9185 | 0.9826 | 0.9865 | 0.9024 |

Frozen tuning choices: HNSW `M=32, efConstruction=160, efSearch=512`; IVF-Flat `nlist=1024, nprobe=256`; IVF-PQ `nlist=1024, m=50, nbits=8, nprobe=1024`. IVF-PQ was the highest-recall tuned choice and did not reach 0.95. Per-query recall uses true source-vector distances and the established tie-aware cutoff. The CSV links each family to its tuning selection and matching main final run.

HNSW increased monotonically across the bins; bin 4 minus bin 1 = +0.0676.
IVF-Flat did not increase monotonically across the bins; bin 4 minus bin 1 = +0.0229.
IVF-PQ increased monotonically across the bins; bin 4 minus bin 1 = +0.0877.
The largest endpoint gap was IVF-PQ (+0.0877).

The prediction was partly supported: all three endpoint changes were positive and IVF-PQ had the largest, but IVF-Flat fell slightly from bin 3 to bin 4. Tightly grouped exact neighbours may be easier to displace in approximate ranking. For IVF-PQ, all 1,024 lists were probed, so missed coarse lists cannot account for its remaining errors; quantized distance ranking is a plausible contributor. HNSW graph search and IVF-Flat's limited list probing have different failure mechanisms. The IVF-Flat dip shows that local contrast alone does not order every query's search difficulty.

Local top-ten contrast does not measure the tenth-to-eleventh gap, IVF centroid boundaries, HNSW graph reachability, or PQ quantization error. Bin associations are descriptive and do not isolate a causal mechanism. The selected search settings also differ by family; their gap sizes do not compare intrinsic sensitivity at equal computational effort.
