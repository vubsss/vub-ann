# Preregistered experiment hypotheses

## Part 3: GloVe query difficulty (Ticket 06)

Written before generating the Part 3 results. Use the full GloVe base and the
5,000 held-out queries (original rows 5,000–9,999). For each query, compute the
true squared Euclidean distances to its first and tenth exact neighbours after
unit normalization. Define local contrast as `(d10 - d1) / d10`. Sort queries
by this score, breaking ties by original query row, and split the sorted list
into four equal bins of 1,250. Bin 1 has the smallest contrast and is predicted
to be hardest; bin 4 has the largest contrast and is predicted to be easiest.

At the already frozen 0.95 tuning selections for HNSW and IVF-Flat, and the
highest-recall tuning selection for IVF-PQ (which did not reach 0.95), predict
that mean tie-aware recall@10 rises from bin 1 to bin 4 for each index. Predict
the largest bin-4 minus bin-1 gap for IVF-PQ: quantized distances can reorder
close neighbours, while HNSW and IVF-Flat retain original vectors. Report the
four per-bin means even if this ordering fails.

This score describes spread *within* the exact top ten. It does not measure the
gap from the tenth to the eleventh neighbour, centroid-boundary position, graph
connectivity, or approximation error. A large score therefore need not imply an
easy search for every index, and the predicted direction may fail.

## Part 4: synthetic cluster imbalance (Ticket 07)

Written before generating Part 4 data or results. Generate 96,000 base vectors
in 32 dimensions from 16 fixed Gaussian clusters, with centre radius 10 and
isotropic within-cluster standard deviation 1. Use seed 20261021. Assign base
vectors to centres in a fixed insertion order using a seeded permutation and
exact cluster counts proportional to 16 geometric weights from 1 to the chosen
largest:smallest population ratio. Test ratios 1 (balanced control), 4, 16,
and 64. Keep the total base count, centres, base noise, query set, query noise,
dimension, and index settings fixed. Generate a separate set of 1,024 queries,
64 from each cluster, regardless of base population. This gives each cluster
equal weight in reported recall and search speed.

Compare HNSW (`M=16`, `efConstruction=100`, `efSearch=32`) with IVF-Flat
(`nlist=64`, `nprobe=4`) using exact Flat top-ten ground truth, tie-aware
recall@10, and one-query-at-a-time speed on one search thread. Use one build
thread and three search-timing repetitions. Predict that increasing imbalance
reduces IVF-Flat recall more than HNSW recall, especially on queries from
small clusters. Global k-means may assign fewer centroids to small clusters;
at fixed `nprobe`, their nearest vectors can be missed. HNSW can also struggle
to enter a small cluster, but its graph is not constrained by a fixed number
of coarse centroids. Record QPS without a directional speed prediction because
list lengths and graph traversals may change in opposing ways.

This design changes population balance while preserving generation rules and
query mix. It does not separate centroid allocation from other effects of
population balance, such as changing nearest-neighbour gaps or graph edges.
