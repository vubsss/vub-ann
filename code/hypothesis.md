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
