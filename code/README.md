# ANN benchmark code

This is the implementation directory for Assignment 3, tracked by the project
Git repository in the parent directory. The package layout is
`annbench/data.py` for loading, `annbench/benchmarks/` for index and timing code,
`annbench/experiments/` for Parts 3 and 4, `annbench/plotting/` for figures,
`scripts/` for commands, and `tests/` for checks. The Part 3/4 experiments are
added in later tickets.

## Environment

Tested on Python 3.13.9, NumPy 2.3.5, h5py 3.15.1, and an Intel Core Ultra 7
155H machine with 22 logical CPUs and about 14 GiB RAM. Dependencies are pinned
in `requirements.txt`. FAISS is the planned implementation for exact search,
HNSW, IVF-Flat, and IVF-PQ; hnswlib is not needed. Run all timed benchmark
searches with one FAISS/OpenMP thread, and set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Record the exact versions and
machine settings with each result; the data smoke command prints the current
machine metadata. The reported hardware will be the machine that actually runs
the final benchmark.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datasets

Download the original [ann-benchmarks datasets](https://github.com/erikbern/ann-benchmarks#data-sets)
into `../data/` (or any directory supplied with `--data-dir`):

```bash
mkdir -p ../data
curl -fL --retry 3 -o ../data/sift-128-euclidean.hdf5 \
  https://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -fL --retry 3 -o ../data/glove-100-angular.hdf5 \
  https://ann-benchmarks.com/glove-100-angular.hdf5
```

The expected files have 1,000,000 SIFT base vectors of dimension 128 and
1,183,514 GloVe base vectors of dimension 100. Both have 10,000 queries and
stored top-100 neighbor IDs. Dataset files and generated indexes are ignored by
Git and must be omitted from the submission ZIP.

Downloaded files verified on 16 October 2026:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `sift-128-euclidean.hdf5` | 525,128,288 | `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984` |
| `glove-100-angular.hdf5` | 485,413,888 | `544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4` |

The loader reads `train`, `test`, and `neighbors`, validates shapes and ID ranges,
and returns contiguous float32 vectors and int64 IDs. GloVe base and query
vectors are normalized to unit length; all search uses Euclidean distance.
Zero vectors remain zero and are counted for later reference checks. Stored
neighbor distances are not reused after this transformation.

Queries at zero-based rows `0:5000` are **tuning only**. Rows `5000:10000` are
held out for **final reporting**. Settings must be selected on the first half.

```bash
python -m scripts.smoke_data --data-dir ../data
python -m pytest -q
```

The smoke command loads both complete files, one at a time, and prints JSON with
validated shapes, ID ranges, normalization counts, versions, CPU, RAM, current
thread settings, and the split. It may use more than 1 GiB of memory while loading each file. To inspect
one file, pass `--dataset sift` or `--dataset glove`.

## Exact baseline

From `code/`, run:

```bash
python -m scripts.run_exact_baseline --data-dir ../data --repetitions 3
```

This builds one FAISS `IndexFlatL2` per dataset and checks its top-10 results
against the stored neighbors and distances for **all 10,000 queries**. The
reference check accepts a different ID only when its true distance ties at the
tenth-neighbor boundary. For SIFT, the file's Euclidean distances are compared
with the square root of squared L2; for normalized GloVe, angular distance is
compared with half of squared L2. Recall uses each returned ID's true distance
from the source vectors, which will also work for compressed ANN indexes later.

The command then times the held-out rows `5000:10000` three times in each mode:
one query per FAISS call and all queries in one FAISS call. It forces FAISS,
OpenMP, OpenBLAS, and MKL to one thread. It writes one row per timing repetition
to `../results/results.csv` (12 rows for both datasets at three repetitions),
including elapsed time, per-run QPS, group median seconds and QPS, split,
dataset checksum, reference check, and machine versions.
Rerunning replaces the previous exact-baseline rows for the selected dataset.
Use `--validation-only` to run the full reference check without timing or
modifying the CSV.

## Reproduction

The assignment's `bash run_all.sh` and `bash run_all.sh --quick` commands will be
added with the benchmark implementation in later tickets. Full runs will read
these same two source files; quick runs will use a smaller base subset and the
same held-out query split.

## ANN benchmark runner

`scripts.run_benchmark` measures one fixed HNSW, IVF-Flat, or IVF-PQ setting.
The JSON configuration names the dataset and family, build and search settings,
seed, quick/full mode, and repetition count. See `configs/quick_hnsw_sift.json`.
Run from `code/`:

```bash
python -m scripts.run_benchmark \
  --config configs/quick_hnsw_sift.json \
  --selection-out ../results/quick_hnsw_sift_selection.json \
  --output ../results/benchmark_smoke.csv
python -m scripts.run_benchmark \
  --selection-file ../results/quick_hnsw_sift_selection.json \
  --output ../results/benchmark_smoke.csv
```

`--config` always uses tuning queries. `--selection-file` is required for a
held-out run and freezes the setting, seed, mode, and source checksum from the
tuning run; it also records the tuning run ID in each held-out row. Choose a
setting using tuning results before running the held-out command. Use
`--output ../results/results.csv` for benchmark results intended for the final
report; the smoke CSV contains only a small protocol check.

Quick mode uses the first 20,000 base vectors and 50 queries from **each**
original half by default. Exact top-10 distances are recomputed for that base
subset. Full mode uses all base vectors and all 5,000 queries in the selected
half; its stored top-10 reference was validated by the exact baseline. ANN
recall uses each returned vector's true squared L2 distance, so compressed
index scores cannot inflate recall. Timed searches issue one query per FAISS
call with one thread. Each CSV row is one timing repetition and includes its
group median QPS, recall@10, build time, serialized index bytes, bytes/vector,
original query bounds, settings, seed, checksum, and machine/library versions.
Tuning and held-out rows have distinct `experiment` and `query_split` values.

## Build and search sweeps

From `code/`, run the Ticket 04 quick sweep:

```bash
python -m scripts.run_sweeps --plan configs/quick_sweep.json
```

The plan covers both datasets and all three families. It tries three HNSW
builds (`M`), three IVF-Flat builds (`nlist`), and two IVF-PQ builds (different
`m`, `nlist`, and eight-bit codes) per dataset. Each index is trained and added
once; its search settings are swept from low effort to a high-effort endpoint.
The quick sweep uses 20,000 base vectors and 500 queries from the original
**tuning** half only. Search timing remains one query per call, one thread, and
three repetitions. Re-running the same plan replaces its prior sweep rows.

`../results/sweeps_quick.csv` contains one row per repetition, with 110 distinct
settings across both datasets. `../results/sweep_summary_quick.json` records
whether each family met recall 0.90 and 0.95 and its highest observed recall.
The 12 files in `../results/selections/` freeze the tuning-selected setting for
each dataset/family/threshold. A threshold miss stores the highest-recall
setting and explicitly marks `target_reached: false`. The selection files also
carry the search grid measured for the selected build, for later held-out
trade-off curves. These quick-mode choices describe the subset only; full-size
settings must be tuned on the full base and tuning queries before final reporting.

## Main benchmark (Ticket 05)

The full-size plan in `configs/full_sweep.json` searches two construction
settings per family and dataset. It builds indexes on all base vectors, uses
the first 1,000 queries of the tuning half to choose settings, and repeats
each tuning search twice. The remaining 4,000 tuning queries are unused.
Full-size IVF centroid training uses at most 50 training vectors per centroid;
IVF-PQ codebook training uses at most 100 per centroid. FAISS samples those
vectors from the full base using the recorded seed. All base vectors are then
added to the index. Full-size HNSW construction uses four FAISS threads;
every timed search uses one. Build thread count, training caps, and the FAISS
version are recorded with runs.
The held-out phase uses **all** original query rows 5,000:10,000 and three
timing repetitions. No setting is chosen from held-out recall or QPS.

From `code/`:

```bash
python -m scripts.run_sweeps --plan configs/full_sweep.json \
  --output ../results/results.csv \
  --summary ../results/sweep_summary_full.json
python -m scripts.run_main_final
python -m scripts.analyze_main
```

The first command records tuning runs and frozen selections and keeps
temporary serialized full-size indexes in `../results/index_cache/`.
The second loads the selected tuned builds (or rebuilds them if the cache is
absent) and measures its tuning-approved search grid through the first setting
that reaches 0.97 tuning recall, or the full grid if none does, on the
5,000 held-out queries. This endpoint rule uses only tuning data. The cache
has checksum and FAISS-version checks and
must be omitted from the submission ZIP. The final runs record the corresponding
tuning run ID for every final setting. It replaces prior final runs for the
same dataset and family when rerun. The analysis command requires complete
final runs and writes two figures to `../results/figures/`, threshold and
build/memory CSV tables, a plotted-point provenance CSV, and
`../results/main_analysis.md` from `../results/results.csv`.

The 0.90 and 0.95 threshold entries use the settings frozen by tuning, even
if the held-out recall falls below the target. Such a miss is marked
`selected_below_target` if another tuning-approved setting reaches it and
`unreachable` if none does. There is no interpolation or extrapolation, and
held-out QPS is never used to choose a setting. The plotted family curves show measured
Pareto-frontier points. Exact Flat is a single-query reference point, not
the much faster batch-search number.

## Part 3: GloVe query difficulty (Ticket 06)

The prediction and bin definition were committed in `hypothesis.md` as commit
`82d9cdc` before the experiment was run. To reproduce the four-bin analysis
after the full benchmark and its temporary index cache are available, run from
`code/`:

```bash
.venv/bin/python -m scripts.run_query_difficulty
```

This uses all 5,000 held-out GloVe queries and the full base. The score is
`(d10-d1)/d10` from true squared L2 distances to the validated exact top-ten
neighbours. Queries are sorted by score, then original query row, into four
bins of 1,250. Each index uses its frozen full-size 0.95 selection (the
highest-recall tuned setting for IVF-PQ), and its per-query tie-aware recall
must agree with the corresponding aggregate main benchmark run. The command
writes `../results/query_difficulty.csv`,
`../results/query_difficulty_analysis.md`, and PNG/PDF figures under
`../results/figures/`. The CSV records bin counts, score ranges, settings,
dataset checksum, and tuning and final run IDs.

## Part 4: synthetic cluster imbalance (Ticket 07)

The Part 4 prediction was committed in `hypothesis.md` as `379f0d9` before
synthetic data or results were generated. Run from `code/`:

```bash
.venv/bin/python -m scripts.run_cluster_imbalance
.venv/bin/python -m scripts.analyze_cluster_imbalance
```

The seeded generator uses 96,000 vectors in 32 dimensions, 16 fixed Gaussian
centres, and a separate fixed set of 1,024 queries. The four population ratios
are 1:1, 4:1, 16:1, and 64:1; the first is the balanced control. Only cluster
membership counts change. Base noise, row insertion order, centres, queries,
and index settings remain fixed. Query clusters have 64 queries each at every
ratio. HNSW uses `M=16, efConstruction=100, efSearch=32`; IVF-Flat uses
`nlist=64, nprobe=4`. Both build and search with one FAISS thread; IVF k-means
uses seed 20261021 and at most 100 training points per centroid.

For each ratio, exact Flat search supplies top-ten neighbours. Results use the
same tie-aware recall code as the main benchmark. Search timing runs one query
per FAISS call, repeats the query set 20 times per repetition, and reports the
median QPS of three repetitions. The runner saves 24 raw repetition rows under
`experiment=synthetic_cluster_imbalance` in `../results/results.csv`. It skips
complete matching runs on restart; `--force` reruns them. The analyzer writes
`../results/cluster_imbalance_summary.csv`,
`../results/cluster_imbalance_analysis.md`, and the single main PNG/PDF figure
in `../results/figures/`. Source, query, and exact-neighbour hashes document
reproducibility. Data hashes and all eight recall values matched on a second
full run; QPS can vary with system load.

## Five-page report (Ticket 08)

After generating the main, Part 3, and Part 4 result tables and figures, run
from `code/`:

```bash
python -m scripts.build_report
```

This writes `../report.pdf`. The command checks main threshold and memory
tables, plotted-point IDs, Part 3 bin means, and Part 4 summary rows against
their run-level CSV records before authoring the PDF. It also checks the
measured values quoted in the narrative. The report includes the two main
trade-off plots, exact-search comparison, threshold and build/memory tables,
both preregistered experiments, and all five discussion answers. ReportLab is
pinned in `requirements.txt`; the script uses Lato when available and falls
back to built-in Helvetica.
