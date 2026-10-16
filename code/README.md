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
