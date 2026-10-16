# ANN benchmark code

This is the implementation directory for Assignment 3, tracked by the project
Git repository in the parent directory. The package layout is
`annbench/data.py` for loading, `annbench/benchmarks/` for index and timing code,
`annbench/experiments/` for Parts 3 and 4, `annbench/plotting/` for figures,
`scripts/` for commands, and `tests/` for checks. The benchmark and experiment
modules are filled in by later tickets.

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
