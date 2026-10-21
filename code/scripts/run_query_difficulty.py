"""Part 3: evaluate frozen GloVe indexes by exact-neighbour local contrast."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "annbench-mplconfig"))
import faiss
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from annbench.benchmarks.exact import recall_per_query, squared_l2_for_ids
from annbench.benchmarks.index_cache import load_index
from annbench.benchmarks.runner import BenchmarkConfig, set_search_settings
from annbench.data import DATASETS, load_dataset
from scripts.run_benchmark import sha256


FAMILIES = ("hnsw", "ivf_flat", "ivf_pq")
LABELS = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat", "ivf_pq": "IVF-PQ"}
COLORS = {"hnsw": "#0072B2", "ivf_flat": "#D55E00", "ivf_pq": "#009E73"}


def contrast_bins(exact_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return local contrast and stable, equal-count quartile labels (1..4)."""
    if exact_distances.ndim != 2 or exact_distances.shape[1] != 10:
        raise ValueError("expected ten exact-neighbour distances per query")
    d1 = exact_distances.min(axis=1)
    d10 = exact_distances.max(axis=1)
    if np.any(~np.isfinite(exact_distances)) or np.any(d10 <= 0):
        raise ValueError("exact distances must be finite with positive d10")
    contrast = (d10 - d1) / d10
    # lexsort uses the original row as the deterministic secondary key.
    order = np.lexsort((np.arange(len(contrast)), contrast))
    bins = np.empty(len(contrast), dtype=np.int8)
    for label, part in enumerate(np.array_split(order, 4), start=1):
        bins[part] = label
    return contrast, bins


def selected_setting(
    family: str, selection_dir: Path, benchmark_rows: list[dict[str, str]],
    source_hash: str,
) -> tuple[BenchmarkConfig, str, str]:
    path = selection_dir / f"full_glove_{family}_95.json"
    selection = json.loads(path.read_text())
    config = BenchmarkConfig(**selection["config"])
    config.validate()
    if (config.dataset, config.index_family, config.mode) != ("glove", family, "full"):
        raise ValueError(f"invalid frozen configuration: {path}")
    if selection["query_split"] != "tune" or selection["source_sha256"] != source_hash:
        raise ValueError(f"selection source or split mismatch: {path}")
    tune_run_id = selection["tune_run_id"]
    if not any(row["run_id"] == tune_run_id and row["experiment"] == "ann_tuning_sweep"
               for row in benchmark_rows):
        raise ValueError(f"frozen tuning run missing from results.csv: {tune_run_id}")
    matches = [row for row in benchmark_rows if
               row["experiment"] == "ann_final" and row["dataset"] == "glove"
               and row["index_family"] == family and row["query_split"] == "report"
               and row["selected_tune_run_id"] == tune_run_id
               and row["build_settings"] == json.dumps(config.build_settings, sort_keys=True)
               and row["search_settings"] == json.dumps(config.search_settings, sort_keys=True)
               and row["source_sha256"] == source_hash]
    if len(matches) != 3 or {int(row["repeat_index"]) for row in matches} != {1, 2, 3}:
        raise ValueError(f"expected three final repetitions for {family} at frozen setting")
    if any((int(row["query_start"]), int(row["query_end"])) != (5000, 10000) for row in matches):
        raise ValueError(f"wrong held-out query range for {family}")
    return config, tune_run_id, matches[0]["run_id"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--selection-dir", type=Path, default=root / "results/selections")
    parser.add_argument("--index-cache-dir", type=Path, default=root / "results/index_cache")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    args = parser.parse_args()
    source_path = args.data_dir / DATASETS["glove"].filename
    source_hash = sha256(source_path)
    with args.results.open(newline="") as source:
        benchmark_rows = list(csv.DictReader(source))
    faiss.omp_set_num_threads(1)
    data = load_dataset(source_path, DATASETS["glove"])
    queries = data.report_queries
    exact_distances = squared_l2_for_ids(data.base, queries, data.report_neighbors[:, :10])
    cutoff = exact_distances.max(axis=1)
    contrast, bins = contrast_bins(exact_distances)
    if len(queries) != 5000 or any(np.count_nonzero(bins == b) != 1250 for b in range(1, 5)):
        raise ValueError("expected four bins of 1,250 held-out queries")

    rows: list[dict[str, object]] = []
    overall: dict[str, float] = {}
    for family in FAMILIES:
        config, tune_run_id, final_run_id = selected_setting(
            family, args.selection_dir, benchmark_rows, source_hash,
        )
        cached = load_index(
            args.index_cache_dir, config, source_sha256=source_hash,
            expected_base_count=len(data.base),
        )
        if cached is None:
            raise FileNotFoundError(f"missing tuned {family} index cache; rerun full sweep")
        index = cached[0]
        set_search_settings(index, config)
        found_ids = np.empty((len(queries), 10), dtype=np.int64)
        for i, query in enumerate(queries):
            found_ids[i] = index.search(query[None, :], 10)[1][0]
        true_distances = squared_l2_for_ids(data.base, queries, found_ids)
        recalls = recall_per_query(found_ids, true_distances, cutoff, k=10)
        overall[family] = float(recalls.mean())
        recorded = next(row for row in benchmark_rows if row["run_id"] == final_run_id)
        if not np.isclose(overall[family], float(recorded["recall_at_10"]), atol=1e-10):
            raise ValueError(f"per-query recall disagrees with main final run for {family}")
        for bin_number in range(1, 5):
            members = bins == bin_number
            rows.append({
                "dataset": "glove", "query_split": "report", "query_start": 5000,
                "query_end": 10000, "index_family": family, "bin": bin_number,
                "query_count": int(members.sum()),
                "score_min": float(contrast[members].min()),
                "score_max": float(contrast[members].max()),
                "mean_recall_at_10": float(recalls[members].mean()),
                "overall_recall_at_10": overall[family],
                "score": "(d10-d1)/d10, true squared L2",
                "build_settings": json.dumps(config.build_settings, sort_keys=True),
                "search_settings": json.dumps(config.search_settings, sort_keys=True),
                "source_sha256": source_hash,
                "selected_tune_run_id": tune_run_id,
                "main_final_run_id": final_run_id,
            })
        print(f"{LABELS[family]}: " + ", ".join(
            f"bin {b}={float(recalls[bins == b].mean()):.4f}" for b in range(1, 5)
        ), flush=True)
        del index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "query_difficulty.csv"
    with table_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for family in FAMILIES:
        group = [row for row in rows if row["index_family"] == family]
        ax.plot([int(row["bin"]) for row in group],
                [float(row["mean_recall_at_10"]) for row in group],
                marker="o", linewidth=2, label=LABELS[family], color=COLORS[family])
    ax.set(xlabel="Local contrast quartile (1 = smallest)", ylabel="Mean recall@10",
           title="GloVe-100: held-out recall by exact-neighbour contrast")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "glove_query_difficulty.png", dpi=180)
    fig.savefig(figure_dir / "glove_query_difficulty.pdf")
    plt.close(fig)

    lines = [
        "# Part 3: GloVe query difficulty", "",
        "Pre-result hypothesis: `code/hypothesis.md` (commit `82d9cdc`).",
        "The 5,000 held-out GloVe queries are ordered by `(d10-d1)/d10`, where distances are true squared L2 to exact neighbours of unit-normalized vectors. Ties are broken by original query row. Four equal-count bins contain 1,250 queries each; bin 1 has the smallest local contrast.",
        "", "| Bin | Count | Contrast range | HNSW | IVF-Flat | IVF-PQ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in range(1, 5):
        group = {row["index_family"]: row for row in rows if row["bin"] == b}
        low, high = group["hnsw"]["score_min"], group["hnsw"]["score_max"]
        lines.append(f"| {b} | 1,250 | {low:.4f}–{high:.4f} | " + " | ".join(
            f"{float(group[family]['mean_recall_at_10']):.4f}" for family in FAMILIES
        ) + " |")
    lines += ["", "Frozen tuning choices: HNSW `M=32, efConstruction=160, efSearch=512`; IVF-Flat `nlist=1024, nprobe=256`; IVF-PQ `nlist=1024, m=50, nbits=8, nprobe=1024`. IVF-PQ was the highest-recall tuned choice and did not reach 0.95. Per-query recall uses true source-vector distances and the established tie-aware cutoff. The CSV links each family to its tuning selection and matching main final run.", ""]
    gaps = {}
    for family in FAMILIES:
        vals = [float(row["mean_recall_at_10"]) for row in rows if row["index_family"] == family]
        gaps[family] = vals[-1] - vals[0]
        direction = "increased monotonically" if all(a <= b for a, b in zip(vals, vals[1:])) else "did not increase monotonically"
        lines.append(f"{LABELS[family]} {direction} across the bins; bin 4 minus bin 1 = {gaps[family]:+.4f}.")
    largest = max(gaps, key=gaps.get)
    lines += [f"The largest endpoint gap was {LABELS[largest]} ({gaps[largest]:+.4f}).", "",
              "The prediction was partly supported: all three endpoint changes were positive and IVF-PQ had the largest, but IVF-Flat fell slightly from bin 3 to bin 4. Tightly grouped exact neighbours may be easier to displace in approximate ranking. For IVF-PQ, all 1,024 lists were probed, so missed coarse lists cannot account for its remaining errors; quantized distance ranking is a plausible contributor. HNSW graph search and IVF-Flat's limited list probing have different failure mechanisms. The IVF-Flat dip shows that local contrast alone does not order every query's search difficulty.", "",
              "Local top-ten contrast does not measure the tenth-to-eleventh gap, IVF centroid boundaries, HNSW graph reachability, or PQ quantization error. Bin associations are descriptive and do not isolate a causal mechanism. The selected search settings also differ by family; their gap sizes do not compare intrinsic sensitivity at equal computational effort.", ""]
    (args.output_dir / "query_difficulty_analysis.md").write_text("\n".join(lines))
    print(f"Wrote {table_path}, figure, and analysis", flush=True)


if __name__ == "__main__":
    main()
