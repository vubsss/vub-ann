"""Build the Part 4 table, one main figure, and one-page analysis from results.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "annbench-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from annbench.experiments.cluster_imbalance import RATIOS


FAMILIES = ("hnsw", "ivf_flat")
LABELS = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat"}
COLORS = {"hnsw": "#0072B2", "ivf_flat": "#D55E00"}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row.get("experiment") == "synthetic_cluster_imbalance"]
    summary = []
    query_hashes = set()
    for ratio in RATIOS:
        base_hashes = set()
        for family in FAMILIES:
            group = [row for row in rows if row["population_ratio"] == str(ratio)
                     and row["index_family"] == family]
            if len(group) != 3 or {int(row["repeat_index"]) for row in group} != {1, 2, 3}:
                raise ValueError(f"missing three repetitions for ratio {ratio}, {family}")
            if len({row["run_id"] for row in group}) != 1:
                raise ValueError("repetitions have inconsistent run IDs")
            for field in ("recall_at_10", "median_qps", "source_sha256", "query_sha256"):
                if len({row[field] for row in group}) != 1:
                    raise ValueError(f"repetitions disagree about {field}")
            chosen = group[0]
            if (int(chosen["base_count"]), int(chosen["reference_query_count"]),
                int(chosen["faiss_threads"]), int(chosen["timed_query_count"])) != (96_000, 1024, 1, 20_480):
                raise ValueError("unexpected vector/query count or search threads")
            base_hashes.add(chosen["source_sha256"])
            query_hashes.add(chosen["query_sha256"])
            summary.append({field: chosen[field] for field in (
                "run_id", "population_ratio", "index_family", "recall_at_10", "median_qps",
                "rare_quartile_recall_at_10", "common_quartile_recall_at_10",
                "build_seconds", "index_size_bytes", "cluster_min_count", "cluster_max_count",
                "cluster_counts_json", "query_cluster_counts_json", "source_sha256",
                "query_sha256", "exact_ids_sha256", "build_settings", "search_settings",
                "seed", "faiss_version", "numpy_version",
                "ivf_centroid_counts_json", "rare_probed_candidates",
                "common_probed_candidates", "timed_query_count",
            )})
        if len(base_hashes) != 1:
            raise ValueError(f"families used different bases at ratio {ratio}")
    if len(query_hashes) != 1:
        raise ValueError("the query set changed across imbalance settings")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    args = parser.parse_args()
    summary = load_summary(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cluster_imbalance_summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    fig, (recall_ax, qps_ax) = plt.subplots(1, 2, figsize=(9.3, 3.7))
    xs = list(range(len(RATIOS)))
    for family in FAMILIES:
        group = [row for row in summary if row["index_family"] == family]
        recall_ax.plot(xs, [float(row["recall_at_10"]) for row in group],
                       marker="o", linewidth=2, color=COLORS[family], label=LABELS[family])
        qps_ax.plot(xs, [float(row["median_qps"]) for row in group],
                    marker="o", linewidth=2, color=COLORS[family], label=LABELS[family])
    for ax in (recall_ax, qps_ax):
        ax.set_xticks(xs, [f"{ratio}:1" for ratio in RATIOS])
        ax.set_xlabel("Largest:smallest cluster population")
        ax.grid(True, alpha=0.25)
    recall_ax.set_ylabel("Recall@10")
    recall_ax.set_ylim(0.85, 1.005)
    recall_ax.legend(loc="best")
    qps_ax.set_ylabel("Single-query QPS (median of 3)")
    fig.suptitle("Synthetic clusters: fixed queries and index settings")
    fig.tight_layout()
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "cluster_imbalance.png", dpi=180)
    fig.savefig(figure_dir / "cluster_imbalance.pdf")
    plt.close(fig)

    by_key = {(row["index_family"], int(row["population_ratio"])): row for row in summary}
    h_drop = float(by_key["hnsw", 1]["recall_at_10"]) - float(by_key["hnsw", 64]["recall_at_10"])
    i_drop = float(by_key["ivf_flat", 1]["recall_at_10"]) - float(by_key["ivf_flat", 64]["recall_at_10"])
    supported = i_drop > h_drop and i_drop > 0
    lines = [
        "# Part 4: Synthetic cluster imbalance", "",
        "Pre-result prediction: `code/hypothesis.md`, commit `379f0d9`. Seed 20261021 generates 96,000 vectors in 32 dimensions from 16 Gaussian clusters (centre radius 10, spread 1). Only population counts change; base noise, insertion order, and 1,024 separate queries (64 per cluster) stay fixed. HNSW uses `M=16, efConstruction=100, efSearch=32`; IVF-Flat uses `nlist=64, nprobe=4`. Exact Flat top-ten results define tie-aware recall. QPS uses one query per call, one thread, 20 passes per repetition, and three timing repetitions.",
        "", "| Ratio | Min–max cluster size | HNSW recall | HNSW QPS | IVF-Flat recall | IVF-Flat QPS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ratio in RATIOS:
        h, i = by_key["hnsw", ratio], by_key["ivf_flat", ratio]
        lines.append(
            f"| {ratio}:1 | {h['cluster_min_count']}–{h['cluster_max_count']} | "
            f"{float(h['recall_at_10']):.4f} | {float(h['median_qps']):,.0f} | "
            f"{float(i['recall_at_10']):.4f} | {float(i['median_qps']):,.0f} |"
        )
    lines += ["", f"From 1:1 to 64:1, HNSW recall changed by {-h_drop:+.4f} and IVF-Flat by {-i_drop:+.4f}. "
              + ("The predicted larger IVF-Flat decline occurred." if supported
                 else "The predicted larger IVF-Flat decline did not occur."), ""]
    control_ivf, extreme_ivf = by_key["ivf_flat", 1], by_key["ivf_flat", 64]
    control_centroids = json.loads(control_ivf["ivf_centroid_counts_json"])
    extreme_centroids = json.loads(extreme_ivf["ivf_centroid_counts_json"])
    lines += ["The predicted sparse-cluster failure did not occur: IVF-Flat recall on queries from the four smallest clusters stayed at 1.0000, while recall on the four largest fell from "
              f"{float(control_ivf['common_quartile_recall_at_10']):.4f} to {float(extreme_ivf['common_quartile_recall_at_10']):.4f}. "
              "Assigning learned IVF centroids to the nearest generating centre, the four smallest clusters had "
              f"{sum(control_centroids[:4])} then {sum(extreme_centroids[:4])} centroids; the four largest had {sum(control_centroids[-4:])} then {sum(extreme_centroids[-4:])}. "
              "Few lists can cover a sparse cluster, while fixed `nprobe=4` covers a smaller share of a large cluster. "
              f"Mean vectors scanned for large-cluster queries rose from {float(control_ivf['common_probed_candidates']):,.0f} to {float(extreme_ivf['common_probed_candidates']):,.0f}, so scan work alone did not preserve recall. "
              "Centroid assignment and scanned-list size support this interpretation but do not prove which lists held missed neighbours. HNSW graph paths were not inspected. The seed reproduced all data hashes and recall on a second run; QPS varied. Dimension, spread, query mix, and search effort were not varied. The summary CSV records settings and run IDs.", ""]
    (args.output_dir / "cluster_imbalance_analysis.md").write_text("\n".join(lines))
    print("Wrote cluster imbalance summary, one figure, and analysis")


if __name__ == "__main__":
    main()
