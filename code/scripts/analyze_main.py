"""Generate main-result plots and tables solely from results/results.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "annbench-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from annbench.benchmarks.sweeps import FAMILIES, TARGETS
from annbench.data import DATASETS


LABELS = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat", "ivf_pq": "IVF-PQ", "IndexFlatL2": "Exact Flat"}
DATASET_LABELS = {"sift": "SIFT-1M", "glove": "GloVe-100"}
COLORS = {"hnsw": "#0072B2", "ivf_flat": "#D55E00", "ivf_pq": "#009E73", "IndexFlatL2": "#222222"}


def load_measurements(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    with path.open(newline="") as source:
        raw = list(csv.DictReader(source))
    tuning_ids = {r["run_id"] for r in raw if r["experiment"] == "ann_tuning_sweep" and r["query_split"] == "tune"}
    groups = defaultdict(list)
    for row in raw:
        if row["experiment"] == "ann_final" or (
            row["experiment"] == "exact_baseline" and row["timing_mode"] == "single"
        ):
            if row["query_split"] != "report" or int(row["query_start"]) != 5000 or int(row["query_end"]) != 10000:
                raise ValueError(f"non-held-out row included: {row['run_id']}")
            if row["experiment"] == "ann_final":
                if row["selected_tune_run_id"] not in tuning_ids or row["benchmark_mode"] != "full":
                    raise ValueError(f"final row lacks full-size tuning provenance: {row['run_id']}")
                if not row["index_size_bytes"] or not row["bytes_per_vector"]:
                    raise ValueError(f"index size is missing: {row['run_id']}")
            groups[(row["dataset"], row["index_family"], row["run_id"])].append(row)
    measurements = defaultdict(list)
    for (dataset, family, run_id), rows in groups.items():
        expected = int(rows[0]["repetitions"])
        if len(rows) != expected or {int(r["repeat_index"]) for r in rows} != set(range(1, expected + 1)):
            raise ValueError(f"incomplete repetitions for {run_id}")
        if any(r["median_qps"] != rows[0]["median_qps"] for r in rows):
            raise ValueError(f"inconsistent median QPS for {run_id}")
        measurements[(dataset, family)].append(rows[0])
    for dataset in DATASETS:
        for family in (*FAMILIES, "IndexFlatL2"):
            if not measurements[(dataset, family)]:
                raise ValueError(f"missing final measurement for {dataset}/{family}")
    return measurements


def frontier(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep observed points that are not slower and less accurate than another."""
    return sorted((row for row in rows if not any(
        other["run_id"] != row["run_id"]
        and float(other["recall_at_10"]) >= float(row["recall_at_10"])
        and float(other["median_qps"]) >= float(row["median_qps"])
        for other in rows
    )), key=lambda row: float(row["recall_at_10"]))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    measurements = load_measurements(args.results)
    threshold_rows, memory_rows, plot_rows = [], [], []
    lines = ["# Main benchmark results", "", "Generated from `results.csv`; QPS uses one query per FAISS call and the median of repeated timings. All reported points use held-out queries 5,001–10,000.", ""]
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        lines += [f"## {DATASET_LABELS[dataset]}", "", "| Family | QPS at 0.90 | QPS at 0.95 | Highest recall |", "| --- | ---: | ---: | ---: |"]
        summary = {}
        for family in FAMILIES:
            rows = measurements[(dataset, family)]
            best = frontier(rows)
            xs = [float(r["recall_at_10"]) for r in best]
            ys = [float(r["median_qps"]) for r in best]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, label=LABELS[family], color=COLORS[family])
            for r in best:
                plot_rows.append({
                    "dataset": dataset, "family": family, "run_id": r["run_id"],
                    "selected_tune_run_id": r["selected_tune_run_id"],
                    "recall_at_10": r["recall_at_10"], "median_qps": r["median_qps"],
                    "build_settings": r["build_settings"], "search_settings": r["search_settings"],
                })
            summary[family] = []
            for target in TARGETS:
                qualifying = [r for r in rows if float(r["recall_at_10"]) >= target]
                highest = max(rows, key=lambda r: (float(r["recall_at_10"]), float(r["median_qps"])))
                chosen = [r for r in rows if str(target) in r["selected_targets"].split(";")]
                if len(chosen) != 1:
                    raise ValueError(f"expected one frozen {target} selection for {dataset}/{family}")
                winner = chosen[0]
                reached = float(winner["recall_at_10"]) >= target
                status = "reached" if reached else ("selected_below_target" if qualifying else "unreachable")
                threshold_rows.append({
                    "dataset": dataset, "family": family, "target_recall": target,
                    "status": status,
                    "median_qps": winner["median_qps"] if reached else "",
                    "measured_recall": winner["recall_at_10"],
                    "highest_measured_recall": highest["recall_at_10"],
                    "run_id": winner["run_id"],
                    "build_settings": winner["build_settings"],
                    "search_settings": winner["search_settings"],
                })
                summary[family].append(f"{float(winner['median_qps']):,.0f}" if reached else ("unreachable" if not qualifying else "selected below target"))
                if target == 0.95:
                    memory_rows.append({
                        "dataset": dataset, "family": family,
                        "status": status,
                        "measured_recall": winner["recall_at_10"],
                        "build_seconds": winner["build_seconds"] if reached else "",
                        "index_size_bytes": winner["index_size_bytes"] if reached else "",
                        "bytes_per_vector": winner["bytes_per_vector"] if reached else "",
                        "run_id": winner["run_id"],
                        "build_settings": winner["build_settings"],
                        "search_settings": winner["search_settings"],
                    })
            lines.append(f"| {LABELS[family]} | {summary[family][0]} | {summary[family][1]} | {max(float(r['recall_at_10']) for r in rows):.4f} |")
        exact = measurements[(dataset, "IndexFlatL2")][0]
        for target in TARGETS:
            threshold_rows.append({
                "dataset": dataset, "family": "IndexFlatL2", "target_recall": target,
                "status": "reached", "median_qps": exact["median_qps"],
                "measured_recall": exact["recall_at_10"],
                "highest_measured_recall": exact["recall_at_10"],
                "run_id": exact["run_id"], "build_settings": "", "search_settings": "",
            })
        ax.scatter([1.0], [float(exact["median_qps"])], marker="*", s=120, color=COLORS["IndexFlatL2"], label="Exact Flat", zorder=5)
        plot_rows.append({"dataset": dataset, "family": "IndexFlatL2", "run_id": exact["run_id"], "selected_tune_run_id": "", "recall_at_10": "1.0", "median_qps": exact["median_qps"], "build_settings": "", "search_settings": ""})
        ax.set(xlabel="Recall@10", ylabel="Single-query QPS", title=f"{DATASET_LABELS[dataset]}: held-out ANN trade-offs")
        ax.set_yscale("log")
        ax.set_xlim(left=0, right=1.015)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_recall_qps.png", dpi=180)
        fig.savefig(figures / f"{dataset}_recall_qps.pdf")
        plt.close(fig)
        lines += [f"| Exact Flat | {float(exact['median_qps']):,.0f} | {float(exact['median_qps']):,.0f} | 1.0000 |", "", f"Exact Flat: {float(exact['median_qps']):,.1f} single-query QPS at recall 1.0 (run `{exact['run_id']}`).", ""]
        lines += ["| Family at 0.95 | Status | Build (s) | Index (MiB) | Bytes/vector |", "| --- | --- | ---: | ---: | ---: |"]
        for entry in (r for r in memory_rows if r["dataset"] == dataset):
            if entry["status"] == "reached":
                build = f"{float(entry['build_seconds']):,.2f}"
                size = f"{int(entry['index_size_bytes']) / 2**20:,.1f}"
                per_vector = f"{float(entry['bytes_per_vector']):,.1f}"
            else:
                build = size = per_vector = "—"
            lines.append(f"| {LABELS[entry['family']]} | {entry['status']} | {build} | {size} | {per_vector} |")
        lines.append("")
        pq_rows = measurements[(dataset, "ivf_pq")]
        pq_highest = max(pq_rows, key=lambda row: float(row["recall_at_10"]))
        if float(pq_highest["recall_at_10"]) < 0.95:
            exhaustive = [row for row in pq_rows if
                json.loads(row["search_settings"])["nprobe"] == json.loads(row["build_settings"])["nlist"]]
            if exhaustive:
                best_exhaustive = max(exhaustive, key=lambda row: float(row["recall_at_10"]))
                lines.append(
                    f"IVF-PQ reached at most {float(pq_highest['recall_at_10']):.4f} recall. "
                    f"With every coarse list probed, its best measured recall was "
                    f"{float(best_exhaustive['recall_at_10']):.4f} at "
                    f"{float(best_exhaustive['bytes_per_vector']):.1f} serialized bytes/vector "
                    f"(run `{best_exhaustive['run_id']}`). "
                    "Scanning every list removes missed coarse lists as the sole explanation; "
                    "the 8-bit product-quantized distance estimates can still misrank candidates."
                )
                lines.append("")
    write_csv(args.output_dir / "main_qps_thresholds.csv", list(threshold_rows[0]), threshold_rows)
    write_csv(args.output_dir / "main_build_memory.csv", list(memory_rows[0]), memory_rows)
    write_csv(args.output_dir / "main_plot_points.csv", list(plot_rows[0]), plot_rows)
    lines += ["The two threshold tables and plotted point IDs are in `main_qps_thresholds.csv`, `main_build_memory.csv`, and `main_plot_points.csv`. Threshold QPS and memory use the frozen tuning selection, never a setting chosen from held-out performance. `unreachable` means no measured setting in the tuning-approved grid reached the threshold; `selected_below_target` means the frozen selection missed it although another measured setting reached it. Index size is serialized FAISS bytes; bytes/vector divides by the full base count.", ""]
    (args.output_dir / "main_analysis.md").write_text("\n".join(lines))
    print("Wrote two plots, threshold and memory tables, point provenance, and main_analysis.md")


if __name__ == "__main__":
    main()
