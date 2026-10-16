"""Run tuning-only ANN build/search sweeps and freeze threshold selections."""

from __future__ import annotations

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import gc
import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import faiss

from annbench.benchmarks.runner import benchmark_prebuilt, build_index, query_slice, reference_cutoffs
from annbench.benchmarks.sweeps import FAMILIES, TARGETS, choose_setting, configurations, selection_document
from annbench.data import DATASETS, load_dataset
from scripts.run_benchmark import save_rows, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[1] / "configs/quick_sweep.json")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output", type=Path, help="run-level CSV (defaults to results/sweeps_MODE.csv)")
    parser.add_argument("--summary", type=Path, help="threshold JSON (defaults to results/sweep_summary_MODE.json)")
    parser.add_argument("--selection-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/selections")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    args = parser.parse_args()

    raw_plan = args.plan.read_bytes()
    plan = json.loads(raw_plan)
    if plan["mode"] not in {"quick", "full"}:
        raise ValueError("plan mode must be quick or full")
    results_dir = Path(__file__).resolve().parents[2] / "results"
    if args.output is None:
        args.output = results_dir / f"sweeps_{plan['mode']}.csv"
    if args.summary is None:
        args.summary = results_dir / f"sweep_summary_{plan['mode']}.json"
    if set(plan["datasets"]) != set(DATASETS):
        raise ValueError("plan must define both assignment datasets")
    if any(set(families) != set(FAMILIES) for families in plan["datasets"].values()):
        raise ValueError("plan must define all three families for each dataset")
    plan_sha256 = hashlib.sha256(raw_plan).hexdigest()
    sweep_id = str(uuid.uuid4())
    names = DATASETS if args.dataset == "both" else [args.dataset]
    families = FAMILIES if args.family == "all" else [args.family]
    summary: dict[str, object] = {
        "sweep_id": sweep_id, "plan_sha256": plan_sha256,
        "mode": plan["mode"], "datasets": {},
    }
    faiss.omp_set_num_threads(1)
    for name in names:
        source_path = args.data_dir / DATASETS[name].filename
        source_hash = sha256(source_path)
        print(f"Loading {name} for tuning sweep", flush=True)
        data = load_dataset(source_path, DATASETS[name])
        dataset_summary: dict[str, object] = {}
        summary["datasets"][name] = dataset_summary
        for family in families:
            configs = configurations(plan, name, family)
            base = data.base[:configs[0].quick_base_count] if configs[0].mode == "quick" else data.base
            queries, stored_ids, _, _ = query_slice(data, configs[0], "tune")
            cutoffs = reference_cutoffs(base, queries, stored_ids, configs[0])
            measurements = []
            family_rows = []
            build_groups = {}
            for config in configs:
                key = json.dumps(config.build_settings, sort_keys=True)
                build_groups.setdefault(key, []).append(config)
            for build_key, group in build_groups.items():
                index, build_seconds = build_index(group[0], base)
                index_size_bytes = int(faiss.serialize_index(index).nbytes)
                print(f"{name} {family} build {build_key}: {build_seconds:.2f}s", flush=True)
                for config in group:
                    rows = benchmark_prebuilt(
                        config, data, split="tune", source_sha256=source_hash,
                        index=index, build_seconds=build_seconds,
                        index_size_bytes=index_size_bytes, cutoffs=cutoffs,
                    )
                    for row in rows:
                        row["experiment"] = "ann_tuning_sweep"
                        row["sweep_id"] = sweep_id
                        row["sweep_plan_sha256"] = plan_sha256
                    family_rows.extend(rows)
                    measurements.append((config, rows[0]))
                    print(
                        f"  {config.search_settings}: recall={rows[0]['recall_at_10']:.4f}, "
                        f"median QPS={rows[0]['median_qps']:.0f}", flush=True,
                    )
                del index
                gc.collect()
            save_rows(
                args.output, family_rows,
                replace_filter=lambda old: (
                    old.get("experiment") == "ann_tuning_sweep"
                    and old.get("sweep_plan_sha256") == plan_sha256
                    and old.get("dataset") == name
                    and old.get("index_family") == family
                ),
            )
            family_summary = {}
            dataset_summary[family] = family_summary
            for target in TARGETS:
                selected, row, reached, highest = choose_setting(measurements, target)
                approved_search = [
                    config.search_settings for config, _ in measurements
                    if config.build_settings == selected.build_settings
                ]
                selection = selection_document(
                    selected, row, target=target, reached=reached,
                    highest_recall=highest, approved_search=approved_search,
                )
                args.selection_dir.mkdir(parents=True, exist_ok=True)
                selection_path = args.selection_dir / f"{plan['mode']}_{name}_{family}_{int(target*100):02d}.json"
                selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
                family_summary[str(target)] = {
                    "reached": reached,
                    "highest_observed_recall": highest,
                    "selected_recall": row["recall_at_10"],
                    "selected_median_qps": row["median_qps"],
                    "selected_tune_run_id": row["run_id"],
                    "selection_file": os.path.relpath(selection_path, args.summary.parent),
                }
                print(
                    f"{name} {family} target {target:.2f}: "
                    f"{'reached' if reached else 'unreachable'}; highest recall={highest:.4f}",
                    flush=True,
                )
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        del data
        gc.collect()


if __name__ == "__main__":
    main()
