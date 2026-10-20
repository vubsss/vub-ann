"""Measure tuning-approved full-size search grids on all held-out queries."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import gc
import json
from dataclasses import replace
from pathlib import Path

import faiss

from annbench.benchmarks.runner import BenchmarkConfig, benchmark_prebuilt, build_index, query_slice, reference_cutoffs
from annbench.benchmarks.index_cache import load_index
from annbench.benchmarks.sweeps import FAMILIES, TARGETS
from annbench.data import DATASETS, load_dataset
from scripts.run_benchmark import save_rows, sha256


def selected_groups(selection_dir: Path, tuning_rows: list[dict[str, str]], dataset: str, family: str):
    """Return each selected build and its approved settings with tuning provenance."""
    by_id = {row["run_id"]: row for row in tuning_rows if row.get("experiment") == "ann_tuning_sweep"}
    groups = {}
    for target in TARGETS:
        path = selection_dir / f"full_{dataset}_{family}_{int(target * 100):02d}.json"
        selection = json.loads(path.read_text())
        config = BenchmarkConfig(**selection["config"])
        config.validate()
        if config.dataset != dataset or config.index_family != family or config.mode != "full":
            raise ValueError(f"wrong dataset, family, or mode in {path}")
        if selection["query_split"] != "tune" or selection["selection_target"] != target:
            raise ValueError(f"invalid tuning selection in {path}")
        selected = by_id.get(selection["tune_run_id"])
        if selected is None or selected["query_split"] != "tune" or selected["source_sha256"] != selection["source_sha256"]:
            raise ValueError(f"selected tuning run is missing from results.csv: {path}")
        build_key = json.dumps(config.build_settings, sort_keys=True)
        group = groups.setdefault(build_key, {"config": config, "settings": {}, "source_sha256": selection["source_sha256"], "target_run_ids": {}})
        group["target_run_ids"][str(target)] = selection["tune_run_id"]
        if group["source_sha256"] != selection["source_sha256"]:
            raise ValueError("selected builds disagree about the source file")
        for search in selection["approved_search_settings"]:
            matches = [row for row in tuning_rows if
                row.get("experiment") == "ann_tuning_sweep"
                and row.get("sweep_id") == selection["sweep_id"]
                and row.get("dataset") == dataset and row.get("index_family") == family
                and row.get("build_settings") == build_key
                and row.get("search_settings") == json.dumps(search, sort_keys=True)
                and row.get("query_split") == "tune"
                and row.get("repeat_index") == "1"
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one tuning run for {dataset}/{family}/{build_key}/{search}")
            setting_key = json.dumps(search, sort_keys=True)
            group["settings"][setting_key] = (
                search, matches[0]["run_id"], selection["sweep_id"],
                selection["sweep_plan_sha256"], float(matches[0]["recall_at_10"]),
            )
    # The tuning half alone sets the curve endpoint. A 0.97 margin avoids
    # extrapolating a near-threshold tuning hit to the held-out half.
    for group in groups.values():
        settings = sorted(group["settings"].items(), key=lambda pair: next(iter(pair[1][0].values())))
        first_margin = next((i for i, (_, item) in enumerate(settings) if item[4] >= 0.97), None)
        if first_margin is not None:
            group["settings"] = dict(settings[:first_margin + 1])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--results", type=Path, default=root / "results/results.csv")
    parser.add_argument("--selection-dir", type=Path, default=root / "results/selections")
    parser.add_argument("--index-cache-dir", type=Path, default=root / "results/index_cache")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    args = parser.parse_args()
    with args.results.open(newline="") as source:
        tuning_rows = list(csv.DictReader(source))
    names = DATASETS if args.dataset == "both" else [args.dataset]
    families = FAMILIES if args.family == "all" else [args.family]
    faiss.omp_set_num_threads(1)
    for name in names:
        source_path = args.data_dir / DATASETS[name].filename
        source_hash = sha256(source_path)
        print(f"Loading {name} for held-out measurement", flush=True)
        data = load_dataset(source_path, DATASETS[name])
        for family in families:
            groups = selected_groups(args.selection_dir, tuning_rows, name, family)
            completed_keys = set()
            for build_key, group in groups.items():
                config = replace(group["config"], repetitions=3)
                if group["source_sha256"] != source_hash:
                    raise ValueError("dataset checksum changed since tuning")
                base = data.base
                queries, stored_ids, _, _ = query_slice(data, config, "report")
                if len(queries) != 5000:
                    raise ValueError("final measurement must use all 5000 held-out queries")
                cutoffs = reference_cutoffs(base, queries, stored_ids, config)
                cached = load_index(
                    args.index_cache_dir, config, source_sha256=source_hash,
                    expected_base_count=len(base),
                )
                if cached is None:
                    index, build_seconds = build_index(config, base)
                    index_size_bytes = int(faiss.serialize_index(index).nbytes)
                    print(f"{name} {family} rebuilt {build_key}: {build_seconds:.2f}s", flush=True)
                else:
                    index, build_seconds, index_size_bytes = cached
                    print(f"{name} {family} loaded tuned build {build_key}: {build_seconds:.2f}s", flush=True)
                for search, tune_run_id, sweep_id, plan_hash, _ in group["settings"].values():
                    run_config = replace(config, search_settings=search)
                    rows = benchmark_prebuilt(
                        run_config, data, split="report", source_sha256=source_hash,
                        index=index, build_seconds=build_seconds,
                        index_size_bytes=index_size_bytes, cutoffs=cutoffs,
                    )
                    for row in rows:
                        row["selected_tune_run_id"] = tune_run_id
                        row["selected_targets"] = ";".join(
                            target for target, chosen in group["target_run_ids"].items()
                            if chosen == tune_run_id
                        )
                        row["sweep_id"] = sweep_id
                        row["sweep_plan_sha256"] = plan_hash
                    setting_key = json.dumps(search, sort_keys=True)
                    completed_keys.add((build_key, setting_key))
                    save_rows(
                        args.results, rows,
                        replace_filter=lambda old: old.get("experiment") == "ann_final"
                        and old.get("dataset") == name and old.get("index_family") == family
                        and old.get("build_settings") == build_key
                        and old.get("search_settings") == setting_key,
                    )
                    print(f"  {search}: recall={rows[0]['recall_at_10']:.4f}, median QPS={rows[0]['median_qps']:.1f}", flush=True)
                del index
                gc.collect()
            save_rows(
                args.results, [],
                replace_filter=lambda old: old.get("experiment") == "ann_final"
                and old.get("dataset") == name and old.get("index_family") == family
                and (old.get("build_settings"), old.get("search_settings")) not in completed_keys,
            )
        del data
        gc.collect()


if __name__ == "__main__":
    main()
