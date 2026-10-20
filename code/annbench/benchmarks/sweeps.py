"""Expand tuning grids and select settings without inspecting held-out queries."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from annbench.benchmarks.runner import BenchmarkConfig


FAMILIES = ("hnsw", "ivf_flat", "ivf_pq")
TARGETS = (0.90, 0.95)


def configurations(plan: dict[str, Any], dataset: str, family: str) -> list[BenchmarkConfig]:
    """Expand build/search grid, skipping probes above the chosen nlist."""
    grid = plan["datasets"][dataset][family]
    configs: list[BenchmarkConfig] = []
    if not grid["build"] or not grid["search"]:
        raise ValueError("each family needs build and search settings")
    for build in grid["build"]:
        for search_value in grid["search"]:
            if family == "hnsw":
                search = {"efSearch": search_value}
            else:
                if search_value > build["nlist"]:
                    continue
                search = {"nprobe": search_value}
            config = BenchmarkConfig(
                dataset=dataset, index_family=family,
                build_settings=build, search_settings=search,
                seed=plan["seed"], mode=plan["mode"],
                repetitions=plan["repetitions"],
                quick_base_count=plan["quick_base_count"],
                quick_query_count=plan["quick_query_count"],
                tune_query_count=plan.get("tune_query_count"),
            )
            config.validate()
            configs.append(config)
    if not configs:
        raise ValueError("sweep has no valid configurations")
    return configs


def choose_setting(
    measurements: list[tuple[BenchmarkConfig, dict[str, Any]]], target: float,
) -> tuple[BenchmarkConfig, dict[str, Any], bool, float]:
    """Pick fastest qualifying tuning setting, or highest recall if unreachable."""
    if not measurements or any(row["query_split"] != "tune" for _, row in measurements):
        raise ValueError("selection requires nonempty tuning measurements only")
    highest = max(float(row["recall_at_10"]) for _, row in measurements)
    qualifying = [pair for pair in measurements if float(pair[1]["recall_at_10"]) >= target]
    if qualifying:
        chosen = max(qualifying, key=lambda pair: (float(pair[1]["median_qps"]), float(pair[1]["recall_at_10"])))
    else:
        chosen = max(measurements, key=lambda pair: (float(pair[1]["recall_at_10"]), float(pair[1]["median_qps"])))
    return chosen[0], chosen[1], bool(qualifying), highest


def selection_document(
    config: BenchmarkConfig, row: dict[str, Any], *, target: float,
    reached: bool, highest_recall: float, approved_search: list[dict[str, int]],
) -> dict[str, Any]:
    """A frozen single setting plus the tuning-approved search grid for its build."""
    return {
        "query_split": "tune",
        "tune_run_id": row["run_id"],
        "source_sha256": row["source_sha256"],
        "config": asdict(config),
        "selection_target": target,
        "target_reached": reached,
        "highest_observed_recall": highest_recall,
        "approved_search_settings": approved_search,
        "sweep_id": row["sweep_id"],
        "sweep_plan_sha256": row["sweep_plan_sha256"],
    }
