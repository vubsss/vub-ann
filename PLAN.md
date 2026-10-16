# Assignment 3 plan: Benchmarking Approximate Nearest Neighbour Search

**Deadline:** 25 October 2026, 23:59 IST. **Submission:** one `<rollnumber>-A3.zip` on Moodle. **Work:** individual.

## Goal and decisions

Measure the recall, single-query throughput, build time, and memory trade-offs of HNSW, IVF-Flat, and IVF-PQ on SIFT-1M and GloVe-100. Explain the mechanisms behind the results in a report of at most five pages.

- **Part 3 choice:** Query difficulty on GloVe-100. Define a difficulty score from exact-neighbour distances, split the held-out queries into four bins, and compare each index's recall by bin. Write and commit a hypothesis before running this experiment.
- **Part 4 choice:** Synthetic cluster imbalance. Use equal-size clusters as the control, then three increasingly imbalanced cluster-size settings. Keep the total vector count, dimension, cluster centres/spread, query-generation rule, and index settings fixed. Compare at least HNSW and IVF-Flat. Seed the generator and use at most 500,000 vectors. Write and commit a separate hypothesis before running this experiment.
- These choices are a working design. Any change must still satisfy the PDF's Part 3 and Part 4 requirements and preserve the pre-result hypothesis commits.

## Measurement protocol

1. Download the SIFT and GloVe-100 files from ann-benchmarks. Normalize GloVe base and query vectors, then use Euclidean distance throughout.
2. Reserve queries 1-5,000 for tuning and queries 5,001-10,000 for final reported measurements. Select build/search settings on the tuning half; do not select them using the final half.
3. Build a FAISS exact-search reference. Implement recall@10 ourselves, including correct treatment of ties at the tenth-neighbour distance, and unit-test it. Check the reference against the neighbours stored in each dataset.
4. Time exact search both one query at a time and as a batch. Time ANN search one query at a time on one thread, repeat timings a few times, and report median queries per second. Record hardware, software versions, settings, vector counts, timings, recall, and index size with every run.
5. For each ANN family and dataset, try several build settings and sweep `efSearch` or `nprobe` from low recall to approximately 0.95 or higher where feasible. Use the final split to plot the selected trade-off curves and report performance at recall 0.90 and 0.95. Explain any family that cannot reach a threshold.
6. Measure build time and serialized index size; calculate bytes per vector. Keep raw run-level data in `results/results.csv` so every reported number is traceable.

## Phases and schedule

| Phase | Target | Ticket(s) | Exit criterion |
| --- | --- | --- | --- |
| 1. Setup and data | Oct 16-17 | [01](tickets/01-setup-and-data.md) | Reproducible environment; both datasets load; machine and version metadata recorded. |
| 2. Exact baseline | Oct 18 | [02](tickets/02-exact-baseline.md) | Tie-aware recall tests and exact/reference checks pass; single and batch timings saved. |
| 3. Benchmark engine | Oct 19 | [03](tickets/03-benchmark-runner.md), [04](tickets/04-index-sweeps.md) | Single-thread, repeated timing and configurable sweeps work on a small subset. |
| 4. Main benchmark and analysis | Oct 20-21 | [05](tickets/05-main-results.md) | Both datasets have final plots, threshold tables, and build/memory results. |
| 5. Standard-data experiment | Oct 22 | [06](tickets/06-query-difficulty.md) | Part 3 hypothesis predates results; four-bin analysis and figure are complete. |
| 6. Own-data experiment | Oct 23 | [07](tickets/07-cluster-imbalance.md) | Part 4 hypothesis predates results; seeded control plus three settings and one main figure are complete. |
| 7. Report and reproduction | Oct 24 | [08](tickets/08-report.md), [09](tickets/09-reproduction.md) | Report fits five pages and both reproduction commands work. |
| 8. Final audit and submission | Oct 25 | [10](tickets/10-submission-audit.md) | ZIP contains exactly the required top-level contents and is submitted before 23:59 IST. |

## Report outline

1. Method: datasets, split, exact reference, recall/timing protocol, machine and library versions.
2. Main results: one recall-versus-QPS plot per dataset; QPS at recall 0.90 and 0.95; build time, index size, and bytes/vector at recall 0.95.
3. Part 3: hypothesis, query-difficulty definition, four-bin result, mechanism, and whether the hypothesis held.
4. Part 4: hypothesis, controlled generator and settings, one main figure, mechanism, limitations, and whether the hypothesis held (at most one page).
5. Discussion: fastest index at 0.95 on each dataset; PQ's savings and cost; why headline numbers can mislead; 150,000-vector and 10-million-vector choices under 100 ms/query and 16 GB RAM; two limitations of this benchmark and whether they were tested.

## Submission contract

The final ZIP must contain only `report.pdf`, `code/`, `results/`, and `ai_log/`. The `code/` directory must include its `.git` history, all source (including the Part 4 generator), `README.md`, `requirements.txt`, `hypothesis.md`, and `run_all.sh`. Both hypotheses must be committed before their experiments' results. `bash run_all.sh` must regenerate reported numbers, figures, and tables from the data files; `bash run_all.sh --quick` must do so on a small subset in under 15 minutes on a laptop. `results/` must contain `results.csv` and `figures/`. Do not package datasets or index files. Keep this working `PLAN.md` and `tickets/` outside the final ZIP unless explicitly incorporated under `code/`.

Working-repository decision (16 October): keep only the cwd Git repository and
track `code/` there. At final packaging, export the `code/` history into a
standalone repository in a temporary staging directory, include its `.git` in
the ZIP's `code/`, and verify that both hypothesis commits precede the related
experiment results. Do not create a nested Git repository in the working tree.

Because AI was used for planning, preserve the prompts and responses in `ai_log/` and maintain `ai_log/ai_statement.md` with accurate source-file classifications.
