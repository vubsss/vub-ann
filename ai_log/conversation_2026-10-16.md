# AI conversation log — 16 October 2026

This records the assignment-related user prompts and user-visible AI responses in this conversation. Tool outputs and hidden instructions are not user-facing responses. Append later assignment-related AI exchanges to this log.

## Exchange 1

### User prompt

> look at Assignment 3 pdf here, 1) create a plan for this assignment and 2) divide it into phases

### Assistant progress response

> I’ll read the Assignment 3 PDF and turn its requirements into a practical plan with clear phases and deliverables. I’m using the PDF skill to inspect the document.

### Assistant final response

I read the three-page assignment. It is due **25 October 2026 at 23:59 IST** and requires an individual benchmark, two experiments, a report of at most five pages, and a reproducible submission. :codex-file-citation{path="/home/vibhu/Desktop/M26/IRE/ass3/Assignment 3 Benchmarking Approximate Nearest Neighbour Search.pdf" purpose="source"}

## Recommended plan

Use **query difficulty** for Part 3 and **cluster imbalance in generated data** for Part 4. They ask different questions, and both can reuse the benchmark code from Part 2.

| Phase | Target dates | Work | Done when |
|---|---|---|---|
| 1. Set up | Oct 16–17 | Create the Git repo and project structure; download SIFT and GloVe-100; record machine specs and library versions. Set up a results CSV and a seeded experiment configuration. | Both datasets load, GloVe vectors are normalized, and the first 5,000 queries are reserved for tuning. |
| 2. Exact baseline | Oct 18 | Implement tie-aware recall@10 and unit tests. Build FAISS exact search, compare its neighbours with those in the datasets, and time single-query and batch search. | Correctness checks pass and exact-search measurements are saved. |
| 3. Main benchmark | Oct 19–21 | Build HNSW, IVF-Flat and IVF-PQ on both datasets. Try several build settings and sweep `efSearch` or `nprobe`. Time one query at a time on one thread, repeat timings, and use medians. | The held-out 5,000 queries produce both trade-off plots and the recall 0.90/0.95, build-time and memory tables. |
| 4. Standard-data experiment | Oct 22 | **Before running it**, write and commit the Part 3 hypothesis. Divide GloVe queries into four difficulty bins using an exact-neighbour distance measure, then compare recall by bin. | A figure and explanation show whether difficult queries affect the indexes differently. |
| 5. Own-data experiment | Oct 23 | **Before running it**, write and commit the Part 4 hypothesis. Generate a seeded dataset with equal-size clusters as the control and three increasing levels of cluster imbalance. Hold vector count, dimensions and other settings fixed; compare at least HNSW and IVF-Flat. | One main figure explains what changed, what happened, and whether the hypothesis held. |
| 6. Report and reproduction | Oct 24 | Write the five-page report, including the five required discussion answers. Make `code/run_all.sh` reproduce the reported outputs and `--quick` run on a subset in under 15 minutes. | A clean run produces the CSV, figures and report inputs; the README explains both commands. |
| 7. Submission check | Oct 25 | Check the Git history, page count, CSV coverage, AI log, and ZIP contents; upload before 23:59 IST. | `<rollnumber>-A3.zip` contains only the required deliverables, with no datasets or index files. |

**Measurement rule to protect throughout:** choose settings using queries 1–5,000, then report final results using queries 5,001–10,000. Keep every benchmark run behind a reported number in `results/results.csv`.

Because this planning conversation uses an AI tool, include its prompt and response in `ai_log/` and mark source files accurately in `ai_statement.md`.

## Exchange 2

### User prompt

> write this in PLAN.md, + the tickets in tickets/* + write the ai_log thing also

### Assistant progress response

> I’ll add the plan and phase tickets to the repository, then record this AI-assisted planning exchange in `ai_log/`. I’ll check the existing files first so the names and structure fit what’s already here.

### Assistant final response

Created [PLAN.md](/home/vibhu/Desktop/M26/IRE/ass3/PLAN.md), ten phase tickets starting with [Ticket 01](/home/vibhu/Desktop/M26/IRE/ass3/tickets/01-setup-and-data.md), and the [AI conversation log](/home/vibhu/Desktop/M26/IRE/ass3/ai_log/conversation_2026-10-16.md) with its [AI use statement](/home/vibhu/Desktop/M26/IRE/ass3/ai_log/ai_statement.md). The tickets cover setup through submission, including the required hypothesis commits and reproduction checks.
