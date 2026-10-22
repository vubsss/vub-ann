# Ticket 08: Write the five-page report

**Phase:** 7 | **Target:** Oct 24 | **Depends on:** 05, 06, 07

## Tasks

- [x] Include method, machine specs, library versions, exact checks, and explanation of single versus batch exact timing.
- [x] Include the two main trade-off plots and required threshold and build/memory tables.
- [x] Present Parts 3 and 4 with hypotheses, results, mechanisms, and whether predictions held; keep Part 4 to one main figure and at most one page.
- [x] Answer all five Part 5 questions, including 150,000 versus 10-million-vector recommendations under 100 ms/query and 16 GB RAM.
- [x] Identify two ways the benchmark could mislead and state whether each was tested.
- [x] Verify every number and figure against the results CSV and proofread the final PDF.

## Acceptance

- [x] `report.pdf` is at most five pages, readable, and includes all required figures and tables.

Completed 22 October 2026. `code/scripts/build_report.py` audits source run
IDs and values behind the main tables and plotted points, Part 3 bin means,
and Part 4 settings before writing `report.pdf`. It checks quoted numerical
claims against the CSV records. The PDF has five A4 pages. Page 1 covers the
method, exact baseline, and SIFT plot; page 2 covers the GloVe plot and main
tables; page 3 covers Part 3; page 4 contains all of Part 4 with one main
figure; and page 5 answers all five Part 5 questions. All pages were rendered
with Poppler and visually inspected for legibility, overlap, and clipping.
