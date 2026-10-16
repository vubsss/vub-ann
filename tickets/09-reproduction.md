# Ticket 09: Make one-command reproduction work

**Phase:** 7 | **Target:** Oct 24 | **Depends on:** 05, 06, 07

## Tasks

- [ ] Implement `code/run_all.sh` so `bash run_all.sh` regenerates every reported number, figure, and table from the data files.
- [ ] Implement `bash run_all.sh --quick` on a small subset and keep it under 15 minutes on a laptop.
- [ ] Document setup, data locations, full and quick commands, expected outputs, and approximate hardware/runtime needs in `code/README.md`.
- [ ] Run both commands from a clean output state; verify output paths and CSV/figure generation.

## Acceptance

- [ ] Full reproduction finishes successfully on the benchmark machine.
- [ ] Quick mode is timed and documented; both commands are explained in the README.
