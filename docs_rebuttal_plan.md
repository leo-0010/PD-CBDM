# Rebuttal experiment plan (1.5-day execution window)

## Goal
Address reviewer requests with the minimum sufficient set of additional experiments.

## Priority and scope

### P0 (must): stronger ablation isolation
- Run a compact ablation matrix to isolate each module contribution.
- Suggested rows: baseline, +PD, +reweight, +attention, +PD+reweight, full model.
- Metrics: FID (primary), plus class-wise/tail metrics already used in the paper.

### P1 (must): one extra imbalance setting + strong baseline comparison
- Add at least one imbalance ratio not currently reported.
- Re-run full model and at least one strong baseline under that setting.

### P2 (low cost, high impact): efficiency table
- Record training throughput (iter/s), sampling time (s/1k images), and parameter count.
- Report relative overhead versus baseline.

### P3 (optional): lightweight generalization evidence
- If time remains, add a small-scale transfer sanity check or short high-resolution trial.
- Otherwise address as limitation + future work in the response letter.

## Execution schedule

### Day 1 AM
- Launch P0 runs in parallel where hardware permits.
- Draft response skeleton aligned to reviewer comments.

### Day 1 PM
- Aggregate P0 results.
- Launch P1 new-ratio runs.

### Day 2 AM
- Finish P1 collection and run P2 efficiency profiling.
- Build revised tables/figures.

### Day 2 PM
- Finalize response letter and manuscript updates.

## Multi-config runner
Use `run_multi_config.py` to execute sequential runs from a single JSON plan:

```bash
python run_multi_config.py --plan config/experiments/rebuttal_1p5day_plan.json
```

Dry run (command check only):

```bash
python run_multi_config.py --plan config/experiments/rebuttal_1p5day_plan.json --dry_run
```

Continue remaining jobs when one fails:

```bash
python run_multi_config.py --plan config/experiments/rebuttal_1p5day_plan.json --continue_on_error
```
