# Stage09 full-grid development results

**Status:** Complete (PASS). Test unopened.

- Grid: 2,268 hyperparameter policies.
- Fit count: 3,240 independently seeded model trajectories.
- Selected policy: w32_lr0300_a025_010_000_e025.
- Source, tests, the 2,268-row score table, plots, manifests and the
  selected ten-seed checkpoints/predictions are versioned in Git.
- The full intermediate trajectory checkpoint bank is approximately
  19.9 GiB. It remains local and is NOT pushed to GitHub. The entire
  policy-wise score table remains in dev_all_2268.csv. Reconstructing
  per-seed forecasts for unselected policies requires rerunning them.
- Frozen data and source hashes appear in design.json. The previous
  metadata-only source-signature update is recorded separately.

These results are historical development evidence, not a test-set
generalization claim. A future canonical pipeline must use fresh
study signatures and newly generated results.
