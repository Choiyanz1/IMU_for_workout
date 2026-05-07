# Configs

The root `config.yaml` remains the default all-in-one config.

Use these copies when you want to keep experiment changes scoped by task:

- `base.yaml`: current shared default.
- `action_classification.yaml`: action-classification experiments.
- `phase_segmentation.yaml`: phase-segmentation experiments.
- `rep_segmentation.yaml`: SDTW repetition-segmentation experiments.
- `micro_macro_recognition.yaml`: DS-MS-TCN-style micro/macro experiments for
  fixed-order `concentric -> eccentric` rep segmentation and action recognition.
  Use `micro_macro.micro_source` to run `both` Stage 1 sources, or switch to
  only `tcn` or only `dtw`. Its resource defaults are automatic: `device: auto`
  prefers CUDA, then MPS, then CPU; `num_workers: auto` and `pin_memory: auto`
  tune the DataLoader for the resolved backend. Splits are subject-wise by
  subject folder, and DTW search knobs are exposed under `micro_macro.dtw`.

The training scripts do not require a specific filename; pass the file you want with `--config`.

Most training/evaluation entrypoints now write standardized comparison files
next to their model-specific artifacts:

- `report.md`
- `metrics/summary.json`
- `metadata/run_manifest.json`
- `metadata/config_snapshot.yaml`

Use `python -m evaluation.compare_runs --root artifacts` to collect these into
one comparison CSV.

Use `python -m evaluation.model_suite --models ds_ms_tcn --mode sets` when you
want one command to run selected comparable models on the shared configured
dataset and save both separate model artifacts and a shared comparison table.
