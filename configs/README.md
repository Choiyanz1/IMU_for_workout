# Configs

The root `config.yaml` remains the default all-in-one config.

Use these copies when you want to keep experiment changes scoped by task:

- `base.yaml`: current shared default.
- `action_classification.yaml`: action-classification experiments.
- `phase_segmentation.yaml`: phase-segmentation experiments.
- `rep_segmentation.yaml`: SDTW repetition-segmentation experiments.

The training scripts do not require a specific filename; pass the file you want with `--config`.
