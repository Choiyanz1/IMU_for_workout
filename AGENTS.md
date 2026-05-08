# AGENTS.md

## Long-Term Project Rules

- Treat `artifacts/` as generated evaluation output. Do not commit large model checkpoints unless explicitly requested.
- Record model-impacting changes in `docs/dev-log.md` and add experiment-specific results under `docs/experiments/`.
- Keep system-level requirements in `docs/specs/system.md` and model architecture decisions in `docs/specs/model.md`.
- Use subject-wise splits for reported model quality. Do not mix the same subject across train and test for headline metrics.
- For the current shell environment, prefer the explicit conda path when running the `imu` environment:
  `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu <command>`.
- Before saying a model is ready for the development board, verify both offline quality and streaming-style inference on held-out subject data.
- Deployment-readiness requires a deployable artifact path, normalization stats, label maps/classes, and a replay or inference command documented with the result.

## Current Deployment Gate

- The only complete current DS-MS-TCN checkpoint found during the 2026-05-08 review is `artifacts/micro_macro_recognition/20260508_143504/tcn`.
- Newer folders `artifacts/micro_macro_recognition/20260508_145027/tcn` and `artifacts/micro_macro_recognition/board_100hz_l6_20260508/tcn` have empty `models/` and `metrics/` directories, so they are not valid deploy candidates.
- The current complete checkpoint is not recommended for the development board as a rep-counting model because held-out rep F1 is too low and recall is very poor.
