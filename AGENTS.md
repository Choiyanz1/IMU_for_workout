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
- **Reference existing designs before creating new ones**: When designing a new model, experiment, or script, always review existing implementations and results first. If an existing design already solves the problem well (even if imperfect), prefer extending or fixing it over rewriting from scratch. Document why the existing design was kept or modified.

## Current Deployment Gate

- **NEW CANDIDATE (2026-05-17)**: Per-Action Plain RF at `artifacts/baseline_comparison/per_action_plain_rf_7fold/`
  - 7-fold LOSO: Rep F1 = **0.850**, Precision = 0.869, Recall = 0.831, IoU-F1@50 = 0.706
  - 8 per-action models (100 trees, depth 15, 1.0s trailing window), total ~1.6MB
  - **Viable for development board deployment** pending streaming inference verification
- **DEPRECATED**: DS-MS-TCN checkpoint `artifacts/micro_macro_recognition/20260508_143504/tcn` (Rep F1 too low, poor recall)
- Newer folders `artifacts/micro_macro_recognition/20260508_145027/tcn` and `artifacts/micro_macro_recognition/board_100hz_l6_20260508/tcn` have empty `models/` and `metrics/` directories, so they are not valid deploy candidates.

### Deployment Checklist for Per-Action RF
- [x] Offline quality verified (7-fold LOSO, 0.850 F1)
- [ ] Streaming-style inference on held-out subject data
- [ ] Deployable artifact path documented
- [ ] Normalization stats per-action exported
- [ ] Label maps/classes documented
- [ ] Replay/inference command documented
