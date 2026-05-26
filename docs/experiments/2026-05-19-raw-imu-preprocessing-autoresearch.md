# Raw IMU Preprocessing Autoresearch Note

Date: 2026-05-19

## Question

Should the current raw 6-axis IMU C/E segmentation pipeline add feature preprocessing before the 1D causal CNN?

## Current Pipeline Evidence

The current main C/E model is already not fully unnormalized raw input. It uses train-fold normalization:

- Input channels: `ax, ay, az, gx, gy, gz`
- Input shape: `[batch, 6, 300]`
- Preprocessing: train-fold per-channel mean/std normalization from active segments
- No explicit low-pass/high-pass filtering
- No gravity/body-acceleration separation
- No jerk channels
- No magnitude channels in the CNN input
- No orientation-invariant transforms beyond what the CNN learns from raw axes

Relevant code:

- `scripts/new_c_pipeline/test_pca_input.py`: `normalize(segments)` computes train-fold mean/std and applies `(seg - mean) / std`.
- `scripts/new_c_pipeline/raw6_cnn_comprehensive_9fold.py`: `train_raw6_model()` extracts active segments, normalizes them, and trains the C/E CNN.
- `scripts/new_c_pipeline/compare_phase_models.py`: the RF active detector uses handcrafted window statistics, magnitude, and total variation, but this is separate from the CNN phase input.

## Existing Project Results

| Probe | Result | Decision |
|---|---|---|
| Raw 6-axis CNN | Rep F1 0.860, Count MAE 1.505, Phase IoU-F1@50 0.720, C/E MAE 0.670 | Current base model |
| Raw 6-axis CNN + `top5_p5` | Rep F1 0.874, Count MAE 0.973, Phase IoU-F1@50 0.716, C/E MAE 0.601 | Current best structured pipeline |
| PCA-4 formal GPU | Raw Rep F1 0.851 / PCA-4 0.845; same Exact and MAE | Do not replace raw axes with PCA |
| PCA-1 formal GPU | Rep F1 0.616, Count MAE 3.88 | Rejected |
| 9-axis IMU with magnetometer | Rep F1 0.787, Count MAE 2.05 | Rejected |
| RF active detector rich features | Useful for active/rest gating | Keep as gating model, not evidence that CNN needs handcrafted features |

## Literature/Method Signal

Wearable HAR and exercise-counting literature is mixed: classical models often rely on handcrafted features, while recent deep models commonly consume standardized raw IMU windows with minimal preprocessing.

Examples from the current related-work pass:

- RecoFit and ExerSense support practical pipelines with segmentation/classification/counting components and engineered signal processing.
- Soro et al. and Prabhu et al. support CNN-style learning directly from wearable exercise signals.
- Recent HAR papers often emphasize raw or minimally preprocessed IMU windows standardized by `StandardScaler`, avoiding expensive time-frequency transforms for real-time deployment.

The literature therefore does **not** imply that heavy feature engineering is necessary for the C/E CNN. It does suggest trying small, streaming-safe channels if they encode invariances the CNN may not learn robustly from limited subject data.

## Recommendation

Do **not** replace the raw 6-axis CNN input with a handcrafted feature pipeline.

Instead, keep raw 6-axis + train-fold z-score as the main input and test only lightweight, causal, deployment-safe augmentations as ablations.

Recommended priority:

| Priority | Preprocessing candidate | Why test it | Risk |
|---:|---|---|---|
| 1 | Add magnitude channels: `acc_mag`, `gyro_mag` | Orientation/load-invariant energy cues may help count stability and active transitions | May duplicate information and slightly increase model size |
| 2 | Add jerk/delta channels: first difference of accel/gyro | Reps are phase transitions; derivatives may sharpen C/E boundaries | Can amplify noise, may hurt C/E ratio |
| 3 | Add causal low-pass-smoothed copy or EMA channels | May reduce sensor noise and short false phase fragments | Adds latency/smoothing; could blur transitions |
| 4 | Gravity/body acceleration split | Could improve orientation robustness | Needs careful causal filter; may be brittle across exercises |
| 5 | Frequency/STFT features | Literature uses them sometimes for HAR | Too heavy and not aligned with low-latency board goal; low priority |
| 6 | PCA or magnetometer channels | Already tested | Rejected unless a new action-specific reason appears |

## Proposed Experiments

Run these as strict LOSO ablations against the frozen current pipeline. Each experiment should use the same excluded sessions and metrics.

| Experiment | Input columns | Acceptance gate |
|---|---|---|
| `raw6_plus_mag` | raw 6 + `acc_mag`, `gyro_mag` | Rep F1 and Phase IoU-F1 must not drop; Count MAE or C/E MAE should improve |
| `raw6_plus_delta` | raw 6 + first-difference 6 | Must improve transition/rep stability without increasing C/E MAE |
| `raw6_plus_mag_delta` | raw 6 + magnitude + delta | Only keep if it beats both smaller ablations |
| `raw6_plus_ema` | raw 6 + causal EMA-smoothed 6 | Must reduce over-fragmentation without adding unacceptable latency |

Primary metrics:

- Count MAE
- Rep IoU-F1@50
- Phase IoU-F1@50
- C/E Ratio MAE

Secondary metrics:

- Exact Count
- Within-1 Count
- Over/under rate
- Transition MAE
- Runtime/model-size impact

## Decision

Feature preprocessing is worth testing only as **small channel augmentation**, not as a replacement for raw IMU.

The best next ablation is `raw6_plus_mag` because it is simple, causal, cheap, and aligned with both prior RF feature usefulness and classic IMU counting heuristics. If that fails, try `raw6_plus_delta`. Avoid PCA, magnetometer expansion, and heavy frequency transforms for now.

## Follow-Up Result

This recommendation was tested in `docs/experiments/2026-05-19-derived-imu-feature-cnn-ablation.md`.

Fast probes were promising, especially `raw6_plus_delta`, but formal hidden=64/20-epoch GPU runs did **not** pass the gate. Delta and mag+delta improved Phase IoU-F1@50 and C/E MAE, but degraded Rep F1 and Count MAE.

Current decision after formal follow-up: keep raw6 + train-fold z-score as the main phase model input. Treat derived delta/magnitude channels as useful future signals for auxiliary boundary heads or decoder features, not as a replacement main CNN input.
