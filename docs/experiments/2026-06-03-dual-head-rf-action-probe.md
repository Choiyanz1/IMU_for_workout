# Dual-Head RF Action Probe

## Goal
- Test a first simple parallel action-recognition branch before integrating it with the `raw6 CNN + top5_p5` rep pipeline.
- Evaluate whether a two-head window RF can distinguish `workout_action` vs `non_action` and classify the 8 known actions early enough for set-level action locking.

## Design
- Script: `scripts/evaluate_dual_head_rf_action_loso.py`
- Input: raw6 IMU columns `ax, ay, az, gx, gy, gz`.
- Window: `200` samples at 100 Hz, stride `100` samples.
- Head 1: binary RF for `workout_action` vs `non_action`.
- Head 2: 8-class RF trained only on true workout windows.
- Non-action negatives: `big_rest`, `rest_after_set*`, plus windows where `phase` is not `concentric/eccentric`.
- Split: 9-fold held-out subject using the same excluded light sessions policy.

## Command
```sh
python -u scripts/evaluate_dual_head_rf_action_loso.py \
  --n-estimators 50 \
  --max-depth 12 \
  --stride-samples 100 \
  --output artifacts/action_recognition/dual_head_rf_loso/summary_fast.json
```

## Result
- Set streams: `220`
- Rest/non-action streams: `274`
- Active head:
  - Accuracy: `0.934`
  - Precision: `0.830`
  - Recall: `0.886`
  - F1: `0.854`
  - Macro F1: `0.905`
- Action head on true-active windows:
  - Accuracy: `0.810`
  - Macro F1: `0.790`
- Gated 9-class output:
  - Accuracy: `0.894`
  - Macro F1: `0.689`
- Set-level action lock:
  - Action lock rate: `0.845`
  - Accuracy among locked action streams: `0.912`
  - Median lock time: `4.0s`
  - Non-action false-lock rate: `0.315`

## Lock Policy Search
To reduce false locks, the same RF probabilities were evaluated with stricter temporal lock policies.

Command:
```sh
python -u scripts/evaluate_dual_head_rf_action_loso.py \
  --n-estimators 50 \
  --max-depth 12 \
  --stride-samples 100 \
  --lock-grid \
  --output artifacts/action_recognition/dual_head_rf_loso/summary_lock_grid.json
```

| Policy | Action lock rate | Locked action accuracy | Median lock time | Non-action false lock |
|---|---:|---:|---:|---:|
| balanced `a055_p055_m010_s3` | `0.845` | `0.912` | `4.0s` | `0.315` |
| strict `a070_p065_m015_s4` | `0.718` | `0.945` | `5.0s` | `0.195` |
| stricter `a075_p070_m020_s4` | `0.641` | `0.956` | `5.0s` | `0.135` |
| very strict `a080_p075_m020_s5` | `0.528` | `0.985` | `6.28s` | `0.082` |
| late `a080_p075_m025_s6` | `0.514` | `0.985` | `7.28s` | `0.075` |
| ultra `a085_p080_m025_s5` | `0.432` | `0.990` | `6.28s` | `0.056` |

Suggested operating points:
- Balanced candidate: `stricter_a075_p070_m020_s4`, because it cuts false locks from `31.5%` to `13.5%` while keeping `64.1%` action lock rate.
- Safety candidate: `very_strict_a080_p075_m020_s5`, because it cuts false locks to `8.2%` with high locked accuracy `98.5%`, but only locks about half of action streams.
- The lowest false-lock policy is `ultra_a085_p080_m025_s5` at `5.6%`, but the `43.2%` lock rate is likely too conservative for driving action-conditioned online decoding.

## Interpretation
- The RF probe is viable as a baseline for the parallel action branch: it reaches useful active detection and action classification on held-out subjects without waiting for completed reps.
- The current lock policy is too permissive for deployment because non-action streams falsely lock to an action about 31.5% of the time.
- The action accuracy is lower than the rep-level RF results from ThomasYang-style features, which is expected because this model uses early windows rather than completed rep segments.
- Lock-policy tuning can reduce false locks substantially, but the current RF probabilities have a clear safety/latency/coverage tradeoff. This should be treated as a reject/unknown calibration problem, not only a threshold problem.

## Next Step
- Keep the dual-head design, but improve rejection/calibration before using it to drive `top5_p5`:
  - tune stricter lock thresholds and margin/stability requirements;
  - train a stronger `non_action` class with preparation/transition negatives;
  - compare against a dual-head tiny CNN after this RF baseline;
  - rerun the full rep pipeline with predicted action context only after false locks are controlled.
