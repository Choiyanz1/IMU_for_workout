# Dual-Head CNN Action Probe

## Goal
- Test whether a tiny causal CNN is better than the dual-head RF probe for the planned parallel active/action branch.
- Keep the same window setup, non-action negatives, LOSO protocol, and lock metrics as the RF probe.

## Design
- Script: `scripts/evaluate_dual_head_cnn_action_loso.py`
- Input: raw6 IMU columns `ax, ay, az, gx, gy, gz`.
- Window: `200` samples at 100 Hz, stride `100` samples.
- Model: small causal Conv1D encoder with two heads:
  - active head: `non_action` vs `workout_action`
  - action head: 8 known workout actions, trained only on active windows
- Non-action negatives: `big_rest`, `rest_after_set*`, and windows where `phase` is not `concentric/eccentric`.
- Split: 9-fold held-out subject with the same excluded light sessions policy.

## Commands
Default lock probe:
```sh
python -u scripts/evaluate_dual_head_cnn_action_loso.py \
  --epochs 3 \
  --hidden 32 \
  --batch-size 256 \
  --output artifacts/action_recognition/dual_head_cnn_loso/summary_e3_h32.json
```

Conservative active weighting / strict lock probe:
```sh
python -u scripts/evaluate_dual_head_cnn_action_loso.py \
  --epochs 3 \
  --hidden 32 \
  --batch-size 256 \
  --active-positive-weight-scale 0.5 \
  --eval-active-threshold 0.6 \
  --lock-active-threshold 0.70 \
  --lock-threshold 0.65 \
  --lock-margin 0.15 \
  --stable-windows 4 \
  --min-lock-windows 4 \
  --output artifacts/action_recognition/dual_head_cnn_loso/summary_e3_h32_pos05_strict.json
```

## Results
| Model / policy | Active F1 | True-active action acc | True-active action macro F1 | Lock rate | Lock acc | Non-action false lock |
|---|---:|---:|---:|---:|---:|---:|
| RF, default lock | `0.854` | `0.810` | `0.790` | `0.845` | `0.912` | `0.315` |
| RF, strict lock | `0.854` | `0.810` | `0.790` | `0.718` | `0.945` | `0.195` |
| CNN, default lock | `0.801` | `0.813` | `0.793` | `0.964` | `0.882` | `0.609` |
| CNN, conservative active + strict lock | `0.798` | `0.818` | `0.803` | `0.795` | `0.928` | `0.334` |

## Interpretation
- The CNN slightly improves true-active action classification (`0.818` vs RF `0.810`), but this gain is small.
- The CNN active head is worse than RF for the deployment-critical rejection problem. Even with conservative active weighting and stricter lock thresholds, non-action false lock remains `0.334`, worse than RF strict lock `0.195`.
- The CNN appears biased toward high active recall / low rejection under this small 3-epoch probe. That is unsafe for controlling `top5_p5` because false action locks can create false reps during rest/prep periods.

## Decision
- Do not replace the RF action branch baseline with this CNN yet.
- Keep RF as the current practical baseline for active/action gating while exploring CNN improvements.
- If continuing CNN work, prioritize active calibration and hard non-action negatives before increasing model capacity:
  - tune active class weighting and threshold calibration;
  - add more preparation/transition negatives;
  - add explicit unknown/reject objective or focal loss;
  - only then test longer training / hidden=48 or 64.
