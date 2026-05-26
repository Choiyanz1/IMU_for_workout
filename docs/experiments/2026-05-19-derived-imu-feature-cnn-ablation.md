# Derived IMU Feature CNN Ablation

Date: 2026-05-19

## Goal

Test whether previously identified important IMU features improve the current raw6 C/E phase segmentation CNN.

The feature-importance notes suggested:

- single raw axes are important, especially `ax`
- magnitude alone loses axis-specific information
- velocity/change-rate features may help because rep boundaries are motion-state changes

Therefore these probes **keep raw6** and add derived channels instead of replacing the raw IMU input.

## Implementation

Script:

```bash
python scripts/new_c_pipeline/derived_feature_cnn_9fold.py
```

Feature modes:

| Mode | Added channels | Total channels |
|---|---|---:|
| `mag` | `acc_mag`, `gyro_mag` | 8 |
| `delta` | `d_ax`, `d_ay`, `d_az`, `d_gx`, `d_gy`, `d_gz` | 12 |
| `mag_delta` | magnitude + delta channels | 14 |

All derived features are causal at sample `t`:

- magnitude uses the current sample only
- delta uses `x[t] - x[t-1]`
- train-fold z-score normalization is still applied after feature construction

The active detector remains the same raw6 RF active detector so only the CNN phase input changes.

## Fast Probe Results

Fast setting: 9-fold LOSO, hidden=32, epochs=5, same exclusions.

| Run | Method | Rep F1 | Exact | Count MAE | Phase IoU-F1@50 | C/E MAE | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `mag` fast | Raw6 baseline | 0.830 | 0.586 | 1.495 | 0.694 | 0.810 | Baseline for same run |
| `mag` fast | Raw6 + magnitude | 0.851 | 0.532 | 1.409 | 0.737 | 0.665 | Mixed; Exact drops |
| `delta` fast | Raw6 baseline | 0.835 | 0.573 | 1.500 | 0.710 | 0.684 | Baseline for same run |
| `delta` fast | Raw6 + delta | **0.859** | **0.609** | **1.273** | **0.727** | **0.652** | Promising; formal follow-up required |
| `mag_delta` fast | Raw6 baseline | 0.852 | 0.532 | 1.473 | 0.693 | 0.694 | Baseline for same run |
| `mag_delta` fast | Raw6 + mag + delta | **0.870** | **0.595** | **1.336** | **0.751** | **0.686** | Promising but more complex |

Fast probes suggested that `delta` carried most of the useful signal. `mag_delta` looked strongest in fast mode but may overfit or destabilize count when model capacity increases.

## Formal GPU Results

Formal setting: 9-fold LOSO, hidden=64, epochs=20, CUDA GPU, same exclusions.

| Run | Method | Rep F1 | Exact | Count MAE | Phase IoU-F1@50 | C/E MAE | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `delta` formal | Raw6 baseline | **0.867** | **0.523** | **1.364** | 0.721 | 0.676 | Baseline for same run |
| `delta` formal | Raw6 + delta | 0.849 | 0.500 | 1.659 | **0.726** | **0.641** | Reject as main model: count/Rep F1 worse |
| `mag_delta` formal | Raw6 baseline | **0.859** | **0.523** | **1.541** | 0.717 | 0.665 | Baseline for same run |
| `mag_delta` formal | Raw6 + mag + delta | 0.848 | 0.514 | 1.691 | **0.737** | **0.620** | Reject as main model: count/Rep F1 worse |

## Interpretation

The derived features consistently improve some phase-structure metrics in formal runs:

- `delta`: Phase IoU-F1@50 `0.721 -> 0.726`, C/E MAE `0.676 -> 0.641`
- `mag_delta`: Phase IoU-F1@50 `0.717 -> 0.737`, C/E MAE `0.665 -> 0.620`

However, both formal variants make rep grouping/count worse:

- `delta`: Rep F1 `0.867 -> 0.849`, Count MAE `1.364 -> 1.659`
- `mag_delta`: Rep F1 `0.859 -> 0.848`, Count MAE `1.541 -> 1.691`

This does not pass the current acceptance gate because Rep F1 and Count MAE are not allowed to degrade.

## Decision

Do **not** replace the current raw6 C/E CNN input with `delta` or `mag_delta` yet.

The derived channels are useful evidence that motion-change information helps C/E phase shape, but the current CNN/decoder turns the sharper phase signal into worse count stability. If revisited, use one of these safer directions:

- add delta channels only to an auxiliary boundary/transition head, not the main phase logits
- use prediction confidence or short-fragment features in the decoder instead of exposing deltas directly to the whole CNN
- test stronger regularization or lower capacity for derived-channel inputs before another formal run
- evaluate derived inputs with `top5_p5` selective merge, since raw decoder count stability is the failure mode

For now, keep **raw6 + train-fold z-score** as the main phase model input.
