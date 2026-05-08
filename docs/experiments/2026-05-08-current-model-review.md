# 2026-05-08 Current Model Review

## Objective

Evaluate the current available model for development-board readiness and identify concrete improvements.

## Environment

- Conda environment requested: `imu`.
- Working conda executable: `/opt/homebrew/Caskroom/miniconda/base/bin/conda`.
- Shell note: the current tool shell did not include `/bin` and `/usr/bin` in `PATH`, so commands used an explicit PATH prefix.

## Candidate Checkpoints

| Candidate | Status |
| --- | --- |
| `artifacts/micro_macro_recognition/20260508_143504/tcn` | Complete checkpoint, metrics, reports, plots |
| `artifacts/micro_macro_recognition/20260508_145027/tcn` | Empty `models/` and `metrics/` |
| `artifacts/micro_macro_recognition/board_100hz_l6_20260508/tcn` | Empty `models/` and `metrics/` |

## Headline Metrics

Source: `artifacts/micro_macro_recognition/20260508_143504/tcn/metrics/summary.json`.

| Metric | Value |
| --- | ---: |
| rep precision | 0.4545 |
| rep recall | 0.1136 |
| rep F1 | 0.1818 |
| start MAE ms | 592.58 |
| end MAE ms | 878.90 |
| transition MAE ms | 933.28 |
| rep action accuracy | 0.2667 |
| micro sample accuracy | 0.5125 |
| micro sample macro F1 | 0.2746 |
| macro sample accuracy | 0.2424 |
| macro sample macro F1 | 0.0919 |
| micro F1@50 | 0.1344 |
| macro F1@50 | 0.1818 |

## Post-Processing Grid

Source: `artifacts/micro_macro_recognition/20260508_143504/tcn/postprocess_grid/best_tcn_postprocess.json`.

Best settings on the evaluated grid:

| Setting | Value |
| --- | ---: |
| min phase samples | 3 |
| max phase gap samples | 3 |
| min rep duration seconds | 1.2 |
| min rep confidence | 0.0 |
| precision | 0.4583 |
| recall | 0.2500 |
| F1 | 0.3235 |

This improves F1 over the saved run summary but still misses most true reps.

## Streaming Replay

Input: `datasets/raw_data/kevin/db_weighted_crunch/set0`.

Fast causal replay:

| Metric | Value |
| --- | ---: |
| samples | 3000 |
| elapsed seconds | 0.2278 |
| inferred sample rate Hz | 111.0001 |
| buffer size samples | 4089 |
| buffer seconds | 36.84 |
| micro accuracy | 0.6627 |
| macro accuracy | 0.9790 |

Step replay:

| Metric | Value |
| --- | ---: |
| samples | 300 |
| elapsed seconds | 2.3376 |
| approximate throughput | 128 samples/s |

Updated actual step replay after streaming instrumentation:

| Runtime buffer | Buffer seconds | Elapsed s | Samples/s | Real-time factor | Micro acc | Macro acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4089 | 36.84 | 76.882 | 39.02 | 0.352 | 0.6627 | 0.9790 |
| 512 | 4.61 | 37.779 | 79.41 | 0.715 | 0.6630 | 0.9790 |
| 256 | 2.31 | 23.485 | 127.74 | 1.151 | 0.6417 | 0.9790 |
| 128 | 1.15 | 21.401 | 140.18 | 1.263 | 0.6553 | 0.9790 |

For this replay, the full 4089-sample buffer is not real time on CPU. `--buffer-size 128` and `--buffer-size 256` are real time and preserve macro sample accuracy on this action, but this is not enough to declare the model deployment-ready.

## Error Pattern

- Pairing diagnostics show many `unexpected_phase_before_concentric` errors.
- Pairing diagnostics also show many `missing_eccentric_after_concentric` errors.
- The model often emits long or fragmented one-phase runs that cannot be paired into valid reps.
- The representative streaming replay has strong macro accuracy for one action, but full held-out macro metrics are poor, so it should not be generalized.

## Deployment Readiness

Verdict: not ready for development-board rep counting.

Reasons:

- Held-out rep recall is 0.1136, so most reps are missed.
- Best available post-processing still reaches only 0.25 recall.
- Full held-out macro sample F1 is 0.0919.
- The only complete checkpoint is an older 9-layer, 40 s slice model, not the current board-style 6-layer, 20 s, 100 Hz config.
- There is no DS-MS-TCN ONNX/export path in `deploy/`; existing export code targets `InertialStudent`.
- Online predictor still recomputes the active buffer per sample. Smaller runtime buffers can exceed real time on CPU for this replay, but the implementation is still not a board-optimized stateful TCN.

## Recommended Improvements

1. Complete a clean board-candidate training run using the current config: 100 Hz, 20 s slices, 6 layers, causal true.
2. Add a held-out validation subject separate from final test subject for post-processing threshold tuning.
3. Improve recall by training with stronger phase supervision or class balancing for short/transition phases.
4. Add action-conditioned rep pairing or action-specific thresholds because the dominant error modes differ by movement.
5. Add DS-MS-TCN export or runtime support before treating it as a board candidate.
6. Optimize streaming inference with cached convolution state or smaller receptive field; avoid recomputing the full buffer on every sample for deployment.
7. Re-run model comparison after the board-candidate checkpoint exists and require rep F1/recall gates before shipping.
