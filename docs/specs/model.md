# Model Specification

## Current Micro/Macro Model

- Architecture: DS-MS-TCN in `models/ds_ms_tcn.py`.
- Stage 1 predicts micro labels: `other`, `concentric`, `eccentric`.
- Stages 2 to 4 refine macro/action labels from micro probabilities.
- Current complete checkpoint: `artifacts/micro_macro_recognition/20260508_143504/tcn/models/ds_ms_tcn.pt`.
- Checkpoint config snapshot:
  - sample rate: 50 Hz in the saved config snapshot, but evaluated streams infer about 111 Hz after replay/input loading.
  - causal: true.
  - filters/layers: 64/9.
  - slice seconds: 40.0.
  - training epochs: 30.
  - held-out subject: `kevin`.

## Current Config Direction

- `configs/micro_macro_recognition.yaml` now targets board-style settings: 100 Hz, 20 s slices, 6 TCN layers, causal inference.
- No complete checkpoint was found for this newer board-style config during the 2026-05-08 review.

## Online Inference Behavior

- `OnlineDSMSTCNPredictor` recomputes the rolling buffer on every sample.
- For the reviewed 9-layer checkpoint, the default total receptive-field buffer is 4089 samples, about 36.8 s at 111 Hz.
- The streaming evaluator now writes sample accuracies, label counts, throughput, and real-time factor into `streaming_summary.json`.
- A 2026-05-08 step replay on `kevin/db_weighted_crunch/set0` showed the default 4089-sample buffer is slower than real time on CPU. Runtime buffers of 256 or 128 samples exceeded real time on the same replay, with similar macro accuracy and slightly lower micro accuracy.
- This implementation is useful for validating online behavior, but it is not optimized for a constrained board because it still recomputes the rolling buffer each sample.

## Known Model Issues

- Rep detection is recall-limited on held-out `kevin` data.
- Macro/action sample prediction is weak in the full held-out evaluation even when one representative replay has high macro accuracy.
- Phase ordering errors appear frequently in pairing diagnostics, especially unexpected eccentric-before-concentric runs and missing eccentric-after-concentric runs.
- The deploy tooling does not yet export DS-MS-TCN to ONNX or another board runtime format.

## Improvement Direction

- Train and evaluate the current 100 Hz, 6-layer, 20 s causal config to completion before comparing it with the old 9-layer checkpoint.
- Tune rep post-processing on validation subjects, not on the held-out subject used for final reporting.
- Reduce online buffer size through the 6-layer model, smaller kernels, or a stateful/streaming TCN implementation. Re-test with `--method step` because `--method fast` does not measure per-sample runtime.
- Add a DS-MS-TCN export path or switch the board candidate to an already exportable student model if only action classification is required.
