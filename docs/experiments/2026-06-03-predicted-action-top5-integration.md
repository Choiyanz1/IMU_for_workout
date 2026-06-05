# Predicted-Action Top5 Integration

## Goal
- Connect the parallel dual-head RF action branch to the existing `raw6 CNN + top5_p5` rep decoder.
- Measure whether predicted action context is already safe/useful enough to replace ground-truth action labels in the selective merge decoder.

## Design
- Script: `scripts/evaluate_predicted_action_top5_pipeline.py`
- Split: 9-fold held-out subject, using the same excluded light sessions policy.
- Phase model: raw6 Causal CNN, trained per fold with `epochs=5`, `hidden=64` for an integration-quality run.
- Action branch: dual-head RF with 200-sample windows and 100-sample stride.
- Active detector: corrected to a global action-agnostic RF by default. It trains on all train-fold set streams plus rest/non-action streams and does not select a model using the true action at inference.
- Decoder variants evaluated on the same phase probabilities per fold:
  - `raw`: base rep parser without selective duration merge.
  - `oracle_top5`: `top5_p5` merge using ground-truth action labels.
  - `predicted_top5`: `top5_p5` merge only when the RF branch locks an action; otherwise falls back to `raw`.
  - `soft_top5`: aggregates RF action posterior over the set and uses a weighted top5 duration threshold only when posterior confidence/mass/margin pass thresholds; otherwise falls back to `raw`.

## Commands
Balanced candidate:
```sh
python scripts/evaluate_predicted_action_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --lock-policy stricter \
  --active-detector global \
  --output artifacts/action_recognition/predicted_action_top5_pipeline/summary_stricter_soft_global_active_e5_h64.json
```

Safety candidate:
```sh
python scripts/evaluate_predicted_action_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --lock-policy very_strict \
  --output artifacts/action_recognition/predicted_action_top5_pipeline/summary_very_strict_e5_h64.json
```

## Results
These are integration runs, not the final 20-epoch deployment-quality CNN results. Each policy command retrains the CNN, so small raw/oracle differences between the two rows should not be interpreted as policy effects.

Important correction: an earlier integration run used the legacy per-action active detector from `compare_phase_models.py`, which selected the active model using the true action embedded in `stream_id`. That is useful as an optimistic/debug comparison, but it is not a valid fully automatic pipeline. The main table below uses the corrected global action-agnostic active detector.

Corrected global-active results:

| Active gate | Policy | Variant | Rep IoU-F1@50 | Exact Count | Within-1 | Count MAE | Phase IoU-F1@50 | C/E MAE | Over-rate | Under-rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global | `stricter` | raw | `0.801` | `0.459` | `0.600` | `1.973` | `0.639` | `0.861` | `0.327` | `0.214` |
| global | `stricter` | oracle top5 | `0.827` | `0.482` | `0.677` | `1.641` | `0.639` | `0.803` | `0.100` | `0.418` |
| global | `stricter` | hard predicted top5 | `0.802` | `0.473` | `0.605` | `1.968` | `0.639` | `0.835` | `0.205` | `0.323` |
| global | `stricter` | soft top5 | `0.823` | `0.468` | `0.664` | `1.691` | `0.639` | `0.806` | `0.118` | `0.414` |

Legacy per-action-active debug results:

| Policy | Variant | Rep IoU-F1@50 | Exact Count | Within-1 | Count MAE | Phase IoU-F1@50 | C/E MAE | Over-rate | Under-rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stricter` | raw | `0.837` | `0.555` | `0.691` | `1.586` | `0.701` | `0.652` | `0.409` | `0.036` |
| `stricter` | oracle top5 | `0.867` | `0.582` | `0.782` | `1.027` | `0.701` | `0.579` | `0.155` | `0.268` |
| `stricter` | hard predicted top5 | `0.837` | `0.559` | `0.705` | `1.445` | `0.701` | `0.612` | `0.277` | `0.164` |
| `stricter` | soft top5 | `0.860` | `0.577` | `0.755` | `1.118` | `0.701` | `0.585` | `0.182` | `0.241` |
| `very_strict` | raw | `0.838` | `0.541` | `0.700` | `1.591` | `0.705` | `0.663` | `0.405` | `0.055` |
| `very_strict` | oracle top5 | `0.868` | `0.564` | `0.773` | `1.086` | `0.705` | `0.605` | `0.155` | `0.282` |
| `very_strict` | predicted top5 | `0.839` | `0.523` | `0.705` | `1.532` | `0.705` | `0.653` | `0.336` | `0.141` |

Action lock summary on set streams:

| Policy | Action lock rate | Locked action accuracy | Median lock time |
|---|---:|---:|---:|
| `stricter` | `0.641` | `0.957` | `5.0s` |
| `very_strict` | `0.527` | `0.983` | `6.0s` |

## Interpretation
- With the corrected global active detector, the automatic pipeline is materially harder: raw Rep F1 drops to `0.801` and Phase IoU-F1@50 drops to `0.639` in this 5-epoch run.
- Soft top5 remains better than hard action locking: Count MAE improves from `1.968` to `1.691`, Rep F1 improves from `0.802` to `0.823`, and C/E MAE improves from `0.835` to `0.806`.
- Soft top5 nearly matches the corrected oracle top5 Count MAE (`1.691` vs `1.641`), so the action-conditioned merge idea is still useful after removing the active-gate leakage.
- The remaining gap versus the earlier legacy run shows that active segmentation is now a major bottleneck for fully automatic deployment.
- `stricter` remains the better integration operating point so far: it keeps more action coverage and gives enough posterior evidence for soft action-conditioned merging.
- `very_strict` reduces wrong locks, but the lower lock coverage means it barely changes Rep F1 and worsens Exact Count versus raw in this run.
- Incorrect action locks can be worse than no action context because they trigger the wrong selective merge. This was visible in subject folds such as `hsianshun`, `thomas`, and `yushuan`.

## Decision
- Replace hard action-lock decoding as the next research path with soft posterior-gated top5 merging.
- Keep unknown/raw fallback: if the action posterior mass/confidence/margin is insufficient, do not force a top5 merge.
- Use `stricter_a075_p070_m020_s4` plus soft top5 as the next engineering integration candidate, but only with the global active detector.
- Do not use per-action active detector results for automatic deployment claims.
- Before deployment-readiness, improve the global/rest-aware active detector, rerun soft top5 with the same 20-epoch phase settings used for the deploy artifact, and add full-session rest/prep false-positive checks.
