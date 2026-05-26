# Action Recognition Architecture Decision

Date: 2026-05-19

## Goal

Decide how exercise action recognition should connect to the current single-IMU rep segmentation pipeline:

- independent/parallel action recognition after active detection
- rep-first action recognition after predicted reps are already segmented
- shared encoder with separate action and C/E phase heads
- action-gated hierarchical decoding

This matters because the current best decoder, `top5_p5`, uses action context for selective duration merge.

## Decision

Use **parallel action recognition from active windows** as the deployable architecture target.

The action branch should run after active/rest gating and in parallel with the C/E phase segmenter. It should produce an early action posterior and confidence lock for the active set. The rep parser and selective merge policy may then consume this action context without waiting for completed reps.

Rep-level action classification can still be used as a secondary stabilizer or audit signal, but it should not be the only source of action context.

## Recommended Pipeline

| Stage | Role | Current status |
|---|---|---|
| Raw 6-axis IMU stream | 100 Hz `ax ay az gx gy gz` input | Validated as best input among raw/PCA/9-axis probes |
| Rest-aware active detector | Suppress rest/prep segments before phase/action inference | Prototype validated on held-out set+rest snippets; needs full 9-fold/full-session integration |
| Parallel action recognizer | Estimate action from active windows or set prefix | Not integrated yet; required for full deployment claim |
| 1D causal CNN C/E model | Predict concentric/eccentric phase logits | Current validated core |
| Online rep parser | Convert phase sequence to rep candidates and C/E boundaries | Current validated core |
| Selective duration merge | Action-conditioned correction for over-segmented reps | Current best decoder, `top5_p5`, assumes action context |
| Output layer | Live count, rep boundaries, C/E ratio, action label | Partially demonstrated by replay; action label pending |

## Why Parallel Action Recognition

| Option | Verdict | Reason |
|---|---|---|
| Rep-first action recognition | Reject as primary architecture | It creates a circular dependency: `top5_p5` needs action context to merge reps, but rep-first recognition needs the reps to already be segmented. It also delays action feedback until after enough reps are complete. |
| Parallel independent action branch | Select for next implementation | It supplies action context early enough for active gating, selective merge, and live display while preserving the current C/E segmentation model. |
| Shared encoder, multi-head action + phase | Keep as a later clean-up | This is architecturally elegant, but it is a model change and should be validated against the fixed baselines. It is not required to unblock deployment-oriented action context. |
| Hierarchical action-gated decoder | Use only after action branch is validated | The decoder can consume action posterior/confidence, but action errors must not catastrophically corrupt rep segmentation. Use confidence thresholds and fallback policies. |

## Literature Grounding

The closest systems generally treat exercise recognition/classification as a parallel or upstream task relative to counting/segmentation, not as something that must wait for completed rep segments.

| Work | Relevant design signal |
|---|---|
| RecoFit, Morris et al. 2014 | The wearable pipeline explicitly frames the problem as find, recognize, and count repetitive exercises. Recognition is a first-class pipeline component, not a post-hoc rep-only label. |
| ExerSense, Ishii et al. 2020 | Real-time IMU exercise segmentation, classification, and counting are handled as a streaming pipeline. This supports action context during online inference. |
| MM-Fit, Stromback et al. 2020 | Activity segmentation, exercise recognition, and repetition counting are all benchmarked as related but distinct outputs. This supports separate evaluation of action and rep tasks. |
| Soro et al. 2019 and Prabhu et al. 2020 | CNN-style wearable models are used for recognition and repetition counting, suggesting action recognition can share input windows with counting/phase models. |
| DS-MS-TCN, Shang et al. 2024 | Micro/macro sequence labeling supports multi-label temporal modeling from IMU streams; this is consistent with future shared encoder or multi-head sequence models. |

## Integration Requirements

Before claiming full action-aware deployment readiness, validate these items:

- Train and evaluate action recognition under the same 9-fold LOSO subject-wise split.
- Report action accuracy and macro F1, both per-window and per-set after confidence locking.
- Evaluate the full pipeline with predicted action context, not ground-truth action labels.
- Measure how action errors affect Count MAE, Rep IoU-F1@50, Phase IoU-F1@50, and C/E Ratio MAE.
- Keep a fallback mode when action confidence is low, such as raw decoder or non-action-specific merge.
- Combine with the rest-aware active detector and rerun full-session or set+rest streaming replay.

## Figure Output

Updated architecture diagrams are generated by:

```bash
python scripts/new_c_pipeline/plot_current_model_architecture.py
```

Outputs:

- `artifacts/figures/current_model_architecture/current_model_architecture.png`
- `artifacts/figures/current_model_architecture/current_model_architecture.pdf`
- `artifacts/figures/current_model_architecture/current_model_architecture.svg`

## Takeaway

The current best segmentation pipeline should be described as **action-context dependent but action-recognition pending**. The next implementation should add a parallel streaming action recognizer and then rerun the full `raw6 CNN + top5_p5` pipeline using predicted action context.
