# Action-Conditioned Decoder Policy Search

Date: 2026-05-20

## Goal

Test whether a more complete action-conditioned decoder can improve over the current universal decoder and the narrower `top5_p5` selective duration merge.

The model is unchanged:

- Global raw6 1D causal CNN phase model
- Same 9-fold LOSO split and excluded light-weight sessions
- Only decoder parameters change per action

## Method

New script:

```bash
python scripts/new_c_pipeline/action_conditioned_decoder_9fold.py
```

Policy search is train-fold only. For each held-out subject, the script:

1. Trains the global raw6 CNN on the training subjects.
2. Predicts raw per-sample C/E probabilities before MA/Viterbi decoding.
3. Selects a decoder policy per action using train-fold streams only.
4. Applies selected action policies to the held-out subject.

Reduced formal grid:

| Parameter | Values |
|---|---|
| MA window | `25`, `35` |
| Viterbi penalty | `0.3`, `0.5`, `0.7` |
| Min phase samples | `3` |
| Duration merge | none or train-fold per-action `p5` |
| Max merge gap | 50 samples |

Policy selection gate:

- Rep F1 cannot drop by more than 0.02 on train-fold policy evaluation.
- Exact count cannot drop on train-fold policy evaluation.
- Count MAE must improve by at least 0.05.
- If no candidate passes, use baseline `MA25 + Viterbi 0.3 + no merge` for that action.

## Failed Wider/Sampled Probe

The first version searched a larger grid (`MA 15/25/35`, Viterbi `0.3/0.5/0.7`, min phase `3/5`, merge none/p5/p10). Evaluating the full train fold was too slow. A sampled train-fold version completed, but overfit policy selection:

| Probe | Rep F1 | Exact | Count MAE | C/E MAE | Decision |
|---|---:|---:|---:|---:|---|
| Baseline fast sampled | 0.854 | 0.550 | 1.47 | 0.922 | Reference |
| Action-conditioned sampled | 0.838 | 0.491 | 1.47 | 0.838 | Rejected; Exact and Rep F1 worse |

## Reduced Grid Results

Fast setting: hidden=32, epochs=5, all train streams used for policy selection.

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over | Under | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MA25/Viterbi | 0.854 | 0.545 | 0.718 | 1.464 | +1.21 | 0.405 | 0.050 | 0.703 | 0.886 |
| Action-conditioned reduced | **0.859** | **0.582** | **0.755** | **1.132** | **+0.12** | **0.259** | 0.159 | 0.701 | **0.812** |

Formal setting: hidden=64, epochs=20, CUDA GPU, all train streams used for policy selection.

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over | Under | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MA25/Viterbi | 0.860 | 0.518 | 0.691 | 1.523 | +1.40 | 0.445 | 0.036 | 0.721 | 0.642 |
| Action-conditioned reduced | **0.869** | **0.545** | **0.709** | **1.341** | **+1.17** | **0.400** | 0.055 | 0.718 | **0.606** |

## Interpretation

The reduced action-conditioned decoder does improve over the universal raw decoder in the formal run:

- Rep F1 improves `0.860 -> 0.869`
- Exact improves `0.518 -> 0.545`
- Count MAE improves `1.523 -> 1.341`
- C/E MAE improves `0.642 -> 0.606`
- Over-rate drops `0.445 -> 0.400`

However, it does **not** beat the existing `top5_p5` selective merge pipeline:

| Decoder | Rep F1 | Exact | Count MAE | C/E MAE | Notes |
|---|---:|---:|---:|---:|---|
| `top5_p5` selective merge | 0.874 | 0.591 | 0.973 | 0.601 | Current best balanced decoder |
| Action-conditioned reduced | 0.869 | 0.545 | 1.341 | 0.606 | Better than raw, worse than `top5_p5` |

The broader conclusion is that action-conditioned decoding helps, but naive per-action policy search is not enough. The strongest improvement still comes from the hand-selected over-count-prone action set in `top5_p5`.

## Decision

Do not replace `top5_p5` with this action-conditioned decoder.

Keep this script and artifact as evidence that:

- action-conditioned decoder parameters are useful compared with a universal decoder
- train-fold policy search can be unstable if the grid is too broad or validation set too small
- the current best strategy remains selective merge on known over-segmented actions

Next better direction: constrain policy search to the `top5_p5` action set and search only the merge/anti-fragmentation parameters, rather than letting every action change smoothing/Viterbi behavior.

## Artifacts

- Fast reduced all-train: `artifacts/cnn_variant_comparison/action_conditioned_decoder_9fold_fast_reduced_alltrain.json`
- Formal reduced all-train: `artifacts/cnn_variant_comparison/action_conditioned_decoder_9fold_gpu_h64e20_reduced_alltrain.json`
- Failed sampled probe: `artifacts/cnn_variant_comparison/action_conditioned_decoder_9fold_fast_guarded.json`
