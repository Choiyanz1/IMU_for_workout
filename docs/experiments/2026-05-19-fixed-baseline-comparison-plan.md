# Fixed Baseline Comparison Plan

Date: 2026-05-19

## Decision

The current cutting model is **strong enough for internal evidence and a related-work positioning draft**, but **not sufficient by itself for a paper-quality comparison**.

The reason is simple: the current result shows our pipeline is useful, but reviewers will expect at least a few common methods evaluated on the **same dataset, same subject-wise split, and same metrics**. Cross-paper literature numbers are helpful context, but they cannot replace same-dataset baselines because sensor placement, exercise set, subject split, and count/segmentation definitions differ.

## What Is Already Strong

Current best structured result:

| Method | Split | Count MAE | Exact | Within-1 | Rep F1 | Phase IoU-F1@50 | C/E MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw6 1D Causal CNN + `top5_p5` | 9-fold LOSO, 220 streams | 0.973 | 59.1% | 75.5% | 87.4% | 71.6% | 0.601 |

This is enough to claim:

- The model emits useful rep boundaries, not only a final count.
- The decoder improves over the raw CNN on Rep F1, Exact Count, Within-1 Count, Count MAE, over-count rate, and C/E ratio MAE.
- The output is richer than most count-only wearable exercise systems.

## What Is Not Yet Enough

It is not yet enough to claim broad model superiority because:

- The strongest current result is mostly compared against our own raw decoder and rejected variants.
- Literature rows are not apples-to-apples.
- Existing internal RF/BiLSTM/1D CNN/DS-MS-TCN results are useful but partly legacy and not fully aligned with the current 9-fold raw6 protocol.
- The action-recognition branch is not integrated into the current `top5_p5` pipeline, so action-recognition numbers should not be borrowed from old DS-MS-TCN experiments.

## Recommended Same-Dataset Baselines To Freeze

Minimum set for a credible paper table:

| Priority | Baseline | Why include it | Freeze rule |
|---:|---|---|---|
| 1 | Peak / threshold counting | Classical count baseline reviewers expect | Run once under 9-fold LOSO, save predictions and metrics |
| 2 | DTW / template matching | Closest to ExerSense-style deployable segmentation | Run once under 9-fold LOSO, save predictions and metrics |
| 3 | Sliding-window Random Forest | Lightweight wearable-HAR baseline | Run once under same input/split, save metrics |
| 4 | BiLSTM or DeepConvLSTM | Standard deep wearable sequence baseline | Run once under same input/split, save metrics |
| 5 | Causal TCN / DS-MS-TCN-style model | Closest common temporal segmentation architecture | Run once under same input/split, save metrics |

Optional if time allows:

| Baseline | Why optional |
|---|---|
| Transformer/HART-like sensor model | Modern sensor-sequence baseline, but more engineering/time |
| Non-causal CNN/TCN | Useful ablation to show cost of causality, but less deployment-realistic |
| Count-only calibration model | Useful for count table, but should be separated from boundary/phase claims |

## Fixed Baseline Workflow

Use this rule to avoid moving goalposts:

1. Choose the baseline set and protocol.
2. Run each baseline once under the locked protocol.
3. Save baseline predictions and metrics under a dated artifact folder.
4. Copy final numbers into `docs/experiments/2026-05-19-fixed-baseline-registry.json` or a new dated registry.
5. Do not rerun or tune baseline rows unless explicitly creating a new locked protocol.
6. Continue improving only the `Current ours` rows by rerunning our model and regenerating the table.

Current support files:

- Fixed registry: `docs/experiments/2026-05-19-fixed-baseline-registry.json`
- Render script: `scripts/new_c_pipeline/render_fixed_comparison_table.py`
- Rendered output: `docs/experiments/2026-05-19-fixed-baseline-comparison-table.md`

Regenerate the comparison table after rerunning our method:

```bash
python scripts/new_c_pipeline/render_fixed_comparison_table.py
```

## Paper Positioning Recommendation

The paper should not be framed as "we beat every exercise counting method." A stronger and safer claim is:

> Compared with count-only wearable exercise systems, our single-IMU causal pipeline predicts a structured repetition trace: live count, rep boundaries, concentric/eccentric phase segmentation, and per-rep C/E balance. Same-dataset baselines are used to separate the value of the causal CNN and selective rep merge from classical counting, RF, recurrent, and TCN alternatives.

## Next Experimental Step

The next best step is to create a **formal frozen same-dataset baseline run** using the minimum set above. After that, model development can focus only on improving our current rows while the baseline rows remain fixed.
