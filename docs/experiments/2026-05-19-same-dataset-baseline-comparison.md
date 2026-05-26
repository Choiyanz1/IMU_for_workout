# Same-Dataset Baseline Comparison

Date: 2026-05-19

## Protocol

All same-dataset baselines below use the current 9-fold LOSO split over 220 `sets` streams, with light-weight sessions excluded:

- `yanz/1000`
- `thomas/thomas`
- `thomas/thomas_2`
- `kevin/kevin`

Output artifact:

- `artifacts/fixed_baseline_comparison/same_dataset_9fold_20260519.json`

Core metrics:

- Count MAE
- Rep IoU-F1@50
- Phase IoU-F1@50
- C/E Ratio MAE

Additional metrics:

- Exact Count
- Within-1 Count
- Rep Precision / Recall
- Phase accuracy / macro F1

## Main Comparison

| Method | Count MAE | Exact | Within-1 | Rep IoU-F1@50 | Phase IoU-F1@50 | C/E Ratio MAE | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Peak accel magnitude | **0.927** | 48.2% | **77.7%** | 74.8% | N/A | N/A | Count/rep-boundary baseline only; does not predict C/E transition |
| Peak 6-axis magnitude | 2.200 | 32.3% | 59.1% | 60.5% | N/A | N/A | Count/rep-boundary baseline only; over-counts heavily |
| RF active + RF C/E phase | 1.427 | 50.0% | 67.7% | 82.4% | 63.6% | 1.675 | Good phase IoU relative to peak, but count and C/E ratio are unstable |
| Causal TCN-lite + shared active detector | 3.214 | 37.7% | 53.2% | 77.8% | 67.1% | 1.238 | Learns phase reasonably, but severe over-counting makes it weak as a rep segmenter |
| BiLSTM + shared active detector | 2.077 | 43.2% | 66.8% | 78.0% | 62.7% | 1.331 | Standard sequence baseline; non-causal and still worse than CNN on structured metrics |
| Raw6 1D Causal CNN + MA25/Viterbi | 1.505 | 51.8% | 67.3% | 86.0% | **72.0%** | 0.670 | Strong boundary/phase baseline, but over-counts |
| Raw6 1D Causal CNN + `top5_p5` selective merge | 0.973 | **59.1%** | 75.5% | **87.4%** | 71.6% | 0.601 | Best structured model: strongest Rep F1 and Exact while improving Count MAE and C/E MAE |
| Raw6 CNN + action-linear count calibration | **0.759** | 57.3% | **85.9%** | same as raw boundaries | same as raw phase | same as raw phase | Best count display result, but not a segmentation/boundary solution |

## Interpretation

The current cutting model is competitive enough to keep as the main pipeline, but the comparison reveals a nuance:

- **Peak accel is a strong count-only baseline.** It reaches Count MAE 0.927 and within-1 77.7%, close to `top5_p5` on count. However, its Rep IoU-F1@50 is only 74.8%, so it is weaker as a rep-boundary method.
- **Peak is excluded from C/E phase tasks.** The peak baselines do not truly predict C/E transition points. They should not be evaluated on Phase IoU-F1@50 or C/E Ratio MAE unless a separate peak-transition heuristic is explicitly implemented and locked.
- **The CNN is much better for structured segmentation.** `top5_p5` improves Rep IoU-F1@50 to 87.4% and Phase IoU-F1@50 to 71.6%, while keeping Count MAE near the peak baseline.
- **RF phase is not enough.** It improves phase segmentation over peak, but C/E ratio MAE is 1.675, much worse than the CNN variants.
- **Generic sequence models are not enough without the current CNN/decoder design.** TCN-lite and BiLSTM learn useful phase labels, but both over-count substantially and underperform `top5_p5` on Count MAE, Rep IoU-F1@50, and C/E Ratio MAE.
- **Count calibration should be reported separately.** It has the best Count MAE and within-1 count, but it does not improve rep boundaries, phase boundaries, or C/E phase structure.

## Recommendation

For the paper or final report, use `top5_p5` as the main segmentation model and report action-linear count calibration as an optional display-only count correction.

The key claim should be:

> A simple peak detector can count many sets reasonably well, but it does not solve C/E phase segmentation. The raw6 causal CNN with selective merge provides the best balance across count, rep boundaries, phase IoU, and C/E ratio error among models that actually predict phase structure.

## Frozen Baseline Rule

Treat the same-dataset baseline rows in these artifacts as frozen:

- `artifacts/fixed_baseline_comparison/same_dataset_9fold_20260519.json`
- `artifacts/fixed_baseline_comparison/deep_same_dataset_9fold_20260519.json`

Future model development should update only our current rows unless a new baseline protocol is explicitly locked.
