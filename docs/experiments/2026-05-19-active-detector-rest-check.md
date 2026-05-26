# Active Detector Rest-Period Check

Date: 2026-05-19

## Goal

Check whether the current per-action active detector stays inactive during rest periods, because false active segments can send rest/prep samples into the C/E phase model and inflate rep counts.

## Method

- Script: `scripts/new_c_pipeline/plot_active_detector_rest_examples.py`
- Held-out subject: `yushuan`
- Streams: `set0 + 20s rest_after_set0` for `db_rdl`, `db_weighted_crunch`, `db_biceps_curl`, and `db_shoulder_press`
- Current detector: per-action `RandomForestClassifier`, 1.0s windows, 0.1s stride, threshold 0.5
- Training variants:
  - Current behavior: train on cropped `sets` only
  - Rest-aware probe: append rest-after-set negatives to train streams before training the active detector

## Commands

```bash
python scripts\new_c_pipeline\plot_active_detector_rest_examples.py --subject yushuan --actions db_rdl,db_weighted_crunch,db_biceps_curl,db_shoulder_press --max-per-action 1 --append-rest-sec 20 --output-dir artifacts\figures\active_detector_rest_examples
python scripts\new_c_pipeline\plot_active_detector_rest_examples.py --subject yushuan --actions db_rdl,db_weighted_crunch,db_biceps_curl,db_shoulder_press --max-per-action 1 --append-rest-sec 20 --train-rest-sec 20 --threshold 0.6 --output-dir artifacts\figures\active_detector_rest_aware_thr06_examples
```

## Results

| Active detector setup | Threshold | Active F1 | Precision | Recall | False-active rest | Missed active |
|-----------------------|----------:|----------:|----------:|-------:|------------------:|--------------:|
| Current set-only training | 0.5 | 0.819 | 0.694 | 1.000 | 19.95s / 20s | 0.00s |
| Rest-aware training, 20s rest | 0.5 | 0.963 | 0.918 | 1.000 | 3.84s / 20s | 0.00s |
| Rest-aware training, 20s rest | 0.6 | **0.973** | 0.949 | 0.999 | 2.62s / 20s | 0.08s |
| Rest-aware training, 20s rest | 0.7 | 0.972 | **0.959** | 0.986 | **2.04s / 20s** | 0.73s |

## Interpretation

- The current active detector is not a valid rest suppressor when trained only on cropped set streams; it marks almost all appended rest as active.
- The reason is data construction, not the CNN: cropped `sets` contain no rest labels, so the RF active detector sees little to no negative class for each action.
- Adding rest-after-set negatives greatly improves rest suppression while preserving active recall, especially around threshold 0.6.
- Threshold 0.7 further reduces false-active rest but begins to miss more active samples on RDL and weighted crunch.

## Decision

Do not treat the current active detector as deployment-ready for full-session replay. The next deployment-readiness check should retrain or refactor active detection with rest negatives, then rerun streaming-style inference before changing the phase CNN or count decoder.
