# Model Specification

## Current Rep-Boundary Model

目前做 held-out rep cutting 比較時，repo 內最實際、最穩定的模型家族已經不是
DS-MS-TCN，而是 **causal RF + boundary refiner**。

- Main benchmark script:
  - `scripts/benchmark_per_action_rf_refiner.py`
- RF phase model:
  - `scripts/evaluate_causal_rf.py`
- Boundary refiner utilities:
  - `scripts/train_rf_boundary_refiner.py`

Current preferred evaluation layout:

- subject-wise outer held-out evaluation
- train-only inner selection
- causal RF phase detector
- optional boundary refiner on top of RF coarse reps
- rep-count consistency treated as a first-class constraint

Current default stable configuration:

- IMU channels: full 6-axis default `ax, ay, az, gx, gy, gz`
- RF trailing window: `50`
- refiner edge window: `20`
- train stride: `10`
- RF hyperparameters:
  - `n_estimators = 50`
  - `max_depth = 15`
  - `max_samples = 0.7`

Current practical reading:

- full default 6-axis input is the safest baseline
- free modality search can improve some folds, but often causes under-segmentation
  or missing reps
- therefore modality selection is now treated as a guarded ablation, not a free
  default optimization
- action recognition is done at the per-rep level using a rep-complete classifier,
  not from a short IMU prefix

Current guarded held-out reference result:

- output:
  - `artifacts/baseline_comparison/modality_count_guardrail_yoru_v1`
- held-out subject: `yoru`
- tuned outcome after count-aware guardrails:
  - Precision `0.8648`
  - Recall `0.8869`
  - Rep F1 `0.8757`
  - `micro_f1@50 = 0.8126`
  - exact-count streams: `19 / 25`
  - over-segmented streams: `6 / 25`
  - under-segmented streams: `0 / 25`
- important reading:
  - all actions fell back to `baseline_reference`
  - on this fold, guarded modality search found no better option than the
    default 6-axis RF setting

Current rep-cutting stack:

```text
Raw IMU (6-axis)
   |
   v
+------------------------------+
| Train-fold z-score           |
+------------------------------+
   |
   v
+------------------------------+
| Causal RF phase detector     |
| other / concentric / ecc     |
+------------------------------+
   |
   | per-sample phase probs
   v
+------------------------------+
| Phase decoder + rep pairing  |
+------------------------------+
   |
   | coarse reps
   v
+------------------------------+
| Boundary refiner             |
| start / transition / end     |
+------------------------------+
   |
   v
Refined reps
   |
   v
+--------------------------------------+
| Per-rep feature extraction          |
| -> Rep-Complete Action Classifier   |
| -> Hybrid (macro + classifier)      |
+--------------------------------------+
   |
   v
Stable action label + rep count
```

Latest guarded held-out `yoru` reference architecture in more explicit form:

```text
Input stream
  sensor_ts + ax ay az gx gy gz
          |
          v
+-------------------------------+
| Train-fold z-score stats      |
| apply same stats at test time |
+-------------------------------+
          |
          v
+----------------------------------------------+
| Per-action causal RF phase detector          |
| window_size = 50, stride = 10               |
| predicts: other / concentric / eccentric    |
+----------------------------------------------+
          |
          | dense phase probabilities
          v
+----------------------------------------------+
| Fixed phase decoder / rep pairing           |
| concentric -> eccentric -> coarse rep       |
+----------------------------------------------+
          |
          | coarse rep boundaries
          v
+----------------------------------------------+
| Boundary refiner                            |
| edge_window = 20                            |
| predicts small corrections for:             |
|   - rep start                               |
|   - rep transition                          |
|   - rep end                                 |
+----------------------------------------------+
          |
          v
+----------------------------------------------+
| Refined reps                                |
+----------------------------------------------+
          |
          | per completed rep
          v
+----------------------------------------------+
| Per-rep feature extraction                  |
| -> Rep-Complete Action Classifier           |
| -> Hybrid (macro + classifier confidence)   |
+----------------------------------------------+
          |
          v
Stable action label + rep count + phase split
```

Important implementation note for the latest guarded held-out run:

- benchmark result aggregation is per held-out subject
- model fitting is per action
- guarded modality search fell back to the default full 6-axis input on `yoru`
- so the latest strong held-out result is effectively the default full-modality
  RF + boundary-refiner design, not a narrower modality-specialized variant
- action recognition is not part of this benchmark; it is handled upstream by a
  separate per-rep rep-complete classifier that runs after a rep is fully cut

## Current Modality Selection Policy

The current benchmark code supports a stricter modality-only selection mode, but
it is intentionally conservative.

When `--modality-only-search` is enabled:

- single-modality candidates are excluded by default
- candidate windows stay fixed at the baseline defaults unless explicitly
  overridden
- selection defaults to `rep_f1`, not `micro_f1_at_50`
- candidate ranking also considers:
  - exact-count consistency
  - mean absolute count difference
  - recall
  - under-segmentation
- a `baseline_reference` candidate is evaluated using the default 6-axis modality
  with fixed window settings and no duration prior
- a narrower modality subset is only kept if it beats that baseline-like
  reference without hurting recall or count consistency beyond the configured
  guardrails

This means the current architecture decision is:

```text
Prefer full 6-axis default
unless a narrower modality subset is clearly better
and does not break rep count behavior
```

As of the current guarded `yoru` held-out test, all actions fell back to
`baseline_reference`, which means no searched modality subset beat the default
6-axis configuration once count consistency was enforced.

## Current Deployment Reading

For offline held-out evaluation, the RF + boundary-refiner family is currently
the strongest practical rep-cutting branch in this repo.

For Luckfox Pico Zero style real-time deployment, however, the current state is
still **not deployment-ready as a full mainline rep counter**.

Why:

- the current preferred RF benchmark is implemented in Python / sklearn
- the repo does not yet export the RF + refiner path to ONNX / RKNN / a small
  C++ runtime
- the current benchmark logic is per-action and effectively assumes the action
  family is already known before rep refinement
- no board-style streaming latency measurement has yet been recorded for this RF
  path on Luckfox-class hardware

Practical implication:

- **accuracy** on at least one held-out fold is now promising enough to justify
  board interest
- **size / RAM / latency** for the RF path on Luckfox are still unverified in a
  deployable runtime
- therefore the RF path should currently be treated as an offline reference
  model, not a confirmed real-time Luckfox deployment artifact

## Current Micro/Macro Model

- Architecture: DS-MS-TCN in `models/ds_ms_tcn.py`.
- Stage 1 predicts micro labels: `other`, `concentric`, `eccentric`.
- Stages 2 to 4 refine macro/action labels from micro probabilities.
- Best current held-out 3-action checkpoint:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn/models/ds_ms_tcn.pt`
- Best current config family in active use:
  - 100 Hz
  - causal
  - 6 layers / 64 filters
  - 20 s slices
  - dual-head Stage 1
  - `semantic_alpha = 0.5`
  - causal micro smoothing window `15`
  - `viterbi` micro decoder

Current deployed-by-default DS-MS-TCN layout in this repo:

```text
Raw IMU
   |
   v
+-------------------------+
| Stage 1 Micro TCN       |
| other / con / ecc       |
+-------------------------+
   |
   | micro probabilities
   v
+-------------------------+
| Stage 2 Macro TCN       |
| other + action_type     |
+-------------------------+
   |
   v
+-------------------------+
| Stage 3/4 Refinement    |
+-------------------------+
   |
   +--> macro labels
   +--> rep decoder: con -> ecc -> rep
```

重點是：這個 repo 目前的 Stage 1 本質上是 **phase head**，不是
action-aware micro head。

## Why The Current 3-Class Micro Head Is Incomplete

目前 Stage 1 刻意設計得很簡單：

```text
micro classes = {other, concentric, eccentric}
```

這有助於現在的 rep decoder，因為它只需要處理一種固定 phase 順序：

```text
concentric -> eccentric -> rep
```

但這也會把語意上不同的動作壓成同一個 phase 類別：

```text
db_rdl::concentric
db_weighted_crunch::concentric
one_arm_db_row::concentric
        |
        v
all map to "concentric"
```

也就是說，Stage 1 可能很會找「是不是 active phase」，但仍然沒有學到
這些 phase 內部的 action-specific 結構。

## Why Full Action-Aware Micro Labels Are Attractive

更符合語意的 Stage 1 會長這樣：

```text
Raw IMU
   |
   v
+--------------------------------------+
| Action-aware micro head              |
| other                                |
| db_rdl::con / db_rdl::ecc            |
| crunch::con / crunch::ecc            |
| row::con / row::ecc                  |
| bench::con / bench::ecc              |
+--------------------------------------+
```

這比較符合領域直覺，因為不同 exercise 的向心/離心訊號確實不同。

最近的實驗也驗證了這件事的優缺點：

- 優點：rep-level action identity 大幅提升；
- 缺點：如果 loss 和 decoder 沒一起升級，phase-level
  `micro_f1_at_50` 反而會下降。

所以 action-aware micro labels 很像是 **長期正確方向**，但不是目前
3-class phase head 的直接替代品。

## Does A Phase-First Design Cause Error Propagation?

如果做成硬式串接，答案是會。

Bad cascade:

```text
Raw IMU
  -> predict con/ecc
  -> hard argmax
  -> use that hard phase label to decide action-aware micro class
```

在這種設計下，前面 phase 判錯，後面 action-aware 判斷就會直接被污染。

這不是目前推薦的下一步設計。

## Preferred Next Architecture

比較安全的下一步，是 shared backbone 搭配平行雙頭：

```text
Raw IMU
   |
   v
+-----------------------------+
| Shared Temporal Backbone    |
+-----------------------------+
        |                |
        |                |
        v                v
+---------------+   +----------------------+
| Phase Head    |   | Action-aware Head    |
| other/con/ecc |   | rdl::con, ...        |
+---------------+   +----------------------+
        |                |
        |                +--> auxiliary loss
        |
        +--> structured decoder / rep pairing
```

這種做法比較安全，原因是：

- phase head 可以繼續專注在穩定的 rep 邊界；
- action-aware head 可以學不同 exercise 的 phase 語意差異；
- action-aware branch 不需要先吃一個 hard phase decision。

也就是說，這不是：

```text
phase head decides first
then action head blindly follows
```

而是：

```text
shared features
-> one head for stable phase boundaries
-> one head for richer action-aware micro semantics
```

## Current Experimental Reading

目前實驗結果比較支持下面這個判斷：

- 單純 3-class phase micro labels 目前仍然是穩定 baseline；
- action-aware micro labels 很可能真的學到了 exercise-specific 結構，
  因為 action identity 提升很明顯；
- 但真正缺的不是只有 label schema，還包括搭配的 loss、decoder 與訓練方式。

因此目前比較合理的工作假設是：

1. 保留穩定的 3-class phase head 負責 rep boundary decoding；
2. 把 action-aware micro supervision 做成平行或輔助分支，而不是直接取代 phase head。

## Current Config Direction

- `configs/micro_macro_recognition.yaml` now targets board-style settings: 100 Hz, 20 s slices, 6 TCN layers, causal inference.
- No complete checkpoint was found for this newer board-style config during the 2026-05-08 review.

## Online Inference Behavior

- `OnlineDSMSTCNPredictor` recomputes the rolling buffer on every sample.
- For the reviewed older 9-layer checkpoint, the default total receptive-field buffer is 4089 samples, about 36.8 s at 111 Hz.
- The streaming evaluator now writes sample accuracies, label counts, throughput, and real-time factor into `streaming_summary.json`.
- As of the 2026-05-13 replay audit, `evaluation/streaming_micro_macro.py --method fast` also applies:
  - the checkpoint's final macro stage via `model.final_macro_logits(...)`
  - the configured micro decoder (`greedy` or `viterbi`) before online rep pairing
- A 2026-05-08 step replay on `kevin/db_weighted_crunch/set0` showed the default 4089-sample buffer is slower than real time on CPU. Runtime buffers of 256 or 128 samples exceeded real time on the same replay, with similar macro accuracy and slightly lower micro accuracy.
- This implementation is useful for validating online behavior, but it is not optimized for a constrained board because it still recomputes the rolling buffer each sample.

## Hybrid Action Classifier (Preferred Strategy)

After comparing multiple action-identification methods on held-out `kevin` data (2026-05-12), the **confidence-based hybrid** is now the default for action assignment:

```text
Online Macro Aggregation  ----->  confidence check  ----->  Rep-complete Classifier
        |                               |                           |
        v                               |                           v
   high conf >= 0.7  ------------------>                   low conf fallback
```

- **Macro stage** is strong on `db_bench_press` (100% accuracy) but poor on `one_arm_db_row` in some sessions.
- **Rep-complete classifier** (logistic regression on rich per-rep features) is perfect on `one_arm_db_row` and `db_rdl` but weaker on some `db_bench_press` sessions.
- **Hybrid** combines both: uses macro when confident (`>= 0.7`), otherwise falls back to the classifier.

Result on 114 IoU-matched reps across 17 kevin test sets:

| Method | Accuracy |
|---|---|
| Online macro aggregation | 86.84% |
| Rep-complete classifier | 92.11% |
| **Confidence hybrid** | **96.49%** |

Implementation:
- `train/hybrid_action_classifier.HybridActionClassifier` trains a lightweight sklearn model (logreg + RF, pick best by training macro F1) on non-test-subject reps.
- `evaluation/streaming_micro_macro.py` now defaults `--hybrid-action` to `True`.
- At runtime the decoder still emits macro-based labels first; the hybrid classifier post-processes each completed rep before writing `online_rep_detections.csv`.

This does **not** improve rep detection itself (precision/recall/F1 remain unchanged); it only improves the *action identity* of already-detected reps.

## Known Model Issues

- Rep detection is recall-limited on held-out `kevin` data.
- Macro/action sample prediction is weak in the full held-out evaluation even when one representative replay has high macro accuracy.
- Phase ordering errors appear frequently in pairing diagnostics, especially unexpected eccentric-before-concentric runs and missing eccentric-after-concentric runs.
- In the latest 3-action all-data replay audit (`20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn`), the main failure is no longer action identity but **action-dependent phase collapse**:
  - `db_bench_press`: usually detected, but often over-segmented
  - `db_rdl`: many streams collapse to all-`eccentric`, producing zero reps
  - `one_arm_db_row`: strong eccentric bias causes severe under-segmentation
- The larger 7-action and 8-action all-data audits show the same core bottleneck at larger scale:
  - adding action classes lowers replay rep F1 much faster than it lowers matched-rep action identity
  - some actions collapse to almost single-phase predictions (`all eccentric` or `all concentric`), which prevents any valid rep pairing
  - representative new failures include `db_biceps_curl`, `db_squat`, and `db_triceps_curl`
- A minimal inference-side probe was added as `semantic_phase_fusion_weight`, which blends the semantic head back into the 3-class phase probabilities before decoding.
- A first probe with `semantic_phase_fusion_weight = 0.25` on representative 3-action failures (`db_rdl`, `one_arm_db_row`) did not recover rep formation, so the current reading is:
  - inference-only semantic fusion is too weak by itself
  - a stronger **training-time** phase-representation change is more likely needed
- The deploy tooling does not yet export DS-MS-TCN to ONNX or another board runtime format.

## Improvement Direction

- Train and evaluate the current 100 Hz, 6-layer, 20 s causal config to completion before comparing it with the old 9-layer checkpoint.
- Tune rep post-processing on validation subjects, not on the held-out subject used for final reporting.
- Reduce online buffer size through the 6-layer model, smaller kernels, or a stateful/streaming TCN implementation. Re-test with `--method step` because `--method fast` does not measure per-sample runtime.
- Add a DS-MS-TCN export path or switch the board candidate to an already exportable student model if only action classification is required.
