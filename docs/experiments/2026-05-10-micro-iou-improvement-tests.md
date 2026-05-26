# 2026-05-10 Micro IoU Improvement Tests

## Goal

測試目前 DS-MS-TCN 管線是否能明顯提升 `micro_f1_at_50`，重點放在：

1. action-aware micro labels
2. micro temporal loss
3. stage1-first pretraining
4. structured decoder
5. whole-session training

## Baseline

基準 checkpoint 與重評估設定：

- checkpoint: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
- re-eval output: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15_after_changes`
- setting: `micro_smoothing_window=15`

Baseline metrics:

基準結果：

| Setting | micro_f1_at_50 | micro sample macro F1 | rep F1 | rep precision | rep recall |
|---|---:|---:|---:|---:|---:|
| baseline `40ep + smooth15` | **0.4582** | 0.4293 | 0.7789 | 0.7255 | 0.8409 |

## Experiments

### 1. Action-aware micro labels

- config: `configs/micro_macro_recognition_stage3_40ep_action_phase.yaml`
- run: `artifacts/micro_macro_recognition/exp_action_phase/tcn`
- evaluation note: training uses action-aware micro classes, but headline micro metrics remain phase-collapsed for comparison with the baseline.
- 補充：訓練時使用 action-aware micro classes，但 headline micro metrics
  仍然以 phase-collapsed 方式計算，才能和 baseline 公平比較。

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.3785 |
| micro sample macro F1 | 0.4173 |
| rep F1 | 0.6331 |
| rep action accuracy | **0.9659** |
| micro semantic f1@50 | 0.2386 |

Interpretation:

解讀：

- Action-aware micro labels strongly improved rep-level action identity.
- But they hurt phase-level micro segmentation and overall rep detection relative to the current baseline.

### 2. Action-aware micro labels + micro temporal loss

- config: `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse.yaml`
- run: `artifacts/micro_macro_recognition/exp_action_phase_microtmse/tcn`

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.4217 |
| micro sample macro F1 | 0.4365 |
| rep F1 | 0.6481 |
| rep action accuracy | 0.9570 |
| micro semantic f1@50 | 0.2519 |

Interpretation:

解讀：

- A light micro TMSE term recovered part of the action-aware drop.
- It still did not beat the original `40ep + smooth15` baseline on phase-level micro IoU-F1@50.

### 3. Action-aware micro labels + micro temporal loss + stage1-first pretraining

- config: `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse_pretrain.yaml`
- run: `artifacts/micro_macro_recognition/exp_action_phase_microtmse_pretrain/tcn`
- training schedule: `10` micro-only pretrain epochs + `30` joint epochs

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.4005 |
| micro sample macro F1 | 0.4382 |
| rep F1 | 0.6502 |
| rep action accuracy | **0.9783** |
| micro semantic f1@50 | **0.2635** |

Interpretation:

解讀：

- Stage1-first pretraining further improved semantic micro labeling and rep action accuracy.
- It did not improve the target phase-level `micro_f1_at_50`; it regressed versus the micro-TMSE run.

### 4. Synthetic whole-session training

- config: `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse_pretrain_whole.yaml`
- run: `artifacts/micro_macro_recognition/exp_action_phase_microtmse_pretrain_whole/tcn`
- loader change: synthetic whole-session streams are assembled by sorting set fragments, `rest_after_set*`, and `big_rest/session*` files by timestamp and splitting sessions when gaps exceed `600s`.
- 載入邏輯：把 set fragments、`rest_after_set*`、`big_rest/session*`
  依時間排序後拼成 synthetic whole-session，當時間間隔超過 `600s`
  就切成新 session。

Raw run metrics (`sets + synthetic whole` evaluation):

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.2642 |
| rep F1 | 0.3507 |

Sets-only re-evaluation for apples-to-apples comparison:

- output: `artifacts/micro_macro_recognition/exp_action_phase_microtmse_pretrain_whole/tcn/reeval_sets_only`

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.2648 |
| micro sample macro F1 | 0.3262 |
| rep F1 | 0.3596 |

Interpretation:

解讀：

- The synthetic whole-session training path worked technically and increased training slices from `917` to `2815`.
- But this particular synthetic whole-session recipe hurt both phase-level micro IoU and rep quality badly.
- The likely issue is that concatenated rep fragments plus partial rest data are still not a good substitute for true continuous whole-session labels.

### 5. Structured decoder on the current best baseline

All decoder tests below reuse the original best checkpoint:

- source checkpoint: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
- source smoothing: `micro_smoothing_window=15`

| Decoder setting | Output dir | micro_f1_at_50 | rep F1 |
|---|---|---:|---:|
| greedy baseline | `reeval_smooth15_after_changes` | 0.4582 | 0.7789 |
| viterbi `switch=0.2 invalid=2.0 min_run=3` | `reeval_viterbi_s015_sw02_inv20_mr3` | 0.4600 | 0.7778 |
| viterbi `switch=0.5 invalid=4.0 min_run=5` | `reeval_viterbi_s015_sw05_inv40_mr5` | 0.4644 | 0.7778 |
| viterbi `switch=1.0 invalid=6.0 min_run=8` | `reeval_viterbi_s015_sw10_inv60_mr8` | 0.4698 | 0.7847 |
| viterbi `switch=1.2 invalid=8.0 min_run=12` | `reeval_viterbi_s015_sw12_inv80_mr12` | 0.4914 | 0.8369 |
| viterbi `switch=1.5 invalid=10.0 min_run=16` | `reeval_viterbi_s015_sw15_inv100_mr16` | **0.4949** | **0.8727** |

Interpretation:

解讀：

- Structured decoding was the strongest improvement among the tested ideas for the target metric.
- The gain was real but modest on `micro_f1_at_50`: `0.4582 -> 0.4949`.
- It also improved rep-level precision/recall/F1 substantially on the same checkpoint.

### 6. Structured decoder on the best new training variant

- source checkpoint: `artifacts/micro_macro_recognition/exp_action_phase_microtmse/tcn`
- output: `artifacts/micro_macro_recognition/exp_action_phase_microtmse/tcn/reeval_viterbi_sw10_inv60_mr8`
- decoder: viterbi `switch=1.0 invalid=6.0 min_run=8`

| Metric | Raw | With decoder |
|---|---:|---:|
| micro_f1_at_50 | 0.4217 | 0.4364 |
| rep F1 | 0.6481 | 0.6597 |

Interpretation:

解讀：

- Structured decoding also helped the best new training variant.
- But even after decoding, it still did not exceed the original baseline.

### 7. Dual-head Stage 1: phase head + auxiliary action-aware head

The next test used a shared Stage 1 temporal backbone with two parallel heads:

接著測試 shared Stage 1 temporal backbone 搭配兩個平行 head：

- phase head: `other / concentric / eccentric`
- semantic head: action-aware labels such as `db_rdl::concentric`

semantic head 只作為輔助 loss，不直接取代 phase head。

#### 7.1 Dual-head auxiliary only

- config: `configs/micro_macro_recognition_stage3_40ep_dual_head.yaml`
- run: `artifacts/micro_macro_recognition/exp_dual_head/tcn`
- semantic branch used only as auxiliary supervision; macro stages still consume
- semantic branch 只作為輔助 supervision；macro stages 仍然只吃 phase head 的輸出。

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.4813 |
| micro sample macro F1 | **0.4688** |
| rep F1 | 0.6360 |
| rep action accuracy | 0.8556 |

Interpretation:

解讀：

- This beat the original `40ep + smooth15` baseline on the target metric:
  - baseline: `0.4582`
  - dual-head raw: `0.4813`
- So a phase head plus auxiliary semantic supervision looks better than either:
  - pure 3-class phase head alone, or
  - replacing the phase head with full action-aware micro labels.

#### 7.2 Dual-head with semantic-to-macro fusion

- config: `configs/micro_macro_recognition_stage3_40ep_dual_head_macro.yaml`
- run: `artifacts/micro_macro_recognition/exp_dual_head_macro/tcn`
- semantic branch is concatenated into the macro stage input.
- semantic branch 的 probability 會一起 concatenate 進 macro stage 輸入。

| Metric | Value |
|---|---:|
| micro_f1_at_50 | 0.4209 |
| micro sample macro F1 | 0.4272 |
| rep F1 | 0.7004 |
| rep action accuracy | 0.9072 |

Interpretation:

解讀：

- Feeding semantic probabilities directly into the macro stage hurt the target
  phase-level micro IoU compared with the simpler auxiliary-only dual-head run.
- So the better current reading is:
  - semantic supervision helps the shared backbone,
  - but forcing semantic probs into the macro stack is not yet beneficial for
    the target metric.

#### 7.3 Dual-head auxiliary + structured decoder

- source run: `artifacts/micro_macro_recognition/exp_dual_head/tcn`
- re-eval: `artifacts/micro_macro_recognition/exp_dual_head/tcn/reeval_viterbi_sw15_inv100_mr16`
- decoder: viterbi `switch=1.5 invalid=10.0 min_run=16`

| Setting | micro_f1_at_50 | rep F1 |
|---|---:|---:|
| dual-head raw | 0.4813 | 0.6360 |
| dual-head + structured decoder | **0.5622** | 0.7172 |

Comparison against the best previous baseline decoder result:

| Setting | micro_f1_at_50 |
|---|---:|
| baseline + best decoder | 0.4949 |
| dual-head auxiliary + best decoder | **0.5622** |

Interpretation:

解讀：

- This is the strongest `micro_f1_at_50` result reached so far in the project.
- The dual-head architecture plus constrained decoding gave a larger gain than
  the earlier action-aware-label replacement experiments.
- It supports the current hypothesis that:
  - a dedicated phase head should remain the main decoding path,
  - while action-aware supervision is better used as an auxiliary branch than
    as a full replacement.

## Overall Conclusion

這輪測試的整體結論：

1. **Structured decoder** gave the best direct improvement to the target metric on the original baseline.
2. **Dual-head Stage 1** was the best architectural change tested so far.
3. **Micro temporal loss** helped recover some of the damage introduced by action-aware micro labels.
4. **Action-aware micro labels as a full replacement** mainly improved action discrimination, not the target phase-level micro IoU.
5. **Stage1-first pretraining** improved semantic micro/action behavior, but not the target `micro_f1_at_50` in the tested formulation.
6. **Synthetic whole-session training** hurt substantially and is not recommended in its current form.

Best metric reached in this test round:

- `micro_f1_at_50 = 0.5622`
- artifact: `artifacts/micro_macro_recognition/exp_dual_head/tcn/reeval_viterbi_sw15_inv100_mr16`

This is meaningfully better than both:

這比下面兩個版本都更好：

- the original best practical baseline (`0.4582`), and
- the original best decoder-tuned baseline (`0.4949`).

But it is still below the target `0.7`.

但距離目標 `0.7` 還有一段距離。

## Recommended Next Steps

If the next goal is specifically to chase `micro_f1_at_50 >= 0.7`, the most promising next directions are:
如果下一步目標仍然是把 `micro_f1_at_50` 推到 `0.7`，目前最值得做的是：

1. keep the **dual-head auxiliary** architecture as the new best structural direction;
2. continue decoder/phase-grammar tuning on top of that dual-head checkpoint;
3. test micro-specific temporal/boundary losses with the dual-head setup;
4. acquire or label true continuous whole-session streams instead of relying on synthetic concatenation;
5. if deployment is Kevin-specific, separate personalized evaluation from strict LOSO reporting before further optimization.
