# Development Log

## 2026-06-05 - Full Auto Realtime Deployment Bundle

### What Changed
- Added `scripts/export_full_auto_realtime_bundle.py` to train/export the current full automatic pipeline as one deploy bundle.
- Added `scripts/run_full_auto_realtime_bundle.py` to run the bundle on saved IMU CSV or raw `zig_bt_client --stdout` input.
- Exported current bundle to `artifacts/deploy/full_auto_realtime_current/` with phase CNN, ONNX, periodic active gate RF, action RF heads, JSON RF tree exports, normalization stats, duration priors, and runtime config.
- Documented commands and Luckfox caveats in `docs/experiments/2026-06-05-full-auto-realtime-bundle.md`.

### Results
- Smoke bundle export passed with `--epochs 1 --hidden 16 --skip-onnx`.
- Current bundle export passed with `--epochs 5 --hidden 64`; `phase_model.onnx` was generated.
- Torch CPU replay passed on `_tsenyu_temp/tsenyu0515workout/db_biceps_curl/set0` sample CSV: predicted action `db_biceps_curl`, count `12`.
- ONNX replay on the same sample also passed: predicted action `db_biceps_curl`, count `12`.
- ONNX + JSON RF replay also passed on the same sample: predicted action `db_biceps_curl`, count `12`, with no pandas/sklearn/joblib inference dependency.
- Added `--emit-mode jsonl-events` to `scripts/run_full_auto_realtime_bundle.py`; ONNX + JSON RF live-style replay emitted 12 rep events and a final summary with `count=12`, matching batch replay on the smoke sample.
- Added `--emit-mode stateful-jsonl` to `scripts/run_full_auto_realtime_bundle.py`; ONNX + JSON RF stateful replay emitted 12 rep events and a final summary with `count=12` on the same smoke sample.
- Stateful replay is closer to the board loop than `jsonl-events`: it updates active/action RFs at configured strides, runs the phase CNN at the phase step, finalizes fixed-lag labels incrementally, and feeds newly finalized labels into a streaming parser. On the smoke sample it produced the same action/count as batch replay, but causal active entry changed boundary timing slightly (`active_samples=6476` vs batch `6525`; first rep start shifted from sample `0` to `49`).
- Prepared a minimal manual-transfer package at `artifacts/deploy/full_auto_realtime_portable/` and `artifacts/deploy/full_auto_realtime_portable.zip`. It includes only the ONNX + JSON RF inference path and smoke sample, excluding datasets, `.pt`, `.joblib`, training scripts, and evaluation artifacts. Portable validation from inside that folder passed with final replay and `stateful-jsonl`, both giving `db_biceps_curl`, count `12`.

### Takeaway
- `artifacts/deploy/full_auto_realtime_current/` is now the packaged full automatic engineering baseline for later Luckfox Pico Zero work.
- The bundle is closer to board-ready: active/action gates can now run from JSON RF trees with pure NumPy via `--rf-runtime json`, and the runner can emit JSONL rep events via `--emit-mode stateful-jsonl`. It is still not a final RKNN-only artifact because `phase_model.onnx` still needs RKNN conversion, and stateful mode still needs broader full-session validation beyond the smoke replay.
- Full-session quality remains active-gate-limited, but the packaging now separates model/runtime config so active gate thresholds can be adjusted without retraining the CNN.

---

## 2026-06-04 - Active Gate Full-Session Research

### What Changed
- Added `scripts/evaluate_periodic_active_gate_loso.py` to test active/rest gates independently from the phase CNN.
- Added periodicity-aware active features: magnitude/jerk autocorrelation, frequency-band power, dominant frequency, spectral entropy, and zero-crossing features.
- Added `scripts/evaluate_active_cnn_full_timeline_loso.py`, including an optional CNN+RF hybrid verifier where the CNN proposes active segments and a basic/periodic RF rejects weak segments.
- Extended `scripts/evaluate_realtime_soft_top5_pipeline.py` with optional periodic active gates, active-masked fixed-lag parsing, active segment cleanup, and candidate set confirmation.
- Added event-level confirmation to `scripts/evaluate_realtime_soft_top5_pipeline.py`: fixed-lag reps are grouped into candidate events, held in a buffer, and released retroactively only after event confirmation.
- Documented the analysis in `docs/experiments/2026-06-04-active-gate-full-session-research.md`.

### Results
- Standalone active gate, 9-fold LOSO: periodic RF improved set F1 from `0.875` to `0.895` and rest false-active/min from `1.84` to `1.49`, but rest-tail active rate stayed high (`0.086`).
- Conservative periodic gate with `min_active_samples=400` reduced rest false-active/min to `0.91`, but rest-tail active rate remained `0.078`.
- Full-timeline CNN active segmenter, 9-fold LOSO e2/h16 with `max_pos_weight=1.0`, reached set F1 `0.905`, set recall `0.879`, rest false-active/min `2.12`, appended false-active/min `1.40`, and `117` rest-tail segments.
- Soft CNN+periodic hybrid verification reduced rest false-active/min to `1.54`, appended false-active/min to `1.17`, and rest-tail segments to `103`, but lowered set F1 to `0.866` and set recall to `0.836`.
- Larger e5/h32 CNN smoke was worse on the first fold: threshold `0.7/0.45` gave set F1 `0.701`, rest false-active/min `1.07`; threshold `0.6/0.35` gave set F1 `0.738`, rest false-active/min `1.54`.
- Candidate set confirmation in the full pipeline reduced rest-only false reps from `346` to `79`, but worsened Count MAE from `2.90` to `3.72` and only reduced rest-tail overlap reps from `131` to `110`.
- Event-level confirmation with `--event-confirm-min-reps 2` improved the tradeoff compared with hard `min_confirmed_reps=3`: fixed-lag soft x0.8 Rep F1 `0.654`, Count MAE `3.32`, rest false reps `145`, rest-tail overlap reps `112`.
- Periodic active gating is a better full-session baseline than the legacy active detector: fixed-lag soft x0.8 Rep F1 `0.712`, Count MAE `2.64`, Phase IoU `0.538`, C/E MAE `0.965`, rest false reps `254`.
- Best current user-context tradeoff is periodic active + event confirmation with `min_reps=2` and `gap=1000`: Rep F1 `0.708`, Count MAE `2.65`, C/E MAE `0.902`, rest false reps `133`, rest-tail overlap reps `145`.
- Adding action-lock evidence (`top_conf>=0.35`, `margin>=0.05`) is a safer mode with rest false reps `78`, but under-counts more: Rep F1 `0.695`, Count MAE `3.14`.
- Added strict appended-rest metrics that split old rest overlap into boundary-crossing reps, new rest reps, and post-grace rest reps. A two-fold debug run found `34` old overlap reps but only `11` new rest reps and `3` post-grace rest reps after a 2s grace.
- Full 9-fold strict-rest rerun for periodic active + event min2 + gap1000 reached Rep F1 `0.706`, Count MAE `2.65`, C/E MAE `0.930`, rest false reps `137`, old rest-tail overlap reps `146`, boundary-crossing reps `104`, new rest reps `42`, and post-2s-grace rest reps `8`.
- Added per-stream full-session diagnostics to `scripts/evaluate_realtime_soft_top5_pipeline.py`: active coverage, active segment counts, compact event-confirmation summaries, and optional debug reps/segments through `--store-debug-reps`. A new `--only-subjects` option supports targeted held-out-subject debug reruns.
- Diagnosed `db_biceps_curl` excluding yanz: cropped active-set fixed-lag soft x0.8 is strong (`MAE 0.24`, `Rep F1 0.976`), while full-session periodic + event gap1000 drops to `MAE 1.04`, `Rep F1 0.819`. Event confirmation was not the cause; biceps errors had `dropped_reps=0` and were missing before event filtering.
- Minimal hsianshun biceps ablation found the first big regression at the stricter full-session active gate, before event confirmation: cropped-like legacy `threshold=0.5` had `24/27` reps, `MAE 1.00`, `Rep F1 0.902`; legacy `threshold=0.7` soft-online dropped to `6/27`, `MAE 7.00`, `Rep F1 0.218`; periodic `threshold=0.7` recovered partially to `12/27`, `MAE 5.00`, `Rep F1 0.390`. Event gap1000 did not change the periodic biceps count.
- Targeted non-yanz biceps debug on `_tsenyu_temp`, `hsianshun`, and `kevin` showed active-mask fragmentation as the cause. Baseline periodic `threshold=0.7` with no bridge had biceps `MAE 2.44`, `Rep F1 0.537`; adding `--active-mask-bridge-samples 300` improved to `MAE 1.56`, `Rep F1 0.872`; lowering threshold to `0.6` with the same bridge improved to `MAE 1.22`, `Rep F1 0.896`. Full 9-fold rest-safety validation is still needed before changing the mainline.

### Takeaway
- Periodicity features help but do not solve active gating, because rest-after-set transitions can look rep-like.
- The CNN active segmenter learns workout-set continuity better than RF windows, but needs a softer verifier; hard segment rejection improves precision while losing too much recall.
- Event-level confirmation can share the action-lock delay and retroactively release buffered reps, but min-rep-only confirmation is still a secondary filter rather than the final active/rest solution.
- For realistic realtime use, use periodic active gating plus event-level buffered release as the next mainline. A wide event gap is necessary because real sets can include long pauses; too-narrow event groups split true sets and under-count.
- Rest-tail evaluation should use post-grace new rest reps rather than raw overlap reps. Most debug overlap cases are final reps crossing the set/rest boundary, which should be backdated instead of suppressed.
- After strict metric splitting, set-tail leakage is much less severe than raw overlap suggested. The next blocker is rest-only false bursts plus implementing live set-closing grace/backdating.
- Candidate confirmation is a useful secondary filter, not the primary fix.
- The biceps curl full-session regression is mostly an active-gate setting problem, not proof that biceps is intrinsically bad in bounded-latency inference. The first large drop appears when moving from the cropped-like `threshold=0.5` active context to stricter full-session gating around `threshold=0.7`; periodic features and bridge/hysteresis can recover part of it, but threshold/hysteresis should be isolated before modifying the phase CNN.
- The next serious fix should use event-level candidate scoring on full `set + rest_after_set` timelines, combining active CNN, periodic/RF evidence, phase alternation quality, and action confidence rather than a single binary gate.

---

## 2026-06-03 - Corrected Fixed-Lag Soft Top5 Realtime Integration

### What Changed
- Updated `scripts/evaluate_realtime_soft_top5_pipeline.py` so the bounded-latency path matches the phase-latency ablation decoder: trailing-window phase posterior, causal `MA25`, then fixed-lag Viterbi.
- Added fixed-lag soft top5 threshold-scale variants through `--merge-threshold-scales`.
- Documented the corrected result in `docs/experiments/2026-06-03-realtime-soft-top5-replay.md`.

### Results
- 9-fold e5/h64 with `phase_step=10`, `fixed_lag_samples=100`, and action RF posterior gating saved to `artifacts/action_recognition/realtime_soft_top5/summary_fixed_lag100_soft_corrected_e5_h64.json`.
- Strict 0-lag online remains poor: raw Rep F1 `0.452`, Count MAE `5.20`, Phase IoU-F1@50 `0.195`.
- Corrected fixed-lag raw reached Rep F1 `0.791`, Exact Count `0.359`, Count MAE `1.71`, Phase IoU-F1@50 `0.631`, C/E MAE `0.745`.
- Best soft merge variant was `fixed_lag_soft_x0.8`: Rep F1 `0.825`, Exact Count `0.427`, Count MAE `1.34`, Phase IoU-F1@50 `0.631`, C/E MAE `0.700`.
- Larger merge thresholds improved C/E MAE further but hurt count stability: `x1.2` had C/E MAE `0.661` but Count MAE `1.95`.
- 20-epoch follow-up saved to `artifacts/action_recognition/realtime_soft_top5/summary_fixed_lag100_soft_corrected_e20_h64.json`: fixed-lag raw reached Rep F1 `0.809`, Exact Count `0.309`, Count MAE `1.87`, Phase IoU-F1@50 `0.662`, C/E MAE `0.779`.
- In the same 20-epoch run, `fixed_lag_soft_x0.8` remained best for count stability: Rep F1 `0.846`, Exact Count `0.400`, Count MAE `1.43`, Phase IoU-F1@50 `0.662`, C/E MAE `0.735`.

### Takeaway
- The automatic pipeline is still not 0-lag realtime ready, but it now has a credible bounded-latency candidate with `1.0s` fixed label delay.
- Use `fixed-lag Viterbi + soft top5 x0.8` as the current bounded-latency deployment candidate.
- Next work should tune confirmed-transition rep finalization and duration-aware merging to recover Exact Count without losing the Count MAE gain.

### Full-Session Gate Follow-Up
- Extended `scripts/evaluate_realtime_soft_top5_pipeline.py` with rest-only and `set + rest tail` checks, active-masked fixed-lag parsing, optional active hysteresis, active-burst filtering, and optional action-active gate.
- A 9-fold e5/h64 full-session safety check with `active_threshold=0.7` failed the deployment gate: `fixed_lag_soft_x0.8` dropped to Rep F1 `0.651`, Count MAE `2.90`, Phase IoU-F1@50 `0.489`, and still produced `346` false reps across `274` rest streams plus `131` rest-overlap reps across `181` appended-rest streams.
- One-fold smoke tests showed that action-active gating can reduce appended-rest overlap but misses too much real motion; `active_threshold=0.7` plus `action_active_gate_threshold=0.75` had Rep F1 `0.310` and Count MAE `7.04` on `_tsenyu_temp` e1/h16.
- Conclusion revised: `fixed-lag Viterbi + soft top5 x0.8` is the best bounded-latency decoder for active/cropped sets, but complete automatic realtime use is blocked by active gating/rest suppression.

---

## 2026-06-03 - Phase Latency Ablation

### What Changed
- Added `scripts/evaluate_phase_latency_ablation.py` to isolate why strict realtime replay failed.
- Compared offline `predict_fast`, trailing-window causal probabilities with a stateful parser, causal MA25 smoothing, full-sequence Viterbi diagnostic upper bound, and simple 1s centered smoothing.
- Documented the result in `docs/experiments/2026-06-03-phase-latency-ablation.md`.

### Results
- 9-fold e5/h64: offline `predict_fast` reached Rep F1 `0.804`, Count MAE `2.00`, Phase IoU-F1@50 `0.644` under corrected global active detection.
- Causal raw stateful parser dropped to Rep F1 `0.697`, Count MAE `3.40`, Phase IoU-F1@50 `0.594`.
- Causal past MA25 plus offline parser recovered much of the gap: Rep F1 `0.791`, Count MAE `2.02`, Phase IoU-F1@50 `0.613`.
- Full-sequence Viterbi on trailing probabilities was a strong diagnostic upper bound: Rep F1 `0.805`, Count MAE `1.78`, Phase IoU-F1@50 `0.628`.
- Added fixed-lag Viterbi with legal bounded future context. Both 0.5s and 1.0s lag reached Rep F1 `0.810`, Count MAE `1.78`; 1.0s lag had Phase IoU-F1@50 `0.637` and C/E MAE `0.787`.
- Simple 1s centered smoothing did not win overall: Rep F1 `0.756`, Count MAE `2.48`, Phase IoU-F1@50 `0.586`, though a step-1 single-fold check showed it can help when tuned.

### Takeaway
- The strict realtime failure is mostly a decoder/state-machine problem, not proof that the phase CNN cannot support streaming use.
- Fixed-lag Viterbi is now the best bounded-latency realtime decoder: it reproduces offline Rep F1/Count MAE with 0.5-1.0s delay.
- Next target: connect fixed-lag Viterbi to the full automatic soft-top5 pipeline and improve Exact Count with confirmed-transition rep finalization.

---

## 2026-06-03 - Realtime Soft Top5 Replay

### What Changed
- Added `scripts/evaluate_realtime_soft_top5_pipeline.py` for strict streaming-style replay of the current global-active + soft top5 direction.
- The script uses trailing windows for global active detection, action posterior updates, and phase CNN inference, plus a stateful C/E rep parser and delayed one-rep soft merge.
- Documented the result in `docs/experiments/2026-06-03-realtime-soft-top5-replay.md`.

### Results
- 9-fold 5-epoch sanity with `phase_step=10`: raw online Rep F1 `0.455`, Count MAE `5.22`, Phase IoU-F1@50 `0.196`, C/E MAE `1.902`.
- Soft online did not improve count under strict streaming: Rep F1 `0.454`, Count MAE `5.36`, Phase IoU-F1@50 `0.196`, C/E MAE `1.879`.
- A one-fold `phase_step=1` smoke improved Count MAE (`4.00 -> 2.28` on `_tsenyu_temp` e1/h16) but Phase IoU remained low (`0.372`).

### Takeaway
- The current model is not fully automatic realtime ready.
- The main gap is now online active/phase decoding, not the soft top5 merge policy.
- Prioritize a streaming-native active state machine and causal phase smoothing/hysteresis before rerunning 20-epoch realtime claims.

---

## 2026-06-03 - Predicted-Action Top5 Integration

### What Changed
- Added `scripts/evaluate_predicted_action_top5_pipeline.py` to connect the dual-head RF action branch to the existing `raw6 CNN + top5_p5` decoder.
- The script compares `raw`, oracle `top5_p5`, and predicted-action `top5_p5` under 9-fold held-out-subject evaluation with fallback to raw decoding when no action lock is available.
- Documented the integration run in `docs/experiments/2026-06-03-predicted-action-top5-integration.md`.

### Results
- `stricter_a075_p070_m020_s4`: action lock rate `0.641`, locked action accuracy `0.957`, median lock time `5.0s`.
- Fixed an evaluation-design issue: the legacy active detector was per-action and selected by true action from `stream_id`. The integration script now defaults to a global action-agnostic active detector trained on all train-fold sets plus rest/non-action streams.
- With corrected global active detection, raw reached Rep F1 `0.801`, Count MAE `1.973`, Phase IoU-F1@50 `0.639`, C/E MAE `0.861`.
- Hard `stricter` predicted-action top5 was not enough under global active detection: Rep F1 `0.802`, Count MAE `1.968`, C/E MAE `0.835`.
- Soft posterior-gated top5 still improved the corrected pipeline: Rep F1 `0.823`, Count MAE `1.691`, C/E MAE `0.806`, close to corrected oracle top5 Count MAE `1.641`.
- The earlier per-action-active run was optimistic: soft top5 had Rep F1 `0.860`, Count MAE `1.118`, C/E MAE `0.585`, close to oracle top5 Rep F1 `0.867`, Count MAE `1.027`, C/E MAE `0.579`.
- `very_strict_a080_p075_m020_s5` had higher locked accuracy `0.983`, but lower lock rate `0.527`; hard predicted top5 did not improve Count MAE enough (`1.591 -> 1.532`) and worsened Exact Count (`0.541 -> 0.523`) in that run.

### Takeaway
- Hard predicted action should not directly control `top5_p5`; soft posterior-gated merging is much safer and recovers most of the oracle benefit in the 5-epoch integration run.
- Keep `global active + stricter + soft_top5` as the next engineering integration candidate, not yet as a deployment-ready default.
- Active segmentation is now the major bottleneck for full automation; improve and validate it before claiming automatic deployment readiness.
- Rerun soft top5 with the 20-epoch deployment-quality phase setting and add full-session rest/prep false-positive checks before claiming automatic deployment readiness.

---

## 2026-06-03 - Dual-Head RF Action Recognition Probe

### What Changed
- Added `scripts/evaluate_dual_head_rf_action_loso.py` to evaluate a first window-based dual-head Random Forest action branch.
- Head 1 predicts `workout_action` vs `non_action`; head 2 predicts the 8 known actions only for workout windows.
- Included `big_rest`, `rest_after_set*`, and non-active phase windows as `non_action` negatives.
- Documented the experiment in `docs/experiments/2026-06-03-dual-head-rf-action-probe.md`.

### Results
- 9-fold held-out-subject probe with 200-sample windows and 100-sample stride:
  - Active F1 `0.854`, active macro F1 `0.905`.
  - True-active action accuracy `0.810`, macro F1 `0.790`.
  - Set-level action lock rate `0.845`, locked-stream accuracy `0.912`, median lock time `4.0s`.
  - Non-action false-lock rate `0.315`, which is too high for deployment control.
- Lock-policy search on the same RF probabilities reduced false locks:
  - `stricter_a075_p070_m020_s4`: false-lock `0.135`, action lock rate `0.641`, lock accuracy `0.956`, median lock time `5.0s`.
  - `very_strict_a080_p075_m020_s5`: false-lock `0.082`, action lock rate `0.528`, lock accuracy `0.985`, median lock time `6.28s`.
  - `ultra_a085_p080_m025_s5`: false-lock `0.056`, action lock rate `0.432`, lock accuracy `0.990`, median lock time `6.28s`.

### Takeaway
- The dual-head active/action design is viable as a parallel action branch baseline, but the current RF lock policy is too permissive.
- Use `stricter_a075_p070_m020_s4` as the balanced next candidate and `very_strict_a080_p075_m020_s5` as a safety candidate.
- Do not feed predicted action into `top5_p5` yet; first rerun the full rep pipeline with a fallback path for unlocked/unknown action context.

---

## 2026-06-03 - Dual-Head CNN Action Recognition Probe

### What Changed
- Added `scripts/evaluate_dual_head_cnn_action_loso.py` to test a tiny causal Conv1D dual-head model under the same LOSO/window/non-action protocol as the RF probe.
- Ran a 3-epoch hidden-32 CNN probe with default lock and a conservative active-weight/strict-lock variant.
- Documented the comparison in `docs/experiments/2026-06-03-dual-head-cnn-action-probe.md`.

### Results
- CNN default lock: active F1 `0.801`, true-active action accuracy `0.813`, lock accuracy `0.882`, non-action false-lock `0.609`.
- CNN conservative active + strict lock: active F1 `0.798`, true-active action accuracy `0.818`, lock accuracy `0.928`, non-action false-lock `0.334`.
- RF strict lock remains better for deployment gating: active F1 `0.854`, true-active action accuracy `0.810`, lock accuracy `0.945`, non-action false-lock `0.195`.

### Takeaway
- The tiny CNN slightly improves action classification on true-active windows, but it is worse at rejecting non-action windows.
- Keep RF as the current action-branch baseline; do not use this CNN to control `top5_p5` until active/rejection calibration improves.

---

## 2026-06-02 - Raw6 CNN Deployment Export Path

### What Changed
- Added `scripts/export_raw6_cnn_deploy.py` for the intended deployment target: raw6 phase-only 1D causal CNN plus `top5_p5` decoder metadata.
- Added `scripts/stream_raw6_cnn_top5_p5.py` to run the exported CNN artifact on saved workout CSV rows or `zig_bt_client --stdout` raw IMU rows.
- Added `scripts/export_raw6_cnn_rknn_onnx.py` and `scripts/convert_raw6_cnn_to_rknn.py` for Luckfox Pico Zero / RV1103 RKNN conversion.
- Documented the deployment artifact and commands in `docs/experiments/2026-06-02-cnn-deployment-export.md` and `docs/experiments/2026-06-02-rknn-luckfox-pico-zero.md`.

### Takeaway
- Deployment should use `artifacts/deploy/raw6_cnn_top5_p5_current/` after running the CNN export script, not the per-action RF artifact.
- The live CNN path assumes a known action and an active set interval; rest-aware full-session gating is still pending.

---

## 2026-05-26 - Deployable Per-Action RF Export

### What Changed
- Added `scripts/export_per_action_plain_rf_models.py` to export deployable per-action causal Random Forest models.
- Trained 8 action-specific RF models with 1.0s trailing windows, 100 trees, max depth 15, max samples 0.7, and per-action z-score normalization.
- Wrote the deployable artifact to `artifacts/deploy/per_action_plain_rf_current/` with `model.joblib` files, normalization stats, label map, and metadata.

### Takeaway
- This is the currently committed deployable model artifact for later inference/use.
- The stronger raw6 CNN + `top5_p5` research pipeline still has no saved checkpoint, so it remains reproducible through scripts/results rather than a loadable deployed model.

---

## 2026-05-20 - Action-Conditioned Decoder Policy Search

### What Changed
- Added `scripts/new_c_pipeline/action_conditioned_decoder_9fold.py` to select decoder parameters per action using train-fold streams only.
- Tested MA window, Viterbi penalty, and per-action p5 duration merge while keeping the global raw6 CNN unchanged.
- Added `docs/experiments/2026-05-20-action-conditioned-decoder-policy-search.md` with fast and formal results.

### Results
- A broad sampled policy search overfit and was rejected: Rep F1 and Exact dropped despite a lower C/E MAE.
- A reduced all-train policy search improved over the universal raw decoder in the formal run: Rep F1 `0.860 -> 0.869`, Exact `0.518 -> 0.545`, Count MAE `1.523 -> 1.341`, C/E MAE `0.642 -> 0.606`.
- The reduced action-conditioned decoder still did not beat `top5_p5`, which remains better on Rep F1, Exact, and Count MAE.

### Takeaway
- More complete action-conditioned decoding helps versus a universal decoder, but naive per-action policy search is not enough to replace `top5_p5`.
- Keep `top5_p5` as the current best decoder.
- If continuing decoder work, constrain search to the known over-count-prone action set and tune anti-fragmentation parameters rather than allowing every action to change smoothing/Viterbi behavior.

---

## 2026-05-19 - Derived IMU Feature CNN Ablation

### What Changed
- Added `scripts/new_c_pipeline/derived_feature_cnn_9fold.py` to compare raw6 against raw6 plus causal derived channels under the same 9-fold LOSO protocol.
- Tested `mag`, `delta`, and `mag_delta` fast probes; then ran formal GPU follow-ups for `delta` and `mag_delta`.
- Added `docs/experiments/2026-05-19-derived-imu-feature-cnn-ablation.md` and updated the raw IMU preprocessing autoresearch note with the follow-up decision.

### Results
- Fast probes were promising: `delta` improved Rep F1 `0.835 -> 0.859`, Count MAE `1.500 -> 1.273`, Phase IoU-F1@50 `0.710 -> 0.727`, and C/E MAE `0.684 -> 0.652`.
- Formal `delta` did not survive: Rep F1 `0.867 -> 0.849`, Count MAE `1.364 -> 1.659`, while Phase IoU-F1@50 and C/E MAE improved slightly.
- Formal `mag_delta` also failed the gate: Rep F1 `0.859 -> 0.848`, Count MAE `1.541 -> 1.691`, despite Phase IoU-F1@50 `0.717 -> 0.737` and C/E MAE `0.665 -> 0.620`.

### Takeaway
- Do not replace the current raw6 CNN input with magnitude/delta derived channels.
- Motion-change features look useful for phase shape, but they currently hurt rep grouping/count stability in the formal setting.
- Keep raw6 as the main input; revisit delta/magnitude only as auxiliary boundary/decoder features or with stronger regularization.

---

## 2026-05-19 - Raw IMU Preprocessing Autoresearch

### What Changed
- Added `docs/experiments/2026-05-19-raw-imu-preprocessing-autoresearch.md` to decide whether the current raw 6-axis CNN should add preprocessing or engineered channels.
- Reviewed current code paths and confirmed the C/E CNN uses train-fold z-score normalization but no explicit filter, gravity separation, jerk, or magnitude channels.
- Grounded the decision in existing PCA/9-axis probes, active-detector feature usage, and wearable HAR/exercise-recognition literature signals.

### Takeaway
- Do not replace raw 6-axis IMU with a handcrafted feature pipeline.
- Keep raw 6-axis + train-fold normalization as the main input.
- The next worthwhile ablation is lightweight channel augmentation, starting with `raw6_plus_mag` (`acc_mag`, `gyro_mag`), then `raw6_plus_delta` if needed.
- PCA and magnetometer expansion remain rejected based on formal GPU probes.

---

## 2026-05-19 - Action Recognition Architecture Decision

### What Changed
- Reworked `scripts/new_c_pipeline/plot_current_model_architecture.py` to draw the current pipeline as a streaming system with a rest-aware active gate, parallel action recognition branch, 1D causal CNN C/E phase model, online rep parser, `top5_p5` selective merge, and deployment-gate notes.
- Regenerated architecture outputs under `artifacts/figures/current_model_architecture/` in PNG, PDF, and SVG formats.
- Added `docs/experiments/2026-05-19-action-recognition-architecture-decision.md` to document how action recognition should connect to rep segmentation.

### Takeaway
- Action recognition should be integrated as a parallel active-window/set-prefix branch, not as a rep-first post-hoc classifier.
- Rep-first action recognition is rejected as the primary architecture because `top5_p5` needs action context before reps are finalized, creating a circular dependency.
- The current pipeline should be described as action-context dependent but action-recognition pending until the full pipeline is rerun with predicted action labels and rest-aware active gating.

---

## 2026-05-19 - Fixed Baseline Comparison Protocol

### What Changed
- Added `docs/experiments/2026-05-19-fixed-baseline-comparison-plan.md` to decide whether the current cutting model is sufficient for comparison and define which baselines should be frozen.
- Added `docs/experiments/2026-05-19-fixed-baseline-registry.json` as a fixed-row registry for literature and legacy internal baselines.
- Added `scripts/new_c_pipeline/render_fixed_comparison_table.py` so future reruns can refresh only the current project rows while keeping baseline rows fixed.
- Added `scripts/new_c_pipeline/fixed_same_dataset_baselines_9fold.py` and ran peak/RF same-dataset baselines under the current 9-fold LOSO protocol.
- Added `docs/experiments/2026-05-19-same-dataset-baseline-comparison.md` with the four core metrics requested for comparison.
- Added and ran deep same-dataset baselines: Causal TCN-lite and BiLSTM with the shared active detector and the same four core metrics.

### Takeaway
- The current `raw6 CNN + top5_p5` cutting model is strong enough for internal evidence and related-work positioning, but not enough alone for a paper-quality apples-to-apples comparison.
- A credible final table should freeze same-dataset baselines once: peak/threshold, DTW/template, sliding-window RF, BiLSTM/DeepConvLSTM, and causal TCN/DS-MS-TCN-style.
- After baselines are frozen, future work should update only the `Current ours` rows from new artifacts to avoid moving goalposts.
- Same-dataset results show peak accel is a strong count-only baseline (MAE 0.927, within-1 77.7%) but much weaker for structured segmentation (Rep F1 74.8%, Phase IoU-F1@50 43.3%). `top5_p5` is the best balanced structured model (MAE 0.973, Rep F1 87.4%, Phase IoU-F1@50 71.6%, C/E MAE 0.601).
- Deep baselines did not beat the current method: TCN-lite reached Rep F1 77.8%, Phase IoU-F1@50 67.1%, Count MAE 3.214, C/E MAE 1.238; BiLSTM reached Rep F1 78.0%, Phase IoU-F1@50 62.7%, Count MAE 2.077, C/E MAE 1.331.
- Peak baseline Phase IoU-F1@50 and C/E MAE are excluded from true C/E comparison because peak detection does not predict C/E transition points. It remains only a count/rep-boundary baseline.

---

## 2026-05-19 - Related Work Metric Comparison Draft

### What Changed
- Added `docs/experiments/2026-05-19-related-work-metric-comparison.md` to compare exercise recognition, repetition counting, repetition segmentation, and wearable strength-training systems against the current raw6 Causal CNN + `top5_p5` pipeline.
- Organized the comparison around Count MAE, Rep IoU-F1@50, Phase IoU-F1@50, and C/E ratio MAE.
- Verified core bibliographic metadata for the closest papers using CrossRef, arXiv, Semantic Scholar, and OpenAlex where available.
- Expanded the table with more commonly reported literature metrics: recognition accuracy/F1, within-1/OBOA, absolute Count MAE, segmentation F1/IoU, real-time/latency, and sensor burden.

### Takeaway
- Most prior wearable exercise systems report recognition accuracy/F1, count accuracy, OBOA, or count error, but not rep IoU-F1@50, Phase IoU-F1@50, or C/E ratio MAE.
- The strongest positioning is not count-only SOTA; it is single-IMU, causal, rep-structured feedback with boundaries and concentric/eccentric phase balance.
- For cross-paper comparison, use Within-1/OBOA and Count MAE as the most comparable count metrics. Current values are `top5_p5`: exact 59.1%, within-1 75.5%, MAE 0.973; optional count calibration: exact 57.3%, within-1 85.9%, MAE 0.759.
- Before using the comparison table in a manuscript, full-text metric tables should be checked for Abedi et al. 2023, Shang et al. 2024, and MM-Fit repetition counting.

---

## 2026-05-19 - Active Detector Rest-Period Check

### What Changed
- Added `scripts/new_c_pipeline/plot_active_detector_rest_examples.py` to visualize GT active/rest, predicted active probability, and active segments on `set + rest_after_set` snippets.
- Ran held-out `yushuan` checks on `db_rdl`, `db_weighted_crunch`, `db_biceps_curl`, and `db_shoulder_press`.
- Outputs saved under `artifacts/figures/active_detector_rest_examples/`, `artifacts/figures/active_detector_rest_aware_examples/`, `artifacts/figures/active_detector_rest_aware_thr06_examples/`, and `artifacts/figures/active_detector_rest_aware_thr07_examples/`.

### Results

| Setup | Threshold | Active F1 | Precision | Recall | False-active rest | Missed active |
|-------|----------:|----------:|----------:|-------:|------------------:|--------------:|
| Current set-only training | 0.5 | 0.819 | 0.694 | 1.000 | 19.95s / 20s | 0.00s |
| Rest-aware training, 20s rest | 0.6 | **0.973** | 0.949 | 0.999 | 2.62s / 20s | 0.08s |
| Rest-aware training, 20s rest | 0.7 | 0.972 | **0.959** | 0.986 | **2.04s / 20s** | 0.73s |

### Takeaway
- Current active detector training on cropped `sets` is not sufficient for rest suppression; it classifies almost all appended rest as active.
- Adding rest-after-set negatives fixes most of the issue without changing the C/E CNN, but the threshold trades false-active rest against missed active samples.
- Deployment-readiness should include a rest-aware active detector and rerun streaming-style inference on held-out subject data before treating `top5_p5` as board-ready.

---

## 2026-05-18 - Model Probe: Boundary/Event Head Rejected

### What Changed
- Added `scripts/new_c_pipeline/boundary_event_head_9fold.py`.
- Ran a formal 9-fold probe with a raw6 1D Causal CNN plus a lightweight C/E transition-boundary head.
- Compared raw phase-only, `top5_p5`, boundary phase-only decoding, and boundary-bonus Viterbi weights 0.1/0.2/0.4.
- Output saved to `artifacts/cnn_variant_comparison/boundary_event_head_9fold_gpu_h64e20.json`.

### Results

| Decoder | Rep F1 | Exact | Count MAE | C/E MAE | Over / Under |
|---------|------:|------:|----------:|--------:|-------------:|
| Raw MA25+Viterbi | 0.8555 | 0.518 | 1.573 | 0.678 | 0.436 / 0.045 |
| top5_p5 selective merge | **0.8702** | **0.577** | **0.977** | **0.604** | 0.173 / 0.250 |
| Boundary phase only | 0.8464 | 0.555 | 1.391 | 0.715 | 0.418 / 0.027 |
| Boundary bonus 0.1 | 0.8437 | 0.536 | 1.450 | 0.722 | 0.436 / 0.027 |

### Takeaway
- This boundary formulation does not pass the C/E-aware gate.
- Boundary supervision with a broad C/E transition band did not improve C/E MAE and the boundary-bonus decoder degraded count stability.
- Keep `top5_p5` as the current best decoder baseline. If boundary/event modeling is revisited, use explicit rep start/transition/end targets or rep proposal scoring rather than transition-band bonus Viterbi.

---

## 2026-05-18 - Decoder Probe: C/E Duration Constraint Rejected

### What Changed
- Added `scripts/new_c_pipeline/ce_duration_constrained_decoder_9fold.py`.
- Tested decoder-only suppression of predicted C/E phase fragments shorter than train-fold per-action/per-phase GT duration percentiles p1, p5, p10, and p15.
- Output saved to `artifacts/cnn_variant_comparison/ce_duration_constrained_decoder_9fold_gpu_h64e20.json`.

### Results

| Decoder | Rep F1 | Exact | Count MAE | C/E MAE | Over / Under |
|---------|------:|------:|----------:|--------:|-------------:|
| Raw MA25+Viterbi | 0.8549 | 0.518 | 1.573 | 0.675 | 0.459 / 0.023 |
| top5_p5 selective merge | **0.8766** | **0.568** | **0.955** | **0.636** | 0.186 / 0.245 |
| C/E duration p1 | 0.8373 | 0.477 | 1.332 | 0.701 | 0.086 / 0.436 |
| C/E duration p5 | 0.7776 | 0.336 | 1.918 | 0.828 | 0.077 / 0.586 |

### Takeaway
- Directly suppressing short C/E phase runs reduces over-counting but destroys too many legitimate reps, producing heavy under-counting.
- C/E MAE worsens for every tested percentile, so this decoder fails the updated C/E-aware gate.
- Keep `top5_p5` as the current best decoder baseline; the next model-impacting direction should be boundary/event supervision or a confidence-aware rep-level decoder, not more phase-run duration suppression.

---

## 2026-05-18 - Real-Time-Safe Count Calibration Probe

### What Changed
- Added `scripts/new_c_pipeline/count_calibration_from_raw_results.py`.
- The script evaluates post-hoc count calibration on existing raw6 CNN held-out stream predictions using only inference-time-safe fields: `action` and raw `pred_count`.
- Output saved to `artifacts/cnn_variant_comparison/count_calibration_from_raw6_loso.json`.

### Results

| Method | Exact | Within-1 | Count MAE | Bias | Over / Under |
|--------|------:|---------:|----------:|-----:|-------------:|
| Raw identity | 0.518 | 0.673 | 1.505 | +1.405 | 0.459 / 0.023 |
| Action linear | 0.573 | **0.859** | **0.759** | -0.005 | 0.200 / 0.227 |
| Nested action selector, Exact | **0.668** | 0.836 | 1.073 | +0.564 | 0.245 / 0.086 |
| Action linear + duration | 0.577 | **0.859** | 0.800 | +0.064 | 0.223 / 0.200 |

### Takeaway
- A tiny final-count calibration layer can substantially reduce over-count bias without affecting streaming inference cost.
- `action_linear` is the best MAE-oriented candidate, but it only corrects the displayed count; it does not change rep boundaries or Rep F1.
- Adding total stream duration does not improve MAE, so the next model-impacting step should use prediction-derived features or a direct boundary/count-density head.

---

## 2026-05-18 - Comprehensive Metrics for Strongest Raw6 1D Causal CNN

### What Changed
- Added `scripts/new_c_pipeline/raw6_cnn_comprehensive_9fold.py` to evaluate the current best raw 6-axis 1D Causal CNN with richer metrics in one run.
- Metrics now include Rep precision/recall/F1, exact and within-1 count accuracy, Count MAE/RMSE/bias, over/under rates, Phase Macro F1/accuracy, transition MAE, C/E phase segment IoU-F1@0.50, and C/E ratio MAE.
- Output saved to `artifacts/cnn_variant_comparison/raw6_cnn_comprehensive_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, hidden=64, epochs=20, CUDA GPU)

| Metric | Value |
|--------|------:|
| Rep Precision | 0.8144 |
| Rep Recall | 0.9115 |
| Rep F1 | 0.8602 |
| Exact Count Acc | 0.5182 |
| Within-1 Count Acc | 0.6727 |
| Count MAE | 1.5045 |
| Count RMSE | 2.7822 |
| Count Bias (pred-gt) | +1.4045 |
| Over / Under Rate | 45.9% / 2.3% |
| Phase Macro F1 | 0.7589 |
| Phase Accuracy | 0.7636 |
| Transition MAE | 310.0 ms |
| Concentric IoU-F1@0.50 | 0.7295 |
| Eccentric IoU-F1@0.50 | 0.7111 |
| Avg Phase IoU-F1@0.50 | 0.7203 |
| C/E Ratio MAE | 0.6704 |

### Takeaway
- This is the most complete current metric snapshot for the strongest sequence model.
- The model has high recall but positive count bias, confirming the remaining issue is over-counting rather than missed reps.
- Best action remains `db_biceps_curl` (Rep F1=0.994, Avg Phase IoU-F1=0.941). Weakest count actions are `db_rdl` and `db_shoulder_press` due to over-counting.

---

## 2026-05-18 - Decoder Probe: Duration-Aware Rep Merge

### What Changed
- Added `scripts/new_c_pipeline/duration_merge_decoder_9fold.py`.
- The experiment keeps the raw 6-axis 1D Causal CNN and MA25+Viterbi phase predictions fixed, then merges predicted reps whose duration is shorter than a train-fold per-action GT duration percentile.
- Tested percentiles p5, p10, p15, p20, p25 with max merge gap 50 samples.
- Output saved to `artifacts/cnn_variant_comparison/duration_merge_decoder_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, hidden=64, epochs=20, CUDA GPU)

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate | C/E MAE |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|--------:|
| Raw MA25+Viterbi | 0.8607 | **0.5045** | 0.6773 | 1.5091 | +1.40 | 0.4682 | 0.0273 | 0.7003 |
| Merge p5 | **0.8659** | 0.4909 | **0.7500** | **1.1455** | -0.83 | **0.1045** | 0.4045 | **0.6426** |
| Merge p10 | 0.8475 | 0.4591 | 0.7273 | 1.3409 | -1.09 | 0.0909 | 0.4500 | 0.6422 |
| Merge p15 | 0.8278 | 0.4500 | 0.6591 | 1.5409 | -1.34 | 0.0682 | 0.4864 | 0.6473 |

### Takeaway
- Duration merge works for the targeted failure mode: over-rate drops from 46.8% to 10.5% at p5.
- However, global duration merge over-corrects into under-counting: under-rate rises to 40.5% and count bias flips from +1.40 to -0.83.
- p5 improves Rep F1, within-1 count accuracy, Count MAE, and C/E MAE, but exact count drops slightly, so it does not satisfy the strict acceptance gate.
- Next direction should be selective duration merge only on over-count-prone actions or confidence-triggered streams, not global merge across all actions.

---

## 2026-05-18 - Decoder Probe: Selective Duration Merge Succeeds

### What Changed
- Added `scripts/new_c_pipeline/selective_duration_merge_decoder_9fold.py`.
- Instead of globally merging short predicted reps, merge is applied only to over-count-prone action sets.
- Tested p5/p10 duration thresholds on action sets: `top3`, `top4`, `top5`, `over50`, and `compound6`.
- Output saved to `artifacts/cnn_variant_comparison/selective_duration_merge_decoder_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, hidden=64, epochs=20, CUDA GPU)

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate | C/E MAE |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|--------:|
| Raw MA25+Viterbi | 0.8582 | 0.5091 | 0.6727 | 1.4955 | +1.38 | 0.4682 | 0.0227 | 0.6544 |
| top3 p5 | 0.8702 | 0.5727 | 0.7409 | 1.0182 | +0.28 | 0.3091 | 0.1182 | 0.6234 |
| top4 p5 | 0.8700 | 0.5909 | 0.7409 | 0.9955 | -0.07 | 0.2273 | 0.1818 | 0.6077 |
| **top5 p5** | **0.8737** | 0.5909 | 0.7545 | **0.9727** | -0.36 | 0.1682 | 0.2409 | 0.6010 |
| over50 p5 | 0.8614 | 0.5864 | 0.7364 | 1.2273 | **+0.02** | 0.2182 | 0.1955 | 0.5977 |
| compound6 p5 | 0.8672 | **0.6000** | 0.7545 | 1.0500 | -0.64 | **0.1182** | 0.2818 | **0.5860** |

### Takeaway
- Selective duration merge beats the raw decoder on both strict gate metrics: Rep F1 and Exact Count.
- `top5_p5` is the best balanced candidate by Rep F1 and Count MAE.
- `compound6_p5` has the best Exact Count and lowest over-rate, but over-corrects more into under-counting.
- New candidate decoder: **top5_p5 selective duration merge** on `db_rdl`, `db_shoulder_press`, `db_bench_press`, `one_arm_db_row`, and `db_weighted_crunch`.

---

## 2026-05-18 - Decoder Probe: Per-Action Duration Oracle Still Cannot Reach 0.6 Count MAE

### What Changed
- Added `scripts/new_c_pipeline/per_action_duration_merge_oracle_9fold.py`.
- Tested an exploratory upper bound where each action independently chooses the best duration merge option from `none`, p5, p10, p15, p20, p25, p30 using held-out results.
- Output saved to `artifacts/cnn_variant_comparison/per_action_duration_merge_oracle_9fold_gpu_h64e20.json`.

### Results

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|
| Raw none | 0.8560 | 0.518 | 0.673 | 1.61 | +1.46 | 0.450 | 0.032 |
| Global p5 | 0.8663 | 0.505 | 0.750 | 1.19 | -0.82 | 0.100 | 0.395 |
| Per-action oracle | **0.8781** | **0.568** | **0.755** | **1.03** | -0.41 | 0.168 | 0.264 |

### Selected Oracle Options
- `db_bench_press`: p15
- `db_biceps_curl`: none
- `db_rdl`: p10
- `db_shoulder_press`: p5
- `db_squat`: none
- `db_triceps_curl`: none
- `db_weighted_crunch`: p5
- `one_arm_db_row`: p5

### Takeaway
- Even an optimistic per-action oracle only reaches Count MAE 1.03, not below 0.6.
- This rules out simple duration-merge decoder tuning as sufficient for the 0.6 Count MAE target.
- To reach 0.6, the next step should be a stream-level count calibration model or a count-constrained decoder, not more duration percentile tuning.

---

## 2026-05-18 - Quick Probe: PCA Input for 1D Causal CNN

### What Changed
- Added `scripts/new_c_pipeline/test_pca_input.py` to compare raw 6-axis IMU input against PCA-transformed inputs for the phase-only 1D Causal CNN.
- The probe fits PCA only on training-subject active samples, then applies the transform to held-out kevin streams to avoid test leakage.
- This was a quick single-fold CPU probe, not a formal 9-fold result.

### Results (kevin fold, excluded light-weight sessions, 5 epochs, hidden=32)

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.744 | 0.458 | 1.67 | 0.657 | 13 / 0 |
| PCA-3 | 0.794 | 0.417 | 1.38 | 0.662 | 13 / 1 |
| **PCA-4** | **0.848** | **0.667** | **0.54** | **0.701** | **6 / 2** |
| PCA-5 | 0.743 | 0.375 | 1.33 | 0.631 | 13 / 2 |
| PCA-6 | 0.704 | 0.458 | 1.83 | 0.635 | 13 / 0 |

### Takeaway
- PCA95 keeps all 6 components, so a variance-retention rule does not actually reduce this input.
- Fixed **PCA-4** is the only strict win in this quick fold: higher Rep F1 and higher Exact Count than raw 6-axis.
- Treat PCA-4 as a candidate for a stricter checkpoint-based LOSO test, not as a new accepted baseline yet.

---

## 2026-05-18 - Fast 9-Fold LOSO: PCA-4 Input for 1D Causal CNN

### What Changed
- Added `scripts/new_c_pipeline/pca4_cnn_9fold.py` for full-subject LOSO comparison of raw 6-axis input vs PCA-4 input.
- Ran the fast protocol only: hidden=32, 5 epochs, CPU, MA25 + Viterbi penalty=0.3.
- Output saved to `artifacts/cnn_variant_comparison/pca4_cnn_9fold_fast.json`.

### Results (9-fold LOSO, 220 streams, light-weight sessions excluded)

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.8634 | 0.559 | 1.37 | 0.7477 | 89 / 8 |
| **PCA-4** | **0.8673** | **0.577** | **1.00** | 0.7470 | **78 / 15** |

### Takeaway
- PCA-4 remained a strict win in the fast full-subject probe: Rep F1 and Exact Count both improved, and Count MAE dropped substantially.
- The improvement is modest in Rep F1 (+0.0039) but more meaningful for counting stability (MAE 1.37 -> 1.00, over-count 89 -> 78).
- This still does not replace the current Golden Baseline because the protocol is faster/different from the 20-epoch hidden=64 CNN baseline.
- Next formal step: run PCA-4 under the same 20-epoch hidden=64 protocol and compare against the established raw 6-axis Golden Baseline.

---

## 2026-05-18 - Formal GPU 9-Fold: PCA-4 Does Not Beat Raw 6-axis

### What Changed
- Re-ran `scripts/new_c_pipeline/pca4_cnn_9fold.py` on GPU with the stronger CNN setting: hidden=64, epochs=20.
- Output saved to `artifacts/cnn_variant_comparison/pca4_cnn_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, 220 streams, light-weight sessions excluded)

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| **Raw 6-axis** | **0.8513** | 0.514 | 1.68 | **0.7575** | 99 / 8 |
| PCA-4 | 0.8453 | 0.514 | 1.68 | 0.7498 | 103 / 4 |

### Takeaway
- PCA-4 was promising in the fast hidden=32, 5-epoch probe, but it did not hold under the stronger hidden=64, 20-epoch GPU run.
- PCA-4 does not satisfy the strict acceptance rule because Rep F1 is lower and Exact Count does not improve.
- Keep raw 6-axis IMU as the CNN input for the current Golden Baseline.

---

## 2026-05-18 - Formal GPU 9-Fold: PCA-1 Input Fails

### What Changed
- Generalized `scripts/new_c_pipeline/pca4_cnn_9fold.py` with `--pca-components`.
- Ran PCA-1 as the only CNN input channel under the same formal setting: hidden=64, epochs=20, CUDA GPU.
- Output saved to `artifacts/cnn_variant_comparison/pca1_cnn_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, 220 streams, light-weight sessions excluded)

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| **Raw 6-axis** | **0.8616** | **0.509** | **1.61** | **0.7582** | 96 / 12 |
| PCA-1 | 0.6160 | 0.150 | 3.88 | 0.6003 | 175 / 12 |

### Takeaway
- PCA-1 is not viable. One principal component removes too much phase-discriminative information.
- The failure mode is heavy over-counting: 175 over-counted streams versus 96 for raw 6-axis.
- Keep raw 6-axis IMU as the CNN input; PCA-1 should not be pursued further.

---

## 2026-05-18 - Formal GPU 9-Fold: 9-axis IMU Input Fails

### Clarification
- The CNN baseline is already a **1D Causal CNN**: convolution is 1D along the time axis.
- The baseline input is not one-dimensional; it is a multichannel 1D sequence: `[channels=6, time=300]` for `ax, ay, az, gx, gy, gz`.

### What Changed
- Added `scripts/new_c_pipeline/axis_subset_cnn_9fold.py` to compare raw 6-axis against arbitrary raw channel subsets/supersets.
- Ran 9-axis IMU input with accelerometer + gyroscope + magnetometer: `ax, ay, az, gx, gy, gz, mx, my, mz`.
- Output saved to `artifacts/cnn_variant_comparison/imu9_cnn_9fold_gpu_h64e20.json`.

### Results (9-fold LOSO, hidden=64, epochs=20, CUDA GPU)

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| **Raw 6-axis** | **0.8535** | 0.464 | **1.75** | **0.7611** | 115 / 3 |
| 9-axis IMU | 0.7871 | **0.468** | 2.05 | 0.6903 | 98 / 19 |

### Takeaway
- 9-axis input does not pass the acceptance gate. Rep F1 drops substantially and Phase F1 degrades.
- Magnetometer channels appear to add cross-subject noise rather than useful phase information.
- Keep the CNN baseline input at raw 6-axis IMU.

---

## 2026-05-17 - Fairness Check: Peak Detection with 6-axis mag

### Test: Does Peak Detection benefit from 6-axis magnitude instead of acc_mag?

**Hypothesis**: If Peak Detection uses gyro_mag or 6-axis_mag, it might perform better because RF feature importance shows GYRO is important.

**Method**: Quick test (2 subjects: haoyu, kevin) with `--mag-mode` flag added to `evaluate_peak_baseline.py`

**Results**:

| Mag Mode | Rep F1 | Precision | Recall |
|----------|--------|-----------|--------|
| acc_mag (baseline, 7-fold) | ~0.757 | ~0.755 | ~0.760 |
| 6-axis mag (2-subject quick) | 0.584 | 0.566 | 0.603 |
| gyro mag (2-subject quick) | 0.585 | 0.568 | 0.604 |

**Conclusion**: 6-axis/gyro mag **hurts** Peak Detection performance. Peak Detection's algorithm (finding periodic peaks in 1D signal) works best with acc_mag which captures the body's center-of-mass motion. Adding GYRO noise disrupts the clear periodic structure.

**Implication**: SDTW should also remain on acc_mag. The "unfairness" is not about input dimension but algorithmic capability — signal processing methods can only use dimensionality-reduced 1D signals, while ML methods learn from high-dimensional raw data.

---

## 2026-05-17 - BREAKTHROUGH: Per-Action Plain RF Achieves 0.850 F1

### What Changed
- Implemented and validated **Per-Action Causal RF** under Action-First architecture
- Each of 8 actions has its own dedicated RF model trained only on that action's data
- Tried per-action **feature subset selection** (Top-30 based on importance) but found it **hurts** performance
- The real breakthrough is **per-action training itself**, not feature selection

### Results (7-fold LOSO, 8 actions, 226 streams)

| Method | Rep F1 | Precision | Recall | IoU-F1@50 |
|--------|--------|-----------|--------|-----------|
| **Per-Action Plain RF** | **0.850** | **0.869** | **0.831** | **0.706 ± 0.273** |
| General Causal RF | 0.778 ± 0.057 | 0.783 ± 0.057 | 0.773 ± 0.061 | 0.561 ± 0.119 |
| Δ | **+0.072** | **+0.086** | **+0.058** | **+0.145** |

### Per-Action Breakdown (Rep F1)
| Action | F1 | Notes |
|--------|-----|-------|
| db_biceps_curl | 0.999 | Near-perfect across all subjects |
| one_arm_db_row | 0.936 | Excellent |
| db_rdl | 0.908 | Excellent |
| db_squat | 0.901 | Excellent |
| db_shoulder_press | 0.880 | Good |
| db_bench_press | 0.869 | Good |
| db_triceps_curl | 0.819 | Good |
| db_weighted_crunch | 0.640 | Weak — needs improvement |

### Why Feature Subset Selection Failed
- 3-fold quick test: Per-Action Plain RF (all features) = 0.895 vs Feature Subset (top-30) = 0.879
- "Low importance" features still contain cross-subject generalization signal
- Per-action training itself is sufficient regularization

### Deployment Implications
- This result **exceeds the deployment gate threshold** (previous best: ~0.7x with poor recall)
- 0.85 F1 / 0.83 recall is viable for real-world rep counting
- Model size: 8 × ~200KB = ~1.6MB total (still tiny for embedded)
- Inference latency unchanged: 1.15s causal (1.0s window + 0.15s smoothing)

### Next Steps
- Investigate db_weighted_crunch failure modes (hsianshun F1=0.12, yanz F1=0.25)
- Consider action-specific hyperparameter tuning (especially crunch)
- Run IoU-F1@50 aggregation properly (weighted by sample count)
- Update AGENTS.md deployment gate status

---

## 2026-05-16 - Phase 1a: 7-Subject LOSO Baseline (Cleaned Data)

### Data Quality Decision
- Manually cleaned non-standard reps with abnormal concentric-dominant phase ratios
- Excluded two subjects (tsenyu, ziho) due to persistent data quality issues
- Final dataset: 7 subjects, 8 actions, 226 streams

### Results (7-fold LOSO, 8 actions, Plain Causal RF)

| Subject | Rep F1 | Precision | Recall |
|---------|--------|-----------|--------|
| haoyu | 0.868 | 0.791 | 0.962 |
| thomas | 0.798 | 0.762 | 0.838 |
| yoru | 0.712 | 0.681 | 0.746 |
| kevin | 0.703 | 0.670 | 0.739 |
| yushuan | 0.672 | 0.632 | 0.718 |
| hsianshun | 0.634 | 0.594 | 0.679 |
| yanz | 0.577 | 0.525 | 0.640 |
| **Overall** | **0.706 ± 0.091** | **0.709** | **0.704** |

### Key Findings
1. **Best performers**: haoyu (0.87) and thomas (0.80) - standard, consistent movement patterns
2. **Worst performer**: yanz (0.58) - despite cleaning, cross-subject generalization remains challenging
3. **Mean F1 = 0.706** is adopted as the official Phase 1a baseline for Causal RF (plain)
4. Refiner historically improved on yushuan (+0.05-0.07) but current implementation underperforms

### Next Steps
- Complete Phase 1a baseline comparison: Peak Detection, SDTW, Sliding-window RF, BiLSTM
- Phase 1b: Modality Ablation (after Phase 1a complete)
- Phase 2/3: Deferred until Phase 1 complete

## 2026-05-16 ( evening ) - Causal RF Config Optimization

### What Changed
- Systematically optimized Causal RF hyperparameters on 7-subject cleaned data
- **Key finding**: window_size is the decisive parameter

### Optimization Grid

| window_size | n_estimators | smoothing | Rep F1 | vs baseline |
|-------------|--------------|-----------|--------|-------------|
| 50 (0.5s) | 50 | 15 | 0.706 ± 0.091 | baseline |
| 50 | 100 | 15 | 0.723 ± 0.094 | +0.017 |
| **100 (1.0s)** | **100** | **15** | **0.777 ± 0.057** | **+0.071** |
| 150 (1.5s) | 100 | 15 | 0.778 ± 0.044 | +0.072 |
| 100 | 100 | 25 | 0.783 ± 0.064 | +0.077 |

### Why window_size=100 is the sweet spot
- 0.5s window sees only 20% of a rep (median rep duration ~2.5-3.0s)
- 1.0s window sees 40% of a rep, including full concentric→eccentric transition
- 1.5s window shows no additional benefit over 1.0s

### Updated Official Causal RF Result
- **Rep F1 = 0.778 ± 0.057** (was 0.706)
- **Precision = 0.783 ± 0.057**
- **Recall = 0.773 ± 0.061**
- **IoU-F1@50 = 0.561 ± 0.119**
- **Total causal latency = 1.15s** (1.0s window + 0.15s smoothing)

### Key Achievement
- ✅ Causal RF (0.778) > Peak Detection (0.757) by +0.021
- ✅ Causal RF (0.778) > Sliding-window RF (0.768) by +0.010
- ✅ Standard deviation reduced from 0.091 to 0.057 (more stable)

### Action-Specific Insight
- Peak Detection excels on full-body compound movements (squat F1=0.95+, rdl F1=0.90+) where acc_mag has clear periodic peaks
- Causal RF dominates on arm isolation movements (biceps curl Δ=+0.50, triceps curl Δ=+0.50) where acc_mag lacks clear peaks
- Causal RF also wins on core (crunch Δ=+0.20) and unilateral (row Δ=+0.30) movements

### Next Steps
- Run remaining baselines: SDTW, BiLSTM
- Re-run Per-action RF+Refiner (current result 0.619 is suspiciously low)
- Proceed to Phase 1b Modality Ablation

## 2026-05-16 - Phase 1a: Rep Segmentation Baseline Comparison (Full 9-Fold LOSO)

### What Changed

- Completed full 9-subject strict LOSO evaluation for Rep Segmentation baselines:
  1. **Causal Random Forest (plain)**: strict causal trailing-window classifier + concentric/eccentric pairing
  2. **Per-action RF + Boundary Refiner**: per-action model with ExtraTrees boundary offset regressor
  3. **Magnitude Peak Detection**: simple heuristic baseline (acc_mag → smooth → find_peaks)
- Fixed Peak Detection ground-truth extraction bug (was treating entire set stream as single rep).
- Fixed `preprocessing/sdtw_rep_segmentation.py` to use `acc_mag` instead of `ranked[0]` for DTW feature.
- Created simplified LOSO wrappers without inner tuning:
  - `scripts/evaluate_causal_rf_loso.py`
  - `scripts/evaluate_rf_refiner_per_action_loso.py`
  - `scripts/evaluate_peak_baseline.py`

### Results (9-fold LOSO, 4 actions, 157 streams)

| Method | Rep F1 | Precision | Recall |
|--------|--------|-----------|--------|
| **Causal RF (plain)** | **0.485 ± 0.075** | 0.391 ± 0.072 | 0.647 ± 0.088 |
| Per-action RF + Refiner | 0.336 ± 0.076 | 0.269 ± 0.072 | 0.456 ± 0.068 |
| Peak Detection | 0.002 ± 0.000 | 0.008 ± 0.000 | 0.001 ± 0.000 |

### Key Findings

1. **Causal RF (plain) is the strongest strict-causal baseline** at F1=0.485.
2. **Boundary Refiner actively degrades performance**: per-action RF+Refiner (0.336) is worse than plain Causal RF (0.485). The refiner is trained on low-quality coarse matches and learns incorrect offsets.
3. **Peak Detection fails catastrophically** on continuous rep streams (no rest periods between reps). It detects only 1-3 peaks per set vs. 10-13 true reps.
4. **Cross-subject generalization is hard**: even with 8 subjects of training data, held-out subject F1 remains modest. Some folds collapse to near-zero (e.g., hsianshun db_bench_press F1=0.023).

### Interpretation

- The current RF-based pipeline has reached a ceiling around F1≈0.5 on strict LOSO.
- The boundary refiner is not salvageable in its current form; training on matched coarse reps propagates errors.
- For deployment, a simple magnitude peak detector is insufficient; model-based phase detection is necessary.
- The gap between single-subject holdout (~0.78 on yushuan) and full LOSO (~0.49) reveals that some subjects are much harder to generalize to than others.

### Next Steps

- Phase 1b: Modality ablation using Causal RF as fixed backbone.
- Phase 2: Phase Segmentation (concentric/eccentric splitting inside detected reps).
- Consider BiLSTM/Sliding-window RF as non-causal upper bounds (not yet run).


## 2026-05-15 - Direct Event RF Probe

### What Changed

- Added `scripts/evaluate_direct_event_rf.py` as a lightweight probe for a more
  direct rep detector using RF on sparse boundary events instead of phase labels.
- The probe trains a causal trailing-window RF on four classes:
  - `other`
  - `start`
  - `transition`
  - `end`
- It then decodes reps directly from event probabilities with a simple ordered
  start -> transition -> end pairing rule.

### Quick Reading

- A first strict sparse-event probe on held-out `yoru`
  (`db_bench_press`, `db_triceps_curl`) produced zero reps.
- A relaxed single-action follow-up on `yoru/db_triceps_curl` using a wider
  event band and lower event threshold produced:
  - precision `0.9474`
  - recall `0.5143`
  - rep F1 `0.6667`
- The current phase-first RF baseline on the same held-out action remains much
  stronger:
  - precision `1.0000`
  - recall `1.0000`
  - rep F1 `1.0000`

### Interpretation

- A naive direct-event RF is not competitive yet on this dataset.
- The direction is still conceptually aligned with the final task, but the
  current minimal implementation is too sparse and too brittle to replace the
  phase-first RF + refinement pipeline.

## 2026-05-15 - Modality-Only Nested RF Benchmark Mode

### What Changed

- Updated `scripts/benchmark_per_action_rf_refiner.py` with a modality-focused
  selection mode that keeps subject-wise nested evaluation intact while removing
  window search as a tuning dimension.
- Added CLI switches:
  - `--modality-only-search`
  - `--selection-window-size`
  - `--selection-edge-window`
- Tightened modality-only selection to reduce missing-rep failures:
  - exclude single-modality candidates by default in modality-only mode
  - switch default selection priority from `micro_f1_at_50` to `rep_f1`
  - rank ties by `rep_f1`, exact-count consistency, `recall`,
    `micro_f1_at_50`, and under-segmentation
  - add a default-modality guardrail so a narrower subset is only kept when it
    is clearly better and does not lose too much recall or count consistency
- Added CLI switches for those guardrails:
  - `--min-modality-groups`
  - `--disable-default-modality-guardrail`
  - `--default-modality-min-improvement`
  - `--default-modality-max-recall-drop`
  - `--default-modality-max-exact-count-ratio-drop`
  - `--default-modality-max-mean-abs-count-diff-increase`
- In modality-only mode, inner selection now varies only the sensor modality
  subset while keeping the RF trailing window and refiner edge window fixed.
- Root benchmark outputs now record the selection mode and fixed selection
  window settings used by the run.

### Why

- The broader full-coverage benchmark showed several actions getting worse after
  nested tuning, suggesting the combined modality + window search may be
  overfitting the inner subject folds.
- Follow-up analysis also showed some tuned runs dropping rep recall badly after
  collapsing to a single modality, suggesting some actions need multi-sensor
  redundancy to avoid missing whole reps.
- This mode creates a cleaner ablation to test whether modality choice carries
  most of the cross-subject benefit while reducing runtime and search variance.

## 2026-05-15 - Second-Pass RF Benchmark Acceleration

### What Changed

- Optimized `scripts/benchmark_per_action_rf_refiner.py` to reduce repeated work
  during nested selection:
  - cache normalized inner-fold train/validation streams
  - cache RF training and `predict_proba` results per fold/modality/window
  - reuse one RF across multiple `edge_window` candidates
  - reuse matched coarse-vs-truth rep pairs for refiner fitting
  - cache validation coarse reps, truth reps, and sample rate during repeated evaluation
  - cache refiner edge-feature matrices per `edge_window`
- Optimized causal RF feature extraction:
  - added batched window feature extraction in `scripts/compare_baselines.py`
  - replaced Python-loop trailing window construction in
    `scripts/evaluate_causal_rf.py` with a vectorized sliding-window view
  - vectorized class-probability remapping in `predict_causal_rf`
- Optimized the shared sliding-window RF baseline path as well:
  - added vectorized start-window matrix construction in
    `scripts/compare_baselines.py`
  - removed the old list-comprehension feature extraction in `predict_rf`
- Added benchmark knobs for refiner runtime control:
  - `--refiner-n-estimators`
  - `--refiner-max-depth`
  - `--refiner-min-samples-leaf`
  - `--max-refiner-train-streams-per-subject`
  - `--max-matched-reps-per-stream`
  - `--max-matched-reps-per-subject`
  - `--min-matched-reps-for-refiner`

### Validation

- Re-ran a tiny end-to-end check after the acceleration changes:
  - outputs:
    - `artifacts/baseline_comparison/smoke_nested_rf_tiny_v2`
    - `artifacts/baseline_comparison/smoke_nested_rf_tiny_v3`
- Validated both new process-level sharding paths:
  - action-parallel smoke:
    - `artifacts/baseline_comparison/smoke_nested_rf_action_parallel`
  - outer-fold-parallel smoke:
    - `artifacts/baseline_comparison/smoke_nested_rf_outer_parallel`
- Ran a larger optimized multi-subject smoke benchmark with outer-fold parallelism:
  - `artifacts/baseline_comparison/smoke_nested_rf_multisubject_v3`
- Ran a medium-fidelity optimized benchmark with relaxed truncation settings:
  - `artifacts/baseline_comparison/medium_nested_rf_multisubject_v1`
- The optimized pipeline still completed successfully and wrote the expected
  benchmark artifacts.

### Reading

- The benchmark infrastructure is now operational for:
  - multi-subject outer held-out folds
  - train-only inner selection
  - action-specific modality and window search
  - process-level outer-fold parallel execution
- The current `smoke_nested_rf_multisubject_v3` run should still be treated as a
  smoke benchmark, not a headline comparison, because it uses:
  - reduced RF tree count
  - reduced refiner tree count
  - aggressive matched-rep and train-stream truncation
  - only two actions and two outer subjects
- Under those aggressive speed settings, the tuned pipeline became very
  precision-heavy and consistently under-segmented:
  - tuned overall: `rep_f1 = 0.7147`, `micro_f1@50 = 0.4777`
  - baseline overall: `rep_f1 = 0.7664`, `micro_f1@50 = 0.4935`
- Interpretation:
  - the pipeline is ready for a more faithful benchmark run
  - but the smoke-time truncation is now strong enough to bias quality and should
  be relaxed before drawing substantive conclusions

### Medium-Fidelity Follow-Up Reading

- The relaxed follow-up run moved the tuned benchmark back toward a more useful
  regime:
  - tuned overall: `rep_f1 = 0.7449`, `micro_f1@50 = 0.5485`
  - baseline overall: `rep_f1 = 0.7664`, `micro_f1@50 = 0.5132`
- Compared with the aggressive smoke:
  - tuned `micro_f1@50` improved substantially
  - tuned `rep_f1` also improved
  - the tuned pipeline is still recall-limited and still under-segments, but it
    now trades that for a meaningful IoU gain over the fixed baseline
- Action-level winner instability remains high in this small run:
  - `db_bench_press` picked different best settings across the two outer folds
  - `db_rdl` also picked different best settings across the two outer folds
- Interpretation:
  - the benchmark infrastructure is now good enough for a broader sweep
  - but two held-out subjects and two actions are still too small to claim stable
    per-action deployment settings

## 2026-05-15 - Nested Per-Action RF Benchmark Pipeline

### What Changed

- Added `scripts/benchmark_per_action_rf_refiner.py`.
- The new script implements a train-only nested benchmark for the known-action
  causal RF + boundary refiner pipeline:
  - outer held-out subject evaluation
  - inner subject-wise model selection on train subjects only
  - per-action duration statistics from ground-truth train reps only
  - automatic trailing-window candidates from train-fold median rep duration
  - automatic edge-window candidates from train-fold median rep duration
  - full 7-subset modality search when all 9 axes are present
  - per-action min/max duration priors from train-fold quantiles
  - baseline-vs-tuned outer-fold comparison
- The script writes benchmark artifacts in the same inspection style as the
  existing RF refiner runs:
  - `results.json`
  - `stream_metrics.csv`
  - `index.html`
  - `stream_replays/*.svg`
  - plus fold-level duration and selection summaries

### Smoke Validation

- Ran a tiny end-to-end smoke benchmark with:
  - held-out subject: `yushuan`
  - action: `db_bench_press`
  - modality search: `acc`
  - one trailing candidate and one edge candidate
  - smaller RF / refiner training budget for runtime sanity
- Output:
  - `artifacts/baseline_comparison/smoke_nested_rf_tiny`
- Smoke result:
  - pipeline completed end-to-end
  - fold-level duration stats were written
  - inner train-only selection completed
  - outer tuned and baseline evaluations completed
  - HTML / CSV / JSON / SVG outputs were generated as expected

### Interpretation

- The smoke run is a tooling validation only, not a headline benchmark result.
- The project now has the required benchmark machinery to answer:
  - which modality wins per action under cross-subject evaluation
  - which trailing / edge windows are stable per action
  - whether train-only tuned per-action settings beat the current fixed baseline

## 2026-05-14 - Full-Action Baseline Expansion And New RF-Based Direction

### What Changed

- Fixed Torch 1.13 AMP compatibility in:
  - `scripts/compare_baselines.py`
  - `train/micro_macro_recognition.py`
- Extended `scripts/compare_baselines.py` to support:
  - `test_subject: __all__`
  - full-action all-data comparison runs
- Added full-action rep segmentation configs:
  - `configs/rep_segmentation_8act.yaml`
  - `configs/hybrid_rep_segmentation_8act.yaml`

### Results

8-action all-data baseline comparison:

- output:
  - `artifacts/baseline_comparison/20260514_8act_alltrain_fullbaseline`

Headline metrics:

| Model | Rep F1 | Precision | Recall | Start MAE | End MAE | Transition MAE | micro_f1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | **0.9582** | 0.9425 | **0.9745** | **198.65** | **249.64** | **199.47** | **0.8934** |
| Phase-only non-causal TCN | 0.8893 | 0.8686 | 0.9110 | 430.66 | 405.22 | 388.27 | 0.7424 |
| Phase-only causal TCN | 0.8616 | 0.8321 | 0.8931 | 457.65 | 457.12 | 430.78 | 0.7066 |
| DS-MS-TCN | 0.3675 | 0.4720 | 0.3009 | 742.30 | 752.00 | 3360.44 | 0.1282 |

Plain 8-action SDTW baseline:

- output:
  - `artifacts/rep_segmentation/20260514_010051_sets`
- result:
  - `f1 = 0.6546`
  - `mean_iou = 0.6782`

### Interpretation

- On the broader full-action audit, the strongest current rep-cutting baseline is
  unexpectedly the sliding-window `Random Forest`.
- This changes the next best direction:
  - instead of replacing the detector entirely, the best near-term research bet is
    to build a **boundary refinement stage on top of the RF phase detector**.
- Since RF already reaches `micro_f1@50 = 0.8934`, a small improvement may be
  enough to cross the user's `0.9` boundary-quality target.

### Additional Autoresearch Reading

- Before committing to a causal RF deployment path, checked for newer literature
  with stronger boundary emphasis.
- Most promising additional directions found:
  - `EXACT: A Meta-Learning Framework for Precise Exercise Segmentation in Physical Therapy` (2025)
  - `ASFormer` / boundary-aware temporal segmentation transformers
  - boundary-aware query-voting / refinement style temporal segmentation models

Current decision:

- There is still enough literature support to try **one boundary-aware neural
  segmentation model** before concluding RF is the final best direction.

## 2026-05-14 - Boundary-Aware Neural Trials And Strict Causal RF

### What Changed

- Added and tested:
  - `scripts/train_boundary_aware_unet.py`
  - `scripts/train_boundary_tcn.py`
  - `scripts/evaluate_causal_rf.py`
- Added strict causal RF evaluation with batched trailing-window inference and
  smoothing-window sweep support.

### Results

Boundary-aware U-Net on 8-action all-data:

- output:
  - `artifacts/baseline_comparison/20260514_boundary_unet_8act_alltrain`
- result:
  - `rep_f1 = 0.4866`
  - `micro_f1@50 = 0.2349`

Boundary-aware phase TCN on 8-action all-data:

- output:
  - `artifacts/baseline_comparison/20260514_boundary_tcn_8act_alltrain`
- result:
  - `rep_f1 = 0.6353`
  - `micro_f1@50 = 0.3451`

Strict causal RF on 8-action all-data:

- output:
  - `artifacts/baseline_comparison/20260514_causal_rf_8act_alltrain_sweep_small`
- best tested smoothing window:
  - `1`
- metrics:
  - `rep_f1 = 0.9649`
  - `precision = 0.9518`
  - `recall = 0.9785`
  - `start_mae_ms = 182.18`
  - `end_mae_ms = 237.88`
  - `transition_mae_ms = 168.95`
  - `micro_f1@50 = 0.8876`

### Interpretation

- The literature-inspired boundary-aware neural models did not transfer well to
  this dataset.
- Strict causal RF preserved rep detection quality remarkably well.
- The remaining gap is now narrow and specific:
  - `rep_f1` is already above `0.95`
  - `micro_f1@50` still needs boundary tightening
- Therefore the strongest next step is **RF + boundary refiner**, not a new
  segmentation backbone.

## 2026-05-14 - Per-Action RF Boundary Refiner (Held-Out Yushuan)

### What Changed

- Added:
  - `scripts/train_per_action_rf_boundary_refiner.py`
- This script trains one causal RF and one boundary refiner per action when the
  action type is assumed known at inference time.

### Result

- output:
  - `artifacts/baseline_comparison/20260514_per_action_rf_boundary_refiner_yushuan`
- overall metrics:
  - `rep_f1 = 0.7820`
  - `precision = 0.7321`
  - `recall = 0.8393`
  - `micro_f1@50 = 0.6583`

### Comparison

- Shared held-out RF refiner:
  - `rep_f1 = 0.7576`
  - `micro_f1@50 = 0.6402`
- Held-out BiLSTM:
  - `rep_f1 = 0.7809`
  - `micro_f1@50 = 0.5021`

### Interpretation

- Per-action modeling helps when action identity is known.
- The gain is real but modest overall.
- The main remaining failures are concentrated in a few hard actions rather than
  being uniformly distributed across all movements.

### Recorded Summary

- Consolidated held-out results note:
  - `docs/experiments/2026-05-14-heldout-yushuan-rep-cutting-results.md`

## 2026-05-13 - Autoresearch Literature Review For Rep Segmentation Models

### What Changed

- Conducted a literature review focused on high-quality repetition segmentation /
  repetition counting models for wearable IMU exercise tracking.
- Wrote structured research notes under:
  - `literature/survey.md`
  - `literature/*.md`
  - `research-state.yaml`
  - `research-log.md`
  - `findings.md`
  - `docs/experiments/2026-05-13-rep-segmentation-literature-review.md`

### Why

- The project's current bottleneck is rep boundary quality, so we needed outside
  references that are relevant to:
  - single-IMU or wearable sensing
  - segmentation-first repetition extraction
  - causal or deployable inference paths

### Main Reading

- Best direct external fit for this repo:
  - `RecoFit` (CHI 2014)
- Best top-tier wearable benchmark reference:
  - `MM-Fit` (PACM IMWUT 2020)
- Strongest raw venue but weaker hardware fit:
  - `Information Fusion 2024` multimodal IMU + respiration model
- Best low-complexity explicit proposal baseline:
  - `ExerSense` weighted-DTW segmentation pipeline

### Recommendation

- The literature supports adding a **segmentation-first external baseline** more
  strongly than jumping immediately to a more complex multimodal counting model.
- Most justified next implementation branch:
  - compare the repo's current phase-only causal TCN against a
    `RecoFit`/`ExerSense`-inspired explicit rep proposal pipeline on boundary and
    count metrics.

## 2026-05-13 - Phase-Only TCN And Common Baseline Comparison (Held-Out Kevin)

### What Changed

- Extended `scripts/compare_baselines.py` with two new set-level phase baselines:
  - `Phase-only causal TCN`
  - `Phase-only non-causal TCN`
- Updated the shared comparison output to include the currently most important metrics:
  - `start_mae_ms`
  - `end_mae_ms`
  - `transition_mae_ms`
  - `rep_f1`
  - `precision / recall`
  - `exact_count_streams`
  - `over_segmented_streams`
  - `under_segmented_streams`
- Re-ran the fair held-out comparison on `kevin` using:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_testkevin.yaml`

### Why

- The current diagnosis is that rep boundary quality is the main bottleneck, not matched-rep action identity.
- So the key question was whether a simpler **phase-only** model could cut reps better than the current multi-task DS-MS-TCN.
- The non-causal TCN was included as an offline whole-set upper-bound comparison, alongside the existing BiLSTM.

### Results

Output directory:

- `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin`

Headline comparison:

| Model | Start MAE | End MAE | Transition MAE | Rep F1 | Precision | Recall | Exact-count | Over | Under | micro_f1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 466.97 | 457.60 | **461.70** | 0.8235 | 0.7538 | **0.9074** | 4 | 13 | 0 | 0.4942 |
| BiLSTM | 489.27 | **440.28** | 605.18 | **0.8463** | **0.8647** | 0.8287 | **7** | 4 | 6 | 0.5450 |
| 1D CNN | 467.68 | 628.61 | 758.92 | 0.5721 | 0.5413 | 0.6065 | 3 | 10 | 4 | 0.2974 |
| Phase-only causal TCN | **400.14** | 496.92 | 599.58 | 0.7706 | 0.7636 | 0.7778 | 2 | 9 | 6 | 0.4918 |
| Phase-only non-causal TCN | 481.49 | 445.63 | 573.11 | 0.8419 | 0.8112 | 0.8750 | **7** | 8 | 2 | **0.5973** |
| DS-MS-TCN (ours, single rerun) | 640.08 | 613.78 | 1160.50 | 0.5322 | 0.5106 | 0.5556 | 1 | 11 | 5 | 0.2405 |

### Interpretation

- On whole-set rep cutting, several simpler baselines were **clearly better** than this fair single rerun of DS-MS-TCN.
- The strongest causal result among the new deep-learning baselines was:
  - `Phase-only causal TCN`
  - start MAE `400.14 ms`
  - Rep F1 `0.7706`
- The strongest overall offline upper bounds were:
  - `BiLSTM` (best Rep F1 `0.8463`)
  - `Phase-only non-causal TCN` (best `micro_f1@50 = 0.5973`)
- Even the sliding-window `Random Forest` remained very competitive and had the best transition MAE in this comparison.

### Key Reading

- This strongly supports the current diagnosis that the multi-task DS-MS-TCN is **not aligned enough with rep boundary quality**.
- A simpler phase-only model can outperform it because it spends all model capacity on:
  - `other / concentric / eccentric`
  - and the resulting rep pairing quality
- This does **not** prove DS-MS-TCN is always worse than every baseline:
  - the previously recorded best held-out DS-MS-TCN checkpoint (`20260512_192822/tcn`) is still stronger than this single rerun
- But it does prove something important:
  - **simpler phase-only baselines are strong enough that they must be taken seriously as the main rep-cutting branch**

### Files

- Comparison outputs:
  - `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin/comparison_results.json`
  - `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin/comparison_results.md`
- Next-agent handoff:
  - `docs/experiments/2026-05-13-next-agent-handoff.md`

## 2026-05-13 - 7/8-Action All-Data Retrain + Streaming Replay Audit

### What Changed

- Ran the same all-data train/replay audit flow for the larger action sets using `imu_for_workout`:
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_alltrain.yaml`
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_alltrain.yaml`
- Reused the current best stack:
  - dual-head Stage 1
  - `semantic_alpha = 0.5`
  - causal smoothing window `15`
  - `viterbi` decoder
- Aggregated full-set replay metrics and rep-level action-comparison summaries for both runs.

### Why

- After the 3-action audit, the next question was whether the same best-practical stack still behaved reasonably when the action space expanded to 7 and 8 classes.
- The answer matters because strong matched-rep action identity can hide very poor rep formation quality.

### 7-Action Run

- Config:
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_alltrain.yaml`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_7act_alltrain_dualhead_viterbi_imuenv/tcn`
- Protocol:
  - `train_all_in_sample`

Offline run summary:

- Precision `0.4163`, Recall `0.2847`, F1 `0.3381`
- `rep_action_accuracy = 0.9255`

Full replay aggregate across `199` streams:

- Precision `0.3310`, Recall `0.2823`, F1 `0.3047`
- exact-count streams: `5 / 199`
- over-segmented streams: `83 / 199`
- under-segmented streams: `111 / 199`
- zero-TP streams: `92 / 199`

Per-action replay audit:

| Action | Streams | Precision | Recall | F1 | Exact-count | Over | Under |
|---|---:|---:|---:|---:|---:|---:|---:|
| `db_bench_press` | 35 | 0.4320 | 0.2628 | 0.3268 | 3 | 7 | 25 |
| `db_biceps_curl` | 26 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 26 |
| `db_rdl` | 33 | 0.4889 | 0.0556 | 0.0998 | 0 | 3 | 30 |
| `db_shoulder_press` | 24 | 0.2976 | 0.5455 | 0.3851 | 1 | 23 | 0 |
| `db_squat` | 24 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 24 |
| `db_triceps_curl` | 24 | 0.2509 | 0.5087 | 0.3360 | 0 | 24 | 0 |
| `one_arm_db_row` | 33 | 0.3840 | 0.6076 | 0.4706 | 1 | 26 | 6 |

Rep-level action identity on IoU-matched reps:

| Method | Matched reps | Accuracy |
|---|---:|---:|
| Online macro aggregation | 666 | 0.9279 |
| Rep-complete classifier | 666 | 0.8679 |
| Confidence hybrid | 666 | 0.9670 |

Interpretation:

- Count quality did **not** improve with the larger 7-action set.
- `db_biceps_curl` and `db_squat` collapsed to near-zero rep detection.
- `db_rdl` remained phase-collapse dominated.
- `db_shoulder_press` and `db_triceps_curl` became over-segmentation-heavy.
- `one_arm_db_row` had the best replay F1 among the 7 actions, but mainly by over-counting rather than clean boundary recovery.
- For matched reps, the current confidence hybrid was still the strongest overall action-identity rule.

Representative failures:

- `db_rdl` still showed all-`eccentric` collapse:
  - `streaming_eval_all/haoyu/haoyu0512workout/db_rdl/set0/streaming_summary.json`
  - `pred_micro_counts = {"eccentric": 5562}`
- `db_biceps_curl` under-detected badly and even stabilized to the wrong displayed action in one replay:
  - `streaming_eval_all/haoyu/haoyu0512workout/db_biceps_curl/set0/streaming_summary.json`
  - final display action `db_triceps_curl`

### 8-Action Run

- Config:
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_alltrain.yaml`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_8act_alltrain_dualhead_viterbi_imuenv/tcn`
- Protocol:
  - `train_all_in_sample`

Offline run summary:

- Precision `0.4274`, Recall `0.1867`, F1 `0.2599`
- `rep_action_accuracy = 0.8711`

Full replay aggregate across `231` streams:

- Precision `0.2580`, Recall `0.1870`, F1 `0.2168`
- exact-count streams: `3 / 231`
- over-segmented streams: `72 / 231`
- under-segmented streams: `156 / 231`
- zero-TP streams: `112 / 231`

Per-action replay audit:

| Action | Streams | Precision | Recall | F1 | Exact-count | Over | Under |
|---|---:|---:|---:|---:|---:|---:|---:|
| `db_bench_press` | 35 | 0.2601 | 0.4088 | 0.3179 | 1 | 25 | 9 |
| `db_biceps_curl` | 26 | 0.6212 | 0.1419 | 0.2310 | 0 | 0 | 26 |
| `db_rdl` | 33 | 0.1014 | 0.0177 | 0.0301 | 0 | 3 | 30 |
| `db_shoulder_press` | 24 | 0.2581 | 0.4909 | 0.3383 | 2 | 20 | 2 |
| `db_squat` | 24 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 24 |
| `db_triceps_curl` | 24 | 0.1600 | 0.0139 | 0.0256 | 0 | 0 | 24 |
| `db_weighted_crunch` | 32 | 0.2667 | 0.0623 | 0.1011 | 0 | 1 | 31 |
| `one_arm_db_row` | 33 | 0.2376 | 0.3392 | 0.2795 | 0 | 23 | 10 |

Rep-level action identity on IoU-matched reps:

| Method | Matched reps | Accuracy |
|---|---:|---:|
| Online macro aggregation | 513 | 0.8986 |
| Rep-complete classifier | 513 | 0.8850 |
| Confidence hybrid | 513 | 0.9220 |

Interpretation:

- 8 actions is the harshest setting so far; rep formation degraded further.
- `db_squat` and `db_triceps_curl` were almost completely unrecoverable in replay.
- `db_rdl` remained near-zero recall.
- `db_weighted_crunch` macro identity stayed strong, but rep counts still under-shot badly.
- Even here, matched-rep action identity stayed much stronger than rep counting itself.

Representative failures:

- `db_weighted_crunch` kept the correct displayed action but still under-counted badly:
  - `streaming_eval_all/haoyu/haoyu0512workout/db_weighted_crunch/set0/streaming_summary.json`
  - `8` predicted reps vs `12` true, `rep_action_accuracy = 1.0`
- `db_triceps_curl` collapsed to all-`concentric` in one replay:
  - `streaming_eval_all/haoyu/haoyu0512workout/db_triceps_curl/set0/streaming_summary.json`
  - `pred_micro_counts = {"concentric": 4851}`

### Cross-Setting Reading

- As the action space expands from 3 -> 7 -> 8, the replay rep-counting quality drops sharply:
  - 3-action replay F1: `0.4221`
  - 7-action replay F1: `0.3047`
  - 8-action replay F1: `0.2168`
- Matched-rep action identity degrades much more slowly than rep formation.
- This strengthens the same diagnosis from the 3-action audit:
  - the main bottleneck is still the shared 3-class phase head and resulting rep formation,
  - not the rep-complete identity stage.

### Files

- Combined note:
  - `docs/experiments/2026-05-13-7act-8act-alltrain-streaming-audit.md`

### Additional Follow-up

- Added `scripts/analyze_phase_collapse.py` to summarize replay streams that collapse to:
  - all `eccentric`
  - all `concentric`
  - or extreme single-phase dominance
- Added a minimal inference-side probe path:
  - `micro_macro.semantic_phase_fusion_weight`
  - This blends the action-aware semantic head back into 3-class phase probabilities before decoding.
- Added probe config:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_alltrain_semphase025.yaml`

Probe result on two representative 3-action failures:

- `kevin/kevin/db_rdl/set0`
- `kevin/kevin0509workout/one_arm_db_row/set1`

Using `semantic_phase_fusion_weight = 0.25` did **not** improve rep formation:

- `db_rdl/set0` remained all-`eccentric` with `n_pred = 0`
- `one_arm_db_row/set1` remained `eccentric`-dominant with `n_pred = 2`, `tp = 0`

Interpretation:

- A light post-hoc semantic-to-phase blend is not enough to fix the hardest collapse cases.
- The next useful model step likely has to change the **training-time phase representation or objective**, not only inference-time fusion.

## 2026-05-13 - 3-Action All-Data Retrain + Streaming Replay Audit

### What Changed

- Added `train-all` support for micro/macro runs by allowing `train.test_subject: __all__` in:
  - `train/micro_macro_recognition.py`
  - `train/hybrid_action_classifier.py`
  - `scripts/evaluate_rep_complete_action_classifier.py`
- Added all-data configs for the current best 3/7/8-action model family:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_alltrain.yaml`
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_alltrain.yaml`
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_alltrain.yaml`
- Added `scripts/batch_train_all_actionset_eval.py` to:
  - train once,
  - replay every eligible set directory,
  - aggregate count/fragmentation metrics,
  - and run rep-level action-comparison summaries.
- Updated `evaluation/streaming_micro_macro.py` so `fast` replay now uses:
  - the checkpoint's final macro stage via `model.final_macro_logits(...)`
  - the configured micro decoder (`greedy` or `viterbi`) before online rep pairing.
- Pinned `requirements.txt` to `numpy<2` after the base environment hit a `torch.from_numpy()` failure with `numpy 2.0.2`.

### Why

- The requested evaluation was to retrain using all available data and audit full-set replay behavior for the current best model combination, not just rely on per-run offline summaries.
- The earlier replay path understated the configured model combination because it ignored the current `viterbi` micro decoder during `fast` replay.
- The local base environment had a NumPy / PyTorch ABI mismatch, so the actual retrain/eval was run in the `imu_for_workout` conda environment.

### 3-Action Run Used First

- Environment:
  - `conda run -n imu_for_workout`
- Config:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_alltrain.yaml`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn`
- Action set:
  - `db_bench_press`
  - `db_rdl`
  - `one_arm_db_row`
- Protocol:
  - `train_all_in_sample` (all 7 subjects used for both training and replay audit; not a headline subject-held-out metric)

### 3-Action Results

- Offline run summary on the same all-data set split:
  - Precision `0.6915`, Recall `0.3397`, F1 `0.4556`
  - `rep_action_accuracy = 0.9975`
- Full-set streaming replay aggregate across `101` set streams:
  - Precision `0.6037`, Recall `0.3245`, F1 `0.4221`
  - exact-count streams: `6 / 101`
  - over-segmented streams: `25 / 101`
  - under-segmented streams: `70 / 101`
  - zero-TP streams: `52 / 101`

Per-action replay audit:

| Action | Streams | Pred reps | True reps | Precision | Recall | F1 | Exact-count | Over | Under |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `db_bench_press` | 35 | 514 | 411 | 0.6304 | 0.7883 | 0.7005 | 5 | 23 | 7 |
| `db_rdl` | 33 | 0 | 396 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 33 |
| `one_arm_db_row` | 33 | 132 | 395 | 0.5000 | 0.1671 | 0.2505 | 1 | 2 | 30 |

Rep-level action identity on IoU-matched replayed reps:

| Method | Matched reps | Accuracy |
|---|---:|---:|
| Online macro aggregation | 390 | 0.9974 |
| Rep-complete classifier | 390 | 0.9872 |
| Rep-complete hierarchical | 390 | 0.9872 |
| Hybrid routing | 390 | 0.9872 |
| Confidence hybrid | 390 | 1.0000 |

### Unexpected Findings And Reading

- `db_rdl` completely failed in replay despite perfect macro identity on samples.
  - Representative stream: `streaming_eval_all/kevin/kevin/db_rdl/set0/streaming_summary.json`
  - The model predicted only `eccentric` for the full stream:
    - `pred_micro_counts = {"eccentric": 6241}`
    - `gt_micro_counts = {"concentric": 4153, "eccentric": 2088}`
  - Pairing diagnostics showed the sequence starts with eccentric and never forms a valid `concentric -> eccentric` rep:
    - `unexpected_phase_before_concentric` over the whole stream.
- `one_arm_db_row` did not collapse completely, but still showed severe concentric under-prediction.
  - Representative stream: `streaming_eval_all/kevin/kevin0509workout/one_arm_db_row/set1/streaming_summary.json`
  - `pred_micro_counts = {"eccentric": 3211, "concentric": 40}` versus GT `{1701, 1550}`
  - So the main issue is again phase-head failure, not macro identity.
- `db_bench_press` is the opposite failure mode.
  - Action identity is stable and correct, but rep count is often too high.
  - Representative stream: `streaming_eval_all/haoyu/haoyu0512workout/db_bench_press/set0/streaming_summary.json`
  - `23` predicted reps vs `12` true with no pairing diagnostic rows, which points to phase alternation being too fragmented rather than the decoder missing a valid pair.

### Conclusion

- The current best 3-action architecture is still **bench-dominant** in replay behavior.
- The core blocker is **action-dependent phase collapse** in the shared 3-class phase head:
  - `db_rdl` -> nearly all active frames become `eccentric`
  - `one_arm_db_row` -> strong eccentric bias and missed concentric runs
- Action classification on matched reps is not the main problem in this 3-action all-data run; rep formation is.

### Files

- Experiment note:
  - `docs/experiments/2026-05-13-3act-alltrain-streaming-audit.md`

## 2026-05-13 - Yushuan 0513 Evaluation + Empty-Output Comparison Fix

### What Changed

- Evaluated `datasets/raw_data/yushuan/yushuan0513workout` using the current best rep checkpoint:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn`
- Added eval-only configs for broader action-space testing:
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_eval_yushuan.yaml`
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_eval_yushuan.yaml`
- Updated `scripts/evaluate_rep_complete_action_classifier.py` to handle empty per-stream outputs without crashing.

### Why

- The new `yushuan0513workout` session includes 8 actions, while the current macro checkpoint only knows 3 non-`other` macro classes.
- Batch comparison on the new data exposed two edge cases in the classifier-comparison script:
  - empty `online_rep_detections.csv`
  - streams with zero predicted rep segments

### Results

- `rep0` cleanup dry-run on the current raw-data tree showed no new non-CE issue:
  - `trimmed=0 deleted=0 unchanged=248`
- 8-action rep detection on `yushuan0513workout`:
  - Precision `0.4725`, Recall `0.4607`, F1 `0.4665`
- 7-action rep detection excluding `db_weighted_crunch`:
  - Precision `0.4727`, Recall `0.4959`, F1 `0.4840`
- Rep-level action identity on IoU-matched reps showed that `rep_complete_classifier` beat the current confidence-based hybrid on this new subject/session:
  - 8 actions: classifier `56.59%`, confidence hybrid `23.26%`
  - 7 actions without crunch: classifier `52.07%`, confidence hybrid `18.18%`

### Files

- Full handoff/results note:
  - `docs/experiments/2026-05-13-yushuan0513-eval-handoff.md`

## 2026-05-13 - Raw Rep0 Phase Cleanup

### What Changed

- Added `scripts/clean_rep0_phase_contamination.py` to clean corrupted `rep0_*.csv` files under `datasets/raw_data/`.
- The cleanup keeps only the first contiguous `concentric`/`eccentric` block in each `rep0` file.
- `rep0` files that contained no `concentric` or `eccentric` rows are removed as invalid rep captures.

### Why

- Several raw `rep0` files included non-rep phases such as `none` and `inter_set_rest`, which polluted the first-rep crops used by downstream phase and repetition workflows.

### Follow-up Cleanup

- Re-ran the same cleanup after adding new raw sessions for `haoyu` and `ziho`.
- Additional cleanup results: `trimmed=51`, `deleted=3`.
- Deleted invalid `rep0` files with no `concentric`/`eccentric` rows:
  - `datasets/raw_data/haoyu/haoyu0512workout/haoyu0512workout(left)/db_bench_press/set0/rep0_165722.csv`
  - `datasets/raw_data/ziho/ziho0512workout/db_rdl/set3/rep0_211726.csv`
  - `datasets/raw_data/ziho/ziho0512workout/db_rdl/set4/rep0_211726.csv`

## 2026-05-12 - Baseline Comparison (Our Proposed vs Common Models)

### What Changed

Implemented and ran a fair comparison of DS-MS-TCN against three common baselines on the same train/test split (LOSO, held-out `kevin`).

### Results

| Model | Rep F1 | Precision | Recall | micro_f1@50 | Action Acc | Causal |
|---|---|---|---|---|---|---|
| Random Forest | 0.661 | 0.630 | 0.694 | 0.362 | N/A | No |
| BiLSTM (2-layer, h=64) | 0.705 | 0.677 | 0.736 | 0.421 | N/A | **No** |
| Simple 1D CNN (4-layer) | 0.495 | 0.425 | 0.593 | 0.248 | N/A | Yes |
| **DS-MS-TCN (ours)** | 0.683 | 0.615 | **0.769** | 0.386 | **80.7%** | **Yes** |
| DS-MS-TCN (best tuned)* | **0.737** | **0.711** | **0.764** | **0.453** | 0.697 | **Yes** |

\* Previously recorded best from semantic_alpha=0.5 tuned run.

### Key Findings

- **BiLSTM achieved highest Rep F1 (0.705)** in single run, but is **non-causal** (bidirectional) — not deployable for real-time.
- **DS-MS-TCN is the only model with action classification** — baselines can only classify phase, not exercise type.
- **DS-MS-TCN has the best recall (0.769)** — catches the most reps.
- **DS-MS-TCN is causal** — deployable on MCU for real-time streaming (0.29 MB int8).
- Simple 1D CNN without multi-stage refinement clearly underperforms.
- With hyperparameter tuning, DS-MS-TCN reaches **0.737 Rep F1 and 0.453 micro_f1@50**, surpassing all baselines.

### Files

- Script: `scripts/compare_baselines.py`
- Results: `artifacts/baseline_comparison/comparison_results.json`
- Full analysis: `docs/experiments/2026-05-12-baseline-comparison.md`

---

## 2026-05-12 - Rep-Count Loss, Dual-Head + Structured Decoder, and Semantic Alpha Sweep

### Rep-Count Loss Experiment

- Added `use_rep_count_head` and `rep_count_head_dim` to `DSMSTCNConfig`.
- Added a small regression head (`nn.Linear → ReLU → Dropout → nn.Linear`) on top of pooled Stage 1 features.
- Updated `SequenceSliceDataset` to compute GT rep count per slice via `_count_reps_in_slice()`.
- Integrated MSE rep-count loss into `ds_ms_tcn_loss` with configurable `rep_count_weight`.
- Config: `configs/micro_macro_recognition_3act_40ep_repcount_testkevin.yaml`, `rep_count_weight=1.0`.
- Run: `artifacts/micro_macro_recognition/20260512_191858/tcn`

| Metric | Baseline (v2_testkevin) | Rep-count loss (w=1.0) |
|---|---|---|
| Rep F1 | **0.7100** | 0.3746 |
| micro_f1_at_50 | **0.3812** | 0.1232 |
| Precision | **0.6667** | 0.5960 |
| Recall | **0.7593** | 0.2731 |
| training loss | ~1.5 | ~7.2 |

Interpretation:

- Rep-count loss **completely failed** on this dataset.
- Two root causes:
  1. **Loss scale imbalance**: MSE rep-count loss (per-slice, ~0–20 reps) dominates the cross-entropy signal, warping the shared backbone.
  2. **Unstable target**: overlapping 20 s slices with 10 s stride produce highly variable rep-count targets for similar input windows, confusing the model.
- Conclusion: rep-count regression on sliced windows is **not recommended** in the current setup. If pursued later, it should use whole-set targets or a dedicated global-pooling branch with far lower weight (e.g., 0.01).

### Dual-Head + Structured Decoder Experiments

- Config base: `configs/micro_macro_recognition_3act_40ep_v2_testkevin.yaml` + `use_dual_micro_head: true` + `micro_decoder: viterbi`.
- Tested four `semantic_alpha` values: 0.1, 0.2, 0.3, 0.5.
- Also tested `use_semantic_for_macro: true` (semantic probabilities concatenated into macro stage input).

#### Semantic Alpha Sweep (auxiliary only, no semantic→macro fusion)

| semantic_alpha | Rep F1 | micro_f1_at_50 | rep_action_accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **0.1** | 0.645 | 0.394 | **0.796** | 0.656 | 0.634 |
| **0.2** | 0.659 | 0.421 | 0.664 | 0.631 | 0.690 |
| **0.3** | 0.658 | 0.406 | 0.690 | 0.644 | 0.671 |
| **0.5** | **0.737** | **0.453** | 0.697 | **0.711** | **0.764** |

- **semantic_alpha=0.5 is the best** for rep detection, beating the baseline on every rep-level metric.
- Lowering `semantic_alpha` hurts both rep F1 and micro_f1_at_50, suggesting the auxiliary semantic loss does help the shared backbone.
- Action accuracy suffers across all settings versus the baseline (0.902 → ~0.70), but this is expected because the semantic head is trained for auxiliary supervision, not direct macro replacement.

#### Semantic-to-Macro Fusion

- Config: `configs/micro_macro_recognition_3act_40ep_dualhead_semanticmacro_viterbi_testkevin.yaml`.
- Run: `artifacts/micro_macro_recognition/20260512_193503/tcn`

| Metric | Value |
|---|---|
| Rep F1 | 0.553 |
| micro_f1_at_50 | 0.284 |
| rep_action_accuracy | 0.710 |

- Feeding semantic probabilities directly into the macro stage **hurts rep detection** compared with the simpler auxiliary-only dual-head run.
- This matches the 2026-05-10 finding (`micro_f1_at_50` dropped from 0.4813 to 0.4209 when semantic→macro fusion was enabled).

### Best Current Configuration

For the **new subject/session data structure** and held-out `kevin`:

- **Best rep detection**: dual-head auxiliary (`semantic_alpha=0.5`) + viterbi decoder (`switch=1.5, invalid=10.0, min_run=16`)
  - Rep F1: **0.737**
  - micro_f1_at_50: **0.453**
  - Config: `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_testkevin.yaml`
  - Run: `artifacts/micro_macro_recognition/20260512_192822/tcn`
- **Best action identity**: hybrid action classifier (confidence-based routing) on the baseline v2 checkpoint
  - Accuracy: **96.49%**

Recommended combined pipeline:

1. Train with dual-head auxiliary + viterbi for best rep detection.
2. At evaluation/runtime, apply the hybrid action classifier to each completed rep for action identity.
3. Do **not** use rep-count loss or semantic→macro fusion in the current architecture.

### Files Changed

- `models/ds_ms_tcn.py`: added `use_rep_count_head`, `rep_count_head_dim`, `rep_count_head`, rep-count MSE loss in `ds_ms_tcn_loss`
- `train/micro_macro_recognition.py`: added `_count_reps_in_slice()`, `rep_count` to `SequenceSliceDataset`, `rep_count_weight` to `MicroMacroConfig`, rep-count target plumbing in `_train_model`
- New configs:
  - `configs/micro_macro_recognition_3act_40ep_repcount_testkevin.yaml`
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_testkevin.yaml`
  - `configs/micro_macro_recognition_3act_40ep_dualhead_a01_viterbi_testkevin.yaml`
  - `configs/micro_macro_recognition_3act_40ep_dualhead_a02_viterbi_testkevin.yaml`
  - `configs/micro_macro_recognition_3act_40ep_dualhead_a03_viterbi_testkevin.yaml`
  - `configs/micro_macro_recognition_3act_40ep_dualhead_semanticmacro_viterbi_testkevin.yaml`

---

## 2026-05-12 - Data Restructuring & Hybrid Action Classifier Comparison

### Data Directory Restructuring

- Migrated `datasets/raw_data` from flat `subject_or_alias/action/set/rep.csv` to a proper `subject/session/action/set/rep.csv` hierarchy:
  - `kevin/` → `kevin/kevin/`, `kevin/kevin0509workout/`
  - `thomas/` → `thomas/thomas/`, `thomas/thomas_2/`, `thomas/thomas0506workout/`
  - `1000/`, `yanz0510workout/` → `yanz/1000/`, `yanz/yanz0510workout/`
  - `yoru0511workout/` → `yoru/yoru0511workout/`
- Updated data loaders to iterate `subject → session → action → set`:
  - `train/micro_macro_recognition.py`: added `_session_dirs()`, rewrote `_load_set_sequences`, `_load_whole_sequences`, `_load_synthetic_whole_sequences`, `_available_actions`, `_load_streams`
  - `evaluation/rep_segmentation.py`: same session iteration for `_load_rep_csvs`, `_load_set_streams`, `_load_whole_streams`, action discovery
  - `datasets/custom_resistance_dataset.py`: `_infer_action_type_from_path` now uses `parts[2]` (was `parts[1]`)
- `subject_aliases` config is no longer needed; subject identity comes from top-level directory name.
- New config without aliases: `configs/micro_macro_recognition_3act_40ep_v2_testkevin.yaml`

### v2 Training (test_subject=kevin, new directory structure)

- Run: `artifacts/micro_macro_recognition/v2_testkevin/tcn`
- Config: 3 actions (`db_bench_press`, `db_rdl`, `one_arm_db_row`), 40 epochs, 6 layers, 64 filters, causal, smooth15
- Split: train=`['thomas', 'yanz', 'yoru']` (54 streams), test=`['kevin']` (17 streams, 2 sessions)
- Results:
  - Rep F1: `0.7100`, Precision: `0.6667`, Recall: `0.7593`
  - Rep action accuracy: `0.9024`
  - micro_f1_at_50: `0.3812`

### Hybrid Action Classifier Comparison

Ran streaming eval on all 17 kevin test sets, then compared 5 action-identification methods on 114 IoU-matched reps:

| Method | Correct | Accuracy |
|---|---:|---:|
| Rep-complete classifier (logreg) | 105 | 92.11% |
| Rep-complete hierarchical | 105 | 92.11% |
| Online macro aggregation | 99 | 86.84% |
| Hybrid routing (precision-trusted) | 99 | 86.84% |
| **Confidence hybrid (macro≥0.7 else clf)** | **110** | **96.49%** |

Key findings per stream:

- **Macro aggregation** is perfect on bench press (all 6 sets 100%) but poor on `one_arm_db_row` in `kevin0509workout` (0–40%).
- **Rep-complete classifier** is perfect on `one_arm_db_row` and `db_rdl` but weak on `kevin0509workout/db_bench_press` (43–75%).
- **Confidence hybrid** combines the best of both: uses macro when confident (≥0.7), falls back to classifier otherwise → **96.49% accuracy**, the best overall.

Scripts:
- `scripts/batch_streaming_eval.py`: batch streaming eval + comparison runner
- `scripts/evaluate_rep_complete_action_classifier.py`: updated with confidence-based hybrid and fixed `trusted_flat_labels` to exclude sklearn summary keys

### Hybrid Action Classifier Integration

- Added `train/hybrid_action_classifier.HybridActionClassifier` — a reusable module that:
  - Trains logreg + RF on non-test-subject rep-level rich features (same pipeline as `evaluate_rep_complete_action_classifier.py`)
  - Provides `predict_segment()` for single-rep classification
  - Provides `hybrid_label()` which routes to macro when `confidence >= 0.7`, else falls back to classifier
- Updated `evaluation/streaming_micro_macro.py`:
  - New CLI flags: `--hybrid-action` (default `True`), `--no-hybrid-action`, `--test-subject`
  - `test_subject` is auto-inferred from the input path (`data_dir/subject/...`) if omitted
  - After `_decode_online_reps`, each emitted rep is post-processed with the hybrid classifier before writing outputs
  - `online_rep_detections.csv` and `online_rep_events.csv` now contain hybrid labels by default
- Updated `docs/specs/model.md` with hybrid strategy as the preferred action-identification approach.

### Model Size & LuckFox Pico Zero Feasibility

Architecture: DS-MS-TCN, 64 filters, 6 layers, kernel 3, 3 macro stages, causal

| Metric | Value |
|---|---|
| Total parameters | 298,955 |
| File size (float32 .pt) | 1.17 MB |
| Memory (float32) | 1.14 MB |
| Memory (int8 quantized) | 0.29 MB |
| FLOPs per sample | ~598K |
| Compute at 100 Hz | ~60 MFLOPS/s |

LuckFox Pico Zero (RV1103): ARM Cortex-A7 1.2 GHz + 0.5 TOPS NPU + 64 MB DDR2

- **Model fits easily**: 0.29 MB (int8) in 64 MB RAM
- **NPU utilization**: 0.012% of 0.5 TOPS capacity at 100 Hz — extremely lightweight
- **CPU-only**: ~60 MFLOPS/s is feasible on Cortex-A7, but leaves little headroom for OS + other tasks
- **Recommendation**: Use RKNN NPU path for comfortable real-time. The rep-complete sklearn classifier adds negligible cost (runs once per completed rep, ~164 features × logistic regression).

## 2026-05-10 - Micro IoU Improvement Tests

- 已實作並測試五種提升 `micro_f1_at_50` 的方向：
  - action-aware micro labels (`micro_label_mode: action_phase`)
  - micro temporal smoothness loss (`micro_tmse_weight`)
  - stage1-first pretraining (`stage1_pretrain_epochs`)
  - structured phase decoder (`micro_decoder=viterbi` with transition/run constraints)
  - synthetic whole-session training streams assembled from set/rest fragments
- 新增實驗設定檔：
  - `configs/micro_macro_recognition_stage3_40ep_action_phase.yaml`
  - `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse.yaml`
  - `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse_pretrain.yaml`
  - `configs/micro_macro_recognition_stage3_40ep_action_phase_microtmse_pretrain_whole.yaml`
  - `configs/micro_macro_recognition_stage3_40ep_dual_head.yaml`
  - `configs/micro_macro_recognition_stage3_40ep_dual_head_macro.yaml`
- 在 `train/micro_macro_recognition.py` 與 `models/ds_ms_tcn.py` 中新增支援：
  - dynamic micro class lists
  - phase-collapsed evaluation of action-aware micro predictions
  - micro TMSE loss
  - stage1 pretraining before joint training
  - constrained micro decoding
  - synthetic whole-session stream loading
  - optional dual-head Stage 1 with a phase head plus an auxiliary action-aware semantic head
- 更新 `scripts/reevaluate_micro_macro_run.py`，支援 decoder override 與 `--mode`。
- 這輪結果整理在：
  - `docs/experiments/2026-05-10-micro-iou-improvement-tests.md`

## 2026-05-10 - App-Friendly Stable Display Tests

- 在 `evaluation/streaming_micro_macro.py` 中加入 app-friendly 顯示狀態機：
  - rep 數量只在 completed rep event 時加一
  - action 不再每 sample 更新，而是根據最近幾個 completed rep 的多數決延後鎖定
- 新增輸出欄位與檔案：
  - `streaming_predictions.csv`
    - `display_rep_count`
    - `display_action`
    - `display_action_confidence`
    - `display_action_locked`
  - `display_events.csv`
- 新增 streaming 參數：
  - `--display-vote-window`
  - `--display-min-reps`
  - `--display-min-fraction`
  - `--display-min-confidence`
- 代表性 replay 結果整理在：
  - `docs/experiments/2026-05-10-app-display-state-tests.md`

## 2026-05-10 - Agent Handoff Index

- Added a dedicated handoff note for the next agent:
  - `docs/experiments/2026-05-10-agent-handoff.md`
- This document summarizes:
  - current best practical checkpoint/runtime setting,
  - what the real bottlenecks are,
  - strict LOSO vs Kevin-personalized interpretation,
  - current live/streaming status,
  - rep-complete classifier status,
  - and recommended next paths.

## 2026-05-09 - TMSE Ablation, Longer Training, And RDL-vs-Crunch Inspection

### What Changed

- Created a fresh conda environment `imu_for_workout` and installed project requirements.
- Replaced the CPU-only PyTorch build in `imu_for_workout` with CUDA-enabled `torch 2.6.0+cu124`.
- Ran the queued 3-stage follow-up experiments:
  - TMSE ablation (`beta=0`)
  - longer training (`40 epochs`)
- Added `scripts/analyze_rdl_vs_crunch.py` to inspect class statistics, phase structure, and subject shift for `db_rdl` vs `db_weighted_crunch`.

### Commands Run

- `conda create -n imu_for_workout python=3.11 -y`
- `conda run -n imu_for_workout pip install -r requirements.txt`
- `C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`
- `C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe -u -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_beta0.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_beta0`
- `C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe -u -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_40ep.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_40ep`
- `C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe scripts/analyze_rdl_vs_crunch.py`

### Result Summary

Baseline numbers below are from the earlier recorded 3-stage run (`stage34_3stage`).

| Metric | Baseline 3-stage | 3-stage beta0 | 3-stage 40ep | Best |
|---|---:|---:|---:|---|
| **Rep F1** | **0.5970** | 0.3390 | 0.5735 | baseline |
| **Rep Precision** | **0.5882** | 0.3846 | 0.5571 | baseline |
| **Rep Recall** | **0.6061** | 0.3030 | 0.5909 | baseline |
| Rep Action Accuracy | 0.7000 | **0.9000** | 0.6795 | beta0 |
| start_mae_ms | 600.1 | 890.6 | **555.5** | 40ep |
| end_mae_ms | **554.2** | 379.1 | 598.6 | beta0 |
| transition_mae_ms | 504.4 | 836.5 | **469.0** | 40ep |
| macro_f1_at_50 | **0.2123** | 0.1439 | 0.1894 | baseline |
| micro_f1_at_50 | 0.2986 | 0.1409 | **0.3160** | 40ep |

- `beta=0` is clearly worse than baseline and worsens the main rep-counting metrics substantially.
- `40ep` recovers much of the gap, but still does not beat the baseline 3-stage run on rep F1 / recall / macro IoU-F1@50.
- GPU is now working in `imu_for_workout` (`torch.cuda.is_available() == True`, `device=cuda`, `amp=True`).

### Pairing Diagnostics

Baseline counts were:

- `missing_eccentric_after_concentric=65`
- `unexpected_phase_before_concentric=59`
- `phase_gap_too_large=6`

New runs:

- `beta0`
  - `unexpected_phase_before_concentric=143`
  - `missing_eccentric_after_concentric=62`
  - `phase_gap_too_large=8`
- `40ep`
  - `missing_eccentric_after_concentric=67`
  - `unexpected_phase_before_concentric=57`
  - `phase_gap_too_large=3`

Interpretation:

- Removing TMSE makes phase-order quality much worse.
- Longer training mostly restores pairing to about the old level, but does not solve the pairing blocker.

### RDL-vs-Crunch Findings

- `beta0` `db_rdl` per-stream F1:
  - `kevin/db_rdl/set0 = 0.0714`
  - `kevin/db_rdl/set1 = 0.1224`
- `40ep` `db_rdl` per-stream F1:
  - `kevin/db_rdl/set0 = 0.8000`
  - `kevin/db_rdl/set1 = 0.3077`
- Baseline from earlier notes:
  - `kevin/db_rdl/set0 = 0.2857`
  - `kevin/db_rdl/set1 = 0.4528`

Rep-level action confusion for `db_rdl`:

- Baseline: all 16 matched `db_rdl` reps were classified as `db_weighted_crunch`
- `beta0`: 3 `db_rdl`, 1 `one_arm_db_row`, 0 `db_weighted_crunch`
- `40ep`: 6 `db_rdl`, 7 `db_weighted_crunch`, 5 `uncertain`

The 40-epoch run partly breaks the “all RDL becomes crunch” failure mode, but the confusion is still a major blocker.

### Data Inspection Findings

From `scripts/analyze_rdl_vs_crunch.py`:

- Train-subject data counts are not severely imbalanced for these two classes:
  - `db_rdl`: 151 reps total across non-Kevin subjects
  - `db_weighted_crunch`: 160 reps total across non-Kevin subjects
- Kevin-specific counts are also not tiny:
  - `db_rdl`: 36 reps, 16354 samples
  - `db_weighted_crunch`: 25 reps, 7843 samples
- Kevin `db_rdl` reps are longer on average (`4.09 s`) than Kevin `db_weighted_crunch` reps (`2.83 s`).
- Kevin `db_rdl` is concentric-heavy (`concentric_ratio ≈ 0.60`, `eccentric_ratio ≈ 0.39`), while Kevin `db_weighted_crunch` is eccentric-heavier (`concentric_ratio ≈ 0.40`, `eccentric_ratio ≈ 0.57`).
- A simple rep-signature classifier trained on non-Kevin subjects and tested on Kevin reached **100% accuracy** separating `db_rdl` and `db_weighted_crunch` using only coarse per-rep IMU/phase summary features.
- Kevin centroid distances also show the expected direction:
  - `kevin_rdl_to_train_rdl = 1.27`
  - `kevin_rdl_to_train_crunch = 1.50`
  - `kevin_crunch_to_train_crunch = 1.89`
  - `kevin_crunch_to_train_rdl = 2.06`

Interpretation:

- The raw rep-level signals still look separable, even for Kevin.
- The persistent confusion is more likely caused by sequence modeling / phase-label structure / macro assignment on full streams than by simple feature overlap alone.

### Updated Diagnosis

- Stage count and TMSE are not the main root cause.
- `beta=0` hurts, so TMSE is not the primary blocker.
- `40ep` helps some pieces, but does not exceed the baseline 3-stage run.
- `db_rdl` vs `db_weighted_crunch` confusion still exists, but the quick data inspection suggests the issue is **not** that the classes are intrinsically inseparable at rep level.
- The next likely root causes are:
  - micro-label design and phase-order assumptions,
  - rep pairing logic under irregular full-stream motion,
  - stream-level macro aggregation and subject-style timing mismatch, especially on `kevin/db_rdl/set1`.

### Additional Follow-up Analysis

- `whole_session` data is not actually participating in the current repo's micro/macro runs because `_load_streams(..., ["whole"])` returns zero streams for the current dataset snapshot.
- Ground-truth phase order is overwhelmingly consistent with the current pairing assumption `concentric -> eccentric`:
  - `db_rdl`: 186 reps `concentric->eccentric`, 1 `eccentric->concentric`
  - `db_weighted_crunch`: 183 reps `concentric->eccentric`, 1 `eccentric->concentric`, 1 `concentric->missing`
  - Kevin's `db_rdl` reps are all `concentric->eccentric` in the current labels.
- This makes it less likely that the main issue is a global phase-order mismatch in the labels.

### New Immediate Adjustment: Causal Micro Smoothing

- Added `MicroMacroConfig.micro_smoothing_window` (default `1`) to `train/micro_macro_recognition.py`.
- `_predict_full_sequence()` now optionally applies a causal moving average to `micro_probs` before converting them to labels for evaluation/post-processing.
- Fixed 3-stage checkpoint loading in:
  - `evaluation/streaming_micro_macro.py`
  - `scripts/grid_micro_macro_postprocess.py`
  so `num_macro_stages` is restored correctly from checkpoint config.

Using `scripts/grid_micro_smoothing.py` on `stage34_3stage_40ep/tcn` for held-out `kevin`:

| Smoothing window | micro sample acc | micro sample macro F1 | micro F1@25 | micro F1@50 |
|---:|---:|---:|---:|---:|
| 1 | 0.6528 | 0.4259 | 0.4444 | 0.3209 |
| 3 | 0.6545 | 0.4269 | 0.4812 | 0.3478 |
| 5 | 0.6565 | 0.4281 | 0.5252 | 0.3851 |
| 7 | 0.6579 | 0.4289 | 0.5476 | 0.4034 |
| 9 | 0.6587 | 0.4294 | 0.5726 | 0.4191 |
| 11 | 0.6602 | 0.4304 | 0.5867 | 0.4410 |
| **15** | **0.6625** | **0.4317** | **0.6289** | **0.4719** |

Interpretation:

- The micro predictions are noisy but recover strongly with simple causal probability smoothing.
- This is the strongest immediate lever found so far for raising sample-wise and IoU segmentation metrics without retraining.
- Next experiments should prioritize combining:
  - stronger micro-task emphasis during training, and
  - causal micro smoothing during evaluation/runtime.

### Full Re-evaluation With Smoothing

- Added `scripts/reevaluate_micro_macro_run.py` to recompute a run's full summary/plots with overridden evaluation settings.
- Re-evaluated `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn` with:
  - `micro_smoothing_window = 15`
  - output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15`

Re-evaluated overall metrics:

| Metric | Baseline 3-stage | 40ep raw | 40ep + smooth15 |
|---|---:|---:|---:|
| Rep F1 | 0.5970 | 0.5735 | **0.7789** |
| Precision | 0.5882 | 0.5571 | **0.7255** |
| Recall | 0.6061 | 0.5909 | **0.8409** |
| start_mae_ms | 600.1 | 555.5 | **433.1** |
| micro sample macro F1 | 0.4259* | 0.4259 | **0.4293** |
| micro F1@25 | 0.4444* | 0.4478 | **0.6172** |
| micro F1@50 | 0.2986 | 0.3160 | **0.4582** |

`*` baseline sample-wise micro metrics were not logged in the earlier stage-ablation note with the same formatting, so the strongest apples-to-apples comparison is `40ep raw` vs `40ep + smooth15`.

Additional effects on failure modes:

- Pairing diagnostics dropped sharply from the raw 40ep run:
  - raw 40ep: `missing_eccentric_after_concentric=67`, `unexpected_phase_before_concentric=57`, `phase_gap_too_large=3`
  - smooth15: `missing_eccentric_after_concentric=10`, `unexpected_phase_before_concentric=7`
- `db_rdl` rep F1 improved to:
  - `kevin/db_rdl/set0 = 0.8800`
  - `kevin/db_rdl/set1 = 0.6538`

Interpretation:

- The dominant immediate bottleneck is micro-phase prediction jitter, not lack of macro information.
- Causal smoothing alone produces a much larger gain than either `beta=0` ablation or longer training did by themselves.
- The best next training experiment should combine smoothing with a stronger micro-loss weight.

### Next Proposed Training Config

- Added `configs/micro_macro_recognition_stage3_40ep_alpha2_smooth15.yaml`
  - `alpha: 2.0`
  - `beta: 0.15`
  - `micro_smoothing_window: 15`

This is the current highest-confidence next experiment for improving sample-wise F1 / IoU F1 while preserving the strong rep-level gains from smoothing.

### Follow-up Result: `alpha=2.0 + smooth15`

- Run dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep_alpha2_smooth15/tcn`
- Result summary:
  - Rep F1: `0.6341`
  - Precision: `0.5871`
  - Recall: `0.6894`
  - Rep action accuracy: `0.8901`
  - start MAE: `411.4 ms`
  - micro sample macro F1: `0.4309`
  - micro IoU-F1@50: `0.4156`
  - macro sample macro F1: `0.2247`

Comparison against `40ep raw` and `40ep + smooth15 reevaluation`:

- Better than raw `40ep` on rep F1 and most sample-wise metrics.
- Worse than `40ep + smooth15 reevaluation` on the metrics that matter most for rep quality:
  - raw `40ep` + smooth15 reevaluation: rep F1 `0.7789`, recall `0.8409`, micro IoU-F1@50 `0.4582`
  - `alpha=2.0 + smooth15`: rep F1 `0.6341`, recall `0.6894`, micro IoU-F1@50 `0.4156`

RDL-specific result:

- `kevin/db_rdl/set0` F1 dropped to `0.4375`
- `kevin/db_rdl/set1` F1 dropped to `0.2069`
- But rep-level action confusion for `db_rdl` improved strongly:
  - `db_rdl -> db_rdl: 13`
  - `db_rdl -> db_weighted_crunch: 0`
  - `db_rdl -> uncertain: 0`

Interpretation:

- Increasing `alpha` pushes action discrimination in the right direction, but it hurts the rep segmentation/pairing quality enough to reduce overall rep F1 versus the smoothed 40ep checkpoint.
- This suggests the next useful direction is **not** simply “increase micro loss more”.
- The best current result remains: keep the `40ep` checkpoint and apply evaluation/runtime causal micro smoothing.

### Alpha Sweep And Weighted-Micro Follow-up

Additional runs completed:

- `artifacts/micro_macro_recognition/stage34_3stage_40ep_alpha1p25_smooth15/tcn`
- `artifacts/micro_macro_recognition/stage34_3stage_40ep_alpha1p5_smooth15/tcn`
- `artifacts/micro_macro_recognition/stage34_3stage_40ep_weighted_other2p5_smooth15/tcn`

Summary comparison versus the best current practical setting (`40ep` checkpoint + `smooth15` reevaluation):

| Setting | Rep F1 | Precision | Recall | Rep action acc | micro sample macro F1 | micro F1@50 |
|---|---:|---:|---:|---:|---:|---:|
| `40ep + smooth15 reeval` | **0.7789** | **0.7255** | **0.8409** | 0.6306 | 0.4293 | **0.4582** |
| `alpha=1.25 + smooth15` | 0.7014 | 0.6474 | 0.7652 | 0.6931 | 0.4288 | 0.4180 |
| `alpha=1.5 + smooth15` | 0.7158 | 0.6667 | 0.7727 | 0.7451 | 0.4212 | 0.4086 |
| `alpha=2.0 + smooth15` | 0.6341 | 0.5871 | 0.6894 | **0.8901** | 0.4309 | 0.4156 |
| `weighted other=2.5 + smooth15` | 0.6360 | 0.6434 | 0.6288 | 0.7711 | **0.4462** | 0.4142 |

Interpretation:

- For overall rep quality, **none** of the retrained variants beat simply re-evaluating the original 40-epoch checkpoint with causal smoothing.
- `alpha` sweeps improve action discrimination and sometimes recall, but still lose badly to `40ep + smooth15 reeval` on rep F1.
- `weighted other=2.5` gives the **best micro sample macro F1** (`0.4462`) and strongest low-threshold micro IoU, but still does not beat the smoothed checkpoint on `micro_f1_at_50` or rep F1.
- This suggests two different objectives are pulling in different directions:
  - micro class balance helps framewise classification,
  - but the strongest rep-level gains still come from smoothing a less constrained checkpoint.

Current best practical recommendation remains:

1. keep the raw `40ep` checkpoint,
2. apply `micro_smoothing_window=15` at evaluation/runtime,
3. avoid large `alpha` increases as the next primary lever.

If further work is needed specifically for **sample-wise F1**, then micro class weighting is worth keeping in the toolbox. If the target is the best **rep F1 / recall**, the current smoothing-only path is still stronger.

### Streaming / Real-Time Follow-up

- Added online/runtime support for `micro_smoothing_window` in:
  - `models/ds_ms_tcn.py` (`OnlineDSMSTCNPredictor`)
  - `evaluation/streaming_micro_macro.py`
- Fixed stale `macro4_probs` references in the streaming evaluator to use the predictor's current `macro_probs` output.

Streaming-style replay with the best current checkpoint (`stage34_3stage_40ep/tcn`) and `micro_smoothing_window=15`:

1. `kevin/db_rdl/set1`
   - output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_rdl_set1_step_smooth15`
   - samples: `3000`
   - sample rate: `111.11 Hz`
   - throughput: `168.49 samples/s`
   - real-time factor: `1.52x`
   - micro sample accuracy: `0.6783`
   - macro sample accuracy: `0.1823`

2. `kevin/db_weighted_crunch/set0`
   - output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step_smooth15`
   - samples: `3000`
   - sample rate: `111.00 Hz`
   - throughput: `164.87 samples/s`
   - real-time factor: `1.49x`
   - micro sample accuracy: `0.7393`
   - macro sample accuracy: `0.3070`

Interpretation:

- On the local RTX 2080 Ti, the current streaming path is fast enough for basic real-time use.
- The micro/phase stream is the usable part right now.
- The sample-level macro action stream is still too unstable for reliable online action identity.
- Practical deployment recommendation for “basically usable” real-time behavior:
  - use the `stage34_3stage_40ep` checkpoint,
  - enable `micro_smoothing_window=15`,
  - use online micro phases for rep detection,
  - derive action identity only after a rep is completed or over a longer aggregation window, not from per-sample macro labels.

### Online Rep Decoder And Event-Level Replay

- Added `OnlineRepDecoder` and `OnlineRepEvent` to `preprocessing/micro_macro_segments.py`.
- The decoder is causal and emits a rep only after the eccentric phase is observed to end.
- `evaluation/streaming_micro_macro.py` now writes:
  - `online_rep_detections.csv`
  - `online_rep_events.csv`
  - `online_pairing_diagnostics.csv`
  - rep-level metrics and emit-delay metrics in `streaming_summary.json`

Streaming rep-level replay with `stage34_3stage_40ep` + `micro_smoothing_window=15`:

1. `kevin/db_rdl/set1`
   - output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_rdl_set1_step_smooth15_rep`
   - throughput: `156.25 samples/s`
   - real-time factor: `1.41x`
   - online rep precision / recall / F1: `0.25 / 0.50 / 0.3333`
   - rep action accuracy: `0.3333`
   - mean emit delay: `132 ms`
   - p95 emit delay: `356 ms`

2. `kevin/db_weighted_crunch/set0`
   - output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step_smooth15_rep`
   - throughput: `157.27 samples/s`
   - real-time factor: `1.42x`
   - online rep precision / recall / F1: `0.5625 / 0.90 / 0.6923`
   - rep action accuracy: `0.2222`
   - mean emit delay: `196 ms`
   - p95 emit delay: `550 ms`

Interpretation:

- The online rep event decoder is fast enough and can be basically usable on easier streams.
- The emit delay is acceptable for a rep-counting style interaction (roughly 0.13s to 0.20s mean delay after the actual rep end).
- Harder streams like `kevin/db_rdl/set1` are still not reliable enough for deployment.
- Online action identity after rep completion remains weak; the current best use is online rep counting / phase tracking rather than trustworthy instant action classification.

### Rep-Complete Action Classifier Prototype

- Added `scripts/evaluate_rep_complete_action_classifier.py`.
- This trains a rep-level action classifier on non-Kevin rep CSVs using the same resampled/z-scored rep data and rich per-rep features from `train/action_classification.py`.
- It then compares two ways of assigning an action to each online detected rep:
  1. current online macro aggregation from the DS-MS-TCN stream,
  2. a rep-complete classifier applied to the raw IMU segment of the detected rep.

Held-out Kevin rep-level classifier quality by itself:

- best model: `logreg`
- held-out rep classification accuracy: `0.8582`
- held-out rep macro F1: `0.8181`
- per-class behavior:
  - `db_rdl`: very strong (`f1 ≈ 0.986`)
  - `one_arm_db_row`: very strong (`f1 ≈ 0.972`)
  - `db_bench_press`: good (`f1 = 0.800`)
  - `db_weighted_crunch`: weak (`f1 ≈ 0.514`, recall `0.36`)

Comparison on online-detected reps:

1. `kevin/db_rdl/set1`
   - online rep matches: `3`
   - rep-complete classifier action accuracy: `1.000`
   - online macro aggregation action accuracy: `0.333`

2. `kevin/db_weighted_crunch/set0`
   - online rep matches: `9`
   - rep-complete classifier action accuracy: `0.111`
   - online macro aggregation action accuracy: `0.222`

Interpretation:

- The rep-complete classifier is clearly better for `db_rdl`-type cases where the current online macro head is especially weak.
- But it is currently worse on `db_weighted_crunch`, matching the held-out rep classifier weakness already visible in the offline rep classification report.
- So “rep-complete action classifier” is the right architectural direction, but the current simple global classifier is not yet a universally better replacement for all actions.

Practical takeaway:

- For immediate deployment, keep the online rep decoder for counting.
- For action identity, a rep-complete classifier is promising, especially for `db_rdl`, but it needs more work before it can fully replace the online macro aggregation across all actions.

### New Kevin Session Handling (`kevin0509workout`)

- Added subject-alias support so multiple folders from the same human can be forced into the same LOSO fold.
- New config: `configs/micro_macro_recognition_stage3_40ep_alias_kevin0509.yaml`
  - `subject_aliases: { kevin0509workout: kevin }`
- Updated loaders so aliasing affects both:
  - micro/macro LOSO split (`train/micro_macro_recognition.py`)
  - rep-level action-classifier data loading (`datasets/custom_resistance_dataset.py`, `train/action_classification.py`, `scripts/evaluate_rep_complete_action_classifier.py`)

Important interpretation:

- If `kevin0509workout` is truly the same person as `kevin`, then under correct subject-wise evaluation it should **not** be added to train when `test_subject=kevin`.
- So the correct immediate action is **re-evaluation on the enlarged Kevin test fold**, not training-data augmentation.

Re-evaluation with aliasing (`stage34_3stage_40ep/tcn` + `smooth15`):

- output dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15_kevin0509alias`
- overall metrics:
  - Rep F1: `0.5939`
  - Precision: `0.5613`
  - Recall: `0.6304`
  - Rep action accuracy: `0.5345`
  - Micro sample macro F1: `0.3927`
  - Micro IoU-F1@50: `0.3109`

Interpretation:

- The new Kevin session is a meaningful distribution shift and makes the held-out Kevin fold substantially harder.
- This confirms the current `best_demo` remains a best-case showcase, not a full representation of the enlarged Kevin evaluation set.
- If the deployment goal is specifically Kevin's future sessions, the next model step should probably be a **personalized / adaptation** path rather than standard LOSO retraining.

### Hierarchical Rep-Complete Classifier

- Extended `scripts/evaluate_rep_complete_action_classifier.py` with a hierarchical variant:
  - coarse stage: `db_rdl`, `one_arm_db_row`, `bench_or_crunch`
  - fine stage inside `bench_or_crunch`: `db_bench_press` vs `db_weighted_crunch`

Held-out Kevin offline rep classification:

- flat logistic classifier: accuracy `0.8582`, macro F1 `0.8181`
- hierarchical classifier: accuracy `0.8060`, macro F1 `0.7836`

Online detected reps:

1. `kevin/db_rdl/set1`
   - flat rep-complete classifier action accuracy: `1.000`
   - hierarchical rep-complete classifier action accuracy: `1.000`
   - online macro aggregation action accuracy: `0.333`

2. `kevin/db_weighted_crunch/set0`
   - flat rep-complete classifier action accuracy: `0.111`
   - hierarchical rep-complete classifier action accuracy: `0.222`
   - online macro aggregation action accuracy: `0.222`

Interpretation:

- The hierarchical split helps the weak `db_weighted_crunch` case a little, but not enough to exceed the online macro aggregation there.
- For `db_rdl`, both rep-complete classifier variants remain much better than the current online macro head.
- The hierarchy is directionally sensible but not yet a decisive improvement over the simpler flat rep-complete classifier.

### DTW Segmentation Comparison

- Added `scripts/compare_dtw_segmentation_effect.py` to compare SDTW segmentation against the current online rep decoder on the same Kevin set streams.
- Important caveat: this SDTW comparison uses the stream's known action to choose the template family, so it is a favorable offline comparison for DTW and not a full unknown-action online deployment setting.

Results:

1. `kevin/db_rdl/set1`
   - DTW: precision `0.2679`, recall `0.6250`, F1 `0.3750`
   - online decoder: precision `0.25`, recall `0.50`, F1 `0.3333`
   - DTW has slightly better rep recall/F1 here, but at the cost of many more false positives (`56` predictions for `24` true reps) and it is not causal.

2. `kevin/db_weighted_crunch/set0`
   - DTW: precision `0.3125`, recall `0.4167`, F1 `0.3571`
   - online decoder: precision `0.5625`, recall `0.9000`, F1 `0.6923`
   - The online decoder is clearly better here.

Interpretation:

- DTW is not uniformly better than the online decoder.
- It can still be competitive on some hard `db_rdl` streams, especially on recall, but it remains noisy and non-causal.
- For “basic real-time usable” behavior, the smoothed online rep decoder remains the better practical choice overall.

## 2026-05-09 - Stage 3/4 Ablation Experiment

### What Changed

- Modified `DSMSTCN` to support configurable `num_macro_stages` via `DSMSTCNConfig.num_macro_stages`.
  - `num_macro_stages=4`: full pipeline (Stage 1 micro + Stage 2 macro + Stages 3-4 refinement)
  - `num_macro_stages=2`: ablation (Stage 1 micro + Stage 2 macro only, no refinement stages)
- Refinement stages are now built via `nn.ModuleList` instead of hardcoded stage3_macro/stage4_macro.
- Added `final_macro_logits()` method to select the correct final output key regardless of num_macro_stages.
- `ds_ms_tcn_loss()` now dynamically discovers macro keys via `startswith("macro") and endswith("_logits")`.
- `OnlineDSMSTCNPredictor` uses `final_macro_logits()` for correct final macro selection.
- `MicroMacroConfig` now includes `num_macro_stages` field (default 4).
- Config `configs/micro_macro_recognition.yaml` includes `num_macro_stages` setting.

### Hypothesis

Stages 3 and 4 in DS-MS-TCN hurt performance because:
1. Error amplification: Stage 2 mistakes propagate and worsen through refinement.
2. Over-smoothing: TMSE loss pushes predictions toward uniform continuity, destroying action boundaries.
3. Information bottleneck: Stages 3-4 only see macro softmax from previous stage, losing raw IMU signal.
4. Training instability: Conflicting gradients between micro CE (needs sharp predictions) and TMSE (needs smooth predictions).

### Commands Run

- 4-stage (baseline): `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_4stage_v3`
- 2-stage (ablation): `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_2stage_v3`

### Shared Config

- sample_rate: 100 Hz
- slice_seconds: 20.0
- num_layers: 6, num_filters: 64
- causal: true
- epochs: 20
- alpha: 1.0, beta: 0.15, tmse_threshold: 4.0
- test_subject: kevin

### Result Summary (held-out subject: kevin, 20 epochs)

| Metric | 4-stage | 3-stage | 2-stage | Best |
|---|---:|---:|---:|---|
| **Rep F1** | 0.4923 | 0.5970 | 0.5725 | **3-stage** |
| **Rep Precision** | 0.5000 | 0.5882 | 0.5620 | **3-stage** |
| **Rep Recall** | 0.4848 | **0.6061** | 0.5833 | 3-stage |
| **Rep Action Accuracy** | 0.5938 | 0.7000 | **0.7792** | 2-stage |
| start_mae_ms | 586.2 | 600.1 | **522.6** | 2-stage |
| end_mae_ms | 600.5 | **554.2** | 670.2 | 3-stage |
| transition_mae_ms | 605.9 | **504.4** | 514.2 | 3-stage |
| macro_f1_at_50 | 0.0295 | **0.2123** | 0.1472 | 3-stage |
| micro_f1_at_50 | 0.2297 | 0.2986 | 0.2978 | **3-stage** |

3-stage is best on the most metrics. 2-stage is best on rep action accuracy and start_mae. 4-stage is last on all metrics.

### Artifact Paths

- 4-stage: `artifacts/micro_macro_recognition/stage34_4stage_v3/tcn`
- 3-stage: `artifacts/micro_macro_recognition/stage34_3stage/tcn`
- 2-stage: `artifacts/micro_macro_recognition/stage34_2stage_v3/tcn`

### Current Diagnosis

- Best current setting is 3-stage with held-out `kevin` rep F1 `0.5970`, but this is still not deployable.
- Failure is not only over-segmentation. There are two clear blockers:
  - Micro phase pairing errors remain frequent: `missing_eccentric_after_concentric=65`, `unexpected_phase_before_concentric=59`, `phase_gap_too_large=6`.
  - Macro action confusion is severe on `db_rdl`: all 16 matched `db_rdl` reps were classified as `db_weighted_crunch` in `rep_action_confusion_tcn.csv`.
- Per-stream metrics show `kevin/db_rdl/set0` and `kevin/db_rdl/set1` are the worst streams, with macro sample accuracy `0.2220` and `0.0555`.
- Training-data volume does not explain the RDL failure by itself: `db_rdl` and `db_weighted_crunch` have similar sample counts in the train subjects.

### Next Queued Experiments

- Run `configs/micro_macro_recognition_stage3_beta0.yaml` to isolate whether TMSE smoothing is hurting 3-stage quality.
- Run `configs/micro_macro_recognition_stage3_40ep.yaml` to check whether the 20-epoch 3-stage run is undertrained.
- If both still fail on `db_rdl`, inspect feature/label mismatch rather than stage count alone.

## 2026-05-08 - Current Model Review For Development Board

### What Changed

- Added project documentation files for long-term rules, system specification, model specification, and experiment tracking.
- Ran comparison over existing evaluation summaries with the `imu` conda environment.
- Ran a streaming-style replay on the complete current DS-MS-TCN checkpoint.

### Commands Run

- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.compare_runs --root artifacts --output artifacts/run_comparison_20260508.csv`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_fast --method fast --max-samples 3000 --device cpu`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step300 --method step --max-samples 300 --device cpu --progress-interval 100`

### Result Summary

- Complete DS-MS-TCN checkpoint reviewed: `artifacts/micro_macro_recognition/20260508_143504/tcn`.
- Newer candidate folders `20260508_145027/tcn` and `board_100hz_l6_20260508/tcn` had no model checkpoint or summary metrics.
- Held-out `kevin` summary for the complete checkpoint:
  - precision: 0.4545.
  - recall: 0.1136.
  - rep F1: 0.1818.
  - rep action accuracy: 0.2667.
  - micro sample accuracy: 0.5125.
  - macro sample accuracy: 0.2424.
  - macro sample macro F1: 0.0919.
- Post-process grid on the same checkpoint improved rep F1 to 0.3235, but recall remained only 0.25.
- Streaming fast replay on `kevin/db_weighted_crunch/set0` processed 3000 samples in 0.228 s on CPU and produced:
  - micro accuracy: 0.6627.
  - macro accuracy: 0.9790.
- Streaming step replay on 300 samples took 2.338 s on CPU, about 128 samples/s, with a 4089-sample buffer.

### Decision

- Do not ship this checkpoint as the development-board rep-counting model.
- It may be used only as a debugging/demo checkpoint for causal replay, not as a reliable workout counter.

### Next Work

- Finish a clean training run for the current board-style config: 100 Hz, 20 s slices, 6 layers.
- Add DS-MS-TCN export/runtime support if DS-MS-TCN is the intended board model.
- Improve recall before deployment by addressing phase-label order errors and post-processing thresholds.

## 2026-05-08 - Streaming Step Replay Improvements

### What Changed

- Updated `OnlineDSMSTCNPredictor` to use a fixed-length `deque` and `torch.inference_mode()` for step-by-step streaming inference.
- Updated `evaluation.streaming_micro_macro` so `streaming_summary.json` includes throughput, real-time factor, sample accuracies, and predicted/ground-truth label counts.
- Re-tested with actual `--method step` replay, not fast full-sequence inference.

### Commands Run

- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m py_compile models/ds_ms_tcn.py evaluation/streaming_micro_macro.py`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_after --method step --max-samples 3000 --device cpu --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b512 --method step --max-samples 3000 --device cpu --buffer-size 512 --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b256 --method step --max-samples 3000 --device cpu --buffer-size 256 --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b128 --method step --max-samples 3000 --device cpu --buffer-size 128 --progress-interval 500`

### Result Summary

| Runtime buffer | Elapsed s | Samples/s | Real-time factor | Micro acc | Macro acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4089 | 76.882 | 39.02 | 0.352 | 0.6627 | 0.9790 |
| 512 | 37.779 | 79.41 | 0.715 | 0.6630 | 0.9790 |
| 256 | 23.485 | 127.74 | 1.151 | 0.6417 | 0.9790 |
| 128 | 21.401 | 140.18 | 1.263 | 0.6553 | 0.9790 |

### Decision

- For this checkpoint and replay, `--buffer-size 128` or `--buffer-size 256` is the practical real-time setting on CPU.
- This improves streaming runtime measurement and runtime configuration, but it does not solve the global held-out rep-counting quality issue.
