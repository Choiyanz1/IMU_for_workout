# 2026-05-10 Agent Handoff

## Purpose

This note is the quickest way for a new agent to understand the current state
of the project, the main findings from the recent micro/macro experiments, and
where to continue.

## Project Snapshot

- Task focus right now: causal DS-MS-TCN for:
  - micro phase prediction (`other`, `concentric`, `eccentric`)
  - rep detection from paired micro phases
  - macro action assignment on top of the rep/phase stream
- Current common action set for the active micro/macro work:
  - `db_bench_press`
  - `db_rdl`
  - `db_weighted_crunch`
  - `one_arm_db_row`
- Current dataset folders under `datasets/raw_data/`:
  - `1000/`
  - `kevin/`
  - `kevin0509workout/`
  - `thomas/`
  - `thomas0506workout/`
  - `thomas_2/`

## Critical Data-Split Rule

- `kevin0509workout` appears to be a new session from the same human as
  `kevin`.
- For strict subject-wise evaluation, it must be treated as the same held-out
  person, not as an extra train subject.
- Subject alias support was added for this purpose.

Relevant config example:

- `configs/micro_macro_recognition_stage3_40ep_alias_kevin0509.yaml`
- `configs/micro_macro_recognition_stage3_40ep_test1000_alias_kevin0509.yaml`

## Most Important Findings

### 1. Stage count and TMSE are not the main issue

From the earlier stage ablation:

- `4-stage < 2-stage < 3-stage`
- Best original setting before later improvements: `3-stage`

But:

- `beta=0` (remove TMSE) hurt badly.
- `40 epochs` alone did not beat the original best `3-stage` run.

Conclusion:

- The main bottleneck is not simply stage count or TMSE.

### 2. The strongest immediate lever was causal micro smoothing

Current best practical offline result:

- Base checkpoint: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
- Best evaluation/runtime setting: `micro_smoothing_window = 15`
- Re-evaluated output:
  - `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15`

Key metrics:

- Rep F1: `0.7789`
- Precision: `0.7255`
- Recall: `0.8409`
- Micro IoU-F1@50: `0.4582`

Interpretation:

- The model's framewise micro probabilities are noisy.
- A simple causal moving average cleans them up substantially.
- This improves both samplewise/IoU metrics and rep-level quality.

### 3. Macro action quality is weak because the macro path is indirect

Important architecture fact:

- The macro branch does **not** directly consume raw IMU.
- It consumes micro probabilities from Stage 1.

So the macro head is not a direct action classifier. It is trying to infer
action from a noisy phase stream.

That is why:

- rep counting can become usable before action classification does
- sample-level macro labels are much weaker than one might expect from a normal
  action classifier

### 4. Rep-complete action classification is the right direction, but not finished

Using rich rep-level features from the rep-complete segment is more sensible
than trusting sample-level macro predictions.

Current prototype:

- `scripts/evaluate_rep_complete_action_classifier.py`

Key observations:

- For `db_rdl`, rep-complete classification is clearly better than online macro
  aggregation.
- For `db_weighted_crunch`, it is still weak and not reliably better.
- A hierarchical classifier was tried but did not clearly beat the flat
  rep-complete classifier overall.

### 5. DTW is not the current main path

DTW/SDTW was compared on a couple of streams, but it is not the current best
overall deployment direction.

- It can be competitive on some hard `db_rdl` cases.
- It is not uniformly better.
- It is not the main thing to optimize next unless the task explicitly shifts
  back to DTW-based rep boundary proposals.

The user explicitly de-prioritized DTW comparison at the end of the recent
conversation, so do not make that the default next step.

## True Live / Streaming Status

### Important distinction

- `streaming_replay.html` is **post-hoc visualization** of already produced
  outputs.
- True sample-by-sample live mode is in:
  - `evaluation/streaming_micro_macro.py --live`

### Current live path

The current live path now includes:

- `micro_smoothing_window`
- `OnlineRepDecoder`
- rep-complete event emission
- rep-level metrics and emit delay summaries

Files added / modified for this:

- `models/ds_ms_tcn.py`
- `evaluation/streaming_micro_macro.py`
- `preprocessing/micro_macro_segments.py`

### Best live showcase bundle

Single entrypoint:

- `artifacts/best_demo/index.html`

Important:

- This is a **best-case showcase**, not the hardest latest benchmark summary.

Useful live launcher scripts:

- `scripts/run_live_demo_weighted_crunch_set0.cmd`
- `scripts/run_live_demo_rdl_set1.cmd`

### Verified live result from the older Kevin stream

For `kevin/db_weighted_crunch/set0` with the original best checkpoint:

- output: `artifacts/best_demo/live_kevin_db_weighted_crunch_set0/`
- real-time factor: about `0.95x`
- online rep F1: `0.6923`
- mean emit delay: about `196 ms`

Interpretation:

- Rep/phase tracking is close to basically usable.
- Action labels are still much less reliable than rep events.

## New Kevin Session (`kevin0509workout`) Findings

### Strict LOSO-style interpretation

If `kevin0509workout` is the same human as `kevin`, then in a strict subject
split it belongs in the Kevin test fold.

Re-evaluating the old best checkpoint under that enlarged Kevin fold gives a
much harder benchmark:

- output:
  - `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15_kevin0509alias`
- metrics:
  - Rep F1: `0.5939`
  - Recall: `0.6304`
  - Micro IoU-F1@50: `0.3109`

Interpretation:

- The new Kevin session is a meaningful distribution shift.

### Personalized / deployment-style interpretation

If the goal is specifically to support Kevin's future sessions, then it is also
reasonable to hold out someone else (for example `1000`) and allow Kevin's new
session into training.

That model was trained here:

- `artifacts/micro_macro_recognition/stage34_3stage_40ep_test1000_alias_kevin0509/tcn`

Streaming on `kevin0509workout` with that model shows:

- `db_bench_press/set0`: improved and fairly usable
- `db_rdl/set0`: still poor rep detection
- `db_weighted_crunch/set0`: still poor rep detection
- `one_arm_db_row/set1`: still poor rep detection

Interpretation:

- Simply adding the new Kevin session to training is not enough to fix all
  actions.
- It helps some action identity patterns, but rep detection remains the real
  bottleneck for several actions.

## What Is Actually Working Best Right Now

If the goal is **best current practical behavior**:

1. Use the `stage34_3stage_40ep/tcn` checkpoint.
2. Apply `micro_smoothing_window = 15`.
3. Use the online rep decoder for:
   - phase tracking
   - rep complete events
   - rep counting
4. Treat action identity as a secondary / weaker output.

If the goal is **best strict benchmark honesty** for Kevin after the new
session:

1. Alias `kevin0509workout -> kevin`
2. Re-evaluate on the enlarged Kevin fold
3. Expect lower metrics than the showcase demo

If the goal is **best deployment behavior for Kevin specifically**:

1. Allow Kevin's known sessions into train
2. Hold out someone else like `1000`
3. Expect improvement for some actions but not all

## Recommended Next Steps For A New Agent

Pick one of these paths and stay focused:

### Path A: Make the current real-time rep counter more robust

- Batch-evaluate live/step rep metrics across more Kevin streams.
- Identify which action families are already near-usable.
- Improve rep decoder stability for the weak actions:
  - `db_rdl`
  - `db_weighted_crunch`
  - `one_arm_db_row`

### Path B: Improve rep-complete action identity

- Keep rep detection as the first stage.
- Continue improving rep-level action classification, especially for
  `db_weighted_crunch`.
- Consider a hybrid routing scheme only if it clearly improves live action
  labels without hurting the safer current path.

### Path C: Build a personalized Kevin model on purpose

- Stop treating Kevin as a headline held-out subject.
- Retrain and evaluate with a deployment mindset.
- Use `1000` or another subject as held-out instead.

## Files Most Worth Reading First

1. `docs/dev-log.md`
2. `docs/experiments/2026-05-09-stage-ablation-and-handoff.md`
3. `artifacts/best_demo/README.md`
4. `artifacts/best_demo/index.html`
5. `artifacts/rep_complete_action_compare.json`
6. `artifacts/rep_complete_action_compare_test1000_kevin0509.json`

## Minimal Repro Commands

### Best offline evaluation

```bash
C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe scripts\reevaluate_micro_macro_run.py --run-dir artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn --micro-smoothing-window 15 --output-dir artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15 --device cuda
```

### True live sample-by-sample demo

```bash
scripts\run_live_demo_weighted_crunch_set0.cmd
```

### Personalized-style retraining with `1000` held out

```bash
scripts\run_stage34_3stage_40ep_test1000_alias_kevin0509_gpu.cmd
```
