# 2026-05-14 - Rep-Cutting Model Architecture And Gap Analysis

## Purpose

Summarize the current rep-cutting model families in this repo, compare their
practical segmentation pipelines, analyze theoretical difficulty, and explain
why theory and empirical results diverge.

## Shared Downstream Decoder

Most sequence-model baselines in this repo do **not** directly predict reps.
They share a common downstream pipeline:

1. predict sample-level phase labels or probabilities
2. collapse them into `SegmentRun`s
3. pair `concentric -> eccentric` into one rep
4. compute rep metrics and segment IoU metrics

Code references:

- `scripts/compare_baselines.py:328-397`
- `preprocessing/micro_macro_segments.py:144-218`

This means many model differences come from **how well they predict stable phase
runs**, not from a fundamentally different rep decoder.

## Model Families

### 1. Sliding-window Random Forest

Pipeline:

1. extract handcrafted statistical features from a fixed window
2. classify the window as `other / concentric / eccentric`
3. project window probabilities back to the sample axis
4. run common rep pairing

Code:

- feature extraction: `scripts/compare_baselines.py:102-118`
- training: `scripts/compare_baselines.py:121-144`
- prediction: `scripts/compare_baselines.py:147-173`

Key property:

- phase classifier first, rep decoder second

### 2. BiLSTM

Pipeline:

1. sequence -> bidirectional LSTM
2. per-sample phase logits
3. optional smoothing
4. common rep pairing

Code:

- `scripts/compare_baselines.py:167-180`

Key property:

- strongest access to past + future context
- not causal

### 3. Simple 1D CNN

Pipeline:

1. temporal Conv1d stack
2. per-sample phase logits
3. common rep pairing

Code:

- `scripts/compare_baselines.py:183-202`

Key property:

- sequence model, but still just a phase classifier

### 4. Phase-only TCN

Pipeline:

1. dilated temporal convolutions predict sample-level phase
2. smoothing
3. common rep pairing

Code:

- wrapper: `scripts/compare_baselines.py:205-230`
- backbone: `models/ds_ms_tcn.py:53-117`

Key property:

- cleaner than DS-MS-TCN because it removes action modeling

### 5. DS-MS-TCN

Pipeline:

1. Stage 1 predicts micro phase
2. later stages predict/refine macro action labels
3. rep extraction still depends on fixed phase ordering

Code:

- `models/ds_ms_tcn.py:120-239`

Key property:

- multi-task and phase-first
- if phase head collapses, the whole rep pipeline fails

### 6. Plain SDTW

Pipeline:

1. build templates from exemplar repetitions
2. perform DTW/SDTW matching on motion features
3. output candidate rep boundaries directly

Code:

- `preprocessing/sdtw_rep_segmentation.py`
- `evaluation/rep_segmentation.py:1-44`

Key property:

- explicit proposal method
- not a dense phase model

### 7. Hybrid SDTW

Pipeline:

1. SDTW candidate generation
2. classifier filters candidates
3. boundary refiner nudges start/end

Code:

- `train/hybrid_rep_segmentation.py:1-25`
- `README.md:275-315`

Key property:

- closest existing repo pipeline to proposal + refine

## Has The Repo Tested A TCN That Directly Learns Boundaries Instead Of Phase?

### Short answer

- **Not as a mature mainline baseline before this session.**

### What existed before

- The main TCN baselines were phase predictors:
  - phase-only causal TCN
  - phase-only non-causal TCN
- DS-MS-TCN also remained phase-first, even when multi-tasked with macro labels.

### What was added in this session

- `scripts/train_boundary_tcn.py`
  - a **boundary-aware phase TCN**
  - predicts phase plus an auxiliary boundary head
  - still not a pure direct-rep detector

### What does not count as a direct boundary-only TCN

- `use_rep_count_head` in `DSMSTCNConfig`
  - this adds a rep-count regression head
  - it does **not** replace phase decoding with direct boundary prediction
  - it is not a full boundary detector benchmark in current results

### Conclusion on this question

- **A true boundary-only TCN / TCN that directly emits rep boundaries has not yet
  been fully benchmarked in this repo.**
- What has been tested so far is:
  - phase-only TCN
  - boundary-aware phase TCN
  - not a pure boundary-event TCN

## Theoretical Difficulty

### Easiest in principle

1. proposal-based rep detectors
   - direct segment proposal is closer to the final objective
   - examples: SDTW / RecoFit / ExerSense style methods

### Medium

2. phase-only sequence models
   - still indirect because they solve phase first, rep second
   - examples: RF / BiLSTM / TCN

### Hardest

3. multi-task phase + action + rep systems
   - examples: DS-MS-TCN
   - hardest optimization problem and easiest to misalign with boundary quality

## Why Theory And Practice Diverged Here

### 1. More sophisticated sequence models did not beat RF on held-out IoU

Held-out `yushuan` comparison:

- RF: `rep_f1 = 0.7726`, `micro_f1@50 = 0.6348`
- BiLSTM: `rep_f1 = 0.7809`, `micro_f1@50 = 0.5021`
- phase-only non-causal TCN: `rep_f1 = 0.7661`, `micro_f1@50 = 0.5670`

Interpretation:

- BiLSTM is slightly better at finding reps overall
- RF is better at keeping boundaries tighter

### 2. TCN likely smooths labels better than it sharpens boundaries

- TCNs model temporal context well
- but the current objective still mainly teaches dense phase classification
- that can improve stability while still leaving rep boundaries too soft

### 3. DS-MS-TCN is overburdened for this task

- It tries to optimize phase, action semantics, and macro refinement together
- your primary task only cares about rep boundaries first
- therefore optimization objectives are misaligned with the main metric

### 4. Plain SDTW is too rigid

- proposal-first is the right general idea
- but exemplar/template matching alone is too brittle on this dataset
- that is why SDTW is theoretically appealing but empirically weak here

## Current Best Reading

- For this dataset, the main difficulty is no longer “does the model know a rep
  exists?” but “can it place the rep start/end tightly enough?”
- RF is currently the best coarse phase detector.
- The remaining gap is boundary refinement.

## Practical Conclusion

The current most justified next step is not another generic TCN replacement.
It is one of:

1. RF + boundary refiner
2. a true boundary-event model that predicts start/end/transition directly
3. only after that, a pure boundary-only TCN benchmark if needed
